#!/usr/bin/env python3
"""
Step 5 of the plan: chain-test a generated adapter.

1. Compare it tensor by tensor with the published OrcaRouter adapter.
   Orca authored A from the unfolded fp16 HF checkpoint, so a ~1e-4 gap is
   the fp16 storage of that checkpoint, not a bug.  For ssm_out, A is also
   compared after the tiled <-> grouped V-head reorder.
2. Simulate the fork graph (build_lora_mm / build_inp_embd) on random
   inputs and measure how much of each residual write is left along r:
   leak = |r . y| / |y|.  Base model leak is ~1/sqrt(5120) = 1.4e-2;
   a correct adapter at scale 1 drives it to float noise (~1e-7).

  python tools/chain_test.py --gguf model/Ternary-Bonsai-2-27B-PTQ1_0.gguf \
      --mine adapters/orca-test.gguf --ref orca/gguf/bonsai-abliterate-lora.gguf \
      --direction orca/directions/refusal_dir.safetensors
"""
from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np
from gguf import GGUFReader

from make_lora import EMBED, gdn_perm, hadamard_signs, load_direction, normalize, wht
from ptq_decode import Gguf, decode_tensor


def load_adapter(path: str) -> tuple[dict, dict]:
    rd = GGUFReader(path)
    kv = {}
    for f in rd.fields.values():
        if f.name.startswith(("general.", "adapter.")) and f.data:
            v = f.parts[f.data[0]]
            kv[f.name] = bytes(v).decode() if v.dtype == np.uint8 else v.tolist()[0]
    t = {x.name: np.asarray(x.data, dtype=np.float32).reshape(-1) for x in rd.tensors}
    return kv, t


def rel(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


def family(name: str) -> str:
    return EMBED if name.startswith(EMBED) else name.split(".", 2)[2].rsplit(".", 1)[0]


def fork_writer_out(g: Gguf, name: str, w: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Base residual write exactly as build_lora_mm builds it: W @ H(s * P x)."""
    width = w.shape[1]
    xm = x
    if ".ssm_out." in name and g.kv.get("prism.hadamard.gdn_v_grouped"):
        n_v = int(g.kv["qwen35.ssm.time_step_rank"])
        n_k = int(g.kv["qwen35.ssm.group_count"])
        # ggml_reshape_4d(hd, nk, rep) + ggml_permute(0, 2, 1, 3)
        xm = x.reshape(n_v // n_k, n_k, width // n_v).transpose(1, 0, 2).reshape(-1)
    return w @ wht(hadamard_signs(g, width) * xm)


def leak(r: np.ndarray, y: np.ndarray) -> float:
    return float(abs(r @ y) / np.linalg.norm(y))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--mine", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--direction", required=True)
    ap.add_argument("--sim", nargs="*", default=[
        "blk.0.ffn_down.weight", "blk.0.ssm_out.weight", "blk.3.attn_output.weight",
        "blk.40.ssm_out.weight", EMBED])
    args = ap.parse_args()

    kv_m, mine = load_adapter(args.mine)
    kv_r, ref = load_adapter(args.ref)
    print(f"[kv] mine={kv_m}")
    print(f"[kv] ref ={kv_r}")
    if set(mine) != set(ref):
        print(f"[names] MISMATCH only-mine={sorted(set(mine) - set(ref))[:5]} "
              f"only-ref={sorted(set(ref) - set(mine))[:5]}")
    common = sorted(set(mine) & set(ref))
    print(f"[names] mine={len(mine)} ref={len(ref)} common={len(common)}")

    g = Gguf(args.gguf)
    to_grouped = {}  # tiled A -> grouped (HF) order, i.e. undo the fork reorder

    stats = defaultdict(list)
    for name in common:
        a, b = mine[name], ref[name]
        if a.shape != b.shape:
            print(f"[shape] {name}: mine {a.shape} ref {b.shape}")
            continue
        fam = family(name)
        stats[(fam, name.rsplit(".", 1)[1], "as-is")].append(rel(a, b))
        if ".ssm_out." in name and name.endswith("lora_a"):
            p = to_grouped.setdefault(a.size, gdn_perm(g, a.size))
            stats[(fam, "lora_a", "mine->grouped")].append(rel(a[p], b))

    print("\n[chain-test] relative error mine vs published, per family")
    print(f"  {'family':<18}{'tensor':<8}{'order':<15}{'n':>4}{'median':>12}{'max':>12}")
    for (fam, kind, order), v in sorted(stats.items()):
        print(f"  {fam:<18}{kind:<8}{order:<15}{len(v):>4}{np.median(v):>12.3e}{max(v):>12.3e}")

    r = normalize(load_direction(args.direction))
    rng = np.random.default_rng(0)
    print("\n[fork-sim] leak = |r.y|/|y| of the residual write, scale 1")
    print(f"  {'tensor':<28}{'base':>11}{'+mine':>11}{'+ref':>11}")
    for name in args.sim:
        if name == EMBED:
            rows, width = g.shape_out_in(EMBED)
            toks = rng.integers(0, rows, 8)
            raw = g.raw(EMBED)
            row_bytes = len(raw) // rows
            w = np.stack([decode_tensor_row(raw, row_bytes, width, t, g.tensors[EMBED]["type"])
                          for t in toks])
            s = hadamard_signs(g, width)
            base = np.stack([s * wht(z) for z in w])  # h = s * H(z)
            res = {"base": base}
            for tag, ad in (("mine", mine), ("ref", ref)):
                A = ad[f"{EMBED}.lora_a"]
                B = ad[f"{EMBED}.lora_b"]
                res[tag] = base + np.outer(A[toks], B)
            vals = [np.mean([leak(r, y) for y in res[k]]) for k in ("base", "mine", "ref")]
        else:
            w = decode_tensor(g, name)
            x = rng.standard_normal(w.shape[1]).astype(np.float32)
            base = fork_writer_out(g, name, w, x)
            vals = [leak(r, base)]
            for ad in (mine, ref):
                A, B = ad[f"{name}.lora_a"], ad[f"{name}.lora_b"]
                vals.append(leak(r, base + B * (A @ x)))
        print(f"  {name:<28}" + "".join(f"{v:>11.2e}" for v in vals))


def decode_tensor_row(raw: bytes, row_bytes: int, width: int, row: int, typ: int) -> np.ndarray:
    from ptq_decode import decode_bytes
    return decode_bytes(raw[row * row_bytes:(row + 1) * row_bytes], (1, width), int(typ))[0]


if __name__ == "__main__":
    main()
