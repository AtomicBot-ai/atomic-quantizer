#!/usr/bin/env python3
"""
Build the rank-1 LoRA adapter for PrismML Ternary Bonsai 2 PTQ1_0/PQ2_0 GGUF.

The base GGUF remains untouched.  For ordinary residual writers:

    W' = W - r (r^T W)
    A  = r^T W
    B  = -r

For token_embd, llama.cpp stores the factors in the opposite order:

    A = W r        # vocab x 1
    B = -r         # hidden x 1

The base GGUF is stored in PrismML's folded Hadamard basis.  The basis
handling follows the PrismML fork (prism-b10709-9a9394a):

  build_lora_mm:   y = W_fold @ H(s * P x)  +  scale * B (A x)
  build_inp_embd:  h = s * H(W_fold[tok])   +  scale * B A[tok]

H is the normalized Sylvester WHT (block 1024), s the explicit sign vector
for the input width, P the tiled->grouped V-head permutation (ssm_out only,
when prism.hadamard.gdn_v_grouped is set).  The LoRA branch gets the raw
activation x, so A must be in the unfolded input basis:

  writers:     A = P^T (s * H(r^T W_fold)),   B = -r
  token_embd:  A = W_fold @ H(s * r),         B = -r

r lives on the output (residual) side, which is never rotated, so r itself
is not transformed.

Typical:
  python make_lora.py \
      --gguf PTQ1_0.gguf \
      --direction directions/refusal_dir.safetensors \
      --out bonsai-abliterate-lora.gguf

Layer-selective:
  python make_lora.py \
      --gguf PTQ1_0.gguf \
      --direction directions.safetensors \
      --layer 38 \
      --layers 15-63 \
      --no-embd \
      --out layer38.gguf
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from ptq_decode import BLOCK_BYTES, Gguf, decode_bytes, decode_tensor


WRITERS = (
    "ffn_down.weight",
    "ssm_out.weight",
    "attn_output.weight",
)
EMBED = "token_embd.weight"


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def load_direction(path: str | Path) -> np.ndarray:
    path = Path(path)
    if path.suffix == ".safetensors":
        try:
            from safetensors.numpy import load_file
        except ImportError as e:
            raise SystemExit(
                "safetensors is required for .safetensors directions: "
                "pip install safetensors"
            ) from e
        obj = load_file(str(path))
        if "direction" in obj:
            arr = obj["direction"]
        elif len(obj) == 1:
            arr = next(iter(obj.values()))
        else:
            raise ValueError(
                f"{path}: expected key 'direction' (or a single tensor), got {list(obj)}"
            )
    elif path.suffix == ".npy":
        arr = np.load(path)
    else:
        raise ValueError("direction must be .safetensors or .npy")

    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim not in (1, 2):
        raise ValueError(f"direction shape {arr.shape}: expected [hidden] or [layers, hidden]")
    return arr


def normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(v))
    if not np.isfinite(n) or n == 0:
        raise ValueError("direction has zero/non-finite norm")
    return v / n


def parse_layers(spec: str | None) -> set[int] | None:
    if not spec:
        return None
    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            lo, hi = int(a), int(b)
            if lo > hi:
                lo, hi = hi, lo
            out.update(range(lo, hi + 1))
        else:
            out.add(int(part))
    return out


def _fwht_rows(x: np.ndarray) -> np.ndarray:
    """Normalized in-place-friendly Sylvester WHT on the last dimension."""
    y = np.asarray(x, dtype=np.float32).copy()
    n = y.shape[-1]
    if n <= 0 or n & (n - 1):
        raise ValueError(f"WHT length must be a power of two, got {n}")

    h = 1
    while h < n:
        step = h * 2
        z = y.reshape(-1, n)
        for start in range(0, n, step):
            a = z[:, start:start + h].copy()
            b = z[:, start + h:start + step].copy()
            z[:, start:start + h] = a + b
            z[:, start + h:start + step] = a - b
        h = step
    return y / math.sqrt(n)


def wht(v: np.ndarray) -> np.ndarray:
    """Blockwise (1024) normalized Sylvester WHT of a 1-D vector."""
    v = np.asarray(v, dtype=np.float32)
    if v.size % 1024:
        raise ValueError(f"width {v.size} is not divisible by WHT block 1024")
    return _fwht_rows(v.reshape(-1, 1024)).reshape(-1)


def gdn_perm(g: Gguf, width: int) -> np.ndarray | None:
    """
    Index map of the fork's tiled [hd, nk, rep] -> grouped [hd, rep, nk] reorder
    (llama-graph.cpp build_lora_mm): (P x)[i] = x[perm[i]].
    """
    if not g.kv.get("prism.hadamard.gdn_v_grouped", False):
        return None
    n_v = int(g.kv["qwen35.ssm.time_step_rank"])
    n_k = int(g.kv["qwen35.ssm.group_count"])
    if n_v % n_k or width % n_v:
        raise ValueError(f"bad GDN head geometry: n_v={n_v} n_k={n_k} width={width}")
    rep, hd = n_v // n_k, width // n_v
    # grouped position (k, r, h) reads tiled index (r, k, h)
    return np.arange(width).reshape(rep, n_k, hd).transpose(1, 0, 2).reshape(-1)


def unfold_input_row(a_fold: np.ndarray, signs: np.ndarray, perm: np.ndarray | None) -> np.ndarray:
    """
    Map a row vector from the folded input basis to the raw activation basis.

    The base matmul sees H(s * P x), so a_fold . H(s * P x) = A . x with
    A = P^T (s * H a_fold).
    """
    a = wht(a_fold) * signs
    if perm is None:
        return a
    out = np.empty_like(a)
    out[perm] = a
    return out


def hadamard_signs(g: Gguf, width: int) -> np.ndarray:
    widths = g.kv.get("prism.hadamard.sign_widths")
    values = g.kv.get("prism.hadamard.sign_values")
    block = int(g.kv.get("prism.hadamard.block_size", 0))
    mode = g.kv.get("prism.hadamard.sign_mode")

    if block != 1024 or mode != "explicit":
        raise ValueError(
            f"unsupported Hadamard metadata: block={block}, sign_mode={mode!r}"
        )
    if not isinstance(widths, list) or not isinstance(values, list):
        raise ValueError("missing prism.hadamard.sign_widths/sign_values")
    if sum(int(x) for x in widths) != len(values):
        raise ValueError("sign_widths does not match sign_values length")

    # one sign vector per exact input width, as in llama-model.cpp
    offset = 0
    for w in widths:
        w = int(w)
        if w == width:
            s = np.asarray(values[offset:offset + w], dtype=np.float32)
            if not np.all(np.isin(s, (-1.0, 1.0))):
                raise ValueError("Hadamard sign_values must be +-1")
            return s
        offset += w
    raise ValueError(f"no Hadamard sign vector for input width {width}")


def embed_lora_a(g: Gguf, r_e: np.ndarray, chunk_rows: int = 16384) -> np.ndarray:
    """W_fold @ r_e for token_embd, decoded in row chunks to bound memory."""
    t = g.tensors[EMBED]
    rows, width = g.shape_out_in(EMBED)
    row_bytes = width // 128 * BLOCK_BYTES[int(t["type"])]
    raw = g.raw(EMBED)
    out = np.empty(rows, dtype=np.float32)
    for lo in range(0, rows, chunk_rows):
        hi = min(rows, lo + chunk_rows)
        w = decode_bytes(raw[lo * row_bytes:hi * row_bytes], (hi - lo, width), int(t["type"]))
        out[lo:hi] = w @ r_e
    return out


def select_direction(direction: np.ndarray, tensor_layer: int | None, fixed_layer: int | None) -> np.ndarray:
    if direction.ndim == 1:
        return normalize(direction)

    n_layers, hidden = direction.shape
    idx = fixed_layer if fixed_layer is not None else tensor_layer
    if idx is None:
        raise ValueError(
            "2-D direction file needs --layer N (fixed direction) or a tensor layer"
        )
    if not (0 <= idx < n_layers):
        raise ValueError(f"direction layer {idx} outside 0..{n_layers - 1}")
    return normalize(direction[idx])


def tensor_layer(name: str) -> int | None:
    if not name.startswith("blk."):
        return None
    parts = name.split(".")
    if len(parts) >= 3 and parts[1].isdigit():
        return int(parts[1])
    return None


def target_names(g: Gguf, layers: set[int] | None, no_embd: bool) -> list[str]:
    out = []
    for name in sorted(g.tensors):
        if name == EMBED:
            if not no_embd:
                out.append(name)
            continue
        if not any(name.endswith(s) for s in WRITERS):
            continue
        layer = tensor_layer(name)
        if layer is None:
            continue
        if layers is not None and layer not in layers:
            continue
        out.append(name)
    return out


def relerr(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


def build(args) -> None:
    g = Gguf(args.gguf)
    direction = load_direction(args.direction)
    layers = parse_layers(args.layers)

    arch = g.kv.get("general.architecture")
    if arch != "qwen35":
        raise SystemExit(f"expected general.architecture=qwen35, got {arch!r}")

    if g.kv.get("prism.hadamard.version") is None:
        raise SystemExit("base GGUF has no prism.hadamard.* metadata")

    targets = target_names(g, layers, args.no_embd)
    if not targets:
        raise SystemExit("no target tensors matched --layers/--no-embd")

    # We intentionally decode one target at a time: a single 17k x 5k matrix
    # is already hundreds of MB in FP32.
    folded = set(g.kv.get("prism.hadamard.weight_names", []))
    inverse = set(g.kv.get("prism.hadamard.inverse_weight_names", []))

    records = []
    for name in targets:
        layer = tensor_layer(name)
        rows, width = g.shape_out_in(name)
        r = select_direction(direction, layer, args.layer)
        signs = hadamard_signs(g, width)

        if name == EMBED:
            if name not in inverse:
                raise SystemExit(f"{name}: not listed in prism.hadamard.inverse_weight_names")
            if r.size != width:
                raise SystemExit(f"{name}: direction has {r.size} dims, hidden is {width}")
            # h = s * H(z)  =>  h . r = z . H(s * r)
            lora_a = embed_lora_a(g, wht(signs * r)).reshape(-1, 1)
        else:
            if name not in folded:
                raise SystemExit(f"{name}: not listed in prism.hadamard.weight_names")
            if r.size != rows:
                raise SystemExit(f"{name}: output rows {rows} != direction size {r.size}")
            perm = gdn_perm(g, width) if ".ssm_out." in name else None
            a_fold = r @ decode_tensor(g, name)
            lora_a = unfold_input_row(a_fold, signs, perm).reshape(1, -1)
        lora_a = lora_a.astype(np.float32)
        lora_b = (-r).reshape(-1, 1).astype(np.float32)

        if not np.isfinite(lora_a).all() or not np.isfinite(lora_b).all():
            raise SystemExit(f"{name}: non-finite LoRA factor")

        records.append((name, lora_a, lora_b))
        print(
            f"[target] {name:<34} W={(rows, width)} "
            f"A={lora_a.shape} B={lora_b.shape} layer={layer}"
        )

    try:
        import gguf
    except ImportError as e:
        raise SystemExit(
            "gguf is required for writing the adapter. "
            "Install the Prism/llama.cpp gguf-py package or `pip install gguf`."
        ) from e

    writer = gguf.GGUFWriter(args.out, "qwen35")
    writer.add_string("general.type", "adapter")
    writer.add_string("adapter.type", "lora")
    writer.add_float32("adapter.lora_alpha", 1.0)

    # provenance: the file alone should say what it was built from
    writer.add_string("general.name", Path(args.out).stem)
    writer.add_string("prism.abliterate.base_gguf", Path(args.gguf).name)
    writer.add_string("prism.abliterate.base_sha256", sha256(args.gguf))
    writer.add_string("prism.abliterate.direction_file", Path(args.direction).name)
    writer.add_string("prism.abliterate.direction_sha256", sha256(args.direction))
    writer.add_string("prism.abliterate.direction_row", "per-layer" if args.layer is None else str(args.layer))
    writer.add_string("prism.abliterate.target_layers", args.layers or "all")
    writer.add_string("prism.abliterate.target_embedding", "no" if args.no_embd else "yes")
    writer.add_string("prism.abliterate.sites", str(len(records)))
    writer.add_string("prism.abliterate.basis",
                      "unfolded input basis; A = P^T (s * H(r^T W_fold)), B = -r")

    for name, a, b in records:
        writer.add_tensor(f"{name}.lora_a", a)
        writer.add_tensor(f"{name}.lora_b", b)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    size = Path(args.out).stat().st_size
    print(f"[done] {args.out}: {size / 1e6:.2f} MB, {2 * len(records)} tensors")
    print(f"[done] residual-writer sites: {len(records)}")
    print("Use with PrismML's llama.cpp fork:")
    print(f"  llama-cli -m {Path(args.gguf).name} --lora {Path(args.out).name}")
    print(f"  llama-cli -m {Path(args.gguf).name} --lora-scaled {Path(args.out).name}:2")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gguf", required=True, help="PTQ1_0/PQ2_0 Bonsai GGUF")
    ap.add_argument("--direction", required=True, help="refusal direction .safetensors/.npy")
    ap.add_argument("--out", default="bonsai-abliterate-lora.gguf")
    ap.add_argument("--layer", type=int, default=None,
                    help="fixed row from a [65,hidden] direction file; does not restrict target layers")
    ap.add_argument("--layers", default=None,
                    help="target block layers, e.g. 15-63 or 34,38,42")
    ap.add_argument("--no-embd", action="store_true",
                    help="do not include token_embd.weight")
    args = ap.parse_args()
    build(args)


if __name__ == "__main__":
    main()
