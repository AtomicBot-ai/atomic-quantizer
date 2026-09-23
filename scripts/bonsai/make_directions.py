#!/usr/bin/env python3
"""
Estimate own refusal directions on the PTQ1_0 pack (difference of means).

  d[L] = mean(act_harmful[L]) - mean(act_harmless[L]),  normalized

act[L] is the residual stream at the last prompt token (end of the chat
template generation prompt, thinking off), row 0 = embedding output and
row L = input of block L, so the file is [65, 5120] and --layer 38 in
make_lora.py picks row 38.

Train splits estimate the direction; test splits are held out and only used
to score each row (how well projection on d separates harmful from
harmless) and later for A/B / sweep prompts.

  python tools/make_directions.py --out directions/own.safetensors
"""
from __future__ import annotations

import argparse
import json
import subprocess
import urllib.request
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from safetensors.numpy import save_file

from ab_run import MODEL, ROOT, SERVER, post, wait_ready
from make_lora import load_direction, normalize

COLLECT = ROOT / "tools/collect_acts"
DATA = ROOT / "data"


def read_split(name: str) -> list[str]:
    return pq.read_table(DATA / f"{name}.parquet").column("text").to_pylist()


def sample(xs: list[str], n: int, seed: int) -> list[str]:
    if n >= len(xs):
        return list(xs)
    idx = np.random.default_rng(seed).choice(len(xs), n, replace=False)
    return [xs[i] for i in sorted(idx)]


def format_all(sets: dict[str, list[str]], system: str, port: int, work: Path) -> None:
    cmd = [str(SERVER), "-m", str(MODEL), "-ngl", "99", "-c", "4096", "-np", "1",
           "--port", str(port), "--jinja", "--no-webui"]
    log = (work / "format-server.log").open("w")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    base = f"http://127.0.0.1:{port}"
    try:
        wait_ready(base, proc)
        for name, prompts in sets.items():
            out = []
            for p in prompts:
                r = post(f"{base}/apply-template", {
                    "messages": [{"role": "system", "content": system},
                                 {"role": "user", "content": p}],
                    "chat_template_kwargs": {"enable_thinking": False},
                })
                if "\x1e" in r["prompt"]:
                    raise SystemExit("prompt contains the 0x1e separator")
                out.append(r["prompt"])
            (work / f"{name}.rs").write_text("\x1e".join(out))
            print(f"[format] {name}: {len(out)} prompts")
        print("[format] example tail:", repr(out[-1][-80:]))
    finally:
        proc.terminate()
        proc.wait(timeout=30)


def collect(name: str, n: int, work: Path) -> np.ndarray:
    path = work / f"{name}.f32"
    if not path.exists():
        subprocess.run([str(COLLECT), "-m", str(MODEL), "-i", str(work / f"{name}.rs"),
                        "-o", str(path)], check=True)
    acts = np.fromfile(path, dtype=np.float32)
    return acts.reshape(n, -1, 5120)


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    s = np.concatenate([pos, neg])
    ranks = s.argsort().argsort() + 1
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--out", type=Path, default=ROOT / "directions/own.safetensors")
    ap.add_argument("--work", type=Path, default=ROOT / "acts")
    ap.add_argument("--n-train", type=int, default=416, help="prompts per class for estimation")
    ap.add_argument("--n-test", type=int, default=104, help="held-out prompts per class")
    ap.add_argument("--system", default="You are a helpful assistant")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--ref", type=Path, default=ROOT / "orca/directions/refusal_dir.safetensors")
    args = ap.parse_args()

    args.work.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sets = {
        "train_harmful":  sample(read_split("harmful_behaviors_train"), args.n_train, args.seed),
        "train_harmless": sample(read_split("harmless_alpaca_train"), args.n_train, args.seed),
        "test_harmful":   sample(read_split("harmful_behaviors_test"), args.n_test, args.seed),
        "test_harmless":  sample(read_split("harmless_alpaca_test"), args.n_test, args.seed),
    }
    prompts_dir = ROOT / "prompts"
    prompts_dir.mkdir(exist_ok=True)
    for name, xs in sets.items():
        with (prompts_dir / f"{name}.jsonl").open("w") as f:
            for x in xs:
                f.write(json.dumps({"prompt": x}, ensure_ascii=False) + "\n")

    if not all((args.work / f"{n}.rs").exists() for n in sets):
        format_all(sets, args.system, args.port, args.work)
    acts = {n: collect(n, len(xs), args.work) for n, xs in sets.items()}

    mu_b = acts["train_harmless"].mean(0)                                   # [65, 5120]
    diff = acts["train_harmful"].mean(0) - mu_b
    # projected variant: drop the part of d along the harmless mean, so ablation
    # (projection -> 0) leaves an average harmless prompt where it was
    mu_hat = mu_b / np.linalg.norm(mu_b, axis=1, keepdims=True)
    variants = {
        args.out: ("plain", diff),
        args.out.with_name(args.out.stem + "_proj" + args.out.suffix):
            ("projected on harmless mean", diff - (diff * mu_hat).sum(1, keepdims=True) * mu_hat),
    }
    ref = normalize(load_direction(args.ref))

    for path, (kind, raw) in variants.items():
        gap = np.linalg.norm(raw, axis=1)
        # row 0 is the embedding of the same last template token for every prompt: zero gap
        dirs = np.divide(raw, gap[:, None], out=np.zeros_like(raw), where=gap[:, None] > 0)
        save_file({"direction": dirs.astype(np.float32)}, str(path))

        rows = []
        for L in range(dirs.shape[0]):
            d = dirs[L]
            if gap[L] == 0:
                rows.append({"row": L, "cos_orca": 0.0, "test_auroc": 0.5, "test_cohen_d": 0.0,
                             "zero_at": 0.0})
                continue
            ph, pb = acts["test_harmful"][:, L] @ d, acts["test_harmless"][:, L] @ d
            pooled = np.sqrt((ph.var() + pb.var()) / 2) + 1e-12
            rows.append({"row": L, "cos_orca": float(d @ ref), "test_auroc": auroc(ph, pb),
                         "test_cohen_d": float((ph.mean() - pb.mean()) / pooled),
                         # where ablation puts prompts: 0 = harmless mean, 1 = harmful mean
                         "zero_at": float(-pb.mean() / (ph.mean() - pb.mean()))})
        meta = {"method": f"difference of means, last prompt token, thinking off; {kind}",
                "rows": "0 = embedding output, L = input of block L",
                "system": args.system, "seed": args.seed,
                "n": {k: len(v) for k, v in sets.items()},
                "datasets": ["mlabonne/harmful_behaviors", "mlabonne/harmless_alpaca"],
                "per_row": rows}
        path.with_suffix(".json").write_text(json.dumps(meta, indent=2))

        print(f"\n[done] {path} ({kind}) shape={dirs.shape}")
        print(f"{'row':>4}{'cos_orca':>10}{'auroc':>8}{'cohen_d':>9}{'zero_at':>9}")
        for r in rows:
            if r["row"] % 8 == 0 or r["row"] in (34, 38, 42):
                print(f"{r['row']:>4}{r['cos_orca']:>10.3f}{r['test_auroc']:>8.3f}"
                      f"{r['test_cohen_d']:>9.2f}{r['zero_at']:>9.2f}")


if __name__ == "__main__":
    main()
