#!/usr/bin/env python3
"""
Step 2 of the plan: check ptq_decode against the published F16 GGUF.

The F16 file is 53.8 GB, so only its header and the requested tensors are
fetched with HTTP range requests.  Both GGUFs are in the same folded
Hadamard basis, so the decoded PTQ1_0 matrix must equal the F16 matrix
directly (no unfolding).

  python tools/compare_f16.py --gguf model/Ternary-Bonsai-2-27B-PTQ1_0.gguf \
      blk.0.ffn_down.weight blk.0.ssm_out.weight blk.3.attn_output.weight
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import numpy as np

from ptq_decode import Gguf, decode_tensor

F16_URL = ("https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf/"
           "resolve/main/Ternary-Bonsai-2-27B-F16.gguf")


def fetch(url: str, start: int, nbytes: int, out: Path) -> Path:
    if out.exists() and out.stat().st_size == nbytes:
        return out
    subprocess.run(["curl", "-sfL", "--retry", "5", "-r", f"{start}-{start + nbytes - 1}",
                    "-o", str(out), url], check=True)
    if out.stat().st_size != nbytes:
        raise SystemExit(f"{out}: got {out.stat().st_size} bytes, want {nbytes}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--gguf", required=True, help="PTQ1_0/PQ2_0 GGUF")
    ap.add_argument("--cache", default="f16_parts", help="where fetched bytes are kept")
    ap.add_argument("--url", default=F16_URL)
    ap.add_argument("tensors", nargs="+")
    args = ap.parse_args()

    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    header = fetch(args.url, 0, 32 << 20, cache / "header.bin")
    ref = Gguf(header)
    ptq = Gguf(args.gguf)

    for name in args.tensors:
        t = ref.tensors[name]
        if t["type"] != 1:
            raise SystemExit(f"{name}: F16 file stores type {t['type']}, expected F16")
        shape = ref.shape_out_in(name)
        nbytes = shape[0] * shape[1] * 2
        part = fetch(args.url, ref.data_start + t["offset"], nbytes, cache / f"{name}.f16")
        w_ref = np.fromfile(part, dtype="<f2").reshape(shape).astype(np.float32)
        w_ptq = decode_tensor(ptq, name)

        rel = float(np.linalg.norm(w_ptq - w_ref) / np.linalg.norm(w_ref))
        maxabs = float(np.abs(w_ptq - w_ref).max())
        exact = float((w_ptq == w_ref).mean())
        print(f"{name:<28} shape={shape} rel_err={rel:.3e} max_abs={maxabs:.3e} "
              f"exact_frac={exact:.6f}")


if __name__ == "__main__":
    main()
