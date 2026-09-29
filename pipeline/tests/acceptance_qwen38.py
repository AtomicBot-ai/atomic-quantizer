#!/usr/bin/env python3
"""Acceptance of the Qwen3.8-27B rehearsal: the rebuilt ladder against the August release.

    acceptance_qwen38.py --metrics AtomicChat/Qwen3.8-27B-GGUF-metrics-rehearsal
    acceptance_qwen38.py --dir runs/<run>/card          # a folder already fetched by `release.py card`
    acceptance_qwen38.py ... --imatrix ours.gguf --imatrix-ref august.gguf

The answer key is the published metrics repo at a pinned revision: its KLD logs
(mean KLD, top-1) and quantize logs (file size). Pass when every rung is within
10 % of the August KLD (or 0.0001 absolute, for Q8_0 where 10 % is noise), within
0.3 points of its top-1 and within 2 % of its size; when every verify log says ok
and the reference measured against itself is exactly zero. Exit 1 otherwise.

August logs are named before the rename to public labels, hence LOG_LABEL.
"""
import argparse
import json
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "lib"))
from conftest import DENSE, fetch  # noqa: E402

import hub  # noqa: E402
import quantlog  # noqa: E402
import results  # noqa: E402

STEM = "Qwen3.8-27B"
RUNGS = ["Q8_0", "AD-Q6_K", "AD-Q6_K-Q5_K", "AD-Q5_K_M", "AD-Q5_K_M-Q4_K_M", "AD-Q4_K_M", "AD-IQ4_XS",
         "AD-IQ4_XS-IQ3_S", "AD-IQ3_S", "AD-IQ3_S-IQ3_XXS", "AD-IQ3_XXS", "AD-IQ2_S", "AD-IQ2_S-IQ2_XS",
         "AD-IQ2_XS", "AD-IQ2_XXS", "AD-IQ1_M"]
LOG_LABEL = {"AD-Q5_K_M": "AD-Q5_K", "AD-Q5_K_M-Q4_K_M": "AD-Q5_K-Q4_K", "AD-Q4_K_M": "AD-Q4_K"}
KLD_REL, KLD_ABS, TOP1_PP, SIZE_REL = 0.10, 0.0001, 0.3, 0.02


def august():
    """label -> {mean_kld, top1_pct, size_bytes} from the published logs."""
    out = {}
    for label in RUNGS:
        old = LOG_LABEL.get(label, label)
        with open(fetch(*DENSE, f"logs/kld-neutral--{STEM}-{old}.log"), errors="replace") as f:
            q = results.parse_kld(f.read())
        size = quantlog.parse_file(fetch(*DENSE, f"logs/quantize-{old}.log"))["quant_mib"] * 2 ** 20
        out[label] = {"mean_kld": q["mean_kld"], "top1_pct": q.get("top1_pct"), "size_bytes": size}
    return out


def compare(rows, ref):
    """Returns (table lines, failures)."""
    ours = {r["label"]: r for r in rows}
    lines, bad = [], []
    lines.append(f"{'rung':18s} {'KLD aug':>9s} {'KLD ours':>9s} {'d%':>6s} {'top1 d':>7s} {'size d%':>8s}")
    for label in RUNGS:
        a = ref[label]
        r = ours.get(label)
        if r is None:
            bad.append(f"{label}: not in results.json")
            lines.append(f"{label:18s} {a['mean_kld']:9.6f} {'-':>9s}")
            continue
        q = r["quality"].get("neutral", {})
        k = q.get("mean_kld")
        if k is None:
            bad.append(f"{label}: no neutral KLD")
            continue
        dk = (k - a["mean_kld"]) / a["mean_kld"]
        dt = (q.get("top1_pct") or 0) - (a["top1_pct"] or 0)
        ds = (r["size_bytes"] - a["size_bytes"]) / a["size_bytes"]
        flags = []
        if abs(dk) > KLD_REL and abs(k - a["mean_kld"]) > KLD_ABS:
            flags.append("KLD")
        if abs(dt) > TOP1_PP:
            flags.append("top1")
        if abs(ds) > SIZE_REL:
            flags.append("size")
        if flags:
            bad.append(f"{label}: {', '.join(flags)} out of tolerance")
        lines.append(f"{label:18s} {a['mean_kld']:9.6f} {k:9.6f} {100 * dk:+6.1f} {dt:+7.2f} {100 * ds:+8.2f}"
                     f"  {'FAIL ' + ','.join(flags) if flags else 'ok'}")
    return lines, bad


def check_logs(d):
    bad = []
    for label in RUNGS:
        p = os.path.join(d, "logs", f"verify-{label}.txt")
        if not os.path.exists(p):
            bad.append(f"{label}: no verify log")
        elif ": ok," not in open(p).read():
            bad.append(f"{label}: verify did not pass")
    p = os.path.join(d, "logs", "kld-selfcheck.log")
    if os.path.exists(p):
        m = re.search(r"Mean\s+KLD:\s+([-0-9.eE+]+)", open(p, errors="replace").read())
        if not m or abs(float(m.group(1))) > 1e-5:
            bad.append(f"self-check KLD is {m.group(1) if m else 'missing'}")
    else:
        bad.append("no kld-selfcheck.log")
    return bad


def imatrix_cosines(ours, ref):
    """Per tensor cosine of mean squared activations (in_sum2 / counts) between two imatrix files."""
    import numpy as np
    from gguf import GGUFReader

    def load(p):
        r = GGUFReader(p)
        t = {x.name: np.asarray(x.data, dtype=np.float64) for x in r.tensors}
        return {n[:-len(".in_sum2")]: v.reshape(-1) / max(float(t.get(n[:-len(".in_sum2")] + ".counts", [1])[0]), 1.0)
                for n, v in t.items() if n.endswith(".in_sum2")}
    a, b = load(ours), load(ref)
    common = sorted(set(a) & set(b))
    cos = {}
    for n in common:
        x, y = a[n], b[n]
        if x.shape != y.shape:
            continue
        den = math.sqrt(float(x @ x) * float(y @ y))
        cos[n] = float(x @ y) / den if den else 0.0
    return cos, len(a), len(b)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--metrics", help="the rehearsal metrics repo (honours LOCAL_HUB)")
    src.add_argument("--dir", help="a folder with results.json and logs/ at their repo paths")
    ap.add_argument("--imatrix")
    ap.add_argument("--imatrix-ref")
    ap.add_argument("--min-cosine", type=float, default=0.99)
    a = ap.parse_args()

    d = a.dir
    if a.metrics:
        d = os.path.join(HERE, ".cache", "acceptance", a.metrics.replace("/", "--"))
        for pat in ("results.json", "logs/verify-*.txt", "logs/kld-selfcheck.log"):
            try:
                hub.get(a.metrics, "dataset", "-", pat, d)
            except FileNotFoundError:
                pass
    with open(os.path.join(d, "results.json")) as f:
        rows = json.load(f)
    lines, bad = compare(rows, august())
    print("\n".join(lines))
    bad += check_logs(d)
    if a.imatrix and a.imatrix_ref:
        cos, na, nb = imatrix_cosines(a.imatrix, a.imatrix_ref)
        low = {n: c for n, c in cos.items() if c < a.min_cosine}
        print(f"\nimatrix: {na} entries ours, {nb} August, {len(cos)} compared, min cosine "
              f"{min(cos.values()) if cos else float('nan'):.4f}")
        for n, c in sorted(low.items(), key=lambda kv: kv[1])[:10]:
            print(f"  {n}: {c:.4f}")
        if low or not cos:
            bad.append(f"imatrix: {len(low)} tensors below cosine {a.min_cosine}")
    print()
    for b in bad:
        print("FAIL", b)
    print("PASS" if not bad else f"{len(bad)} problems")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
