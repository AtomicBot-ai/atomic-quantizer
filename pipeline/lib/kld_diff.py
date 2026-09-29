#!/usr/bin/env python3
"""Paired difference of two llama-perplexity --kl-divergence logs, chunk by chunk.

    kld_diff.py A.log B.log                 # B - A, e.g. A = mask, B = candidate
    kld_diff.py A.log B.log --gb-saved 0.21 # also per GB that B saves against A
    kld_diff.py A.log B.log --json

Both runs score the same tokens against the same BF16 reference, so the
difference is taken per chunk and its spread is what separates two quants.
The interval is over chunks, never over tokens: tokens inside a 4096 window
are correlated, and a per-token interval is several times too narrow (the
lesson of the NVFP4 review). Two estimates are given, the paired standard
error (sd of the per-chunk differences / sqrt(n)) and a percentile bootstrap
that resamples chunks.

llama-perplexity prints only running means. Every chunk scores the same number
of tokens (the second half of the window), so chunk i's own value is
i*m_i - (i-1)*m_(i-1). The running KLD is printed with 5 decimals, so each
recovered chunk carries rounding noise of about sqrt(2)*i*3e-6 (7e-5 at
chunk 30). The mean over chunks telescopes back to the last running mean and
is exact; only the SE is inflated a little, which errs on the careful side.

Refuses to compare logs that differ in chunk count, context or reference
(Mean PPL(base) is a fingerprint of the reference: same tokens, same logits).
"""
import argparse
import json
import math
import random
import re
import sys

# chunk, PPL, ln(PPL(Q)/PPL(base)), KLD, dp RMS, same top p: each "value ± error"
ROW = re.compile(r"^\s*(\d+)\s+([\d.]+)\s+±\s+[\d.]+\s+(-?[\d.]+)\s+±\s+[\d.]+\s+(-?[\d.]+)\s+±\s+[\d.]+\s+"
                 r"([\d.]+)\s+±\s+[\d.]+\s+%\s+([\d.]+)\s+±\s+[\d.]+\s+%")
HEAD = re.compile(r"kl_divergence: computing over (\d+) chunks, n_ctx=(\d+)")
MEAN_KLD = re.compile(r"^Mean\s+KLD:\s+(-?[\d.]+)\s+±\s+([\d.]+)", re.M)
PPL_BASE = re.compile(r"^Mean PPL\(base\)\s+:\s+([\d.]+)", re.M)
SAME_TOP = re.compile(r"^Same top p:\s+([\d.]+)\s+±\s+([\d.]+)", re.M)


class KldLogError(Exception):
    pass


def per_chunk(running):
    return [(i + 1) * v - i * (running[i - 1] if i else 0.0) for i, v in enumerate(running)]


def parse(text, path="<log>"):
    kld, top = [], []
    for line in text.splitlines():
        m = ROW.match(line)
        if m:
            if int(m.group(1)) != len(kld) + 1:
                raise KldLogError(f"{path}: chunk {m.group(1)} out of order")
            kld.append(float(m.group(4)))
            top.append(float(m.group(6)))
    head = HEAD.search(text)
    mean = MEAN_KLD.search(text)
    if not kld or not mean:
        raise KldLogError(f"{path}: not a finished llama-perplexity --kl-divergence log")
    base = PPL_BASE.search(text)
    same = SAME_TOP.search(text)
    return {
        "path": path,
        "chunks": len(kld),
        "n_ctx": int(head.group(2)) if head else None,
        "mean_kld": float(mean.group(1)),
        "mean_kld_se": float(mean.group(2)),
        "top1_pct": float(same.group(1)) if same else None,
        "ppl_base": float(base.group(1)) if base else None,
        "kld": per_chunk(kld),
        "top1": per_chunk(top),
    }


def parse_file(path):
    with open(path, errors="replace") as f:
        return parse(f.read(), path)


def finished(path):
    try:
        parse_file(path)
        return True
    except (OSError, KldLogError):
        return False


def mean(xs):
    return sum(xs) / len(xs)


def se(xs):
    n = len(xs)
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1) / n)


def bootstrap(stat, n, reps, seed):
    rnd = random.Random(seed)
    out = sorted(stat([rnd.randrange(n) for _ in range(n)]) for _ in range(reps))
    return out[int(0.025 * reps)], out[int(0.975 * reps) - 1]


def check_same_bench(a, b):
    if a["chunks"] != b["chunks"]:
        raise KldLogError(f"chunk counts differ: {a['chunks']} vs {b['chunks']}")
    if a["n_ctx"] != b["n_ctx"]:
        raise KldLogError(f"context differs: {a['n_ctx']} vs {b['n_ctx']}")
    if a["ppl_base"] is not None and b["ppl_base"] is not None and a["ppl_base"] != b["ppl_base"]:
        raise KldLogError(f"different references: Mean PPL(base) {a['ppl_base']} vs {b['ppl_base']}")
    if a["chunks"] < 2:
        raise KldLogError("need at least 2 chunks for an interval")


def paired(a, b, reps=10000, seed=0):
    """B - A per chunk. a, b are parse() results on the same bench."""
    check_same_bench(a, b)
    n = a["chunks"]
    d = [y - x for x, y in zip(a["kld"], b["kld"])]
    dt = [y - x for x, y in zip(a["top1"], b["top1"])]
    ka, kb = a["kld"], b["kld"]
    r = {
        "chunks": n, "n_ctx": a["n_ctx"],
        "a": a["path"], "b": b["path"],
        "a_kld": a["mean_kld"], "b_kld": b["mean_kld"],
        "a_top1": a["top1_pct"], "b_top1": b["top1_pct"],
        "d_kld": b["mean_kld"] - a["mean_kld"],
        "d_kld_se": se(d),
        "d_kld_ci95": bootstrap(lambda ix: mean([d[i] for i in ix]), n, reps, seed),
        "rel": b["mean_kld"] / a["mean_kld"] - 1 if a["mean_kld"] else None,
        "rel_ci95": bootstrap(lambda ix: mean([kb[i] for i in ix]) / mean([ka[i] for i in ix]) - 1, n, reps, seed + 1)
        if a["mean_kld"] else None,
        "b_lower_in": sum(1 for x in d if x < 0),
        "d_top1_pp": mean(dt),
        "d_top1_se": se(dt),
        "d_top1_ci95": bootstrap(lambda ix: mean([dt[i] for i in ix]), n, reps, seed + 2),
    }
    r["z"] = r["d_kld"] / r["d_kld_se"] if r["d_kld_se"] else None
    return r


def per_gb(r, gb_saved):
    """d KLD per GB that B saves against A, with the same intervals scaled."""
    if not gb_saved:
        return None
    return {"gb_saved": gb_saved, "per_gb": r["d_kld"] / gb_saved, "per_gb_se": r["d_kld_se"] / abs(gb_saved),
            "per_gb_ci95": tuple(sorted(x / gb_saved for x in r["d_kld_ci95"]))}


def verdict(r, k=2.0):
    """'better' / 'worse' when B is k SE away from A, else 'tie'."""
    if r["d_kld_se"] == 0 or abs(r["d_kld"]) <= k * r["d_kld_se"]:
        return "tie"
    return "better" if r["d_kld"] < 0 else "worse"


def fmt(r):
    lines = [
        f"chunks {r['chunks']} x {r['n_ctx']}   A {r['a_kld']:.6f} ({r['a_top1']}%)   B {r['b_kld']:.6f} ({r['b_top1']}%)",
        f"B - A  dKLD {r['d_kld']:+.6f} ± {r['d_kld_se']:.6f} SE   95% CI [{r['d_kld_ci95'][0]:+.6f}, {r['d_kld_ci95'][1]:+.6f}]"
        f"   z {r['z']:+.1f}   B lower in {r['b_lower_in']}/{r['chunks']} chunks   -> {verdict(r)} (2 SE)"
        if r["z"] is not None else f"B - A  dKLD {r['d_kld']:+.6f} (no spread)",
    ]
    if r["rel"] is not None:
        lines.append(f"       rel {100 * r['rel']:+.1f}%   95% CI [{100 * r['rel_ci95'][0]:+.1f}, {100 * r['rel_ci95'][1]:+.1f}]")
    lines.append(f"       top-1 {r['d_top1_pp']:+.3f} pp ± {r['d_top1_se']:.3f}   95% CI "
                 f"[{r['d_top1_ci95'][0]:+.3f}, {r['d_top1_ci95'][1]:+.3f}]")
    if r.get("per_gb"):
        g = r["per_gb"]
        lines.append(f"       per GB saved ({g['gb_saved']:.3f} GB): {g['per_gb']:+.6f} ± {g['per_gb_se']:.6f}   "
                     f"95% CI [{g['per_gb_ci95'][0]:+.6f}, {g['per_gb_ci95'][1]:+.6f}]")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a", help="reference side of the difference, e.g. the hand mask")
    ap.add_argument("b", help="candidate; the difference is B - A")
    ap.add_argument("--gb-saved", type=float, help="GB (1e9 bytes) B is smaller than A")
    ap.add_argument("--reps", type=int, default=10000)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    try:
        r = paired(parse_file(args.a), parse_file(args.b), reps=args.reps)
    except KldLogError as e:
        print(f"kld_diff: {e}", file=sys.stderr)
        return 2
    r["per_gb"] = per_gb(r, args.gb_saved)
    r["verdict_2se"] = verdict(r)
    print(json.dumps(r, indent=1) if args.json else fmt(r))
    return 0


if __name__ == "__main__":
    sys.exit(main())
