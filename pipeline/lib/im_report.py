#!/usr/bin/env python3
"""What an importance matrix covers, whether it has converged, where its energy is.

    im_report.py coverage imatrix.gguf [--inventory inventory.json]
    im_report.py converge imatrix-N.gguf imatrix-2N.gguf [--min-cos 0.995]
    im_report.py converge shard-0-of-8.gguf,...,shard-3-of-8.gguf shard-0-of-8.gguf,...,shard-7-of-8.gguf
    im_report.py stats    imatrix.gguf -o imatrix.stats.txt
    im_report.py all      imatrix.gguf [--vs imatrix-N.gguf] [--inventory ...] [--stats-out F] [--json F]

    band_select.py imatrix.stats.txt --inventory ... --profile ... -o bands.json

A llama-imatrix GGUF holds, per matmul weight, `<name>.in_sum2` (the summed
squared input activations, one row per expert) and `<name>.counts` (how many
tokens each expert saw). Everything here is read from those two arrays; no
model, no GPU.

coverage  An expert whose count is zero has no statistics at all: the quantiser
          treats its rows as uniformly important, and at 1-3 bits that expert
          is noise. llama-imatrix only says "partial data (99.80%)". This names
          the experts, flags the ones under --low of the median, and with an
          inventory lists the quantisable weights that have no entry (GET_ROWS
          tables, the MTP block) with their share of the model.
converge  cos(N, 2N) per tensor, on per-expert means (in_sum2 / counts), so an
          expert that got twice the tokens does not read as movement, averaged
          over experts weighted by the tokens they saw. The matrix has
          converged when every tensor is at or above --min-cos. `cos_raw` is
          calib-corpora converge.py's number (flat in_sum2), kept for
          comparison with older reports; on MoE it mixes routing shifts in.
stats     the table `llama-imatrix --show-statistics` prints (Σ(Act²) is the
          sum of per-expert means over experts with data), from the file alone:
          llama-imatrix wants -m for it and loads the model, 354 GB on
          Flash-Next. band_select.py reads the table and picks the bands; this
          prints only the top list and which roles and blocks fill it.

Exit 3 when coverage finds a dead expert or converge a tensor below --min-cos.
"""
import argparse
import json
import math
import re
import sys

import numpy as np

DEAD_EXIT = 3


def load(path):
    """{name: (values[n_mat, row], counts[n_mat])} and the chunk count."""
    from gguf import GGUFReader

    r = GGUFReader(path)
    sums, counts = {}, {}
    for t in r.tensors:
        if t.name.endswith(".in_sum2"):
            sums[t.name[:-len(".in_sum2")]] = np.asarray(t.data, dtype=np.float64).reshape(-1)
        elif t.name.endswith(".counts"):
            counts[t.name[:-len(".counts")]] = np.asarray(t.data, dtype=np.float64).reshape(-1)
    chunks = None
    f = r.fields.get("imatrix.chunk_count")
    if f is not None:
        chunks = int(f.contents()) if hasattr(f, "contents") else int(f.parts[f.data[0]][0])
    out = {}
    for name, v in sums.items():
        c = counts.get(name)
        if c is None or c.size == 0 or v.size % c.size:
            raise SystemExit(f"{path}: {name} has {v.size} values and {0 if c is None else c.size} counts")
        out[name] = (v.reshape(c.size, v.size // c.size), c)
    return out, chunks


def load_sum(paths):
    """Shards over disjoint chunk ranges add up: in_sum2 and counts are sums over tokens.

    So the first half of the shards against all of them is N against 2N chunks,
    with no --save-frequency run and no llama-imatrix merge.
    """
    total, chunks = None, 0
    for p in paths.split(","):
        im, n = load(p)
        chunks += n or 0
        if total is None:
            total = {k: (v.copy(), c.copy()) for k, (v, c) in im.items()}
            continue
        for k, (v, c) in im.items():
            if k in total and total[k][0].shape == v.shape:
                total[k] = (total[k][0] + v, total[k][1] + c)
            else:
                total.setdefault(k, (v.copy(), c.copy()))
    return total, chunks


def block_of(name):
    m = re.match(r"blk\.(\d+)\.", name)
    return int(m.group(1)) if m else None


def role_of(name):
    return re.sub(r"^blk\.\d+\.", "blk.N.", name)


# ------------------------------------------------------------------ coverage

def coverage(im, low=0.01, inventory=None):
    experts, dead, weak = [], [], []
    for name in sorted(im, key=lambda n: (block_of(n) or -1, n)):
        vals, c = im[name]
        if c.size < 2:
            continue
        med = float(np.median(c))
        z = [int(i) for i in np.flatnonzero(c == 0)]
        w = [int(i) for i in np.flatnonzero((c > 0) & (c < low * med))]
        experts.append({"tensor": name, "experts": int(c.size), "dead": z, "weak": w,
                        "min": float(c.min()), "median": med, "max": float(c.max())})
        dead += [(name, i) for i in z]
        weak += [(name, i) for i in w]
    never = sorted(n for n, (v, c) in im.items() if c.size == 1 and c[0] == 0)
    rep = {"entries": len(im), "expert_tensors": len(experts), "dead_experts": len(dead),
           "weak_experts": len(weak), "low_fraction": low, "never_seen": never, "tensors": experts}
    # the same expert id dead in gate, up and down of one block is one hole, not three
    holes = sorted({(block_of(n), i) for n, i in dead})
    rep["dead_by_block"] = {str(b): sorted(i for bb, i in holes if bb == b) for b in sorted({b for b, _ in holes})}
    if inventory:
        rep["uncovered"] = uncovered(im, inventory)
    return rep


def uncovered(im, inv):
    """Quantisable weights with no imatrix entry, and their share of the model."""
    total = sum(math.prod(t["shape"]) for t in inv["tensors"])
    rows = []
    for t in inv["tensors"]:
        name = t["name"]
        dims = sum(1 for d in t["shape"] if d > 1)
        if dims < 2 or not name.endswith(".weight") or name in im:
            continue
        if any(s in name for s in ("_norm.weight", "ffn_gate_inp", "ssm_conv1d", "indexer.k_proj",
                                   "indexer.q_proj", "conv1d")):
            continue
        rows.append((name, math.prod(t["shape"])))
    groups = {}
    for name, n in rows:
        g = groups.setdefault(role_of(name), [0, 0])
        g[0] += 1
        g[1] += n
    return {"params_share": sum(n for _, n in rows) / total if total else 0.0,
            "groups": {k: {"tensors": v[0], "share": v[1] / total, "why": why_uncovered(k)}
                       for k, v in sorted(groups.items(), key=lambda kv: -kv[1][1])}}


def why_uncovered(role):
    # tools/imatrix/imatrix.cpp collect_imatrix: only blk.* weights, output.weight
    # with --process-output; a GET_ROWS lookup is not a matmul at all
    if role in ("token_embd.weight", "per_layer_token_embd.weight"):
        return "GET_ROWS table, no matmul to observe"
    if role == "output.weight":
        return "llama-imatrix skips it without --process-output"
    if not role.startswith("blk."):
        return "outside blk.*, llama-imatrix never collects it"
    return "a blk.* matmul with no entry: never executed (MTP?) or the corpus missed it"


def print_coverage(rep):
    print(f"{rep['entries']} entries, {rep['expert_tensors']} with experts")
    if rep["never_seen"]:
        print(f"NEVER SEEN (count 0): {', '.join(rep['never_seen'][:12])}")
    print(f"dead experts (count 0): {rep['dead_experts']} tensor-expert pairs, "
          f"{sum(len(v) for v in rep['dead_by_block'].values())} distinct (block, expert)")
    for b, ids in rep["dead_by_block"].items():
        print(f"  blk.{b}: experts {ids}")
    print(f"weak experts (< {rep['low_fraction']:.0%} of the median count): {rep['weak_experts']}")
    worst = sorted(rep["tensors"], key=lambda t: t["min"] / max(t["median"], 1))[:8]
    if worst:
        print("lowest min/median:")
        for t in worst:
            print(f"  {t['tensor']:36s} min {t['min']:9.0f}  median {t['median']:9.0f}  max {t['max']:9.0f}")
    if "uncovered" in rep:
        u = rep["uncovered"]
        print(f"quantisable weights with no imatrix entry: {100 * u['params_share']:.2f}% of parameters")
        for k, v in u["groups"].items():
            print(f"  {k:36s} {v['tensors']:4d} tensors  {100 * v['share']:6.2f}%  {v['why']}")


# ------------------------------------------------------------------ converge

def means(vals, counts):
    ok = counts > 0
    m = np.zeros_like(vals)
    m[ok] = vals[ok] / counts[ok, None]
    return m, ok


def cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na and nb else None


def converge(a, b, min_cos=0.995):
    rows = []
    for name in sorted(set(a) & set(b)):
        (va, ca), (vb, cb) = a[name], b[name]
        if va.shape != vb.shape:
            continue
        ma, oka = means(va, ca)
        mb, okb = means(vb, cb)
        both = oka & okb
        if not both.any():
            continue
        # Each expert is quantised with its own row of the matrix, so the question is per
        # expert; weighting by the tokens it saw keeps an expert routed to three times from
        # deciding the verdict (on Flash-Next such experts sit at cos 0.02 either way).
        idx = np.flatnonzero(both)
        per = np.array([cos(ma[i], mb[i]) or 0.0 for i in idx])
        w = np.minimum(ca, cb)[idx]
        c = float((per * w).sum() / w.sum()) if w.sum() else float(per.mean())
        rows.append({"tensor": name, "cos": c, "cos_raw": cos(va.ravel(), vb.ravel()),
                     "expert_min_cos": float(per.min()) if ca.size > 1 else None,
                     "experts_new": int((okb & ~oka).sum()), "experts_lost": int((oka & ~okb).sum())})
    rows = [r for r in rows if r["cos"] is not None]
    bad = [r for r in rows if r["cos"] < min_cos]
    cs = np.array([r["cos"] for r in rows]) if rows else np.array([1.0])
    roles = {}
    for r in rows:
        k = role_of(r["tensor"])
        roles[k] = min(roles.get(k, 1.0), r["cos"])
    return {"tensors": len(rows), "min_cos": min_cos, "mean": float(cs.mean()), "median": float(np.median(cs)),
            "min": float(cs.min()), "below": len(bad), "converged": not bad,
            "experts_new": sum(r["experts_new"] for r in rows),
            "role_min": dict(sorted(roles.items(), key=lambda kv: kv[1])),
            "worst": sorted(rows, key=lambda r: r["cos"])[:12]}


def print_converge(rep, la, lb):
    print(f"{la} vs {lb}: {rep['tensors']} tensors, cos mean {rep['mean']:.6f} median {rep['median']:.6f} "
          f"min {rep['min']:.6f}")
    print(f"below {rep['min_cos']}: {rep['below']}   experts first seen in the larger run: {rep['experts_new']}")
    bad = {k: v for k, v in rep["role_min"].items() if v < rep["min_cos"]}
    if bad:
        print("  roles below: " + ", ".join(f"{k} {v:.4f}" for k, v in bad.items()))
    for r in rep["worst"][:8]:
        e = f"  worst expert {r['expert_min_cos']:.4f}" if r["expert_min_cos"] is not None else ""
        print(f"  {r['cos']:.6f}  {r['tensor']}  (raw {r['cos_raw']:.4f}){e}")
    print("CONVERGED" if rep["converged"] else "NOT CONVERGED: more chunks, or a second build with another seed")


# ------------------------------------------------------------------ stats

def tensor_stats(name, vals, counts):
    """One row of llama-imatrix --show-statistics (tools/imatrix/imatrix.cpp compute_statistics)."""
    m, ok = means(vals, counts)
    act = m[ok].ravel()
    if act.size == 0:
        return None
    total = float(act.sum())
    mean = total / act.size
    std = math.sqrt(max(0.0, float((act * act).sum()) / act.size - mean * mean))
    p = act[act > 0] / total if total > 0 else np.array([])
    entropy = float(-(p * np.log2(p)).sum()) if p.size else 0.0
    zd = float(((act - mean) / std > 1).sum()) / act.size if std > 0 else 0.0
    return {"tensor": name, "sum": total, "min": float(act.min()), "max": float(act.max()), "mean": mean,
            "std": std, "active": 1 - float((np.abs(act) <= 1e-5).sum()) / act.size, "n": int(act.size),
            "entropy": entropy, "entropy_norm": 100 * entropy / math.log2(act.size) if act.size > 1 else 0.0,
            "zd": zd}


def short_names(name):
    """(layer, tensor) the way imatrix.cpp process_tensor_name splits a name."""
    parts = name.split(".")
    layer = parts[parts.index("blk") + 1] if "blk" in parts[:-1] else "-"
    tensor = parts[parts.index("weight") - 1] if "weight" in parts[1:] else name
    return layer, tensor


def stats(im):
    """Every tensor's row, in llama-imatrix's order: by tensor name, then falling Σ(Act²).

    llama-imatrix --show-statistics needs -m and loads the model to print this
    (354 GB for Flash-Next); the numbers only depend on the imatrix, so here they
    come from the file alone. band_select.py reads the text this writes.
    """
    rows = [r for n, (v, c) in im.items() if (r := tensor_stats(n, v, c))]
    prev = {r["tensor"]: r for r in rows}
    for r in rows:
        # CosSim: this tensor against the same tensor one block up, on the raw sums
        b = block_of(r["tensor"])
        other = prev.get(re.sub(r"^blk\.\d+\.", f"blk.{b - 1}.", r["tensor"])) if b else None
        r["cossim"] = cos(im[r["tensor"]][0].ravel(), im[other["tensor"]][0].ravel()) if other else 0.0
        r["cossim"] = r["cossim"] or 0.0
    rows.sort(key=lambda r: (short_names(r["tensor"])[1], -r["sum"]))
    return rows


def stats_text(rows, source):
    out = [f"Computing statistics for {source} ({len(rows)} tensors)", "",
           "\t".join((" Layer", "       Tensor", "          \u03a3(Act\u00b2)", "  Min", "            Max",
                      "           \u03bc", "   \u03c3", " % Active", "N", "   Entropy", "E (norm)", "ZD",
                      "  CosSim")),
           "=" * 120]
    for r in rows:
        layer, tensor = short_names(r["tensor"])
        out.append("%5s\t%-20s\t%10.2f\t%8.4f\t%11.4f\t%6.2f\t%6.2f\t%8.2f%%\t%6d\t%10.4f\t%6.2f%%\t%10.2f%%\t%8.4f"
                   % (layer, tensor, r["sum"], r["min"], r["max"], r["mean"], r["std"], 100 * r["active"], r["n"],
                      r["entropy"], r["entropy_norm"], 100 * r["zd"], r["cossim"]))
    return "\n".join(out) + "\n"


def print_top(rows, top=40):
    ranked = sorted(rows, key=lambda r: -r["sum"])[:top]
    roles, blocks = {}, {}
    for r in ranked:
        roles[role_of(r["tensor"])] = roles.get(role_of(r["tensor"]), 0) + 1
        if (b := block_of(r["tensor"])) is not None:
            blocks[b] = blocks.get(b, 0) + 1
    print(f"top {len(ranked)} tensors by \u03a3(Act\u00b2):")
    for r in ranked[:10]:
        print(f"  {r['sum']:14.2f}  {r['tensor']}")
    print("  roles:  " + ", ".join(f"{k} {v}" for k, v in roles.items()))
    print("  blocks: " + ", ".join(f"{k}:{v}" for k, v in sorted(blocks.items(), key=lambda kv: (-kv[1], kv[0]))))


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=("coverage", "converge", "stats", "all"))
    ap.add_argument("imatrix", nargs="+")
    ap.add_argument("--inventory")
    ap.add_argument("--low", type=float, default=0.01, help="weak expert: count under this fraction of the median")
    ap.add_argument("--min-cos", type=float, default=0.995)
    ap.add_argument("--vs", help="the smaller run, for 'all'")
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("-o", "--stats-out", help="write the --show-statistics table here")
    ap.add_argument("--json")
    a = ap.parse_args()

    inv = None
    if a.inventory:
        with open(a.inventory) as f:
            inv = json.load(f)
    report, rc = {}, 0
    if a.what == "converge":
        if len(a.imatrix) != 2:
            ap.error("converge takes two files: the N-chunk and the 2N-chunk matrix")
        (ia, na), (ib, nb) = load_sum(a.imatrix[0]), load_sum(a.imatrix[1])
        report["converge"] = converge(ia, ib, a.min_cos)
        print_converge(report["converge"], f"{na} chunks", f"{nb} chunks")
        rc = 0 if report["converge"]["converged"] else DEAD_EXIT
    else:
        im, chunks = load(a.imatrix[0])
        report["chunks"] = chunks
        print(f"{a.imatrix[0]}: {chunks} chunks")
        if a.what in ("coverage", "all"):
            report["coverage"] = coverage(im, a.low, inv)
            print_coverage(report["coverage"])
            rc = DEAD_EXIT if report["coverage"]["dead_experts"] or report["coverage"]["never_seen"] else rc
        if a.what in ("stats", "all"):
            print()
            rows = stats(im)
            report["top"] = [{"tensor": r["tensor"], "sum": r["sum"]}
                             for r in sorted(rows, key=lambda r: -r["sum"])[:a.top]]
            print_top(rows, a.top)
            if a.stats_out:
                with open(a.stats_out, "w") as f:
                    f.write(stats_text(rows, a.imatrix[0]))
                print(f"wrote {a.stats_out}: band_select.py reads it")
        if a.what == "all" and a.vs:
            print()
            small, ns = load_sum(a.vs)
            report["converge"] = converge(small, im, a.min_cos)
            print_converge(report["converge"], f"{ns} chunks", f"{chunks} chunks")
            rc = rc or (0 if report["converge"]["converged"] else DEAD_EXIT)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(report, f, indent=1)
            f.write("\n")
    return rc


if __name__ == "__main__":
    sys.exit(main())
