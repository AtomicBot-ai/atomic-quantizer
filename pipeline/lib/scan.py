#!/usr/bin/env python3
"""Sensitivity scan: which tensor groups cost the most KLD per GB saved.

    scan.py plan   --inventory inventory.json --out scan/ [--base q8_0 --down 2]
    scan.py run    scan/ --bf16 M-BF16.gguf --imatrix imatrix.gguf --eval eval.txt --ref base.kld [--bin DIR]
    scan.py report scan/                                   # -> scan/scan.json, table
    scan.py bands  scan/scan.json --inventory inventory.json --profile profiles/dense-hybrid.yaml -o bands.json

The port of img_scan (foundry-image.sh) to text models. Groups come from the
inventory, not from a hand list: every quantizable tensor kind (blk.N.<kind>,
gate+up, k+v and the shared experts merged, since the ladders move them
together) is a role; a role that holds at least --split-share of the weights
and spans four blocks or more is cut into quarters of the depth, the rest stay
whole. Kinds under --min-share (ssm_alpha/beta) are not worth a run and stay
at the base type. The MTP block is pinned: the imatrix never sees it.

Every build is the base type everywhere (q8_0 by default) with one group taken
--down steps down the ladder STEPS with the imatrix (q5_k from q8_0); rows
that are not a multiple of 256 walk the legacy steps instead (q8_0 q5_0 q4_0),
where a k/i type would silently fall back. ladder_gen simulates every rule file
before it is used and the quantize log is checked against the simulation after.

run quantizes the next group on the CPU while the current one is measured on
the GPU, so a group costs max(quantize, KLD), keeps at most two GGUFs on disk
and deletes each after measuring. It resumes: a group with a finished KLD log
and the same rule file is not built again. --part I/N splits the groups across
boxes (the base goes with part 0).

report pairs every group with the base chunk by chunk (kld_diff) and ranks by
dKLD per GB saved, most sensitive first: the order an AD layout should spend
bytes in. bands turns the ranking of the quarters of the band roles (ffn by
default) into edge/mid blocks for ladder_gen --bands-from: the best quarter
fills the edge band first, the next ones follow; inside a quarter the blocks
nearest an end of the network go first (the August prior, which the scan
cannot resolve below a quarter).
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kld_diff  # noqa: E402
import ladder_gen as lg  # noqa: E402
import quantlog  # noqa: E402

STEPS = ["q8_0", "q6_k", "q5_k", "q4_k", "iq4_xs", "iq3_s", "iq3_xxs", "iq2_s", "iq2_xs", "iq2_xxs", "iq1_m"]
LEGACY = ["q8_0", "q5_0", "q4_0"]   # 32 wide blocks: for rows that are not a multiple of 256
MERGE = {"ffn_gate": "ffn_gate_up", "ffn_up": "ffn_gate_up", "attn_k": "attn_kv", "attn_v": "attn_kv",
         "ffn_gate_exps": "exps_gate_up", "ffn_up_exps": "exps_gate_up", "ffn_down_exps": "exps_down",
         "ffn_gate_shexp": "shexp", "ffn_up_shexp": "shexp", "ffn_down_shexp": "shexp",
         "ssm_alpha": "ssm_ab", "ssm_beta": "ssm_ab"}
BLOCK = re.compile(r"^blk\.(\d+)\.(.+)\.weight$")


class ScanError(Exception):
    pass


def bpw(typ):
    be, bb = lg.GGML[typ]
    return 8.0 * bb / be


def step_down(base, down, row):
    """The type `down` steps below `base`; legacy steps when the row does not fit a 256 block."""
    if row % 256 == 0 and base in STEPS:
        return STEPS[min(STEPS.index(base) + down, len(STEPS) - 1)]
    start = next(i for i, t in enumerate(LEGACY) if bpw(t) <= bpw(base) + 1e-9)
    return LEGACY[min(start + down, len(LEGACY) - 1)]


def base_for(base, row):
    """The base type itself, or its legacy neighbour on a row a k/i type cannot take."""
    return base if lg.GGML[base][0] == 1 or row % lg.GGML[base][0] == 0 else step_down(base, 0, row)


def kind_of(name):
    m = BLOCK.match(name)
    k = m.group(2) if m else name[:-len(".weight")] if name.endswith(".weight") else name
    return MERGE.get(k, k), (int(m.group(1)) if m else None)


def elements(t):
    n = 1
    for d in t["shape"]:
        n *= d
    return n


def quarters(blocks, n):
    """Split a sorted block list into n contiguous parts, as even as possible."""
    out, start = [], 0
    for i in range(n):
        size = len(blocks) // n + (1 if i < len(blocks) % n else 0)
        out.append(blocks[start:start + size])
        start += size
    return out


def alt(xs):
    return "(" + "|".join(str(x) for x in xs) + ")"


def group_regex(tensors):
    """An anchored regex that matches exactly these tensor names (one kind family, some blocks)."""
    kinds = sorted({BLOCK.match(n).group(2) for n in tensors if BLOCK.match(n)})
    blocks = sorted({int(BLOCK.match(n).group(1)) for n in tensors if BLOCK.match(n)})
    if not kinds:
        return "^(" + "|".join(re.escape(n) for n in sorted(tensors)) + ")$"
    return rf"^blk\.{alt(blocks)}\.{alt([re.escape(k) for k in kinds])}\.weight$"


def make_groups(inv, split_share=0.08, min_share=0.005, n_quarters=4):
    mtp = set(inv.get("mtp_blocks") or [])
    cand = [t for t in inv["tensors"] if lg.quantizable(t) and t["type"] in ("bf16", "f16", "f32")
            and kind_of(t["name"])[1] not in mtp]
    total = sum(elements(t) for t in cand)
    kinds = {}
    for t in cand:
        k, b = kind_of(t["name"])
        kinds.setdefault(k, []).append((b, t))
    blocks = sorted({b for k in kinds.values() for b, _ in k if b is not None})
    parts = quarters(blocks, n_quarters)
    groups, skipped = [], []
    for k in sorted(kinds, key=lambda k: -sum(elements(t) for _, t in kinds[k])):
        ts = kinds[k]
        share = sum(elements(t) for _, t in ts) / total
        if share < min_share:
            skipped.append({"kind": k, "share": round(share, 5), "tensors": len(ts)})
            continue
        spans = sorted({b for b, _ in ts if b is not None})
        if share >= split_share and len(spans) >= n_quarters:
            for q, part in enumerate(parts, 1):
                sub = [t for b, t in ts if b in set(part)]
                if sub:
                    groups.append({"name": f"{k}-Q{q}", "kind": k, "quarter": q, "blocks": sorted({b for b, t in ts if b in set(part)}),
                                   "tensors": sorted(t["name"] for t in sub), "share": sum(elements(t) for t in sub) / total})
        else:
            groups.append({"name": k, "kind": k, "quarter": None, "blocks": spans,
                           "tensors": sorted(t["name"] for _, t in ts), "share": share})
    return groups, skipped, parts


def group_profile(inv, base, down, group=None):
    """A one rung ladder_gen profile: the group down, everything else at the base, MTP pinned."""
    by_name = {t["name"]: t for t in inv["tensors"]}
    roles, types = [], {}
    if group:
        rows = {by_name[n]["shape"][0] for n in group["tensors"]}
        for i, row in enumerate(sorted(rows)):
            names = [n for n in group["tensors"] if by_name[n]["shape"][0] == row]
            roles.append({"name": f"group{i}", "pattern": group_regex(names)})
            types[f"group{i}"] = step_down(base, down, row)
    odd = sorted(t["name"] for t in inv["tensors"] if lg.quantizable(t) and base_for(base, t["shape"][0]) != base
                 and kind_of(t["name"])[1] not in set(inv.get("mtp_blocks") or []))
    if odd:
        roles.append({"name": "base_legacy", "pattern": "^(" + "|".join(re.escape(n) for n in odd) + ")$"})
        types["base_legacy"] = base_for(base, by_name[odd[0]]["shape"][0])
    roles.append({"name": "base", "pattern": "."})
    types["base"] = base
    label = group["name"] if group else "base"
    return {"name": f"scan-{label}", "ftype": "Q8_0", "roles": roles,
            "mtp": {"type": base if base not in lg.NEEDS_IMATRIX else "q8_0"},
            "rungs": [{"label": label, "types": types}]}


def simulate(inv, base, down, group=None):
    prof = group_profile(inv, base, down, group)
    ladder, problems = lg.build(prof, inv)
    if problems:
        raise ScanError(f"{prof['name']}: llama-quantize would not build it: {problems[prof['rungs'][0]['label']][:3]}")
    rung = ladder["rungs"][0]
    if group:
        hit = {n for n, why in rung["_why"].items() if why.startswith("group")}
        if hit != set(group["tensors"]):
            raise ScanError(f"{group['name']}: rules hit {len(hit)} tensors, the group has {len(group['tensors'])}")
    return rung


# ---------------------------------------------------------------- plan

def plan(inv, out, base="q8_0", down=2, split_share=0.08, min_share=0.005, n_quarters=4):
    base = lg.norm_type(base)
    groups, skipped, parts = make_groups(inv, split_share, min_share, n_quarters)
    os.makedirs(os.path.join(out, "types"), exist_ok=True)
    with open(os.path.join(out, "inventory.json"), "w") as f:
        json.dump(inv, f)
    b = simulate(inv, base, down)
    entries = [{"name": "base", "tensors": [], "rules": b["rules"], "predicted_bytes": b["predicted_bytes"]}]
    for g in groups:
        r = simulate(inv, base, down, g)
        targets = sorted({r["_effective"][n] for n in g["tensors"]})
        entries.append(dict(g, target=targets, rules=r["rules"], predicted_bytes=r["predicted_bytes"],
                            predicted_saved=b["predicted_bytes"] - r["predicted_bytes"]))
    for e in entries:
        text = "".join(f"{p}={t}\n" for p, t in e["rules"])
        e["types_sha256"] = hashlib.sha256(text.encode()).hexdigest()
        with open(os.path.join(out, "types", f"{e['name']}.types"), "w") as f:
            f.write(text)
    doc = {"inventory": inv.get("source"), "arch": inv.get("arch"), "base": base, "down": down,
           "split_share": split_share, "min_share": min_share, "quarters": parts,
           "groups": entries, "skipped": skipped}
    with open(os.path.join(out, "plan.json"), "w") as f:
        json.dump(doc, f, indent=1)
    return doc


# ---------------------------------------------------------------- run

def load_plan(d):
    with open(os.path.join(d, "plan.json")) as f:
        p = json.load(f)
    with open(os.path.join(d, "inventory.json")) as f:
        inv = json.load(f)
    return p, inv


def state_path(d, name):
    return os.path.join(d, "runs", f"{name}.json")


def done(d, g):
    try:
        with open(state_path(d, g["name"])) as f:
            st = json.load(f)
    except OSError:
        return False
    return st.get("types_sha256") == g["types_sha256"] and kld_diff.finished(os.path.join(d, "logs", f"kld-{g['name']}.log"))


def expected(inv, p, g):
    if g["name"] == "base":
        return simulate(inv, p["base"], p["down"])
    return simulate(inv, p["base"], p["down"], g)


def quantize(d, g, inv, p, cfg):
    out = os.path.join(cfg["work"], f"scan-{g['name']}.gguf")
    log = os.path.join(d, "logs", f"quantize-{g['name']}.log")
    cmd = [os.path.join(cfg["bin"], "llama-quantize"), "--imatrix", cfg["imatrix"],
           "--tensor-type-file", os.path.join(d, "types", f"{g['name']}.types"), cfg["bf16"], out, "Q8_0",
           str(cfg["threads"])]
    t0 = time.time()
    with open(log, "w") as f:
        f.write("# " + " ".join(cmd) + "\n")
        f.flush()
        rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT)
    if rc:
        raise ScanError(f"{g['name']}: llama-quantize exit {rc}, see {log}")
    diffs = lg.compare(expected(inv, p, g), quantlog.parse_file(log))
    if diffs:
        raise ScanError(f"{g['name']}: {len(diffs)} tensors are not what the rules say, first {diffs[:3]}")
    return out, os.path.getsize(out), time.time() - t0


def measure(d, g, path, cfg):
    log = os.path.join(d, "logs", f"kld-{g['name']}.log")
    cmd = [os.path.join(cfg["bin"], "llama-perplexity"), "-m", path, "-f", cfg["eval"], "-c", str(cfg["ctx"]),
           "--chunks", str(cfg["chunks"]), "-ngl", str(cfg["ngl"]), "--kl-divergence-base", cfg["ref"], "--kl-divergence"]
    t0 = time.time()
    with open(log, "w") as f:
        f.write("# " + " ".join(cmd) + "\n")
        f.flush()
        rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT)
    if rc or not kld_diff.finished(log):
        raise ScanError(f"{g['name']}: llama-perplexity exit {rc}, see {log}")
    return time.time() - t0


def run(d, cfg, part=(0, 1), say=print):
    p, inv = load_plan(d)
    os.makedirs(os.path.join(d, "logs"), exist_ok=True)
    os.makedirs(os.path.join(d, "runs"), exist_ok=True)
    os.makedirs(cfg["work"], exist_ok=True)
    i, n = part
    mine = [g for k, g in enumerate(p["groups"]) if (k == 0 and i == 0) or (k > 0 and (k - 1) % n == i)]
    todo = [g for g in mine if not done(d, g)]
    say(f"scan {d}: {len(mine)} builds in part {i}/{n}, {len(mine) - len(todo)} already measured")
    t_start = time.time()
    with ThreadPoolExecutor(1) as ex:
        nxt = ex.submit(quantize, d, todo[0], inv, p, cfg) if todo else None
        for k, g in enumerate(todo):
            path, size, tq = nxt.result()
            nxt = ex.submit(quantize, d, todo[k + 1], inv, p, cfg) if k + 1 < len(todo) else None
            say(f"  {g['name']:22s} quantized in {tq:5.0f} s, {size / 1e9:.3f} GB; measuring")
            try:
                tk = measure(d, g, path, cfg)
            finally:
                if not cfg.get("keep"):
                    os.remove(path)
            st = {"name": g["name"], "types_sha256": g["types_sha256"], "size_bytes": size,
                  "quantize_s": round(tq, 1), "kld_s": round(tk, 1),
                  "protocol": {k: cfg[k] for k in ("chunks", "ctx", "ref", "eval", "imatrix", "bf16", "bin")}}
            with open(state_path(d, g["name"]), "w") as f:
                json.dump(st, f, indent=1)
            r = kld_diff.parse_file(os.path.join(d, "logs", f"kld-{g['name']}.log"))
            say(f"  {g['name']:22s} KLD {r['mean_kld']:.6f}  top-1 {r['top1_pct']}%  ({tk:.0f} s)  "
                f"[{k + 1}/{len(todo)}, {(time.time() - t_start) / 60:.1f} min]")
    wall = time.time() - t_start
    with open(os.path.join(d, "runs", f"part-{i}-of-{n}.json"), "w") as f:
        json.dump({"builds": len(todo), "wall_s": round(wall, 1)}, f)
    say(f"scan part {i}/{n}: {len(todo)} builds in {wall / 60:.1f} min")
    return wall


# ---------------------------------------------------------------- report

def report(d):
    p, _ = load_plan(d)

    def load(name):
        with open(state_path(d, name)) as f:
            st = json.load(f)
        return st, kld_diff.parse_file(os.path.join(d, "logs", f"kld-{name}.log"))

    try:
        bst, blog = load("base")
    except OSError:
        raise ScanError("the base is not measured yet: scan.py run")
    rows, missing = [], []
    for g in p["groups"][1:]:
        try:
            st, log = load(g["name"])
        except (OSError, kld_diff.KldLogError):
            missing.append(g["name"])
            continue
        r = kld_diff.paired(blog, log)
        saved = (bst["size_bytes"] - st["size_bytes"]) / 1e9
        pg = kld_diff.per_gb(r, saved) if saved > 1e-4 else None
        rows.append({"group": g["name"], "kind": g["kind"], "quarter": g["quarter"], "blocks": g["blocks"],
                     "target": g["target"], "share": round(g["share"], 5), "gb_saved": round(saved, 5),
                     "kld": log["mean_kld"], "top1_pct": log["top1_pct"],
                     "d_kld": r["d_kld"], "d_kld_se": r["d_kld_se"], "d_kld_ci95": r["d_kld_ci95"],
                     "d_top1_pp": r["d_top1_pp"],
                     "per_gb": pg["per_gb"] if pg else None, "per_gb_se": pg["per_gb_se"] if pg else None,
                     "per_gb_ci95": pg["per_gb_ci95"] if pg else None,
                     "d_chunks": [y - x for x, y in zip(blog["kld"], log["kld"])]})
    rows.sort(key=lambda r: (r["per_gb"] is None, -(r["per_gb"] or 0.0)))   # nothing saved goes last
    walls = []
    for fn in sorted(os.listdir(os.path.join(d, "runs"))):
        if fn.startswith("part-"):
            with open(os.path.join(d, "runs", fn)) as f:
                walls.append(json.load(f)["wall_s"])
    doc = {"protocol": dict(bst["protocol"], base=p["base"], down=p["down"], inventory=p["inventory"]),
           "base": {"kld": blog["mean_kld"], "top1_pct": blog["top1_pct"], "size_bytes": bst["size_bytes"]},
           "quarters": p["quarters"], "groups": rows, "missing": missing, "skipped": p["skipped"],
           "wall_s": walls}
    with open(os.path.join(d, "scan.json"), "w") as f:
        json.dump(doc, f, indent=1)
    return doc


def fmt_report(doc):
    b = doc["base"]
    out = [f"base {doc['protocol']['base']} everywhere: KLD {b['kld']:.6f}, top-1 {b['top1_pct']}%, "
           f"{b['size_bytes'] / 1e9:.3f} GB; each group {doc['protocol']['down']} steps down, "
           f"{doc['protocol']['chunks']} chunks x {doc['protocol']['ctx']}",
           f"{'group':20s} {'to':14s} {'share':>6s} {'GB saved':>8s} {'dKLD':>10s} {'± SE':>9s} {'dKLD / GB':>10s} {'± SE':>9s}"]
    for r in doc["groups"]:
        out.append(f"{r['group']:20s} {','.join(r['target']):14s} {100 * r['share']:5.1f}% {r['gb_saved']:8.3f} "
                   f"{r['d_kld']:+10.6f} {r['d_kld_se']:9.6f} "
                   + (f"{r['per_gb']:+10.6f} {r['per_gb_se']:9.6f}" if r["per_gb"] is not None else "   nothing saved"))
    if doc["missing"]:
        out.append(f"not measured: {' '.join(doc['missing'])}")
    if doc["skipped"]:
        out.append("left at the base (under min share): " + " ".join(s["kind"] for s in doc["skipped"]))
    if doc.get("wall_s"):
        out.append(f"wall time: {' + '.join(f'{w / 60:.1f}' for w in doc['wall_s'])} min")
    return "\n".join(out)


# ---------------------------------------------------------------- bands

def quarter_ranking(doc, kinds):
    """Quarters ranked by the pooled dKLD per GB of the given roles, with a paired SE over chunks."""
    by_q = {}
    for r in doc["groups"]:
        if r["kind"] in kinds and r["quarter"] is not None:
            by_q.setdefault(r["quarter"], []).append(r)
    if not by_q:
        raise ScanError(f"no quartered groups of {sorted(kinds)} in the scan")
    ranked = []
    for q, rs in by_q.items():
        saved = sum(r["gb_saved"] for r in rs)
        dc = [sum(xs) for xs in zip(*(r["d_chunks"] for r in rs))]
        d = sum(dc) / len(dc)
        ranked.append({"quarter": q, "blocks": doc["quarters"][q - 1], "groups": [r["group"] for r in rs],
                       "gb_saved": saved, "per_gb": d / saved, "per_gb_se": kld_diff.se(dc) / saved})
    ranked.sort(key=lambda x: -x["per_gb"])
    return ranked


def scan_bands(doc, eligible, n_edge, n_mid, kinds=("ffn_down", "ffn_gate_up")):
    ranked = quarter_ranking(doc, set(kinds))
    eligible = list(eligible)
    pos = {b: i for i, b in enumerate(eligible)}

    def near_end(b):  # nearest an end first, the tail before the head on a tie
        i = pos[b]
        return (min(i, len(eligible) - 1 - i), -i)

    order = [b for q in ranked for b in sorted((b for b in q["blocks"] if b in pos), key=near_end)]
    if n_edge + n_mid > len(order):
        raise ScanError(f"bands need {n_edge}+{n_mid} blocks, the scanned quarters hold {len(order)}")
    return {"method": "scan", "select": list(kinds), "quarters": ranked,
            "edge": sorted(order[:n_edge]), "mid": sorted(order[n_edge:n_edge + n_mid])}


# ---------------------------------------------------------------- cli

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("plan")
    a.add_argument("--inventory", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--base", default="q8_0")
    a.add_argument("--down", type=int, default=2)
    a.add_argument("--split-share", type=float, default=0.08)
    a.add_argument("--min-share", type=float, default=0.005)
    a.add_argument("--quarters", type=int, default=4)
    a = sub.add_parser("run")
    a.add_argument("dir")
    a.add_argument("--bin", default=os.environ.get("LLAMA_BIN", ""))
    a.add_argument("--bf16", required=True)
    a.add_argument("--imatrix", required=True)
    a.add_argument("--eval", required=True)
    a.add_argument("--ref", required=True, help="--kl-divergence-base file of the BF16 over the same eval")
    a.add_argument("--chunks", type=int, default=32)
    a.add_argument("--ctx", type=int, default=4096)
    a.add_argument("--ngl", type=int, default=99)
    a.add_argument("--threads", type=int, default=8)
    a.add_argument("--work", help="where the GGUFs go while measured (default DIR/gguf)")
    a.add_argument("--part", default="0/1", help="I/N: this box runs every N-th group")
    a.add_argument("--keep", action="store_true", help="keep the GGUFs")
    a = sub.add_parser("report")
    a.add_argument("dir")
    a = sub.add_parser("bands")
    a.add_argument("scan_json")
    a.add_argument("--inventory", required=True)
    a.add_argument("--profile", required=True)
    a.add_argument("--kinds", default="ffn_down,ffn_gate_up", help="roles whose quarters the bands follow")
    a.add_argument("--edge", type=int)
    a.add_argument("--mid", type=int)
    a.add_argument("-o", "--out", default="-")
    args = ap.parse_args()

    try:
        if args.cmd == "plan":
            with open(args.inventory) as f:
                inv = json.load(f)
            doc = plan(inv, args.out, args.base, args.down, args.split_share, args.min_share, args.quarters)
            print(f"{len(doc['groups'])} builds (base + {len(doc['groups']) - 1} groups) in {args.out}")
            for g in doc["groups"][1:]:
                print(f"  {g['name']:20s} {100 * g['share']:5.1f}%  {len(g['tensors']):3d} tensors -> "
                      f"{','.join(g['target']):10s} saves {g['predicted_saved'] / 1e6:7.1f} MB")
            if doc["skipped"]:
                print("  left at the base: " + " ".join(f"{s['kind']} ({100 * s['share']:.2f}%)" for s in doc["skipped"]))
        elif args.cmd == "run":
            i, n = (int(x) for x in args.part.split("/"))
            b = args.bin or os.path.dirname(shutil.which("llama-quantize") or "")
            if not os.path.exists(os.path.join(b, "llama-quantize")):
                raise ScanError("no llama-quantize: give --bin or set LLAMA_BIN")
            cfg = {"bin": b, "bf16": args.bf16, "imatrix": args.imatrix, "eval": args.eval, "ref": args.ref,
                   "chunks": args.chunks, "ctx": args.ctx, "ngl": args.ngl, "threads": args.threads,
                   "work": args.work or os.path.join(args.dir, "gguf"), "keep": args.keep}
            run(args.dir, cfg, (i, n), say=lambda s: print(s, flush=True))
        elif args.cmd == "report":
            print(fmt_report(report(args.dir)))
        elif args.cmd == "bands":
            with open(args.scan_json) as f:
                doc = json.load(f)
            with open(args.inventory) as f:
                inv = json.load(f)
            profile = lg.load_profile(args.profile)
            b = lg.compute_bands(profile, inv)
            n_edge = len(b["edge"]) if args.edge is None else args.edge
            n_mid = len(b["mid"]) if args.mid is None else args.mid
            bands = scan_bands(doc, b["eligible"], n_edge, n_mid, args.kinds.split(","))
            bands["source"] = os.path.basename(args.scan_json)
            for q in bands["quarters"]:
                print(f"Q{q['quarter']} blocks {q['blocks'][0]}-{q['blocks'][-1]}: dKLD/GB {q['per_gb']:+.6f} "
                      f"± {q['per_gb_se']:.6f}", file=sys.stderr)
            print(f"edge {bands['edge']}  mid {bands['mid']}", file=sys.stderr)
            text = json.dumps(bands, indent=1) + "\n"
            if args.out == "-":
                sys.stdout.write(text)
            else:
                with open(args.out, "w") as f:
                    f.write(text)
    except (ScanError, lg.LadderError, kld_diff.KldLogError) as e:
        print(f"scan: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
