#!/usr/bin/env python3
"""Pick the high precision bands from the imatrix statistics instead of from depth fractions.

    llama-imatrix -m MODEL-BF16.gguf --in-file imatrix.gguf --show-statistics > stats.txt
    band_select.py stats.txt --inventory inventory.json --profile profiles/dense-hybrid.yaml -o bands.json
    ladder_gen.py --inventory inventory.json --profile profiles/dense-hybrid.yaml --bands-from bands.json ...

The team rule (Flash-Next, Qwen3.8-27B): the band goes where the imatrix saw
the most activation energy, top tensors by Sum(Act^2), not symmetric around the
ends. This walks the tensors in falling Sum(Act^2), keeps those that match the
band group of the profile (ffn_down for dense-hybrid; --tensors to widen it),
and takes their block numbers in order of first appearance: the first blocks
fill the edge band, the next ones the mid band.

The block counts are those the profile's own fractions give on this inventory,
so the ladder keeps the size it had and only the choice of blocks changes.
--edge/--mid set other counts (a size matched comparison, for instance).

Output, bands.json: {"method", "select", "edge": [...], "mid": [...], "ranking": [[block, sum, tensor], ...]}.
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ladder_gen  # noqa: E402

# "0.00.054.285 I " in front of every line when llama.cpp logs with timestamps
PREFIX = re.compile(r"^\s*\d+\.\d+\.\d+\.\d+\s+[IWEDT]\s")
NUM = re.compile(r"^-?\d+(\.\d+)?([eE][-+]?\d+)?$")


class BandError(Exception):
    pass


def parse_stats(text):
    """Per tensor rows of llama-imatrix --show-statistics: [{block, tensor, sum_act2}].

    A row is 'Layer Tensor Sum(Act^2) Min Max ...' with tabs, spaces or pipes
    between the columns; the per layer summary that follows ('Layer mean ...')
    has a number in the second column and is skipped, as are tensors outside
    the blocks (no integer layer).
    """
    rows = []
    for line in text.splitlines():
        line = PREFIX.sub("", line)
        cols = [c for c in re.split(r"[\s|]+", line.strip()) if c]
        if len(cols) < 3 or not cols[0].isdigit() or NUM.match(cols[1]) or not NUM.match(cols[2]):
            continue
        rows.append({"block": int(cols[0]), "tensor": cols[1], "sum_act2": float(cols[2])})
    if not rows:
        raise BandError("no per tensor rows: is this the output of llama-imatrix --show-statistics?")
    return rows


def full_name(row):
    return f"blk.{row['block']}.{row['tensor']}.weight"


def rank_blocks(rows, select, eligible):
    """Blocks in order of their first tensor in falling Sum(Act^2) among tensors matching select."""
    rx = re.compile(select)
    eligible = set(eligible)
    seen, ranking = set(), []
    for r in sorted(rows, key=lambda r: -r["sum_act2"]):
        if r["block"] in seen or r["block"] not in eligible or not rx.search(full_name(r)):
            continue
        seen.add(r["block"])
        ranking.append([r["block"], r["sum_act2"], r["tensor"]])
    return ranking


def select_bands(ranking, n_edge, n_mid):
    if n_edge + n_mid > len(ranking):
        raise BandError(f"bands need {n_edge}+{n_mid} blocks but the statistics rank only {len(ranking)}")
    blocks = [b for b, _, _ in ranking]
    return sorted(blocks[:n_edge]), sorted(blocks[n_edge:n_edge + n_mid])


def profile_counts(profile, inv):
    """(eligible blocks, edge count, mid count) as the profile's own bands give them on this inventory."""
    spec = profile.get("bands") or {}
    if not spec:
        raise BandError(f"profile {profile['name']} has no bands")
    b = ladder_gen.compute_bands(profile, inv)
    return b["eligible"], len(b["edge"]), len(b["mid"])


def build(stats_text, profile, inv, select=None, n_edge=None, n_mid=None, source=None):
    eligible, pe, pm = profile_counts(profile, inv)
    select = select or profile["bands"]["group"]
    n_edge = pe if n_edge is None else n_edge
    n_mid = pm if n_mid is None else n_mid
    ranking = rank_blocks(parse_stats(stats_text), select, eligible)
    edge, mid = select_bands(ranking, n_edge, n_mid)
    return {"method": "band_select", "source": source, "select": select, "edge": edge, "mid": mid,
            "ranking": ranking}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stats", help="saved output of llama-imatrix --show-statistics")
    ap.add_argument("--inventory", required=True)
    ap.add_argument("--profile", required=True)
    ap.add_argument("--tensors", help="regex on full tensor names to rank by (default: the band group)")
    ap.add_argument("--edge", type=int, help="edge band size in blocks (default: what the profile gives)")
    ap.add_argument("--mid", type=int, help="mid band size in blocks (default: what the profile gives)")
    ap.add_argument("-o", "--out", default="-")
    a = ap.parse_args()
    with open(a.inventory) as f:
        inv = json.load(f)
    with open(a.stats, errors="replace") as f:
        text = f.read()
    try:
        bands = build(text, ladder_gen.load_profile(a.profile), inv, a.tensors, a.edge, a.mid,
                      source=os.path.basename(a.stats))
    except (BandError, ladder_gen.LadderError) as e:
        print(f"band_select: {e}", file=sys.stderr)
        return 2
    print(f"ranked by Sum(Act^2) of {bands['select']}: " +
          " ".join(str(b) for b, _, _ in bands["ranking"]), file=sys.stderr)
    print(f"edge {bands['edge']}  mid {bands['mid']}", file=sys.stderr)
    text = json.dumps(bands, indent=1) + "\n"
    if a.out == "-":
        sys.stdout.write(text)
    else:
        with open(a.out, "w") as f:
            f.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
