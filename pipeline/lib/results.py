#!/usr/bin/env python3
"""results.json: one row per published file, the only thing the model card reads.

    results.py row --name Qwen3.8-27B-AD-Q4_K_M --label AD-Q4_K_M --kld kld.log \
                   --quant-log quantize.log --size-bytes N --file-type 15 --commit SHA -o row.json
    results.py merge rows/*.json [--into results.json] -o results.json

Schema (pinned; foundry's nested variant with sizes always filled):
  {name, label, publisher, size_bytes, size_gb, bpw, file_type, llama_commit,
   measured_on, quality: {<evalset>: {mean_kld, median_kld, q99_kld, q999_kld, max_kld,
   top1_pct, mean_dp_pct, rms_dp_pct, ppl, ppl_base, ppl_ratio, chunks, ctx}}}
"""
import argparse
import json
import os
import re
import socket
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import quantlog  # noqa: E402

NUM = r"([-0-9.eE+]+)"
KLD_FIELDS = {
    "mean_kld": r"Mean\s+KLD:\s+" + NUM,
    "median_kld": r"Median\s+KLD:\s+" + NUM,
    "max_kld": r"Maximum\s+KLD:\s+" + NUM,
    "q99_kld": r"99\.0%\s+KLD:\s+" + NUM,
    "q999_kld": r"99\.9%\s+KLD:\s+" + NUM,
    "top1_pct": r"Same top p:\s+" + NUM,
    "mean_dp_pct": r"Mean\s+Δp:\s+" + NUM,
    "rms_dp_pct": r"RMS\s+Δp\s*:\s+" + NUM,
    "ppl": r"Mean PPL\(Q\)\s+:\s+" + NUM,
    "ppl_base": r"Mean PPL\(base\)\s+:\s+" + NUM,
    "ppl_ratio": r"Mean PPL\(Q\)/PPL\(base\)\s+:\s+" + NUM,
}
CHUNKS = re.compile(r"kl_divergence: computing over (\d+) chunks, n_ctx=(\d+)")


def parse_kld(text):
    out = {k: float(m.group(1)) for k, p in KLD_FIELDS.items() if (m := re.search(p, text))}
    if (m := CHUNKS.search(text)):
        out["chunks"], out["ctx"] = int(m.group(1)), int(m.group(2))
    if "mean_kld" not in out:
        raise ValueError("no 'Mean KLD' in the log: the measurement did not finish")
    return out


def publisher(name):
    if "UD-" in name or name.startswith("unsloth--"):
        return "unsloth"
    return name.split("--")[0] if "--" in name else "atomicchat"


def row(a):
    with open(a.kld, errors="replace") as f:
        q = parse_kld(f.read())
    bpw = None
    if a.quant_log:
        bpw = quantlog.parse_file(a.quant_log)["quant_bpw"]
    return {
        "name": a.name, "label": a.label, "publisher": publisher(a.name),
        "size_bytes": a.size_bytes, "size_gb": round(a.size_bytes / 1e9, 2), "bpw": bpw,
        "file_type": a.file_type, "llama_commit": a.commit, "measured_on": socket.gethostname(),
        "quality": {a.evalset: q},
    }


def merge(paths, into=None):
    rows = {}
    for p in ([into] if into and os.path.exists(into) else []) + list(paths):
        with open(p) as f:
            data = json.load(f)
        for r in data if isinstance(data, list) else [data]:
            old = rows.get(r["name"])
            if old:
                old["quality"].update(r.get("quality", {}))
                old.update({k: v for k, v in r.items() if k != "quality" and v is not None})
            else:
                rows[r["name"]] = r
    for r in rows.values():
        if not r.get("size_bytes"):
            raise ValueError(f"{r['name']}: no size_bytes, the card would print an empty size")
    return sorted(rows.values(), key=lambda r: r["size_bytes"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("row")
    r.add_argument("--name", required=True)
    r.add_argument("--label", required=True)
    r.add_argument("--kld", required=True)
    r.add_argument("--quant-log")
    r.add_argument("--size-bytes", type=int, required=True)
    r.add_argument("--file-type", type=int)
    r.add_argument("--commit")
    r.add_argument("--evalset", default="neutral")
    r.add_argument("-o", "--out", required=True)
    m = sub.add_parser("merge")
    m.add_argument("rows", nargs="+")
    m.add_argument("--into")
    m.add_argument("-o", "--out", required=True)
    a = ap.parse_args()
    data = row(a) if a.cmd == "row" else merge(a.rows, a.into)
    with open(a.out, "w") as f:
        json.dump(data, f, indent=1, ensure_ascii=False)
        f.write("\n")


if __name__ == "__main__":
    main()
