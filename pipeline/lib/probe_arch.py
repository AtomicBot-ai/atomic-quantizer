#!/usr/bin/env python3
"""One row per model of what decides a quant ladder, before anything is downloaded.

    probe_arch.py Qwen/Qwen3.5-4B Qwen/Qwen3.8-Flash-Next --llama-cpp ~/llama.cpp --work /tmp/probe
    probe_arch.py Qwen/Qwen4-XXB --inventory Qwen/Qwen4-XXB=inventory.json --md table.md --json table.json

The laptop half of foundry.sh probe_arch. Per model it reads config.json and the
tokenizer files (tok_fingerprint.py) and the tensor inventory the converter would
write: with --llama-cpp it runs `convert_hf_to_gguf.py --remote REPO --dry-run`,
which fetches only the safetensors headers, so a 354 GB checkpoint costs a few
minutes; --inventory REPO=FILE reuses one already made.

Columns: GGUF arch, blocks (text layers, of them full attention, MTP), row
lengths of the quantisable weights with their share and whether they divide
by 256 (k and i types need it), vocab, the canonical tokenizer and template
hashes, MTP and vision, and the share of GET_ROWS tables (token_embd and the
PLE n-gram table), which no imatrix can calibrate.

If the converter does not know the class the dry run fails; the row then has
only what the config and tokenizer say, and says why.
"""
import argparse
import json
import math
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gguf_inventory  # noqa: E402
import ladder_gen  # noqa: E402
import tok_fingerprint  # noqa: E402

GET_ROWS = ("token_embd.weight", "per_layer_token_embd.weight")


def dry_run(repo, llama_cpp, work):
    """convert_hf_to_gguf.py --remote --dry-run; (log path, None) or (None, why it failed)."""
    os.makedirs(work, exist_ok=True)
    log = os.path.join(work, repo.replace("/", "--") + ".convert.log")
    if os.path.exists(log) and "Dry run, not writing files" in open(log, errors="replace").read():
        return log, None
    env = dict(os.environ, PYTHONPATH=os.path.join(llama_cpp, "gguf-py"))
    out = os.path.join(work, repo.replace("/", "--") + ".gguf")
    with open(log, "w") as f:
        r = subprocess.run([sys.executable, os.path.join(llama_cpp, "convert_hf_to_gguf.py"), "--remote", repo,
                            "--dry-run", "--outtype", "bf16", "--outfile", out],
                           stdout=f, stderr=subprocess.STDOUT, env=env)
    if r.returncode:
        tail = open(log, errors="replace").read().strip().splitlines()[-1:]
        return None, f"converter failed: {tail[0][:120] if tail else r.returncode}"
    return log, None


def gguf_arch(llama_cpp, hf_class):
    """GGUF arch name of a converter class, read from the converter source (no torch import)."""
    if not llama_cpp or not hf_class:
        return None
    for root in (os.path.join(llama_cpp, "conversion"), llama_cpp):
        if not os.path.isdir(root):
            continue
        for fn in sorted(os.listdir(root)):
            if not fn.endswith(".py"):
                continue
            src = open(os.path.join(root, fn), errors="replace").read()
            for m in re.finditer(r"@ModelBase\.register\(([^)]*)\)", src):
                if f'"{hf_class}"' not in m.group(1):
                    continue
                arch = re.search(r"model_arch\s*=\s*gguf\.MODEL_ARCH\.(\w+)", src[m.end():])
                if arch:
                    return arch.group(1).lower()
    return None


def row(repo, cfg, fp, inv, note=None):
    inner = cfg.get("text_config") or cfg
    layers = inner.get("num_hidden_layers")
    types = inner.get("layer_types") or []
    full = sum(1 for t in types if t == "full_attention") if types else None
    mtp = inner.get("mtp_num_hidden_layers") or inner.get("num_nextn_predict_layers") or 0
    r = {"repo": repo, "revision": fp.get("revision"), "hf_class": (cfg.get("architectures") or ["?"])[0],
         "layers": layers, "full_attention": full, "mtp_layers": mtp,
         "experts": inner.get("num_experts"), "experts_used": inner.get("num_experts_per_tok"),
         "hidden": inner.get("hidden_size"), "vocab_size": inner.get("vocab_size"),
         "tokenizer_entries": fp["tokens"]["vocab_entries"] + fp["tokens"]["added"],
         "tokenizer_sha256": fp["tokenizer_sha256"], "template_sha256": fp["template_sha256"],
         "tokenizer_json_sha256": fp["files"].get("tokenizer.json"),
         "chat_template_sha256": fp["files"].get("chat_template.jinja"),
         "vision": "vision_config" in cfg, "tied": bool(cfg.get("tie_word_embeddings", inner.get("tie_word_embeddings"))),
         "note": note}
    if inv:
        total = sum(math.prod(t["shape"]) for t in inv["tensors"])
        rows = {}
        for t in inv["tensors"]:
            if not ladder_gen.quantizable(t):
                continue
            ne0 = t["shape"][0]
            rows.setdefault(ne0, [0, set()])
            rows[ne0][0] += math.prod(t["shape"])
            rows[ne0][1].add(re.sub(r"^blk\.\d+\.", "", t["name"]).replace(".weight", ""))
        r.update({
            "arch": inv.get("arch"), "blocks": inv.get("block_count"), "mtp_blocks": inv.get("mtp_blocks"),
            "params": total, "bf16_gb": sum(math.prod(t["shape"]) * (4 if t["type"] == "f32" else 2)
                                            for t in inv["tensors"]) / 1e9,
            "rows": [{"ne0": k, "share": v[0] / total, "div256": k % 256 == 0, "tensors": sorted(v[1])}
                     for k, v in sorted(rows.items(), key=lambda kv: -kv[1][0])],
            "get_rows_share": {n: math.prod(t["shape"]) / total for t in inv["tensors"]
                               if (n := t["name"]) in GET_ROWS},
            "mtp_in_gguf": bool(inv.get("mtp_blocks")),
        })
    return r


def markdown(rows):
    head = ("| model | arch | blocks (full attn) | MTP | experts | rows not /256 (share) | vocab / tokenizer "
            "| tokenizer.json | chat_template | vision | GET_ROWS: embd + PLE | BF16 |")
    out = [head, "|" + "---|" * (head.count("|") - 1)]
    for r in rows:
        bad = [x for x in r.get("rows", []) if not x["div256"]]
        rows_s = ", ".join(f"{x['ne0']} ({100 * x['share']:.1f}%: {'/'.join(x['tensors'][:3])})" for x in bad) \
            if "rows" in r else "?"
        g = r.get("get_rows_share") or {}
        mtp = f"{r['mtp_layers']}" + ("" if not r["mtp_layers"] else
                                      (" (in GGUF)" if r.get("mtp_in_gguf") else " (dropped)" if "rows" in r else ""))
        out.append("| " + " | ".join([
            r["repo"].split("/")[-1], str(r.get("arch") or r["hf_class"]),
            f"{r['layers']} ({r['full_attention']})" if r["full_attention"] is not None else str(r["layers"]),
            mtp, f"{r['experts']}/{r['experts_used']}" if r["experts"] else "dense",
            rows_s or "none", f"{r['vocab_size']} / {r['tokenizer_entries']}",
            (r["tokenizer_json_sha256"] or "?")[:12], (r["chat_template_sha256"] or "?")[:12],
            "yes" if r["vision"] else "no",
            (f"{100 * g.get('token_embd.weight', 0):.2f}% + {100 * g.get('per_layer_token_embd.weight', 0):.1f}%"
             + (" (tied)" if r["tied"] else "")) if "rows" in r else "?",
            f"{r['bf16_gb']:.1f} GB" if "bf16_gb" in r else "?"]) + " |")
        if r.get("note"):
            out.append(f"| | {r['note']} |" + " |" * (head.count("|") - 3))
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("repos", nargs="+")
    ap.add_argument("--llama-cpp", help="checkout whose convert_hf_to_gguf.py makes the inventory")
    ap.add_argument("--work", default="probe", help="where dry-run logs and inventories are kept")
    ap.add_argument("--inventory", action="append", default=[], help="REPO=inventory.json, instead of a dry run")
    ap.add_argument("--md")
    ap.add_argument("--json")
    a = ap.parse_args()

    given = dict(x.split("=", 1) for x in a.inventory)
    rows = []
    for repo in a.repos:
        from huggingface_hub import HfApi
        rev = HfApi().model_info(repo).sha
        files = tok_fingerprint.read_files(repo, rev)
        fp = tok_fingerprint.fingerprint(files, repo, rev)
        cfg = json.loads(files.get("config.json", b"{}"))
        inv, note = None, None
        if repo in given:
            with open(given[repo]) as f:
                inv = json.load(f)
        elif a.llama_cpp:
            log, note = dry_run(repo, a.llama_cpp, a.work)
            if log:
                hf_class = (cfg.get("architectures") or [None])[0]
                inv = gguf_inventory.from_convert_log(log, gguf_arch(a.llama_cpp, hf_class))
                with open(os.path.join(a.work, repo.replace("/", "--") + ".inventory.json"), "w") as f:
                    json.dump(inv, f, indent=1)
        else:
            note = "no inventory: give --llama-cpp or --inventory"
        rows.append(row(repo, cfg, fp, inv, note))
        print(f"{repo}: {rows[-1].get('arch') or rows[-1]['hf_class']}"
              + (f", {rows[-1]['note']}" if rows[-1]["note"] else ""), file=sys.stderr)
    text = markdown(rows)
    print(text)
    if a.md:
        with open(a.md, "w") as f:
            f.write(text)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(rows, f, indent=1)
            f.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
