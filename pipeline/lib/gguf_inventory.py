#!/usr/bin/env python3
"""Tensor inventory of a model: names, storage types and shapes.

    gguf_inventory.py MODEL-BF16-00001-of-00002.gguf -o inventory.json
    gguf_inventory.py --from-log quantize-AD-Q4_K.log -o inventory.json

The ladder generator works only from this file, so it runs on a laptop without
the weights. Shapes are in ggml order (ne0 first, ne0 is the row length).
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import quantlog  # noqa: E402

KV_ARCH = re.compile(r"general\.architecture\s+str\s+=\s+(\S+)")
KV_BLOCKS = re.compile(r"(\w+)\.block_count\s+u32\s+=\s+(\d+)")
KV_LINE = re.compile(r"- kv\s+\d+:\s+(\S+)\s+(u8|i8|u16|i16|u32|i32|u64|i64|f32|f64|bool|str)\s+=\s+(.*)$", re.M)


def keep_meta(key):
    """Scalar hyperparameters for the model card; the tokenizer tables stay out."""
    return not key.startswith(("tokenizer.", "split.")) or key == "tokenizer.ggml.model"


def scalar(typ, raw):
    raw = raw.strip()
    if typ == "str":
        return raw
    if typ == "bool":
        return raw == "true"
    return float(raw) if typ.startswith("f") else int(raw)


def mtp_blocks(names):
    return sorted({int(m.group(1)) for n in names if (m := re.match(r"blk\.(\d+)\.nextn\.", n))})


def finish(arch, block_count, tensors, source, meta=None):
    return {
        "source": source,
        "arch": arch,
        "block_count": block_count,
        "mtp_blocks": mtp_blocks([t["name"] for t in tensors]),
        "meta": meta or {},
        "tensors": tensors,
    }


def from_log(path):
    with open(path, errors="replace") as f:
        text = f.read()
    log = quantlog.parse(text)
    arch = (m.group(1) if (m := KV_ARCH.search(text)) else None)
    blocks = None
    for a, n in KV_BLOCKS.findall(text):
        if a == arch:
            blocks = int(n)
    tensors = [{"name": n, "type": t["src"], "shape": t["shape"]} for n, t in log["tensors"].items()]
    meta = {k: scalar(t, v) for k, t, v in KV_LINE.findall(text) if keep_meta(k)}
    return finish(arch, blocks, tensors, os.path.basename(path), meta)


def from_gguf(path):
    from gguf import GGUFReader

    r = GGUFReader(path)

    def field(key):
        f = r.fields.get(key)
        if f is None:
            return None
        if hasattr(f, "contents"):
            return f.contents()
        v = f.parts[f.data[0]]
        return bytes(v).decode() if f.types and f.types[0].name == "STRING" else v.tolist()[0]

    arch = field("general.architecture")
    blocks = field(f"{arch}.block_count")
    split_count = field("split.count") or 1
    paths = [path]
    if split_count > 1:
        m = re.match(r"(.*)-(\d{5})-of-(\d{5})\.gguf$", path)
        if not m:
            raise SystemExit(f"{path}: split.count={split_count} but the name is not -NNNNN-of-NNNNN.gguf")
        paths = [f"{m.group(1)}-{i:05d}-of-{m.group(3)}.gguf" for i in range(1, split_count + 1)]
    meta = {}
    for key, f in r.fields.items():
        if not keep_meta(key) or len(f.types) != 1 or f.types[0].name == "ARRAY":
            continue
        try:
            meta[key] = f.contents() if hasattr(f, "contents") else field(key)
        except Exception:  # an odd field must not stop the inventory
            pass
    meta["tokenizer.chat_template"] = "tokenizer.chat_template" in r.fields
    tensors = []
    for p in paths:
        for t in (r if p == path else GGUFReader(p)).tensors:
            shape = [int(x) for x in t.shape] + [1] * (4 - len(t.shape))
            tensors.append({"name": t.name, "type": t.tensor_type.name.lower(), "shape": shape})
    return finish(arch, int(blocks) if blocks is not None else None, tensors, os.path.basename(path), meta)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("gguf", nargs="?")
    ap.add_argument("--from-log")
    ap.add_argument("-o", "--out", default="-")
    a = ap.parse_args()
    if bool(a.gguf) == bool(a.from_log):
        ap.error("give either a GGUF file or --from-log")
    inv = from_log(a.from_log) if a.from_log else from_gguf(a.gguf)
    text = json.dumps(inv, indent=1)
    if a.out == "-":
        print(text)
    else:
        with open(a.out, "w") as f:
            f.write(text + "\n")
        n2 = sum(1 for t in inv["tensors"] if t["type"] in ("bf16", "f16", "f32") and t["shape"][1] > 1)
        print(f"{a.out}: {inv['arch']}, {inv['block_count']} blocks, mtp {inv['mtp_blocks']}, "
              f"{len(inv['tensors'])} tensors ({n2} 2-D)")


if __name__ == "__main__":
    main()
