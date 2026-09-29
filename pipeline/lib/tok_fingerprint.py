#!/usr/bin/env python3
"""Fingerprint of the two things a calibration build is bound to: vocabulary and chat markup.

    tok_fingerprint.py MiniMaxAI/MiniMax-M3 -o fingerprints/minimax-m3.json
    tok_fingerprint.py MiniMaxAI/MiniMax-M3.1-Flash --against fingerprints/minimax-m3.json

Reads only the small tokenizer and template files off the hub, no weights. The
comparison answers the release-day question: can the corpus built for the older
model calibrate the new one?

  same tokenizer, same template   the build is reusable as it is
  same tokenizer, other template  the vocab sweep is reusable, re-render the corpus
  other tokenizer                 new recipe, new sweep, new build

Hashes are over canonical JSON, so a file that is only re-indented or has its
keys reordered does not count as a change. File hashes are kept as well, for
the record.
"""
import argparse
import hashlib
import json
import os
import sys

FILES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "special_tokens_map.json",
         "added_tokens.json", "vocab.json", "merges.txt", "generation_config.json", "config.json")
# the parts of tokenizer.json that decide which ids a text becomes
TOKENIZER_KEYS = ("model", "added_tokens", "normalizer", "pre_tokenizer", "post_processor", "decoder")


def sha(data):
    if not isinstance(data, bytes):
        data = json.dumps(data, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(data).hexdigest()


def read_files(target, revision=None):
    """{file name: bytes} from a local directory or a hub repository; missing files are left out."""
    out = {}
    for fn in FILES:
        if os.path.isdir(target):
            p = os.path.join(target, fn)
            if os.path.exists(p):
                with open(p, "rb") as f:
                    out[fn] = f.read()
            continue
        from huggingface_hub import hf_hub_download
        from huggingface_hub.utils import EntryNotFoundError
        try:
            with open(hf_hub_download(target, fn, revision=revision), "rb") as f:
                out[fn] = f.read()
        except EntryNotFoundError:
            pass
    return out


def template_of(files):
    if "chat_template.jinja" in files:
        return files["chat_template.jinja"].decode()
    tpl = json.loads(files.get("tokenizer_config.json", b"{}")).get("chat_template")
    if isinstance(tpl, list):
        tpl = {t.get("name", str(i)): t.get("template") for i, t in enumerate(tpl)}
    return tpl


def fingerprint(files, repo=None, revision=None):
    if "tokenizer.json" not in files:
        raise SystemExit("no tokenizer.json: this tool reads fast tokenizers only")
    tok = json.loads(files["tokenizer.json"])
    tok_cfg = json.loads(files.get("tokenizer_config.json", b"{}"))
    cfg = json.loads(files.get("config.json", b"{}"))
    gen = json.loads(files.get("generation_config.json", b"{}"))
    inner = cfg.get("text_config") or cfg
    tpl = template_of(files)
    added = sorted((a["content"], a["id"], bool(a.get("special"))) for a in tok.get("added_tokens", []))
    return {
        "repo": repo,
        "revision": revision,
        "tokenizer_sha256": sha({k: tok.get(k) for k in TOKENIZER_KEYS}),
        "vocab_sha256": sha(tok.get("model", {}).get("vocab")),
        "added_tokens_sha256": sha(added),
        "template_sha256": sha(tpl) if tpl else None,
        "tokens": {
            "vocab_size": inner.get("vocab_size"),
            "vocab_entries": len(tok.get("model", {}).get("vocab") or {}),
            "added": len(added),
            "bos": tok_cfg.get("bos_token"),
            "eos": tok_cfg.get("eos_token"),
            "pad": tok_cfg.get("pad_token"),
            "eos_token_id": gen.get("eos_token_id", inner.get("eos_token_id")),
            "add_bos_token": tok_cfg.get("add_bos_token"),
        },
        "files": {fn: sha(b) for fn, b in sorted(files.items())},
    }


def compare(old, new):
    """(verdict, [differences]) for a new model against a saved fingerprint."""
    diffs = [f"{k}: {old['tokens'].get(k)!r} -> {new['tokens'].get(k)!r}"
             for k in sorted(set(old["tokens"]) | set(new["tokens"]))
             if old["tokens"].get(k) != new["tokens"].get(k)]
    if old["tokenizer_sha256"] != new["tokenizer_sha256"]:
        what = []
        if old["vocab_sha256"] != new["vocab_sha256"]:
            what.append("vocabulary")
        if old["added_tokens_sha256"] != new["added_tokens_sha256"]:
            what.append("added tokens")
        what = " and ".join(what) or "merges or pre-tokenizer"
        return f"NEW TOKENIZER ({what} differ): new recipe, new vocab sweep, new build", diffs
    if old["template_sha256"] != new["template_sha256"]:
        return "SAME TOKENIZER, NEW TEMPLATE: the vocab sweep is reusable, re-render the corpus", diffs
    return "SAME TOKENIZER AND TEMPLATE: the build is reusable as it is", diffs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("target", help="hub repository id or a local directory")
    ap.add_argument("--revision")
    ap.add_argument("-o", "--out", help="save the fingerprint here")
    ap.add_argument("--against", help="a saved fingerprint to compare with")
    a = ap.parse_args()

    revision = a.revision
    if not os.path.isdir(a.target) and revision is None:
        from huggingface_hub import HfApi
        revision = HfApi().model_info(a.target).sha
    fp = fingerprint(read_files(a.target, revision), None if os.path.isdir(a.target) else a.target, revision)

    print(f"{a.target} @ {revision or 'local'}")
    for k in ("tokenizer_sha256", "vocab_sha256", "added_tokens_sha256", "template_sha256"):
        print(f"  {k:20s} {(fp[k] or 'none')[:16]}")
    print("  " + ", ".join(f"{k} {v}" for k, v in fp["tokens"].items()))

    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(fp, f, indent=2, ensure_ascii=False)
            f.write("\n")
        print(f"wrote {a.out}")

    if a.against:
        with open(a.against) as f:
            old = json.load(f)
        verdict, diffs = compare(old, fp)
        print(f"\nagainst {old.get('repo')} @ {(old.get('revision') or 'local')[:12]}:")
        print(f"  {verdict}")
        for d in diffs:
            print(f"    {d}")
        return 0 if verdict.startswith("SAME TOKENIZER AND") else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
