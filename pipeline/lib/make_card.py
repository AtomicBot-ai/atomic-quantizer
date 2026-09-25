#!/usr/bin/env python3
"""README.md for a GGUF repo, drafted from what the pipeline measured.

    make_card.py --dir FETCHED --model Qwen/Qwen3.8-27B --stem Qwen3.8-27B \
                 --main AtomicChat/Qwen3.8-27B-GGUF --metrics AtomicChat/Qwen3.8-27B-GGUF-metrics \
                 [--main-files files.txt] [--recommended AD-Q4_K_M] -o README.md

FETCHED holds files from the metrics repo, at their paths there: results.json,
inventory.json, ladder/ladder.json, kld/base-<evalset>.manifest.txt, and when
present logs/base-<evalset>.log, logs/env-node_{quant,base}.txt, logs/llama-version.txt,
imatrix/params.txt, imatrix/corpus-manifest.json.

Every number comes from those files. The sections that need a person (the
comparison with other builds, what we learned, speed, the vision demo) are left
as <!-- TODO(editorial) --> comments, invisible on the hub; the list is printed
so the driver can say what is still missing before the repo goes public. The
layout follows the published AtomicChat/Qwen3.8-27B-GGUF card.
"""
import argparse
import json
import os
import re
import sys

GIB = 1024 ** 3
ASSETS = "https://huggingface.co/AtomicChat/Qwen3.8-27B-GGUF/resolve/main"   # the header buttons live there
UTM = "utm_source=huggingface&utm_medium=referral&utm_campaign={campaign}&utm_content={content}"
CARDS_GIB = [8, 12, 16, 24, 32, 48, 64, 80, 96, 128, 192, 256]
MIN_CTX = 8192
OVERHEAD = 1 * GIB   # compute buffers and the rest of what is not weights or cache


def load(d, rel, kind="json"):
    p = os.path.join(d, rel)
    if not os.path.exists(p):
        return None
    with open(p, errors="replace") as f:
        return json.load(f) if kind == "json" else f.read()


def kv_file(text):
    return dict(line.split(None, 1) for line in (text or "").splitlines() if " " in line.strip())


def display_name(stem):
    """`Qwen3.8-27B` -> `Qwen3.8 27B`."""
    return " ".join(p for p in re.split(r"[-_]", stem) if p)


def gb(n):
    return f"{n / 1e9:.1f} GB"


class Card:
    def __init__(self, a):
        d = a.dir
        self.a = a
        self.rows = [r for r in load(d, "results.json") or [] if not r.get("control")]
        self.inv = load(d, "inventory.json") or {"meta": {}, "tensors": [], "mtp_blocks": []}
        self.meta = self.inv.get("meta") or {}
        self.arch = self.inv.get("arch") or self.meta.get("general.architecture")
        self.ladder = load(d, "ladder/ladder.json") or {"rungs": []}
        self.evalset = a.evalset
        self.manifest = kv_file(load(d, f"kld/base-{a.evalset}.manifest.txt", "text"))
        self.base_log = load(d, f"logs/base-{a.evalset}.log", "text") or ""
        # the quant box measured the table; the reference box is the fallback
        self.env = load(d, "logs/env-node_quant.txt", "text") or load(d, "logs/env-node_base.txt", "text") or ""
        self.version = load(d, "logs/llama-version.txt", "text") or ""
        self.im = kv_file(load(d, "imatrix/params.txt", "text"))
        self.corpus = load(d, "imatrix/corpus-manifest.json")
        self.files = a.main_files or []
        self.todo = []
        self.name = display_name(a.stem)
        self.family = a.stem.split("-")[0]
        self.campaign = "hf_" + re.sub(r"[^a-z0-9]+", "_", a.stem.lower()).strip("_")
        labels = [r["label"] for r in self.rows]
        self.rec = a.recommended if a.recommended in labels else next(
            (x for x in ("AD-Q4_K_M", "AD-Q4_K", "Q4_K_M") if x in labels),
            min(labels, key=lambda x: abs(self.bpw(x) - 4.8)) if labels else None)

    def m(self, key):
        return self.meta.get(f"{self.arch}.{key}")

    def bpw(self, label):
        r = next(r for r in self.rows if r["label"] == label)
        return r.get("bpw") or 0

    def fname(self, label):
        return f"{self.a.stem}-{label}.gguf"

    def utm(self, content):
        return UTM.format(campaign=self.campaign, content=content)

    def note(self, what):
        self.todo.append(what)
        return f"<!-- TODO(editorial): {what} -->"

    # ------------------------------------------------------------------ facts
    def kv_bytes_per_token(self):
        """f16 K and V over the layers that keep a cache; None when we can not tell (MLA)."""
        mtp = set(self.inv.get("mtp_blocks") or [])
        dims = {}
        for t in self.inv.get("tensors", []):
            mt = re.match(r"blk\.(\d+)\.attn_([kv])\.weight$", t["name"])
            if mt and int(mt.group(1)) not in mtp:
                dims[(mt.group(1), mt.group(2))] = t["shape"][1]
        if dims:
            return 2 * sum(dims.values())
        if self.m("attention.kv_lora_rank"):
            return None
        hkv, kl, vl = self.m("attention.head_count_kv"), self.m("attention.key_length"), self.m("attention.value_length")
        layers = {t["name"].split(".")[1] for t in self.inv.get("tensors", [])
                  if re.match(r"blk\.\d+\.attn_(q|qkv)\.weight$", t["name"])} - {str(b) for b in mtp}
        if isinstance(hkv, int) and kl and vl and layers:
            return len(layers) * hkv * (kl + vl) * 2
        return None

    def vocab(self):
        t = next((t for t in self.inv.get("tensors", []) if t["name"] == "token_embd.weight"), None)
        return t["shape"][1] if t else None

    def build_tag(self):
        """`version: 6123 (1692f9e5)` or `version: 0.1.0-dev (build 10431, commit 1692f9e50)` -> b10431 (1692f9e50)."""
        mt = (re.search(r"version:\s*(\d+)\s*\(([0-9a-f]{7,})\)", self.version)
              or re.search(r"version:.*\(build\s+(\d+),\s*commit\s+([0-9a-f]{7,})\)", self.version))
        return f"b{mt.group(1)} ({mt.group(2)})" if mt else None

    def hardware(self):
        gpus = [ln.split(",")[0].strip() for ln in self.env.splitlines() if re.search(r"\d+ MiB", ln)]
        if gpus:
            names = sorted(set(gpus))
            return ", ".join(f"{gpus.count(n)}x {n.replace('NVIDIA GeForce ', '').replace('NVIDIA ', '')}" for n in names)
        return "CPU only" if "no GPU" in self.env else None

    def mmproj(self):
        """F16 first: it is the one most people want."""
        return sorted((f for f in self.files if re.match(rf"mmproj-{re.escape(self.a.stem)}-(F16|BF16)\.gguf$", f)),
                      key=lambda f: f.endswith("-BF16.gguf"))

    # ------------------------------------------------------------------ sections
    def front(self):
        lic = self.meta.get("general.license")
        tags = ["gguf", "llama.cpp", self.family.lower(), self.a.model.split("/")[0].lower(), "imatrix",
                "quantized", "conversational", "atomic-chat"]
        out = ["---", f"base_model: {self.a.model}", "base_model_relation: quantized", "quantized_by: AtomicChat",
               f"pipeline_tag: {'image-text-to-text' if self.mmproj() else 'text-generation'}", "library_name: gguf"]
        if lic:
            out.append(f"license: {lic}")
        out += ["tags:"] + [f"- {t}" for t in dict.fromkeys(tags)] + ["---", ""]
        return out

    def header(self):
        u = self.utm
        return [
            f"# How to Run {self.name} Locally",
            '<p style="margin-top: 0; margin-bottom: 0;">',
            f"  <em>Built from {self.a.model.split('/')[0]}'s original weights with our own importance matrix. The "
            '<a href="https://huggingface.co/datasets/AtomicChat/calib-corpora">calibration corpora</a> '
            "behind our builds are public.</em>",
            "</p>",
            '<div style="display: flex; gap: 8px; align-items: center; margin-top: 10px; margin-bottom: 10px;">',
            f'  <a href="https://atomic.chat/?{u("btn_atomic")}"><img src="{ASSETS}/btn_atomic.png" width="162" alt="Atomic Chat"></a>',
            f'  <a href="https://discord.gg/8wGSsvmg4V"><img src="{ASSETS}/btn_discord.png" width="119" alt="Discord"></a>',
            f'  <a href="https://github.com/AtomicBot-ai/Atomic-Chat"><img src="{ASSETS}/btn_github.png" width="115" alt="GitHub"></a>',
            "</div>",
            '<ul style="margin: 0 0 12px 0;">',
            "  " + self.note("guide bullet: link to the atomic.chat blog guide for this model, if there is one"),
            f'  <li>You can now run {self.family} in <a href="https://atomic.chat/?{u("bullet_app")}">Atomic Chat</a>.</li>',
            "  <li>See our quantization analysis below for measurements and instructions.</li>",
            "</ul>",
            '<hr style="margin: 0 0 16px 0;">',
            "",
            self.note("the KLD-vs-size chart (ours and other publishers) and the sentence under it"),
            "",
        ]

    def pick(self):
        ev = self.evalset
        out = ["## Pick a file", "",
               "Every number below is measured, not estimated. How we measured it is at the",
               "bottom, and the raw logs are in",
               f"[the metrics repo](https://huggingface.co/datasets/{self.a.metrics})",
               "so you can check any of it yourself.", "",
               "`KL divergence` is how far the quantized model's predictions drift from the",
               "original weights. Lower is better, and zero means identical. `top-1` is how",
               "often it picks the same next word the original would have picked.", "",
               "| File | Size | KL divergence | top-1 |", "|---|---:|---:|---:|"]
        for r in sorted(self.rows, key=lambda r: -r["size_bytes"]):
            q = r["quality"].get(ev, {})
            kld = f"{q['mean_kld']:.5f}" if "mean_kld" in q else "not measured"
            top1 = f"{q['top1_pct']:.2f}%" if "top1_pct" in q else "-"
            out.append(f"| `{r['label']}` | {gb(r['size_bytes'])} | {kld} | {top1} |")
        out += ["", "`AD-` marks an **Atomic Dynamic** layout. The name says what the two largest",
                "tensor groups got: `AD-<ffn_down>-<ffn_up>`, collapsed to one name when both",
                "match. Nothing is named after a type it does not contain.", ""]
        return out

    def fits(self):
        kv = self.kv_bytes_per_token()
        ctx_max = self.m("context_length") or 0
        enough = ctx_max or 131072   # without the model's own limit, 128k counts as "all you need"
        if not kv or not self.rows:
            return ["## Which one fits your card", "", self.note("card-size table: the KV cache size is unknown"), ""]
        rows = sorted(self.rows, key=lambda r: r["size_bytes"])
        table, last = [], None
        for card in CARDS_GIB:
            room = lambda r: card * GIB - OVERHEAD - r["size_bytes"]  # noqa: E731
            ok = [r for r in rows if room(r) >= MIN_CTX * kv]
            short = [r for r in rows if room(r) >= 2048 * kv]
            r = ok[-1] if ok else (short[-1] if short else None)
            if r is None:
                continue
            ctx = int(room(r) // kv)
            if r is rows[-1] and ctx >= enough:
                table.append((f"{card} GB and up", r, "full context" if ctx_max else "128k context and more"))
                break
            if last is r:
                continue
            leaves = "short context only" if not ok else f"around {ctx // 8192 * 8}k context"
            table.append((f"{card} GB", r, leaves))
            last = r
        per_tok = f"{kv / 1024:.0f} KB" if kv < 1024 ** 2 else f"{kv / 1024 ** 2:.1f} MB"
        out = ["## Which one fits your card", "",
               f"The file has to fit, and so does the context. This model keeps {per_tok} of",
               f"attention cache per token, which is {kv * 8192 / GIB:.1f} GB at 8k context and "
               f"{kv * 32768 / GIB:.1f} GB at 32k. Budget", "for both.", "",
               "| Your card | File | Leaves room for |", "|---|---|---|"]
        out += [f"| {c} | `{r['label']}` | {leaves} |" for c, r, leaves in table]
        out += ["", "> [!TIP]", "> If you are choosing between two neighbouring files, take the larger one. The",
                "> steps between them cost one or two gigabytes and buy noticeably fewer wrong words.", ""]
        return out

    def running(self):
        f = self.fname(self.rec)
        out = ["## Running it", "", "```bash", f"llama-server -m {f} -ngl 99 -c 8192", "```", "",
               "The chat template is inside every file, llama.cpp applies it on its own.",
               self.note("prompt format block and the thinking toggle, if the family has one"), ""]
        if self.inv.get("mtp_blocks"):
            out += ["The model ships a multi token prediction head. It is inside every file here",
                    "and needs no extra download:", "", "```bash",
                    f"llama-cli -m {f} --spec-type draft-mtp -ngl 99 -c 8192", "```", ""]
        out += ["## Speed", "", self.note("llama-bench prompt and generation t/s at 8k and 32k for the recommended file"), ""]
        return out

    def vision(self):
        mm = self.mmproj()
        if not mm:
            return []
        f16 = next((x for x in mm if x.endswith("-F16.gguf")), mm[0])
        out = ["## Images and video", "",
               "This model reads images, not only text. That needs one extra file, the vision",
               "projector, which is separate from the quant and shared by all of them:", ""]
        for x in mm:
            out.append(f"- `{x}`" + (" for most people" if x.endswith("-F16.gguf") else " if your setup prefers bfloat16"))
        out += ["", "Download it once alongside whichever quant you picked:", "", "```bash",
                "llama-mtmd-cli \\", f"  -m {self.fname(self.rec)} \\", f"  --mmproj {f16} \\",
                "  --image your-photo.jpg -ngl 99 -c 8192 \\", '  -p "What is in this image?"', "```", "",
                self.note("family specific vision flags (e.g. --image-min-tokens) and the demo image test"), ""]
        return out

    def editorial(self):
        return ["## How these compare to other builds", "",
                self.note("other publishers measured against the same reference (kld_ext), same-name table"), "",
                "## What we found while building these", "",
                self.note("findings from this release: layout experiments, bands, groups that mattered"), ""]

    def corpus_section(self):
        c = self.corpus
        if not c:
            return ["## The calibration data", "", self.note("calibration corpus section: no corpus manifest"), ""]
        ct = c.get("calib_train", {})
        names = {"agentic": "agentic and tool use", "longctx": "long context", "vocab_sweep": "vocabulary sweep",
                 "structured": "structured data"}
        recipe = c.get("build") or self.im.get("recipe", "?")
        out = ["## The calibration data", "",
               "Our importance matrix comes from a corpus we built for this model and",
               "published: [AtomicChat/calib-corpora](https://huggingface.co/datasets/AtomicChat/calib-corpora),",
               f"recipe `{recipe}`. It is {ct.get('tokens', 0):,} tokens across {ct.get('documents', 0):,} documents:", "",
               "| Part | Share |", "|---|---:|"]
        shares = ct.get("shares_actual_percent") or {}
        for k, v in sorted(shares.items(), key=lambda kv: -kv[1]):
            out.append(f"| {names.get(k, k)} | {v:.1f}% |")
        out += ["", "Every conversation is rendered through **this model's own chat template**, so",
                "the chat markup appears as the single tokens the model actually sees.", "",
                "> [!IMPORTANT]", "> If you build your own importance matrix from this corpus, pass",
                "> `--parse-special` to `llama-imatrix`. Without it the chat markup is read as",
                "> ordinary punctuation.", ""]
        return out

    def measured(self):
        ev, man = self.evalset, self.manifest
        ctx, chunks = man.get("context", "?"), man.get("chunks", "0")
        if chunks in ("0", "?"):
            row_q = next((r["quality"].get(ev, {}) for r in self.rows if r["quality"].get(ev)), {})
            chunks = str(row_q.get("chunks", "?"))
        ppl = re.search(r"Final estimate: PPL = ([0-9.]+) \+/- ([0-9.]+)", self.base_log)
        hw = self.hardware()
        out = ["## How we measured", "",
               "Reference: the original BF16 weights, converted to GGUF and run unquantized."]
        if ppl:
            out.append(f"Perplexity on the held-out set is {ppl.group(1)} plus or minus {ppl.group(2)}.")
        out += ["", f"Held-out text: `eval_{ev}` from our calibration dataset, never used for",
                f"calibration. {chunks} chunks at {ctx} context.", "",
                "Metric: per-token KL divergence of each file's predictions against the",
                "reference, plus top-1 agreement.", ""]
        out.append(f"Hardware: {hw}." if hw else self.note("hardware line"))
        out += ["", "The raw logits of the reference are published in the metrics repo, split into",
                "parts because of the file size limit. With them you can measure your own build",
                "against exactly the same point we did:", "", "```bash",
                f"cat base-{ev}.kld.*.part > base-{ev}.kld", "",
                f"llama-perplexity -m your-quant.gguf -f eval_{ev}.txt \\",
                f"  --kl-divergence-base base-{ev}.kld --kl-divergence -c {ctx} -ngl 99"
                + (f" --chunks {chunks}" if man.get("chunks", "0") != "0" else ""), "```", ""]
        return out

    def reproduce(self):
        rung = next((r for r in self.ladder.get("rungs", []) if r["label"] == self.rec), None)
        im = self.im
        bf16 = f"{self.a.stem}-BF16.gguf"
        shards = im.get("shards", "N")
        run = (f"  -ngl 99 -c {im.get('ctx', 512)} -b {im.get('batch', 8192)} -ub {im.get('batch', 8192)} "
               "--parse-special")
        if shards == "1":
            out = ["## Reproducing a file", "",
                   "The importance matrix was collected on the BF16 weights rather than on a",
                   "quantized copy, in one run:", "", "```bash",
                   f"llama-imatrix -m {bf16} -f calib_train.txt -o imatrix.gguf \\",
                   run + (f" --chunks {im['chunks']}" if im.get("capped") == "1" else ""), "```", ""]
        else:
            out = ["## Reproducing a file", "",
                   "The importance matrix was collected on the BF16 weights rather than on a",
                   f"quantized copy, split across {shards} workers and merged. Splitting by chunk",
                   "range gives the same result as one long run, because the statistic is a sum:", "", "```bash",
                   f"llama-imatrix -m {bf16} -f calib_train.txt -o shard-0.gguf \\",
                   run + f" --from-chunk 0 --chunks {im.get('per_shard', 'N')}", "",
                   f"llama-imatrix -m {bf16} --in-file shard-0.gguf,shard-1.gguf,... \\", "  -o imatrix.gguf",
                   "```", ""]
        if not rung:
            return out + [self.note(f"quantize command: {self.rec} is not in ladder.json"), ""]
        cmd = ["llama-quantize"] + (["--imatrix imatrix.gguf"] if rung["imatrix"] else [])
        cmd += [f"--tensor-type '{p}={t}'" for p, t in rung["rules"]]
        cmd += rung["flags"]
        if rung.get("file_type") is not None:
            cmd.append(f"--override-kv general.file_type=int:{rung['file_type']}")
        cmd.append(f"{bf16} {self.fname(self.rec)} {rung['ftype']}")
        out += [f"Then, for `{self.rec}`:", "", "```bash", " \\\n  ".join(cmd), "```", "",
                "The rules for every other file are in `ladder/` in the metrics repo, next to",
                "its quantization log. The first rule that matches a tensor wins.", ""]
        return out

    def details(self):
        blocks = (self.inv.get("block_count") or 0) - len(self.inv.get("mtp_blocks") or [])
        bits = [f"{blocks} layers"]
        for key, what in (("embedding_length", "hidden size"), ("feed_forward_length", "feed-forward size")):
            if isinstance(self.m(key), int):
                bits.append(f"{what} {self.m(key)}")
        if self.vocab():
            bits.append(f"vocabulary {self.vocab():,}")
        if self.m("context_length"):
            bits.append(f"context {self.m('context_length'):,}")
        n_mtp = len(self.inv.get("mtp_blocks") or [])
        out = ["## Model details", "", ", ".join(bits) + "."
               + (f" {'One' if n_mtp == 1 else n_mtp} multi token prediction head{'s' if n_mtp > 1 else ''}." if n_mtp else ""),
               ""]
        tag = self.build_tag()
        out.append(f"Needs a recent llama.cpp with `{self.arch}` architecture support."
                   + (f" Built and tested with llama.cpp {tag}." if tag else ""))
        return out

    def render(self):
        if not self.rows:
            raise SystemExit("results.json has no rows: run the results stage first")
        parts = [self.front(), self.header(), self.pick(), self.fits(), self.running(), self.vision(),
                 self.editorial(), self.corpus_section(), self.measured(), self.reproduce(), self.details()]
        text = "\n".join(line for part in parts for line in part).rstrip() + "\n"
        return re.sub(r"\n{3,}", "\n\n", text)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--stem", required=True)
    ap.add_argument("--main", required=True)
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--main-files", help="file listing the main repo, one path per line (mmproj detection)")
    ap.add_argument("--recommended")
    ap.add_argument("--evalset", default="neutral")
    ap.add_argument("-o", "--out", default="-")
    a = ap.parse_args()
    if a.main_files:
        with open(a.main_files) as f:
            a.main_files = [x.strip() for x in f if x.strip()]
    card = Card(a)
    text = card.render()
    if a.out == "-":
        sys.stdout.write(text)
    else:
        with open(a.out, "w") as f:
            f.write(text)
    for t in card.todo:
        print(f"TODO {t}", file=sys.stderr)


if __name__ == "__main__":
    main()
