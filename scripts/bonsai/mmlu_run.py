#!/usr/bin/env python3
"""
Capability cost of the ablation: MMLU accuracy per adapter scale.

Same server plumbing as ab_run.py, so every config shares the seed,
sampling, system prompt and question set; only the adapter scale changes.
A GBNF grammar (root ::= [A-D]) forces the reply to be exactly one letter,
so nothing is lost to the model explaining instead of answering.

  python tools/mmlu_run.py --adapter orca=adapters/orca-all.gguf \
      --config base --config orca:1.5 --config orca:2 --out results/mmlu
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from ab_run import MODEL, ROOT, SERVER, post, sha256, wait_ready

LETTERS = "ABCD"
TEMPLATE = ("{question}\n\n{choices}\n\n"
            "Answer with a single letter: A, B, C, or D.")


def sample_questions(path: Path, n: int, seed: int) -> list[dict]:
    t = pq.read_table(path).to_pylist()
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(t))
    # stratify: walk subjects round-robin so no subject dominates the slice
    by_subject: dict[str, list[int]] = {}
    for i in idx:
        by_subject.setdefault(t[i]["subject"], []).append(i)
    order, subjects = [], sorted(by_subject)
    while len(order) < n:
        added = False
        for s in subjects:
            if by_subject[s]:
                order.append(by_subject[s].pop())
                added = True
                if len(order) == n:
                    break
        if not added:
            break
    return [t[i] for i in order]


def parse_letter(text: str) -> str | None:
    m = re.search(r"\b([ABCD])\b", text.strip().upper())
    return m.group(1) if m else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--adapter", action="append", default=[], help="name=path.gguf")
    ap.add_argument("--config", action="append", required=True, help="base | name:scale")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--questions", type=Path, default=ROOT / "data/mmlu_test.parquet")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--model", type=Path, default=MODEL)
    ap.add_argument("--system", default="You are a helpful assistant")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sample-seed", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=2)
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--port", type=int, default=8091)
    args = ap.parse_args()

    adapters = dict(a.split("=", 1) for a in args.adapter)
    ids = {name: i for i, name in enumerate(adapters)}
    configs = []
    for c in args.config:
        if c == "base":
            configs.append((c, None, 0.0))
            continue
        name, scale = c.rsplit(":", 1)
        if name not in ids:
            raise SystemExit(f"config {c}: unknown adapter {name!r}")
        configs.append((c, name, float(scale)))

    qs = sample_questions(args.questions, args.n, args.sample_seed)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "meta.json").write_text(json.dumps({
        "model": str(args.model),
        "adapters": {n: {"path": p, "sha256": sha256(Path(p))} for n, p in adapters.items()},
        "questions": str(args.questions), "n": len(qs), "sample_seed": args.sample_seed,
        "subjects": len({q["subject"] for q in qs}), "system": args.system,
        "seed": args.seed, "max_tokens": args.max_tokens,
        "configs": [c[0] for c in configs]}, indent=2))

    cmd = [str(SERVER), "-m", str(args.model), "-ngl", "99", "-c", str(args.ctx), "-np", "1",
           "--port", str(args.port), "--jinja", "--no-webui"]
    if adapters:
        cmd += ["--lora", ",".join(adapters.values()), "--lora-init-without-apply"]
    log = (args.out / "server.log").open("w")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    base_url = f"http://127.0.0.1:{args.port}"
    summary = {}
    try:
        wait_ready(base_url, proc)
        for cname, aname, scale in configs:
            lora = [{"id": i, "scale": (scale if n == aname else 0.0)} for n, i in ids.items()]
            rows, t0 = [], time.time()
            for i, q in enumerate(qs):
                choices = "\n".join(f"{LETTERS[j]}. {c}" for j, c in enumerate(q["choices"]))
                r = post(f"{base_url}/v1/chat/completions", {
                    "messages": [{"role": "system", "content": args.system},
                                 {"role": "user", "content": TEMPLATE.format(
                                     question=q["question"], choices=choices)}],
                    "seed": args.seed, "temperature": 0.0, "top_p": 1.0, "top_k": 1,
                    "max_tokens": args.max_tokens, "grammar": "root ::= [A-D]",
                    "chat_template_kwargs": {"enable_thinking": False},
                    "lora": lora})
                text = r["choices"][0]["message"].get("content") or ""
                got = parse_letter(text)
                rows.append({"i": i, "subject": q["subject"], "gold": LETTERS[q["answer"]],
                             "got": got, "response": text, "correct": got == LETTERS[q["answer"]]})
            with (args.out / f"{cname.replace(':', '@')}.jsonl").open("w") as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_ok = sum(r["correct"] for r in rows)
            acc = n_ok / len(rows)
            summary[cname] = {"n": len(rows), "correct": n_ok, "accuracy": acc,
                              "stderr": float(np.sqrt(acc * (1 - acc) / len(rows))),
                              "unparsed": sum(r["got"] is None for r in rows),
                              "seconds": round(time.time() - t0)}
            (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
            print(f"[{cname}] {n_ok}/{len(rows)} = {acc:.1%} in {summary[cname]['seconds']}s",
                  file=sys.stderr)
        print(f"\n{'config':<16}{'n':>5}{'accuracy':>10}{'+-':>7}{'unparsed':>10}")
        for c, s in summary.items():
            print(f"{c:<16}{s['n']:>5}{s['accuracy']:>9.1%}{s['stderr']:>7.1%}{s['unparsed']:>10}")
    finally:
        proc.terminate()
        proc.wait(timeout=30)


if __name__ == "__main__":
    main()
