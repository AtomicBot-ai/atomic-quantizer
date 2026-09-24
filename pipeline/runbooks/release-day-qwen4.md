# Release day: a new Qwen

Order of work and the points where a person decides. Durations are for a 27B
dense model on one 4x RTX 5090 box; replace them with the rehearsal's
`run.jsonl` numbers once it has run.

## 0. Before renting anything (15 min)

```bash
. ~/.venvs/atomic-pipeline/bin/activate && cd pipeline
python driver/release.py status --model Qwen/Qwen4-XXB
```

Check by hand, each one is a stop sign:

- **architecture**: is `architectures` from the model's `config.json` registered
  in `conversion/` of the llama.cpp commit you will pin? If not: find the PR
  that adds it and pin that branch's commit with `--llama-commit` (and
  `--llama-repo` for a fork). No support means no GGUF today.
- **license**: Apache-2.0 or a license that allows derivatives and quantized
  redistribution. `qwen-community` style licenses: read before publishing.
- **calibration build**: is there `builds/<name>/` in `AtomicChat/calib-corpora`
  for this tokenizer? A new tokenizer needs `make_recipe` + `build_corpus` from
  foundry.sh first (about an hour, a person reads the shares).
- **profile**: dense hybrid -> `dense-hybrid`; MoE -> `moe-hybrid`, expect the
  generator to list tensor groups it does not know (next step).

## 1. GGUF (4-5 h)

```bash
python driver/release.py gguf --model Qwen/Qwen4-XXB --recipe <build> --profile <profile> \
    --llama-commit <sha>
```

Runs convert, base reference, imatrix, then stops at the **ladder review**: it
prints every rung with its predicted size and asks before quantizing. If the
generator refused, it prints why (uncovered tensors, rows not a multiple of 256,
low-bit on the MTP block): add a role or change a type in the profile, then
`python driver/release.py ladder --model ... --profile ...` and rerun `gguf`.

The box keeps running during the review; answer promptly or pass `--ladder-ok`
when the profile is already known to fit.

Each rung is verified against the ladder before it is uploaded; a mismatch stops
the run and nothing wrong is published. At the end `results.json` is merged and
printed. Repos are private.

## 2. Abliteration (3-5 h, parallel with step 1 on its own box)

```bash
python driver/release.py ablit --model Qwen/Qwen4-XXB
```

Needs Heretic support for the architecture (its model loading is transformers).
If the gate fails (`refusals <= 5/100`, `KL <= 0.10`), read the Pareto front in
`heretic/heretic.log` of the target repo, pick a trial and rerun:
`--trial-index N --resume --force` (no new trials, only export and evaluation).

Then by hand, before any GGUF of it: 10-20 prompts with thinking on, the refusal
must not survive inside `<think>` and the block must close.

```bash
python driver/release.py gguf --model AtomicChat/Qwen4-XXB-abliterated --recipe <build> \
    --profile <profile> --llama-commit <same sha> --ladder-ok
```

## 3. NVFP4 (1-2 h per model)

```bash
python driver/release.py nvfp4 --model Qwen/Qwen4-XXB --recipe <build>
python driver/release.py nvfp4 --model AtomicChat/Qwen4-XXB-abliterated --recipe <build>
```

The vLLM smoke test only proves the kernels on Blackwell; elsewhere it may fail
without the checkpoint being wrong.

## 4. Before making anything public

- `release.py status` for every model: all stages done.
- `results.json`: KLD grows monotonically as size shrinks; no row without a size.
- `release.py reap`: no box left running.
- Model card: `gguf` ends with the draft (`card/README.draft.md` in the metrics
  repo, and README.md in the main repo if it had none). The stage prints what is
  left for a person: the chart, comparisons with other builds, findings, speed,
  the vision demo. Edit the README on the hub; `release.py card --card-overwrite`
  regenerates it from scratch.
- Flip the repos to public.
