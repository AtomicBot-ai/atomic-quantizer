# Release day: a new Qwen

Order of work and the points where a person decides. Durations and money for
a 27B dense model are measured: the Qwen3.8-27B rehearsal of 2026-09-29
(`rehearsal-qwen3.8-27b.md`, "Measured: the 27B"), vast.ai, one 4x RTX 5090 box
for convert, base and imatrix, then `--quant-boxes 4` of 2x RTX 5090.

The machine that runs `release.py` must stay awake until it prints the
release of the box: a sleeping laptop pauses the driver, not the box, and the
box bills for the whole night. On a Mac every command below goes under
`caffeinate -i`.

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
  generator to list tensor groups it does not know (next step). A small model
  with tied embeddings (no `output.weight` in the inventory) still takes
  `dense-hybrid`: its head gets the output types (`tied_embeddings`), and on
  Qwen3.5-4B that ladder beat the hand masks of `qwen35-masks` by a third.

## 1. GGUF (~3 h, ~$10-11)

```bash
caffeinate -i python driver/release.py gguf --model Qwen/Qwen4-XXB --recipe <build> --profile <profile> \
    --llama-commit <sha> --quant-boxes 4
```

Where the hours go, measured on the 27B (box side, from the nodes' own
`seconds`; the driver's minutes include any time this machine slept):

| stage | where | time | money |
|---|---|---|---|
| rent + probe | 4x RTX 5090, $2.38/h | 2 min | |
| node_convert | same box | 9.0 min (llama.cpp CUDA build included) | |
| node_base | same box | 5.4 min (88.4 GB reference, self-check 0) | |
| node_imatrix | same box, 2 shards on 2 GPU pairs | 98.8 min (9701 chunks of 512) | |
| gguf box total | | 1.93 h | $4.93 (GPU 4.59, disk 0.18, traffic 0.16) |
| quant, first rung on a fresh box | 2x RTX 5090, $1.18-1.31/h | 13-18 min: llama.cpp build, BF16 + reference download (~145 GB), one rung | |
| quant, each further rung | same | 4-5 min (K rungs), 9-16 min (IQ rungs): quantize 49-181 s K, 241-799 s IQ; KLD ~2.5 min | |
| quant stage, 16 rungs on 4 boxes | | ~50 min wall | ~$4.5 GPU + $0.2-0.6 traffic per box |

A run with nothing going wrong: ~2 h 50 min and ~$10-11. The rehearsal itself
cost $20.83 (vast charges for 36 instances, racers included) because of four
things now fixed or written down: a full box disk on the first rung (the nodes
now delete their upload copies), this laptop's own disk filling up (the driver
now survives it), two hosts billing $0.020 and $0.039 per GB of traffic, which
for one box was $2.90 of download against $1.13 of GPU (`vast.search` now asks
for `inet_down_cost<=0.01`), and the laptop sleeping twice with the lid closed
while four boxes waited (~$2.5; `caffeinate -i` does not cover a closed lid).

The imatrix is now two thirds of the wall time; it runs two shards on one box.
Spreading shards over boxes (`--im-boxes`, in progress) is the next saving.

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
