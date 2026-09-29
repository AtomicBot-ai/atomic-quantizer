# Release day: a new Qwen

Order of work and the points where a person decides. Durations are for a 27B
dense model on one 4x RTX 5090 box, scaled from the Qwen3.5-2B smoke run
(`rehearsal-qwen3.8-27b.md`, "Measured"); replace them with the 27B rehearsal's
`run.jsonl` numbers once it has run.

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
- **profile**: dense hybrid -> `dense-hybrid`; MoE with the PLE table and
  hyper-connections (`qwen4exp`) -> `moe-qwen4exp`; another MoE -> `moe-hybrid`,
  expect the generator to list tensor groups it does not know (next step). A small model
  with tied embeddings (no `output.weight` in the inventory) still takes
  `dense-hybrid`: its head gets the output types (`tied_embeddings`), and on
  Qwen3.5-4B that ladder beat the hand masks of `qwen35-masks` by a third.

The same questions answered by tools, on the laptop, in about ten minutes (the
dry run reads only safetensors headers; a llama.cpp checkout at the commit you
will pin must know the class):

```bash
python lib/probe_arch.py Qwen/Qwen4-XXB Qwen/Qwen3.8-27B Qwen/Qwen3.8-Flash-Next \
    --llama-cpp ~/llama.cpp --work /tmp/probe --md /tmp/probe/table.md
python lib/tok_fingerprint.py Qwen/Qwen4-XXB --against fingerprints/qwen3.8-27b.json
python lib/corpus_check.py template Qwen/Qwen4-XXB
python lib/ladder_gen.py --inventory /tmp/probe/Qwen--Qwen4-XXB.inventory.json \
    --profile profiles/moe-qwen4exp.yaml      # or dense-hybrid
```

- `probe_arch`: rows that are not a multiple of 256 and their share, the MTP
  block (kept or dropped by the converter), the GET_ROWS share (the PLE table
  was 28.9% of Flash-Next and no imatrix reaches it).
- `tok_fingerprint`: SAME TOKENIZER AND TEMPLATE -> take the Qwen3.8 build as it
  is. The 3.5 -> 3.8 step appended 7 audio tokens to an identical BPE
  vocabulary: that reads as a new tokenizer, but the sweep only needs those ids
  added. A new template -> re-render (`build.py` with the new model dir).
- `corpus_check template`: FAIL stops the build. Read the notes: whether earlier
  reasoning is kept, the thinking-off tail, the reasoning efforts the template
  accepts (Qwen3.8 refuses `high`), the vision markers.
- `ladder_gen`: no refusal. A refusal names the tensor and the reason; add a
  role or change a type in the profile.

After the imatrix (MoE: before any rung):

```bash
python lib/im_report.py all imatrix.gguf --vs imatrix-half.gguf --inventory inventory.json \
    -o imatrix.stats.txt --json im-report.json      # exit 3: dead expert or not converged
python lib/band_select.py imatrix.stats.txt --inventory inventory.json --profile profiles/moe-qwen4exp.yaml -o bands.json
python lib/corpus_check.py specials calib_train.txt --tokenizer Qwen/Qwen4-XXB
```

A dead expert (count 0) at 1-3 bits is noise: add the domain that routes to it
to the pool and recompute one shard, not the whole corpus. On the August
Flash-Next imatrix, 4000 chunks left 7 experts dead (blk.0 and blk.47) and
`ffn_down_exps` had not converged (per-expert cos 0.93 between 1200 and 4000
chunks; every other role above 0.994).

## 1. GGUF (5-7 h)

```bash
caffeinate -i python driver/release.py gguf --model Qwen/Qwen4-XXB --recipe <build> --profile <profile> \
    --llama-commit <sha>
```

Where the hours go, scaled from the 2B: convert ~30 min (half of it the CUDA
build of llama.cpp), base reference ~10 min, imatrix ~2 h (two shards on four
cards; it is bound by copying activations to the host, not by the GPU), ladder
seconds, 16 rungs at 3-5 min each on 128 cores plus uploads.

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

## The Qwen line on 2026-09-29 (what a new row is compared with)

`probe_arch.py` over the proxies, llama.cpp `957538960`; hashes are sha256 of the
files (first 12). "rows not /256" are the quantisable row lengths that k and i
types cannot take, with their share of the parameters.

| model | arch | blocks (full attn) | MTP | experts | rows not /256 (share) | vocab / tokenizer | tokenizer.json | chat_template | vision | GET_ROWS: embd + PLE | BF16 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3.5-0.8B | qwen35 | 24 (6) | 1 (in GGUF) | dense | none | 248320 / 248070 | 5f9e4d4901a9 | 273d8e0e683b | yes | 32.90% + 0.0% (tied) | 1.5 GB |
| Qwen3.5-2B | qwen35 | 24 (6) | 1 (in GGUF) | dense | none | 248320 / 248070 | 5f9e4d4901a9 | 273d8e0e683b | yes | 26.18% + 0.0% (tied) | 3.9 GB |
| Qwen3.5-4B | qwen35 | 32 (8) | 1 (in GGUF) | dense | none | 248320 / 248070 | 5f9e4d4901a9 | a4aee8afcf2e | yes | 14.69% + 0.0% (tied) | 8.7 GB |
| Qwen3.8-27B | qwen35 | 64 (16) | 1 (in GGUF) | dense | none | 248320 / 248077 | 0997f410c57a | c3cf9e34abf4 | yes | 4.65% + 0.0% | 54.6 GB |
| Qwen3.8-Flash-Next | qwen4exp | 48 (12) | 1 (dropped) | 512/10 | 160 (28.9%: per_layer_token_embd), 640 (22.8%: ffn_down_exps/ffn_down_shexp), 320 (0.2%: hc_attn_up/hc_ffn_up/output_hc_up), 4 (0.0%: ple_conv1d) | 248320 / 248077 | 0997f410c57a | c3cf9e34abf4 | yes | 0.36% + 28.9% | 354.0 GB |

- Tokenizer: one BPE vocabulary through the whole line (vocab.json and merges.txt
  identical); 3.8 appends 7 audio/tts tokens (ids 248070-248076) to 3.5's 26
  added tokens. Qwen3.8-27B and Flash-Next share tokenizer.json and the template
  byte for byte, which is why `builds/qwen3.8-27b` and `builds/qwen3.8-flash-next`
  are the same file.
- Template: 3.5-0.8B/2B think only when asked, 3.5-4B thinks by default, 3.8 adds
  `reasoning_effort` (low, medium, xhigh; xhigh by default, written into the
  system block) and keeps the reasoning of earlier turns (`preserve_thinking`).
- Flash-Next: the converter drops the MTP block (`no_mtp`), the n-gram table is
  28.9% of the parameters in rows of 160, and the pinned `1692f9e50bb2` predates
  `qwen4exp` (added in `6c84c7d5d`, 2026-08-27; hc ops and sparse FA later).
