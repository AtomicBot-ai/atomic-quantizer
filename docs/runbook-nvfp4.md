# Runbook: an NVFP4 release of DeepSeek-V4.1-Flash

`scripts/foundry-nvfp4.sh` is self contained. It does not source `foundry.sh`
and it never builds llama.cpp. Read this page once before renting anything,
because the trap on this route is not technical: it is measuring the wrong
thing on the wrong GPU and publishing a table that means nothing.

## What is different from GGUF and MLX

**The weights are already quantized.** V4.1-Flash ships its routed experts in
MXFP4: E2M1 nibbles with one E8M0 scale per 32 weights, 259 GiB of the 475 GiB
checkpoint. NVFP4 keeps every nibble and rewrites the scales as E4M3 per 16
plus one fp32 per tensor. The conversion is lossless by construction, needs no
GPU, and makes the checkpoint 15.8 GiB larger (17.00 GB: the expert scales
double, 543 G parameters at 1/32 byte each). Nothing on this route shrinks
anything.
The Engram tables, 183 GiB of FP8, stay FP8: no engine loads them any other way.

**The recipe is one scalar per expert projection.** NVIDIA's recipe for V4 is
that cast plus a calibration pass that produces `input_scale`, one fp32 per
expert for w1/w3 and one for w2, about thirty thousand numbers. It places the
E4M3 window for the FP4 activations. That is the whole difference between a
"real" NVFP4 checkpoint and a cast: the scalar's value.

**The scalar is consumed only on Blackwell.** On Hopper vLLM runs NVFP4 experts
through Marlin as W4A16 and drops the activation scales; every NVFP4 build of
this model produces identical bits there. So the calibration can run on H200,
but the measurement must run on B200, on the generic FlashInfer/CUTLASS NVFP4
MoE path, or it measures nothing about the recipe.

**Speed is not on the table.** The three checkpoints this runbook produces are
byte for byte the same layout and run the same kernels; they cannot differ in
throughput. NVFP4 against the native MXFP4 checkpoint is a separate question
and the vLLM branch cannot answer it fairly: the native path goes through the
MegaMoE/DeepGEMM kernel, NVFP4 through the generic fused MoE. Publish quality
numbers, not speed numbers.

## What "better" can honestly mean

Three checkpoints, one code path, one variable:

| name | `input_scale` comes from | tokens per expert |
| --- | --- | --- |
| `flat` | nothing, 1.0 everywhere. What the two community casts on the hub carry | 0 |
| `nvidia` | cnn_dailymail + nemotron-post-training-dataset-v2, 64 samples of 512 | about 500 |
| `atomic` | our calib-corpora build rendered in the model's own chat markup, 512 windows of 4096 | about 33,000 |

Theory, worked through in the chat that produced this route, says the three
will sit within a few percent of each other on KLD. The input of w1/w3 is
bounded at 189 by RMSNorm and the norm weights, the input of w2 at 150 by the
SwiGLU clamps times the 1.5 routing weight (the calibration saw exactly 150.00), and E4M3 with a flat scale of 1.0 covers everything from 0.094
to 2688 with full precision. Calibration helps only the quiet blocks below
0.094 and carries a risk of its own: a per-expert ceiling set by the maximum
seen on a few hundred tokens clips whatever exceeds it later.

So the claims to pre-register, in this order, before the first number exists:

1. **Coverage.** `nvfp4_coverage` counts experts that received no calibration
   token and got a guessed `input_scale`. This number is guaranteed to separate
   `nvidia` from `atomic`, and it is meaningful: a guessed ceiling is where
   clipping comes from.
2. **Divergence.** The KL lower bound, top-1 agreement and perplexity against
   the native checkpoint on `neutral`, `code` and `agentic`, several runs per
   build, paired per-window differences with intervals that resample windows
   and runs. If the interval for `atomic − flat` contains zero, say so: that
   is the finding, and it is consistent with the theory.
3. **Tails.** `p99_kld` and `max_kld` per corpus. If calibration hurts, it hurts
   here, on the tokens that exceeded the calibrated ceiling.

Never publish a table from a Hopper stand, and never a throughput column.

## The two boxes

| box | GPUs | RAM | disk | does |
| --- | --- | --- | --- | --- |
| calib | 8×H200 or 8×B200 | 600 GB | 2 TB NVMe | reshard, three calibrations, three exports, push |
| stand | any Blackwell that holds ~290 GiB of weights: 8×B200, 8×B300, or 8×RTX PRO 6000 (96 GB) | 300 GB | 2 TB NVMe | vLLM branch build, reference, three measurements |

Which Blackwell matters less than that it is Blackwell. vLLM's NVFP4 W4A4
MoE kernels that consume `input_scale` are gated on compute capability: the
FlashInfer TRT-LLM and CuTe DSL backends on the 10.x family (B200, B300,
GB200, GB300), FlashInfer CUTLASS and vLLM's own CUTLASS on 10.x and 12.0
(RTX PRO 6000 Blackwell). Hopper gets Marlin, which runs W4A16 and drops the
activation scales. RTX PRO 6000 is the cheapest stand and the least travelled
path; print the GPU and the selected backend beside every number, because the
activation quantization is implemented separately in each backend.

RAM on the calib box is set by DeepSeek's `convert.py`, which holds every
shard of every rank in memory before writing. Disk on it is the source (475
GiB), the resharded copy (475 GiB, deletable after the calibrations) and the
exports: each one rewrites only the shards that hold experts, about 300 GiB,
and hard links the Engram shards, so three exports are another 900 GiB. Peak
is just under 1.9 TB. Uploading the three is 1.45 TB of egress, which on a
rented box is the line item to look at before the GPU price. RAM on the stand is set by the
Engram tables, which vLLM keeps in pinned host memory by default.

`nvfp4_box calib` and `nvfp4_box stand` print the exact command list for each.

## Calibration box

```bash
export HF_TOKEN=hf_...
git clone https://github.com/AtomicBot-ai/atomic-quantizer /quantizer
source /quantizer/scripts/foundry-nvfp4.sh
nvfp4_setup            # clones modelopt at the pinned commit and patches ptq.py
nvfp4_persist
```

`nvfp4_setup` clones and patches modelopt; `nvfp4_calib_env` builds the
calibration venv: torch from the cu128 index, `tilelang==0.1.8`, modelopt, and
`transformers<5.15` because modelopt pins it there. The venv is deliberate:
the vLLM image ships a newer transformers, and a box that does both roles
must not let one install break the other. Every calibration function picks
the venv up when it exists. Then:

```bash
nvfp4_get src          # 475 GiB
nvfp4_get calib        # builds/dsv41-flash/calib_train.txt
nvfp4_reshard          # convert.py --model-parallel 8 --expert-dtype fp4, CPU, hours
nvfp4_calib_jsonl      # windows of 4096 tokens as jsonl, what modelopt's loader takes
nvfp4_amax_flat        # the no-calibration amax, seconds
nvfp4_calib nvidia     # about an hour, the first tilelang compile is most of it
nvfp4_calib atomic     # a few hours
nvfp4_coverage nvidia ; nvfp4_coverage atomic
nvfp4_export flat ; nvfp4_export nvidia ; nvfp4_export atomic
nvfp4_push flat ; nvfp4_push nvidia ; nvfp4_push atomic
```

> [!IMPORTANT]
> `nvfp4_get calib` needs a `dsv41-flash` build in calib-corpora. V4.1 changed
> the chat markup relative to V4: DSML tags carry a leading space, reasoning
> effort is a number. The existing `dsv4-flash-0731` build renders V4 markup
> and does not apply. Build one with `tools/build.py` and a `dsv41-flash`
> recipe first; the model ships its encoder in `encoding/encoding.py`.

### The two patches to modelopt

modelopt's `examples/deepseek/deepseek_v4/ptq.py` was written for V4. Two
lines stop it on V4.1, and `nvfp4_setup` rewrites both through
`nvfp4_patch_ptq.py`, which refuses to touch a file that is not the pinned one:

- `Transformer(margs)` becomes `Transformer(margs, tokenizer)`. V4.1's Engram
  layers hash n-grams over a compressed token map built from the tokenizer, and
  the constructor needs it.
- The FP8 dequant used for the shared expert during calibration assumed
  128×128 blocks. V4.1 stores 32×32. The block is now read off the scale shape.

Everything else in the recipe carries over: the expert tensor names are the
same (`layers.N.ffn.experts.E.w[123]`), MTP and DSpark experts are excluded by
the quant config and left in MXFP4 by the export, and the export's regexes
match the 48-shard layout without change.

### What the export writes

`quantize_to_nvfp4.py --cast_mxfp4_to_nvfp4` replaces every routed expert's
`.weight` and `.scale` with `.weight` (packed E2M1), `.weight_scale` (E4M3 per
16), `.weight_scale_2` and `.input_scale` (fp32 scalars), hard links every
other file from the source, and rewrites `config.json` with
`moe_quant_algo: NVFP4`. That key is what the vLLM `deepseek_v41` loader keys
on to route experts to the NVFP4 fused MoE. `nvfp4_export` checks the three
config fields the loader reads before declaring success. The `ignore` list the
export also writes is not read by vLLM's FP8 config, which looks for
`ignored_layers`; the attention path stays FP8 as it should.

## Stand box

The vLLM branch that knows `deepseek_v41` is `vllm-project/vllm:dsv41-feat`,
PR #56214, pinned by sha because the branch is force pushed. It changes four
files under `csrc/`, so the precompiled wheel of its merge base cannot be used
and the extension has to be built. The nightly image has every dependency
compiled; vLLM is rebuilt on top of it, about an hour.

On vast.ai the image is the instance. The template:

| field | value |
| --- | --- |
| image | `vllm/vllm-openai:nightly`, or `cu129-nightly` when the host reports Max CUDA 12.x |
| one box for both roles | same image; the calibration deps go into their own venv via `nvfp4_calib_env`, disk 3500 GB or more |
| launch mode | SSH. vast overrides the image ENTRYPOINT in this mode, so `vllm serve` never starts and you land in a shell |
| container disk | 2500 GB or more, set on the search page, cannot be changed after |
| on-start | empty |

Inside the instance `nvfp4_stand` prints the build, and then:

```bash
nvfp4_get src ; nvfp4_get eval
nvfp4_ref              # native checkpoint: the reference, and the smoke test of the branch on this GPU
nvfp4_repeat           # reference against itself: the noise floor
hf download AtomicChat/DeepSeek-V4.1-Flash-NVFP4-flat   --local-dir /nvfp4/DeepSeek-V4.1-Flash-NVFP4-flat
hf download AtomicChat/DeepSeek-V4.1-Flash-NVFP4-nvidia --local-dir /nvfp4/DeepSeek-V4.1-Flash-NVFP4-nvidia
hf download AtomicChat/DeepSeek-V4.1-Flash-NVFP4-atomic --local-dir /nvfp4/DeepSeek-V4.1-Flash-NVFP4-atomic
nvfp4_score /nvfp4/DeepSeek-V4.1-Flash-NVFP4-flat   flat
nvfp4_score /nvfp4/DeepSeek-V4.1-Flash-NVFP4-nvidia nvidia
nvfp4_score /nvfp4/DeepSeek-V4.1-Flash-NVFP4-atomic atomic
nvfp4_kld flat ; nvfp4_kld nvidia ; nvfp4_kld atomic
nvfp4_table
```

Run `nvfp4_ref` before downloading anything else. It is the one step that
depends on nothing from the calib box, and on an RTX PRO 6000 stand it is
also the test of whether the branch runs on SM120 at all.

One `nvfp4_score` is one model load and every corpus in `NVFP4_CORPORA`,
`neutral code agentic` by default, so the table comes out per corpus from five
loads, not fifteen. Loading is the slow step on this stand, scoring is not.

Run every measurement without `--speculative-config`. The export leaves the
DSpark draft experts in MXFP4 while `moe_quant_algo` is a global switch, and
that combination has not been loaded by anyone yet.

## The measurement protocol

The window layout follows the GGUF and MLX tables: chunks of 4096 tokens
starting with BOS, only the second half of each chunk scored, 24 chunks,
reference first in the divergence, raw token ids in, no chat template. The
numbers are still not directly comparable with those tables: a different
engine, a different reference and a coarsened KL. Scoring only the second half
fixes the regime, predictions with 2k to 4k tokens of context; it does not make
the comparison fairer for short contexts, it leaves them out.

vLLM returns top-K log probabilities, not the whole vocabulary, so the KL is a
**lower bound** and is labelled as one. `nvfp4_kld.py` takes S = the
reference's top-512 ids that every compared run also ranked, with exact p and
q on S, and one bucket for everything else on both sides. Coarsening onto a
common partition can only lower the KL, so the number is guaranteed to be at or
below the true divergence. There is no upper bound from top-K data: a token the
candidate ranked below K can carry any amount of divergence, and an earlier
version of this tool that "bracketed" the truth with a uniform-tail guess was
wrong about that. The common-set mass is printed beside every number; on this
model its median is 1.00000 and its first percentile above 0.993.

Uncertainty is a bootstrap over windows, 2048 positions each, with 95 %
percentile intervals of the per-window mean; positions are not independent,
windows are treated as such. Two builds measured against the same reference are
compared with `nvfp4_compare`, the paired per-window difference with its own
interval. With several runs per build, `aggregate.py` in the metrics dataset
gives two intervals: windows only, with the run results fixed, and windows
plus runs, where every draw also resamples each build's runs with replacement.
The second is the one to quote; with three runs per build it is coarse, and it
is wider. An interval that contains zero means no convincing difference was
found, not that the two are equivalent; equivalence would need a tolerance
chosen before looking at the data, and none was.

`nvfp4_repeat` scores the reference against itself. On this stand two runs of
the same model diverge by 0.016 KL and flip 2–4 % of top-1 tokens, while a
build's mean divergence from the reference is stable to 0.0002–0.0006 across
runs. Non-deterministic MoE kernels flipping near-tied experts is the working
hypothesis, not an established cause. A single batch-size-1 run of the
original diverged from the batch-4 reference about as much as the batch-4
repeats did, which argues against batching as the source but does not settle
it: the batch-1 mode's own repeatability was not measured, and the logprob
extraction path was not tested separately. vLLM's batch invariance mode has
not been tried on this branch.

## How long the stand takes

Wall clock on 8×B200, everything going right, with the downloads and the
build overlapped:

| step | hours | notes |
| --- | --- | --- |
| image pull, vLLM build from the branch | 1 | csrc changed, no precompiled wheel |
| four checkpoints in, 475 GiB each | 2–3 | overlaps the build; the calib box's upload was the slow half |
| five model loads | 1.5–2.5 | ref, ref-repeat, flat, nvidia, atomic; 15–30 min each, the first one longest while compile caches fill |
| scoring, three corpora per load | 0.5 | seconds of prefill, minutes of Python turning 50 million log probabilities into arrays |
| KLD and table | 0.25 | |
| total | 4–6 | |

The number that decides the bill is the model load, and it is the least
predictable: the Engram tables go to pinned host memory, the NVFP4 experts get
repacked at load, and nobody had loaded this combination before. Budget a
day of B200 time and expect to use half of it. A 4×B200 stand also fits, at
`NVFP4_TP=4`, with the Engram tables off the GPUs; the loads take the same
time, the hour costs half.

The cheap risk reducer before renting Blackwell: on the calibration box, once
the exports are done, build the same branch and run `nvfp4_ref` and
`nvfp4_score ... flat`. That is about two hours of H200 and it settles whether
the NVFP4 checkpoint loads at all and whether the cast is lossless. It says
nothing about the recipe, which on Hopper is not exercised.

## What the first run measured (2026-09-10)

One box, 4×B200 on vast.ai, both roles, 3 h 40 min from `nvfp4_get src` to
the first table, an hour of fixing what this page now documents, and another
90 minutes of repeat runs. Published:
[AtomicChat/DeepSeek-V4.1-Flash-NVFP4-nvidia](https://huggingface.co/AtomicChat/DeepSeek-V4.1-Flash-NVFP4-nvidia)
and [AtomicChat/DeepSeek-V4.1-Flash-NVFP4-metrics](https://huggingface.co/datasets/AtomicChat/DeepSeek-V4.1-Flash-NVFP4-metrics).

KL is the lower bound described above, coarsened on the set common to all ten
runs (common-set mass: median 1.00000, first percentile above 0.993). Runs:
three of `nvidia`, three of `flat`, two repeats of the original and one with
batch size 1, each a fresh engine start. Reference perplexity 2.9685 neutral,
1.8919 code, 1.3861 agentic.

| corpus | build | runs | KL lower bound, mean of runs (run SD) | top-1 (run SD) | ppl | Δ ppl |
| --- | --- | --- | --- | --- | --- | --- |
| neutral | nvidia | 3 | 0.03548 (0.00020) | 94.10 % (0.05) | 2.9938 | +0.85 % |
| | flat | 3 | 0.03459 (0.00040) | 94.29 % (0.10) | 2.9901 | +0.73 % |
| | original repeats | 2 | 0.01596 (0.00013) | 96.13 % (0.08) | 2.9689 | +0.01 % |
| | original, batch 1 | 1 | 0.01676 | 95.90 % | 2.9645 | −0.13 % |
| code | nvidia | 3 | 0.02008 (0.00031) | 96.65 % (0.09) | 1.8991 | +0.38 % |
| | flat | 3 | 0.01942 (0.00060) | 96.77 % (0.06) | 1.8995 | +0.40 % |
| | original repeats | 2 | 0.01022 (0.00017) | 97.69 % (0.00) | 1.8907 | −0.06 % |
| | original, batch 1 | 1 | 0.01108 | 97.55 % | 1.8922 | +0.02 % |
| agentic | nvidia | 3 | 0.00888 (0.00011) | 98.34 % (0.05) | 1.3873 | +0.09 % |
| | flat | 3 | 0.00856 (0.00019) | 98.37 % (0.04) | 1.3875 | +0.10 % |
| | original repeats | 2 | 0.00537 (0.00020) | 98.67 % (0.05) | 1.3856 | −0.04 % |
| | original, batch 1 | 1 | 0.00585 | 98.65 % | 1.3863 | +0.02 % |

Paired per-window differences, per-window means averaged over runs first,
A − B with 95 % window bootstrap:

Intervals resample windows and, independently per build, runs (the reference
is fixed); the windows-only intervals are in `logs/aggregate.txt`.

| corpus | nvidia − flat, KL | nvidia − flat, top-1 | flat − original, KL | flat − original, top-1 | batch 1 − original, KL |
| --- | --- | --- | --- | --- | --- |
| neutral | +0.0009 [+0.0001, +0.0019] | −0.19 pt [−0.41, +0.03] | +0.0186 [+0.0159, +0.0217] | −1.84 pt [−2.11, −1.59] | +0.0008 [+0.0004, +0.0012] |
| code | +0.0007 [−0.0002, +0.0015] | −0.12 pt [−0.29, +0.04] | +0.0092 [+0.0061, +0.0125] | −0.92 pt [−1.29, −0.58] | +0.0009 [+0.0003, +0.0015] |
| agentic | +0.0003 [−0.0001, +0.0008] | −0.02 pt [−0.13, +0.09] | +0.0032 [+0.0025, +0.0040] | −0.30 pt [−0.40, −0.20] | +0.0005 [−0.0001, +0.0011] |

The pre-registered reading, in order:

1. **Coverage.** 15,246 of 15,360 routed experts calibrated (99.3 %); 342
   expert projections took the fallback scale, the maximum over all
   calibrated experts of the same projection across layers. All 16,986,931,200
   expert blocks had scales inside the exactly representable window, which is
   what the exporter's `cast_blocks_lossless` counts; on top of that, 12
   experts × 3 projections (0.42 G parameters) were dequantized from the source
   and from the export and compared element by element: identical, worst
   difference 0.0. The packed bytes differ in 11 % of positions, every one of
   them a −0 nibble the export normalized to +0. Script and report:
   `logs/cast-check.py`, `logs/cast-check.txt` in the metrics dataset.
2. **Divergence.** The calibration showed no convincing advantage in this
   experiment. The point estimates lean the other way, `nvidia` 0.0009 KL and
   0.19 points of top-1 behind `flat` on neutral, but once run variation is
   in the interval only the neutral KL difference stays clear of zero, and by
   a hair; top-1 on every corpus and KL on code and agentic do not. Code
   perplexity is even marginally lower for `nvidia` (1.8991 against 1.8995).
   The defensible statement is: no measurable benefit from the calibration on
   this model, and a slight lean toward the plain cast that three runs cannot
   confirm. A mechanism consistent with that lean is the one the theory
   section names, a per-expert ceiling from a few hundred tokens clipping what
   it did not see, but it is a hypothesis: no clipping was observed, and a
   changed scale also moves the rounding of every value, so nothing here shows
   `input_scale = 1.0` to be optimal. What is clearly separated from the
   original's own spread on every corpus is the cost of the NVFP4 W4A4 path as
   a whole, native MXFP4×MXFP8 kernels against the FlashInfer TRT-LLM NVFP4
   path: 1.8 points of top-1 and +0.7 to +0.9 % perplexity on neutral, half
   that on code, a tenth on agentic.
3. **Tails.** p99 and max move within what the repeats show; no
   calibration-induced clipping is visible at this resolution.

On the noise: two runs of the original diverge by 0.016 KL and disagree on
4 % of top-1 tokens, but a build's mean divergence from the reference is
stable to 0.0002–0.0006 across runs, so the mean is a usable statistic after
all. One batch-size-1 run left the spread in place (0.0168 against 0.0159 for
the batch-4 repeats), which argues against batching as the source without
settling it. Non-deterministic MoE kernels flipping near-tied experts remains
the working hypothesis.

The `agentic` corpus is agentic dialogue rendered in Muse Glimmer's markup
(`<|start|>`, `atem:function_calls`), not DeepSeek's DSML: V4.1's tokenizer
sees it as plain text with unfamiliar control strings. That is fine for
comparing builds on identical input and wrong to call "the model's own
markup". All three corpora are teacher-forced, scored one token at a time, and
say nothing about tool calls or long free trajectories. Their hashes as used
are in `logs/corpora-sha256.txt` of the metrics dataset. The eval corpora are
disjoint from every calib-corpora build by construction; overlap with the
CNN/DailyMail and Nemotron samples NVIDIA's recipe calibrates on was not
checked. Speed was not measured.

Timings that matter for the next run: reshard 2.5 min (the NVMe did 50 GB/s),
calibration 10 min of forwards plus compile, export 7 min per checkpoint, vLLM
branch build 12 min, the first load of every distinct config 30–40 min of
FlashInfer autotuning of which the NVFP4 MoE kernel alone is 20, and 11–16
min per load once the cache is warm. The autotune cache is keyed by the whole
vLLM config, model path included, so every checkpoint pays it once. Publishing
527 GB took 3.5 minutes: Xet deduplicates against the identical expert nibbles
already on the Hub, only the scales travel.

## What is not verified

The calibration box and the stand ran end to end once, on one 4×B200 box.
What has not:

- The patched `ptq.py` ran once on V4.1 on B200 with the 4-way reshard; other
  MP counts and Hopper have not been tried.
- The NVFP4 V4.1 checkpoint loads and scores in vLLM at `e47aa780` on B200 via
  the FlashInfer TRT-LLM NVFP4 MoE backend, six times over; other backends,
  SM120 and speculative decoding with the MXFP4 draft experts are untried.
- `vllm/vllm-openai:nightly` moves daily. If the build on top of it fails on
  a dependency mismatch, pin the image to the tag printed by `docker pull`.
- The `dsv41-flash` calibration build does not exist yet.
