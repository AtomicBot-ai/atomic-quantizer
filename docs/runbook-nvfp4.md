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
GPU, and makes the file 8 GiB larger. Nothing on this route shrinks anything.
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
| `atomic` | our calib-corpora build, the model's own chat markup, 512 windows of 4096 | about 33,000 |

Theory, worked through in the chat that produced this route, says the three
will sit within a few percent of each other on KLD. The input of w1/w3 is
bounded at 189 by RMSNorm and the norm weights, the input of w2 at 100 by the
SwiGLU clamps, and E4M3 with a flat scale of 1.0 covers everything from 0.094
to 2688 with full precision. Calibration helps only the quiet blocks below
0.094 and carries a risk of its own: a per-expert ceiling set by the maximum
seen on a few hundred tokens clips whatever exceeds it later.

So the claims to pre-register, in this order, before the first number exists:

1. **Coverage.** `nvfp4_coverage` counts experts that received no calibration
   token and got a guessed `input_scale`. This number is guaranteed to separate
   `nvidia` from `atomic`, and it is meaningful: a guessed ceiling is where
   clipping comes from.
2. **Divergence.** Mean and p99 KLD against the native checkpoint, top-1
   agreement, perplexity, on `neutral`, `code` and `agentic`. Report all three
   checkpoints beside the noise floor from `nvfp4_repeat`. If `atomic` is not
   outside three noise floors of `flat`, say so: that is the finding, and it is
   consistent with the theory.
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

`nvfp4_setup` prints the pip line for the calibration deps: modelopt from the
clone, `tilelang==0.1.8`, `torch>=2.10`, `safetensors>=0.7`. Install them, then:

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

```bash
source /quantizer/scripts/foundry-nvfp4.sh
nvfp4_get src ; nvfp4_get eval
nvfp4_stand            # prints the docker run and build commands, run them by hand
```

Inside the container, after the build, `nvfp4_stand`'s last lines re-source
this file with `NVFP4_ROOT=/host`, and then:

```bash
nvfp4_ref              # native checkpoint: the reference logprobs
nvfp4_repeat           # reference against itself: the noise floor
nvfp4_score /host/nvfp4/DeepSeek-V4.1-Flash-NVFP4-flat   flat
nvfp4_score /host/nvfp4/DeepSeek-V4.1-Flash-NVFP4-nvidia nvidia
nvfp4_score /host/nvfp4/DeepSeek-V4.1-Flash-NVFP4-atomic atomic
nvfp4_kld flat ; nvfp4_kld nvidia ; nvfp4_kld atomic
nvfp4_table
```

One `nvfp4_score` is one model load and every corpus in `NVFP4_CORPORA`,
`neutral code agentic` by default, so the table comes out per corpus from five
loads, not fifteen. Loading is the slow step on this stand, scoring is not.

Run every measurement without `--speculative-config`. The export leaves the
DSpark draft experts in MXFP4 while `moe_quant_algo` is a global switch, and
that combination has not been loaded by anyone yet.

## The measurement protocol

Same as the published GGUF and MLX tables so the numbers can sit side by side:
windows of 4096 tokens starting with BOS, only the second half scored, 24
windows, reference first in the divergence, raw token ids in, no chat
template.

One difference has to be printed beside every number. vLLM returns top-K log
probabilities, not the whole vocabulary, so `nvfp4_kld.py` gives a bracket:
`low` assumes every reference token the candidate did not rank sits at the
candidate's K-th probability, `high` spreads the candidate's leftover mass
uniformly over the rest of the vocabulary. The truth is between. With K = 512
on a language model the two agree to the third decimal; on a synthetic flat
distribution with K = 64 the tool bracketed an exact 0.0103 as 0.0089 to
0.0528, which is the worst case and is why K is 512. Top-1 agreement and
perplexity need no bracket and are exact.

`nvfp4_repeat` scores the reference against itself. On a deterministic engine
it is zero; on vLLM with batching it may not be, and whatever it is, three
times it is the smallest gap the table may call a difference.

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
repacked at load, and nobody has loaded this combination before. Budget a
day of B200 time and expect to use half of it. A 4×B200 stand also fits, at
`NVFP4_TP=4`, with the Engram tables off the GPUs; the loads take the same
time, the hour costs half.

The cheap risk reducer before renting Blackwell: on the calibration box, once
the exports are done, build the same branch and run `nvfp4_ref` and
`nvfp4_score ... flat`. That is about two hours of H200 and it settles whether
the NVFP4 checkpoint loads at all and whether the cast is lossless. It says
nothing about the recipe, which on Hopper is not exercised.

## What is not verified

Nothing on this page has run end to end. In particular:

- The patched `ptq.py` has not been executed against V4.1. The two edits are
  the two known incompatibilities; there may be a third.
- No one has loaded an NVFP4 V4.1 checkpoint in vLLM. The loader's name
  mapping and quant config route it, and the same route serves V4 NVFP4
  checkpoints from NVIDIA, but the branch carries no test for it.
- `vllm/vllm-openai:nightly` moves daily. If the build on top of it fails on
  a dependency mismatch, pin the image to the tag printed by `docker pull`.
- The `dsv41-flash` calibration build does not exist yet.
