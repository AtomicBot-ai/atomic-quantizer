# Release day: a new MiniMax (M3.1-Flash)

Prepared on MiniMax-M3 (`MiniMaxAI/MiniMax-M3` @ `f0e1c1e0`) while the new
weights are not public. Everything under "Baseline" was read off the hub with
`probe_arch`, `corpus_check`, `make_recipe` and a ladder dry run; nothing was
downloaded but config, tokenizer and shard headers.

**Go only if** the weights are public and llama.cpp converts them (master, or a
PR pinned by sha in the card). Otherwise the work stops at step 1.

## Baseline: MiniMax-M3

| | |
| --- | --- |
| class | `MiniMaxM3SparseForConditionalGeneration` (`minimax_m3_vl`), text + vision |
| size | 427.0B parameters, 854 GB BF16 on the wire, 59 shards |
| blocks | 60: 3 dense (`ffn_*`, width 12288), then 57 MoE |
| experts | 128 routed, top 4, width 3072, sigmoid router with bias; 1 shared, width 3072. Experts are 96.7% of the weights |
| attention | 64 q heads, 4 kv heads, head 128, qk norm per head, partial rope 64 of 128 |
| sparse attention (MSA) | indexer in the 57 MoE blocks: 4 heads x 128, top 16 blocks of 128. The converter writes the indexer F32 and llama-quantize never quantizes it |
| rows % 256 | all zero: 6144, 3072, 12288, 8192 (o_proj input), vision 1280/5120. No silent fallback to block-32 types |
| MTP | config says `num_nextn_predict_layers 1`, `num_mtp_modules 7`; the checkpoint has **no** MTP tensor |
| activation | `swigluoai`, alpha 1.702, limit 7.0, `routed_scaling_factor 2.0` |
| vocab | 200064 rows (200000 BPE + 61 added), BOS `]~b]`, EOS `[e~[` (200020), pad `]!p~[` |
| dtype | BF16 everywhere, router F32. Not vendor-quantized (a separate `MiniMax-M3-MXFP8` repo exists) |
| license | `minimax-community`: see "Stop signs" |
| llama.cpp | master `526c43b` (2026-09-29) has `conversion/minimax.py` `MiniMaxM3Model` + `MiniMaxM3VisionModel` and `src/models/minimax-m3.cpp` |

Chat markup, from `corpus_check`:

- system prompt is role `root`; `system` and `developer` go to a second, lower
  priority block. With no `root` message the template writes MiniMax's own
  default system prompt;
- thinking is switched by `thinking_mode` = `enabled` / `disabled` / `adaptive`
  (default adaptive), **not** `enable_thinking`. Tags `<mm:think>` `</mm:think>`,
  past turns keep their reasoning;
- tool calls are XML, every element prefixed with `]<]minimax[>[`, inside
  `<tool_call>`; arguments must be a dict (the pool stores dicts);
- 8 special tokens in a render: `]~!b[ ]~b] [e~[ ]<]minimax[>[ <tool_call>
  </tool_call> <mm:think> </mm:think>`. `--parse-special` is mandatory.

Ladder dry run, `moe-hybrid` on an inventory built from the shard headers
(GiB, predicted): AD-Q8_0 424.6, AD-Q4_K_M 251.5, AD-IQ3_S 183.5, AD-IQ2_M 155.6,
AD-IQ2_XXS 124.6, AD-IQ1_S 102.1. No refusal; edges are blocks 3, 4, 57-59 by
count, to be replaced by imatrix energy.

## 0. The moment the repo appears (15 min, laptop)

```bash
. ~/.venvs/atomic-pipeline/bin/activate && cd pipeline
python lib/tok_fingerprint.py MiniMaxAI/<new repo> --against fingerprints/minimax-m3.json
```

- `SAME TOKENIZER AND TEMPLATE`: the M3 recipe and build carry over.
- `SAME TOKENIZER, NEW TEMPLATE`: keep the vocab sweep, re-run `corpus_check`
  and re-render.
- `NEW TOKENIZER`: new recipe and build (`new_model`, `build_corpus`).

Then, with `FROOT` set on a Mac:

```bash
. scripts/foundry.sh
probe_arch MiniMaxAI/<new repo>          # diff against the table above
corpus_check MiniMaxAI/<new repo>
```

## Stop signs, check by hand

- **SwiGLU constants.** llama.cpp hardcodes alpha 1.702 and limit 7.0 for
  `swigluoai` (`llama-graph.cpp`, both the dense and the MoE path) and does not
  read them from the config. If the new config has another `swiglu_alpha` or
  `swiglu_limit`, the graph is silently wrong and every KLD is meaningless.
- **Converter class.** A new `architectures` string needs a register line in
  `conversion/minimax.py`; the old class name with new tensors (for example MTP
  weights, or a changed indexer) needs a converter change too. Compare the
  tensor table of `probe_arch` with M3's.
- **Vendor 4-bit.** If experts arrive in MXFP4/NVFP4/FP8: the base must be
  bit-exact BF16 (dequantize, never go through Q8_0), and experts at 4 bits are
  repacked, not quantized deeper without a measurement. Anything in MXFP4 gets
  no imatrix statistics.
- **Rows % 256** of every expert width. Anything else silently takes a block-32
  type.
- **MTP.** If the new checkpoint ships it, `mtp: q8_0` in the profile applies;
  if only the config announces it, there is nothing to keep.
- **License.** `minimax-community` grants non-commercial use. Commercial use
  (the Atomic Chat app counts) needs "Built with MiniMax M3" displayed and a
  notice to MiniMax, or written authorization above $20M yearly revenue; a
  modified model deployed commercially (the abliterated release) is named
  explicitly. Read the new license and its prohibited-use appendix before any
  repo goes public.

## 1. Hardware

BF16 GGUF is about 2 bytes per parameter. The imatrix wants it in VRAM plus the
logits tensor (`vocab x IM_BATCH x 4`, 1.6 GB at batch 2048) and headroom:

| params | BF16 | imatrix on BF16 | or Q8 proxy (say so in the card) |
| --- | --- | --- | --- |
| 427B (M3) | 854 GB | 8x B200 or 8x H200 | ~456 GB: 8x H100 80G, 8x RTX PRO 6000 |
| ~200B | ~400 GB | 8x H100 80G | ~212 GB: 8x RTX 5090 (256 GB), tight |
| ~110B | ~220 GB | 8x RTX 5090 (256 GB), tight | |

Disk on the quantize box for M3: the peak is the convert, `/src` 854 GB next
to the BF16 855 GB, about 1.7 TB. After `/src` is deleted: BF16 + ~80 GB KLD
reference (200k vocab) + the largest rung in flight (456 GB) is about 1.4 TB.
Ask for 2.5 TB. Traffic for M3: pull 854 GB, push BF16 855 GB and 21 rungs of
~4.8 TB, about 6.5 TB in all; at vast's ~$13/TB that is ~$85 before any rent,
so price the big pushes against a provider that does not bill traffic.

## 2. Corpus

Recipe from `make_recipe` (MoE shares 22/16/12/18/10/12/6/4, pin none,
`add_bos_per_document: false`). `auto_fmt` passes `thinking_mode` alongside
`enable_thinking`, so the recipe's thinking on/off split reaches this template.
Check in the first rendered documents that the system block says
`Current thinking mode: enabled` / `disabled`, not always `adaptive`.

## 3. Then the usual route

`release.py gguf --profile moe-hybrid`, edges from `--show-statistics`, mmproj
from the same checkout (`MiniMaxM3VisionModel`), 5 experiments per rung, KLD
against BF16 on `neutral`, and a behaviour pass on the recommended sampling
(temperature 1.0, top_p 0.95 in M3's `generation_config.json`) with thinking on,
tools and a long context.
