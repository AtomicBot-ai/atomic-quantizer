# Qwen3.8-Flash-Next: corpus and imatrix for the MoE variant of Qwen 4

The `qwen4exp` rehearsal before Qwen 4: a BF16 base, an MoE calibration build, an
imatrix that passes acceptance, and a check of band_select against the band
chosen by hand in August. The GPU part runs after the imatrix fan-out
(`node_imatrix` over several boxes) is merged.

## llama.cpp: `957538960`

Upstream master of 2026-09-23. It contains `6c84c7d5d` (qwen4exp, PR #27742,
2026-08-27) and `3cf03257f` (CUDA sparse FA for qwen4, 2026-09-20), checked with
`gh api repos/ggml-org/llama.cpp/compare/<sha>...957538960` (status `ahead`).
After it, only `ed7ac35e1e` (scheduler reserve, no numerics) touches
`src/models/qwen4exp.cpp`; `conversion/qwen4exp.py` and `tools/imatrix` are
unchanged since the arch landed.

The August numbers were made on the PR branch before `5fdfa62829` (2026-09-06,
GDN normalisation `max` -> `rsqrt`). Activations changed with it, so the
August imatrix and the August KLD are not a baseline for this commit: the BF16
reference and the imatrix are recomputed, and a rung is compared with August
only through a rebuild measured here.

## Hardware

The table does not go to the GPU. `per_layer_token_embd` (51.2B, 102 GB in
BF16) is created with `TENSOR_READ_LAZY` and read from the mmap on the host;
`token_embd` is an input layer tensor and sits in CPU memory. What the GPUs
hold, from the inventory: **250.3 GB** in BF16, about 133 GB in Q8_0.

| box | VRAM | BF16 (250 GB + logits 8 GB at batch 8192 + buffers) | Q8_0 |
|---|---|---|---|
| 8x RTX 5090 | 256 GB | no | yes |
| 4x RTX PRO 6000 | 384 GB | yes, ~120 GB spare | yes |
| 8x H100 80 GB | 640 GB | yes | yes |

4x RTX PRO 6000 is sm_120 like the August box, so the imatrix is BF16 as the
rule says and costs no more than the Q8 route on 8x 5090. The same box also
computes the BF16 reference logits the rungs need, which Q8 cannot give.

`node_imatrix` sizes shards with every tensor at 2 bytes, table included
(`MODEL_BYTES` = 354 GB): it still accepts 4x RTX PRO 6000 at batch 8192
(needs ~366, has ~391) and refuses what really does not fit, but it
over-counts by the 103 GB of GET_ROWS tables.

Host: RAM >= 512 GB (the 102 GB table stays mapped, the rest of the 354 GB file
wants to stay in page cache), disk 1 TB (source 354 GB + BF16 GGUF 354 GB +
imatrix shards, logits, rungs). Convert on the same box: moving 354 GB between
vast boxes costs time and ~$5 of traffic.

## Conversion: lazy

```bash
python convert_hf_to_gguf.py /src --outtype bf16 --outfile /gguf/qwen3.8-flash-next-bf16.gguf
```

The default lazy mode is what `conversion/qwen4exp.py` is written for: the
table concatenates into a `LazyChunkedTensor` with one shard resident at a
time. `--no-lazy` would hold every other tensor in RAM (~250 GB) for nothing.
The NaN that made `--no-lazy` a rule came from DeepSeek's FP8 dequantisation in
lazy mode; this checkpoint is BF16 and nothing is dequantised. Verify anyway:
the first llama.cpp tool that loads the BF16 (the reference run) gets
`--check-tensors`.

## Corpus: `builds/qwen3.8-flash-next-moe`

The August `builds/qwen3.8-flash-next` is byte for byte `builds/qwen3.8-27b`:
dense shares, and its vocabulary sweep never reached the pool. On the Flash-Next
tokenizer it covers 99.12% of the reachable ids. It stays for reproducing
August; the new build replaces it for `qwen4exp`.

Built on a laptop (tokenizer and template only, `FOUNDRY_MODEL_DIR` pointing
at them, pool 70 MB):

```bash
python tools/vocab_sweep.py --tokenizer /src/tokenizer.json --name qwen3.8-flash-next-moe
python tools/purge_overlap.py --whole-pool --build builds/qwen3.8-flash-next-moe \
    --against eval/neutral/eval_neutral.txt eval/code/eval_code.txt eval/code/eval_code_ext.txt
python tools/build.py --recipe recipes/qwen3.8-flash-next-moe.yaml --out builds/qwen3.8-flash-next-moe
```

| check | result |
|---|---|
| shares, actual | agentic 22.7, code 16.7, reasoning 12.4, multilingual 18.7, longctx 10.6, vocab 10.3, structured 4.5, graphics 4.2 |
| tokens | 4,846,205 in 2,755 documents (calib_longctx: 766,514 in 27); ~95k tokens per expert at 10 of 512 |
| `corpus_check coverage` | 246,867 of 246,930 reachable ids, 99.974% |
| `corpus_check specials` | every chat and tool marker >= 1,831 (`<tool_response>`); vision markers 1-2, excepted |
| overlap with eval (13-grams) | neutral 0, code 0, code_ext 0; agentic 4 distinct, all template boilerplate (the tool call instruction of the chat template, 371 times) |
| `calib_train.txt` sha256 | `2534f97f623e...`, same on a rebuild |

Structured and the sweep came up short of their targets (pool exhausted): the
sweep is one full pass over the vocabulary, padding it would only repeat ids.

The vision markers: text-only `llama-imatrix` feeds `<|image_pad|>`'s own
embedding row where inference puts vision encoder output, so no text corpus
gives them honest statistics.

What changed in the pool: 59 Wikipedia articles (many whole) and one code file
that eval_neutral / eval_code_ext were cut from moved to
`pool/_quarantine/eval-residue.jsonl`. `dedupe.py` had let them through, and the
August builds carried 14 of them (17.5k shared 13-grams with eval_neutral).
`purge_overlap.py` got `--whole-pool` (a rebuild draws replacements from the
same source, which overlap too) and stopped overwriting the residue shard: each
run used to drop whatever an earlier run had quarantined.

## imatrix

4,846,205 tokens / 512 = 9,465 chunks. August on 8 sm_120 cards: 12.12 s per
8 chunks, which is ~4 h for the whole build on one such box. Fan it out over
two 4x RTX PRO 6000 boxes; `im_report converge` over half the shards against
all of them is the N vs 2N check.

```bash
caffeinate -i python driver/release.py gguf --model Qwen/Qwen3.8-Flash-Next --recipe qwen3.8-flash-next-moe \
    --profile moe-qwen4exp --llama-commit 957538960 --repo-suffix -rehearsal
```

## Acceptance

```bash
python lib/im_report.py all imatrix.gguf --vs imatrix-half.gguf --inventory inventory.json \
    --stats-out imatrix.stats.txt --json im-report.json      # exit 3: dead expert or not converged
python lib/band_select.py imatrix.stats.txt --inventory inventory.json --profile profiles/moe-qwen4exp.yaml -o bands.json
python lib/band_select.py imatrix.stats.txt --inventory inventory.json --profile profiles/moe-qwen4exp.yaml --tensors attn_gate -o bands-attn-gate.json
```

1. No expert with `counts == 0`.
2. Converged: every tensor at cos >= 0.995 between N and 2N.
3. band_select against the August band, blocks 0-3 and 40-47.

What the August matrix gives (`imatrix-4000.gguf`, 4000 chunks on the dense
build), pinned in `tests/test_flash_next_bands.py`:

- 7 dead experts (blk.0: 181, 193, 244, 271, 413, 424; blk.47: 371), 135 under
  1% of the median count;
- not converged: `ffn_down_exps` at cos 0.93 between 1200 and 4000 chunks
  (worst single experts 0.22-0.45), everything else >= 0.994;
- band_select by the profile's group (`ffn_gate_exps`) picks 36-47. The tail is
  right, the head is not: Sum(Act^2) grows with depth, and blocks 0, 1, 2 rank
  last of 48;
- by `attn_gate`, the ranking Boris read: the first seven are exactly his 40,
  46, 0, 42, 41, 45, 44, but blocks 8-12 go to 17, 18, 32, 33, 34 rather than
  1-3 and 43/47. Seven of twelve agree.

On that matrix the automation does not reproduce the hand band, so for MoE it
is not trusted yet: the profile keeps the August band as depth fractions, and
the day's band comes from measurement (scan, or KLD of AD-IQ2_XS with each
band at the same size). Repeat the check on the new matrix; if band_select
there lands on 0-3 and 40-47, that is the evidence the task asks for.
