# Fixtures kept in the repo

The August releases are replayed from their published metrics repos at pinned
revisions (`conftest.py`, fetched into `.cache`). The runs here were never
published, so their files live in the repo. Absolute paths in the logs are
shortened to `~`; nothing else is edited.

## qwen3.5-4b-masks

Three builds of `Qwen/Qwen3.5-4B` made on 2026-09-24 on an M-series Mac with the
turboquant fork of llama.cpp (`b10269-1.5.1`), imatrix from the 4B BF16 over
`calib-corpora` `builds/qwen3.8-27b/calib_train.txt`, each a hand-written
`--tensor-type-file` on a Q5_K_S base. The first lines of each log carry the mask.

| file | mask | rung in `profiles/qwen35-masks.yaml` | sha256 of the original (first 16) |
|---|---|---|---|
| `quantize-E-Q5_K_S.log` | none, stock Q5_K_S | `Q5_K_S` | `807f78dd80d4636a` |
| `quantize-F1-q5embd.log` | head q5_K, attn_gate/ssm_out q6_K, k/v q8_0 | `F1` | `c4bb1500221cd3a4` |
| `quantize-G-q5edges-embd6.log` | k/v q8_0, ffn_down q6_K on blocks 0-3 and 28-31 | `AD-Q5_K_S` | `1a1d416f33181f8f` |

Their KLD against BF16 is in the header of the profile.

## qwen3.5-2b-smoke

The first run of the pipeline on a rented box, 2026-09-25: `release.py gguf`
on `Qwen/Qwen3.5-2B` with `--local-hub` (nothing on Hugging Face), 2x RTX 4090,
llama.cpp `1692f9e50bb2`, 48 KLD chunks of 4096 over eval/neutral, full
imatrix. Timings and cost are in `runbooks/rehearsal-qwen3.8-27b.md`.

| file | what | sha256 (first 16) |
|---|---|---|
| `quantize-Q8_0.log` | the Q8_0 rung, the source of the tensor inventory | `3a6fadfd6334dcfb` |
| `ladder.json` | the ladder as the box built it, before `tied_embeddings` | `aee6cb2a0fbfdcfe` |
| `results.json` | all 16 rungs as measured | `090bd5861faf6bc8` |
