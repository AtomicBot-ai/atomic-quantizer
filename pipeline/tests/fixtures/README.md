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

## qwen3.5-4b-bands

Band selection on `Qwen/Qwen3.5-4B`, 2026-09-29, the bench of the 4B masks:
same M4 Max, same turboquant build of llama.cpp (binaries of 2026-09-17), same
imatrix (`imatrix-qwen35-4b.gguf`, sha256 `4d8f1897cd5c500f`), same BF16
reference over `eval/neutral`, 30 chunks of 4096 (`base-neutral-30.kld`).

| file | what |
|---|---|
| `imatrix-stats.txt` | `llama-imatrix --show-statistics` of that imatrix, the input of `band_select.py` |
| `kld-D-Q4_K_M.log` | stock Q4_K_M + imatrix (`kld-D-recheck.log` of 2026-09-24) |
| `kld-mask-B2.log` | hand mask B2 on Q4_K_M (`kld-B2-hybrid.log`, 3 105 597 504 bytes) |
| `kld-mask-G.log` | hand mask G on Q5_K_S, the kept one (`kld-G-q5edges-embd6.log`, 3 159 009 344 bytes) |
| `kld-{frac,bs,scan}-{8,12,16}-4.log` | AD-Q5_K_M-Q4_K_M of dense-hybrid with bands of 8, 12, 16 edge + 4 mid blocks from depth fractions, `band_select.py`, `scan.py bands`; 3 041 536 064 / 3 102 730 304 / 3 163 924 544 bytes |
| `kld-q5-noband.log` | AD-Q5_K_M with no bands (every ffn tensor q5_K), 3 168 184 384 bytes |
| `scan.json` | `scan.py report` of the full scan: q8_0 base, 18 groups two steps down to q5_K, 59.1 min |

`kld-frac-8-4.log` is the rung exactly as the profile gives it (step 1 of the task).
`tests/test_bands_4b.py` rebuilds every band set and size from these files and
pins the conclusions.
