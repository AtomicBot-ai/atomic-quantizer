# Rehearsal: rebuild Qwen3.8-27B and compare with August

The published `AtomicChat/Qwen3.8-27B-GGUF` is the answer key. Everything goes
to `-rehearsal` repos, private; the public repos are not touched.

## 0a. Locally, free (~1 h on a laptop CPU)

The same driver and nodes in a Docker container, our repos as folders, nothing
rented; see README "A free run on this machine". Rerun the command: every stage
must be skipped.

## 0b. Smoke on a 2B (~2 h, ~$2-4)

The KLD reference is `2047 x 248320 x 2` bytes per chunk of 4096, about 1 GB,
whatever the model size: the full 87 chunks are ~88 GB, and above 45 GB it is
uploaded in 45 GB parts (deleted after the upload since 2026-09-29: on the 27B
the parts, the BF16 upload shards and the source filled a 328 GB disk before the
first rung). 48 chunks (~49 GB) exercise that split path and fit a 200 GB disk.

Off the hub, results on this machine (`README.md`, "A rented box, nothing on the hub").
`caffeinate -i` because a sleeping laptop pauses the driver while the box keeps billing:

```bash
caffeinate -i python driver/release.py gguf --model Qwen/Qwen3.5-2B --recipe qwen3.8-27b --profile dense-hybrid \
    --llama-commit 1692f9e50bb2 --local-hub ~/hub --ladder-ok --kld-chunks 48 \
    --gpu-query "gpu_name in [RTX_4090,RTX_5090] num_gpus>=2 cuda_max_good>=13.0" --disk-gb 200
python driver/release.py reap
```

On the hub, in `-rehearsal` repos (needs `~/.config/atomic-pipeline/hf_env`):

```bash
python driver/release.py gguf --model Qwen/Qwen3.5-2B --recipe qwen3.8-27b --profile dense-hybrid \
    --llama-commit 1692f9e50bb2 --repo-suffix=-rehearsal --ladder-ok --kld-chunks 48 \
    --gpu-query "gpu_name in [RTX_4090,RTX_5090] num_gpus>=2 cuda_max_good>=13.0" --disk-gb 200
python driver/release.py status --model Qwen/Qwen3.5-2B --repo-suffix=-rehearsal   # all done, rent nothing
python driver/release.py reap
```

Same tokenizer family, so the qwen3.8-27b calibration build is fine for a smoke
test (not for publishing).

## 1. The 27B (~3 h, ~$10-11 when nothing goes wrong)

```bash
caffeinate -i python driver/release.py gguf --model Qwen/Qwen3.8-27B --recipe qwen3.8-27b --profile dense-hybrid \
    --llama-commit 1692f9e50bb2 --repo-suffix=-rehearsal --ladder-ok --quant-boxes 4
```

`1692f9e50` is the commit most of the August files were built with. The ladder
is the August one byte for byte (`tests/test_ladder_replay.py`), hence `--ladder-ok`.

`--quant-boxes 4`: convert, base and imatrix run on one 4x RTX 5090 box; when
the ladder is up, three more boxes are rented side by side (`quant_box_plan`:
2x RTX 5090, >= 32 cores, >= 128 GB RAM, ~300 GB disk) and every box takes rungs
from one queue in the driver, largest first. Quantizing is CPU work, so what an
extra box buys is its cores; the GPU is only for KLD, and it stays the card the
August numbers were measured on. Each extra box downloads the BF16 and the KLD
reference (~145 GB) before its first rung and is released as soon as the queue
is empty. A failed rung stops the others from taking new ones; rerun to resume.

KLD on another card class (H100) is not comparable as is: measure Q8_0 there
first, against August's 0.000640, and write the difference down before reading
any other rung.

## 2. Acceptance

```bash
python tests/acceptance_qwen38.py --metrics AtomicChat/Qwen3.8-27B-GGUF-metrics-rehearsal \
    --imatrix <rehearsal imatrix.gguf> --imatrix-ref <August imatrix/imatrix.gguf>
```

It reads the August KLD and quantize logs at the pinned metrics revision and
checks the rehearsal's `results.json` against them:

| rung | August mean KLD |
|---|---|
| Q8_0 | 0.000640 |
| AD-Q6_K | 0.001068 |
| AD-Q6_K-Q5_K | 0.002521 |
| AD-Q5_K_M | 0.004189 |
| AD-Q5_K_M-Q4_K_M | 0.007296 |
| AD-Q4_K_M | 0.011262 |
| AD-IQ4_XS | 0.012483 |
| AD-IQ4_XS-IQ3_S | 0.026596 |
| AD-IQ3_S | 0.032471 |
| AD-IQ3_S-IQ3_XXS | 0.043368 |
| AD-IQ3_XXS | 0.069718 |
| AD-IQ2_S | 0.098319 |
| AD-IQ2_S-IQ2_XS | 0.138070 |
| AD-IQ2_XS | 0.161702 |
| AD-IQ2_XXS | 0.256633 |
| AD-IQ1_M | 0.342121 |

- every rung within 10% of August (or 0.0001 absolute, the noise floor at Q8_0),
  top-1 within 0.3 points, size within 2% of the August quantize log;
- every `logs/verify-*.txt` says ok (types, overrides, no fallbacks, commit);
- with `--imatrix/--imatrix-ref`: per tensor cosine of the mean squared
  activations >= 0.99 over the common entries (496 on this model);
- the self-check in `logs/kld-selfcheck.log` is exactly 0.

`python -m pytest tests/test_acceptance.py` proves the script passes August
against itself and fails a 20% KLD drift, a 5% size drift and a missing rung.

Expected differences: the MTP block is pinned to q5_k on every rung now (8 of
the August rungs quantized it at the rung types), and `general.file_type` now
says the rung's type instead of Q8_0.

## 3. Write down

From `runs/<stem>-<time>/run.jsonl`: minutes per node, box, $/h, total cost.
Put them into release-day-qwen4.md in place of the estimates.

### Measured: the 27B, 2026-09-29

`acceptance_qwen38.py`: PASS on all 16 rungs. Every verify log ok, self-check 0,
imatrix 496 of 496 entries at cosine >= 0.9982 (median 0.99999997). The
reference is byte for byte the August size (88 448 971 100), BF16 PPL 4.5231
against August's 4.5234.

| rung | GB | KLD August | KLD ours | d% | top-1 % | top-1 d | size d% |
|---|---|---|---|---|---|---|---|
| Q8_0 | 28.89 | 0.000640 | 0.000640 | +0.0 | 98.92 | +0.00 | +0.04 |
| AD-Q6_K | 24.93 | 0.001068 | 0.001059 | -0.8 | 98.67 | +0.00 | -0.27 |
| AD-Q6_K-Q5_K | 23.03 | 0.002521 | 0.002580 | +2.3 | 97.94 | +0.01 | -0.16 |
| AD-Q5_K_M | 20.20 | 0.004189 | 0.004122 | -1.6 | 97.33 | -0.01 | -0.08 |
| AD-Q5_K_M-Q4_K_M | 18.55 | 0.007296 | 0.007288 | -0.1 | 96.41 | -0.03 | +0.07 |
| AD-Q4_K_M | 17.13 | 0.011262 | 0.011258 | -0.0 | 95.57 | -0.02 | +0.14 |
| AD-IQ4_XS | 16.53 | 0.012483 | 0.012433 | -0.4 | 95.39 | +0.01 | +0.19 |
| AD-IQ4_XS-IQ3_S | 14.48 | 0.026596 | 0.026632 | +0.1 | 93.15 | -0.01 | +0.40 |
| AD-IQ3_S | 13.89 | 0.032471 | 0.032478 | +0.0 | 92.42 | +0.01 | +0.48 |
| AD-IQ3_S-IQ3_XXS | 12.98 | 0.043368 | 0.043432 | +0.1 | 91.34 | +0.02 | +0.08 |
| AD-IQ3_XXS | 12.08 | 0.069718 | 0.069711 | -0.0 | 89.12 | -0.01 | +0.09 |
| AD-IQ2_S | 11.14 | 0.098319 | 0.098270 | -0.0 | 87.18 | +0.00 | +0.10 |
| AD-IQ2_S-IQ2_XS | 10.22 | 0.138070 | 0.138042 | -0.0 | 84.69 | -0.07 | +0.11 |
| AD-IQ2_XS | 9.89 | 0.161702 | 0.161685 | -0.0 | 83.51 | +0.03 | +0.11 |
| AD-IQ2_XXS | 8.98 | 0.256633 | 0.256245 | -0.2 | 79.44 | -0.00 | +0.12 |
| AD-IQ1_M | 8.50 | 0.342121 | 0.342070 | -0.0 | 76.29 | -0.06 | +0.13 |

The largest KLD difference is +2.3 % (AD-Q6_K-Q5_K), inside the 10 % tolerance
with room; sizes are up to +0.5 %, from the MTP block now pinned at q5_k.
KLD was measured on RTX 5090 like August (2x on the quant boxes, 4x in August):
Q8_0 lands on August's 0.000640 exactly, so no card offset to write down.

Timings and money are in `release-day-qwen4.md`, step 1. It took three starts
of the same command, each resuming from the hub:

1. convert, base, imatrix on 4x RTX 5090, then Q8_0 died: the box disk (328 GB)
   was full with the source, the BF16, its upload shards, the reference and its
   upload parts. The nodes now delete their upload copies after the upload.
2. 7 rungs on four 2x RTX 5090, then this laptop's disk filled up and the driver
   died on a log write before releasing its boxes; they were released by hand
   once their rungs were on the hub. The driver now ignores failed log writes.
3. the other 9 rungs; the laptop slept twice with the lid closed (~15 min each)
   while the boxes waited.

### Measured: the 2B smoke, 2026-09-25

Off the hub (`--local-hub`, 48 KLD chunks, full imatrix, all 16 rungs), box
2x RTX 4090, 56 cores, $0.82/h, race won in 2.5 min at 177 MB/s.

| stage | box time | notes |
|---|---|---|
| node_convert | 20.0 min | half of it the CUDA build of llama.cpp; BF16 3.9 GB, two mmproj |
| node_base | 3.5 min | 48 chunks: 48.8 GB reference, split into parts, self-check 0 |
| node_imatrix | 35.0 min | 9701 chunks of 512 on one GPU pair: 2,370 tokens/s, 186 entries |
| ladder | 0.4 min | dense-hybrid on 25 blocks: bands edge 0-1 + 19-23, mid 2-4, MTP 24 pinned |
| node_quant | 23.3 min | 16 rungs, ~85 s each: quantize 14-38 s, KLD ~60 s on 48 chunks |
| results + card | 0.3 min | |

Box total 1 h 23 min, about $1.15 of rent; the bill was $7.24 because the
laptop running the driver slept for 7.5 hours while the box waited (hence
`caffeinate -i` above). All 16 verify logs ok, `general.file_type` set per rung.

What this says about the 27B on one box: the imatrix cost scales with the
activation bytes per token (llama-imatrix copies every matmul input to the
host), about 7x the 2B, so one GPU pair needs ~4 h for the 4.97 M token corpus;
a 4x RTX 5090 box runs two shards in parallel (~2 h), a single H100 one
(~4 h). Quantizing 16 rungs of a 27B is CPU time: ~3-5 min per rung on 128
cores, 10-15 min on the 24 cores of a typical 1x H100 offer. Expect 5-7 h and
$25-30 on 4x RTX 5090 ($4.3/h), 9-12 h and $35-45 on 1x H100 ($3.9/h).

The ladder as measured (`tests/fixtures/qwen3.5-2b-smoke/results.json`), 48
chunks of 4096 over neutral against the 2B BF16:

| rung | GB | BPW | mean KLD | top-1 % | head in the smoke |
|---|---|---|---|---|---|
| Q8_0 | 2.05 | 8.41 | 0.00131 | 97.98 | q8_0 |
| AD-Q6_K | 1.83 | 7.50 | 0.00232 | 97.36 | q8_0 |
| AD-Q6_K-Q5_K | 1.56 | 6.37 | 0.00801 | 94.97 | q5_k |
| AD-Q5_K_M | 1.36 | 5.57 | 0.01996 | 91.76 | q4_k |
| AD-Q5_K_M-Q4_K_M | 1.25 | 5.12 | 0.03109 | 89.92 | iq4_xs |
| AD-Q4_K_M | 1.18 | 4.80 | 0.04074 | 88.74 | iq4_xs |
| AD-IQ4_XS | 1.14 | 4.66 | 0.04539 | 88.17 | iq4_xs |
| AD-IQ4_XS-IQ3_S | 1.04 | 4.25 | 0.08206 | 84.80 | iq4_xs |
| AD-IQ3_S | 1.01 | 4.12 | 0.09710 | 83.61 | iq4_xs |
| AD-IQ3_S-IQ3_XXS | 0.96 | 3.89 | 0.13214 | 81.16 | iq4_xs |
| AD-IQ3_XXS | 0.91 | 3.68 | 0.21237 | 76.64 | iq4_xs |
| AD-IQ2_S | 0.85 | 3.47 | 0.35528 | 70.19 | iq4_xs |
| AD-IQ2_S-IQ2_XS | 0.81 | 3.30 | 0.55249 | 63.48 | iq4_xs |
| AD-IQ2_XS | 0.79 | 3.21 | 0.65931 | 60.48 | iq4_xs |
| AD-IQ2_XXS | 0.74 | 3.02 | 1.12851 | 49.55 | iq4_xs |
| AD-IQ1_M | 0.72 | 2.91 | 1.35418 | 45.05 | iq4_xs |

KLD falls and top-1 rises monotonically with size, and every verify log agrees
with the ladder. The head column is the catch: the 2B ties its embedding to the
output, and the August ladder gave that tensor the cheap embedding type: iq4_xs
from AD-Q5_K_M-Q4_K_M down. `dense-hybrid` now has `tied_embeddings: output`,
which gives the head a heavier type on 12 of the 16 rungs (q6_k from
AD-Q5_K_M to AD-IQ4_XS), so these numbers are the "before";
`tests/test_qwen35_small.py` pins both ladders.

What the head costs, measured on Qwen3.5-4B on a laptop (the bench of the
24.09 masks: imatrix on the 4B, 30 chunks, one build), MB / mean KLD / top-1:

| build | MB | mean KLD | top-1 % |
|---|---|---|---|
| AD-Q4_K_M, head iq4_xs (the smoke ladder) | 2 657 | 0.024972 | 91.85 |
| stock Q4_K_M | 2 783 | 0.024006 | 92.78 |
| AD-Q4_K_M, head q6_k | 2 841 | 0.015298 | 94.06 |
| AD-Q5_K_M, head q4_k (the smoke ladder) | 3 117 | 0.012272 | 94.04 |
| stock Q5_K_S | 3 118 | 0.007951 | 95.61 |
| hand mask G (`qwen35-masks` AD-Q5_K_S) | 3 159 | 0.007143 | 95.73 |
| stock Q5_K_M | 3 203 | 0.006731 | 95.87 |
| AD-Q5_K_M, head q6_k | 3 281 | 0.004567 | 96.48 |

With the starved head the ladder lost to stock Q5_K_S by 54 % at the same
size; with the head at q6_k it beats the stock presets and the best hand mask
by about a third for 2-4 % more bytes. None of this touches the 27B, which has
its own `output.weight`: its rehearsal ladder is the August one, byte for byte.
