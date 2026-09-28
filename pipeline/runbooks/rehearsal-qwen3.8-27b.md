# Rehearsal: rebuild Qwen3.8-27B and compare with August

The published `AtomicChat/Qwen3.8-27B-GGUF` is the answer key. Everything goes
to `-rehearsal` repos, private; the public repos are not touched.

## 0a. Locally, free (~1 h on a laptop CPU)

The same driver and nodes in a Docker container, our repos as folders, nothing
rented; see README "A free run on this machine". Rerun the command: every stage
must be skipped.

## 0b. Smoke on a 2B (~2 h, ~$2-4)

The KLD reference is `2047 x 248320 x 2` bytes per chunk of 4096, about 1 GB,
whatever the model size: the full 87 chunks are ~88 GB, and above 45 GB
`node_base` also keeps the 45 GB parts it uploads. 48 chunks (~49 GB) exercise
that split path and fit a 200 GB disk; the full protocol needs 250 GB.

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
    --llama-commit 1692f9e50bb2 --repo-suffix -rehearsal --ladder-ok --kld-chunks 48 \
    --gpu-query "gpu_name in [RTX_4090,RTX_5090] num_gpus>=2 cuda_max_good>=13.0" --disk-gb 200
python driver/release.py status --model Qwen/Qwen3.5-2B --repo-suffix -rehearsal   # all done, rent nothing
python driver/release.py reap
```

Same tokenizer family, so the qwen3.8-27b calibration build is fine for a smoke
test (not for publishing).

## 1. The 27B (~4-5 h, ~$11-15)

```bash
python driver/release.py gguf --model Qwen/Qwen3.8-27B --recipe qwen3.8-27b --profile dense-hybrid \
    --llama-commit 1692f9e50bb2 --repo-suffix -rehearsal
```

`1692f9e50` is the commit most of the August files were built with.

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
