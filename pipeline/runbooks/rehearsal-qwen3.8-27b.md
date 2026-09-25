# Rehearsal: rebuild Qwen3.8-27B and compare with August

The published `AtomicChat/Qwen3.8-27B-GGUF` is the answer key. Everything goes
to `-rehearsal` repos, private; the public repos are not touched.

## 0a. Locally, free (~1 h on a laptop CPU)

The same driver and nodes in a Docker container, our repos as folders, nothing
rented; see README "A free run on this machine". Rerun the command: every stage
must be skipped.

## 0b. Smoke on a 2B (~40 min, ~$1)

```bash
python driver/release.py gguf --model Qwen/Qwen3.5-2B --recipe qwen3.8-27b --profile dense-hybrid \
    --llama-commit 1692f9e50bb2 --repo-suffix -rehearsal --ladder-ok \
    --gpu-query "gpu_name in [RTX_4090,RTX_5090] num_gpus>=2 cuda_max_good>=13.0" --disk-gb 120
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
