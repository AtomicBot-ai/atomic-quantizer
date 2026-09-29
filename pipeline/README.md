# pipeline: a new model in, AD GGUF + abliterated + NVFP4 out

`scripts/foundry.sh` is how the August releases were made: a person on a rented
box, one function at a time. This directory is the same work without the person
in the loop, so a release on the day a model lands (Qwen4) is a few commands:

```
release.py gguf  --model Qwen/Qwen4-XXB --recipe <calib build> --profile dense-hybrid --llama-commit <sha>
release.py ablit --model Qwen/Qwen4-XXB
release.py gguf  --model AtomicChat/Qwen4-XXB-abliterated --recipe <same> --profile <same> --llama-commit <same>
release.py nvfp4 --model Qwen/Qwen4-XXB --recipe <same>
```

Every stage checks the hub first and skips what is already there, so a run that
died is resumed by running the same command again. The hub is the only state.

## What is in here

| path | what |
|---|---|
| `lib/ladder_gen.py` | the AD recipe generator: tensor inventory + profile -> ordered llama-quantize rules per rung. Simulates llama-quantize and refuses anything it would silently change |
| `profiles/dense-hybrid.yaml` | the Qwen3.8-27B August ladder (16 rungs) as roles |
| `profiles/moe-hybrid.yaml` | the Ling-3.0-flash August ladder (21 rungs + 2 controls) as roles |
| `profiles/qwen35-masks.yaml` | the hand masks measured on Qwen3.5-4B (stock Q5_K_S, F1, the kept mask as AD-Q5_K_S) for small tied Qwen3.5 |
| `lib/gguf_inventory.py` | tensor names, types and shapes from a BF16 GGUF (or a quantize log) |
| `lib/verify_quant.py` | after every quantize: each tensor's type, the override set, no fallbacks, commit, size |
| `lib/results.py` | KLD log -> row, rows -> `results.json` (pinned schema, sizes always filled) |
| `lib/make_card.py` | README draft from the metrics repo: file table, card-size table, measurement protocol, reproduce commands; editorial sections left as TODO comments |
| `lib/hub.py` | every read and write of our repos; with `LOCAL_HUB` they are plain folders |
| `lib/quantlog.py` | parser of llama-quantize logs |
| `lib/node_common.sh` | the node contract, hub helpers, pinned llama.cpp build |
| `nodes/node_convert.sh` | HF -> BF16 GGUF (MTP kept) + mmproj + inventory, published to the metrics repo |
| `nodes/node_base.sh` | KLD reference over eval/neutral, with the self-check (reference vs itself = 0) |
| `nodes/node_prepare.sh` | a freshly rented extra box: packages, venv, the pinned llama.cpp build, while the model still converts elsewhere |
| `nodes/node_imatrix.sh` | imatrix shards on GPU pairs or several boxes (`IM_MERGE=0` computes only, `only` merges what is on the hub), merge + statistics + coverage/convergence report (`lib/im_report.py`) |
| `nodes/node_quant.sh` | per rung: quantize -> verify -> KLD -> upload -> delete; the next rung quantizes (CPU) while the previous one is measured (GPU), at most two on disk |
| `nodes/node_abliterate.sh` | Heretic, integrity of the saved checkpoint, gate on refusals and KL |
| `nodes/node_nvfp4.sh` | llm-compressor NVFP4 calibrated on the same corpus build |
| `driver/release.py` | stages, hub state, box rental, run log (`runs/<stem>-<time>/run.jsonl`); `--im-boxes N` spreads the imatrix shards over N boxes rented while the model converts, `--quant-boxes N` the rungs, `--quant-batch` rungs per node call |
| `driver/vast.py`, `driver/remote.py` | rent/probe/destroy (port of atomic-forge rent_race.sh); ssh or `docker exec`, tmux, follow |
| `tests/` | replay of the published releases and of the Qwen3.5-4B masks, the 2B smoke ladder, refusal cases, results schema, card |
| `tests/fixtures/` | logs of runs that were never published (the 4B masks, the 2B smoke run), with their provenance |
| `tests/acceptance_qwen38.py` | the rehearsal against the August Qwen3.8-27B release (KLD, top-1, size, verify logs, imatrix cosine) |
| `runbooks/` | release day, rehearsal |

## One time setup (Mac or any Linux)

```bash
python3 -m venv ~/.venvs/atomic-pipeline && . ~/.venvs/atomic-pipeline/bin/activate
pip install "huggingface_hub>=1.0" pyyaml pytest vastai numpy
pip install -e <a llama.cpp checkout>/gguf-py
vastai set api-key <key>                                    # stored by vastai in your home, never by us
mkdir -p ~/.config/atomic-pipeline && umask 077 \
  && echo 'export HF_TOKEN=hf_...' > ~/.config/atomic-pipeline/hf_env   # write access to AtomicChat
```

The token goes to a box as `/root/.hf_env` (0600, piped over ssh) and nowhere else: not into a
script, not into a log (`scan_secrets` runs before every folder upload). Boxes
are labelled `release:<run>` and destroyed on exit and on Ctrl-C;
`release.py reap` finds any that survived.

## A free run on this machine

Before renting anything, the whole chain runs in a local container: no token,
no vast, our repos kept as folders under `--local-hub`, upstream models and
calib-corpora still read from Hugging Face. On a laptop CPU a 0.8B model goes
through every stage with shortened measurements:

```bash
python driver/release.py gguf --model Qwen/Qwen3.5-0.8B --recipe qwen3.8-27b --profile dense-hybrid \
    --llama-commit 1692f9e50bb2 --local-box --local-hub /tmp/localhub --ladder-ok \
    --only-rungs Q8_0 AD-Q4_K_M AD-IQ2_XXS --ctx 512 --kld-chunks 16 --im-max-chunks 16
python driver/release.py status --model Qwen/Qwen3.5-0.8B --local-hub /tmp/localhub
```

Needs Docker (ubuntu:24.04 is pulled). `/tmp/localhub/AtomicChat--Qwen3.5-0.8B-GGUF*`
then hold exactly what the hub would. `--im-max-chunks` marks the imatrix as
capped in its params, it is not for publishing.

## A rented box, nothing on the hub

`--local-hub DIR` without `--local-box` rents the box as usual but keeps the
repos on it, under `/hub`; no token leaves this machine and nothing is written
to Hugging Face. When the box is released, every file under `--pull-max-mb`
(default 200: logs, `results.json`, the ladder, the imatrix, the manifests) is
copied into `DIR`; the GGUFs and the KLD reference stay on the box and die with
it, `--pull-max-mb 0` brings them too. The pull happens on failure as well, so
the logs of a broken node come home. Such a run is not resumable once the box
is gone, and `status --local-hub DIR` afterwards reports only what was pulled.

```bash
python driver/release.py gguf --model Qwen/Qwen3.5-2B --recipe qwen3.8-27b --profile dense-hybrid \
    --llama-commit 1692f9e50bb2 --local-hub ~/hub --ladder-ok --kld-chunks 48 --disk-gb 200
```

## The recipe generator

```bash
python lib/gguf_inventory.py Model-BF16.gguf -o inventory.json
python lib/ladder_gen.py --inventory inventory.json --profile profiles/dense-hybrid.yaml --explain AD-Q4_K_M
python lib/ladder_gen.py --inventory inventory.json --profile profiles/dense-hybrid.yaml --out ladder/
```

A profile names tensor roles with anchored regexes and gives each rung a type
per role. llama-quantize takes the first matching rule, so order matters, and
it matches by substring, so every pattern is anchored: the August rules
`output=` and `attn_q=` also caught `attn_output` and `attn_qkv`, and those side
effects are now explicit roles instead of accidents.

The generator refuses (exit 2) when llama-quantize would not build what the
ladder says:

- a quantizable tensor no rule covers, where the fallback would silently apply
  (the trap: dense rules on a MoE model leave every expert at the base type);
- under a mixture ftype (Q4_K_M, IQ4_XS...) any uncovered tensor, because
  llama.cpp's own per tensor logic would choose;
- a k/i type on a row that is not a multiple of 256 (MoE `moe_intermediate_size`
  640 on Qwen3.8-Flash-Next): llama.cpp falls back without an error;
- iq3_xxs/iq2_*/iq1_* on a tensor the imatrix never sees (the MTP block), which
  aborts llama-quantize halfway through;
- a rule that a flag overrides: `--token-embedding-type` also hits the n-gram
  table `per_layer_token_embd`.

Bands (edge and mid blocks that get more bits) are counted among the blocks that
carry the band tensor, MTP excluded; the dense profile uses fractions of depth
that give exactly the August 4/12/8 on 64 blocks.

A model with tied embeddings (no `output.weight`: the small Qwen3.5 models)
uses `token_embd.weight` as its head. `tied_embeddings: output` in a profile
gives it the output role's type there, which is what llama.cpp does on its own;
without it the head takes the cheap embedding type (iq4_xs on the 2B smoke run).
The key does nothing on a model that has an `output.weight`.

`python -m pytest tests` replays both August releases: for all 16 Qwen3.8-27B
rungs and 19 Ling rungs, the simulated type of every tensor, the set of tensors
that printed "applying manual override", and the file size match the published
quantize logs (fixtures are fetched at pinned dataset revisions into
`tests/.cache`). Eight of the dense rungs were built before the MTP pin fix, so
there the MTP block alone is left out of the comparison.

## Node contract

Environment in, hub out. The last line is `NODE_DONE <node> {json}` or
`NODE_FAIL <node> <reason>`; exit 1 error, 2 model or toolchain not supported,
3 a check refused the result. A node whose outputs exist exits at once unless
`FORCE=1`. Every node prints a heartbeat every 5 minutes with the tail of the
log that moved last, so a long imatrix run is not mistaken for a hang; a
heartbeat that repeats the same tail for 90 minutes is one, and the driver
gives the box back. Downloads run under a watchdog (`lib/hub.py`): no byte in
5 minutes restarts them, the last try over plain HTTP instead of Xet. pip,
apt and the llama.cpp clone retry, because box networks drop.

Pinned inputs: llama.cpp `LLAMA_COMMIT` (the driver refuses to run without one,
every node builds the same commit and every log carries it), calib-corpora
`CALIB_REV`, the Heretic commit inside node_abliterate.

## Layout on the hub

- `AtomicChat/<Stem>-GGUF`: `<Stem>-<LABEL>.gguf` flat at the root (45 GB shards
  `-0000N-of-0000M.gguf` above 48 GB), `mmproj-<Stem>-{F16,BF16}.gguf`.
- `AtomicChat/<Stem>-GGUF-metrics` (dataset): `bf16/`, `inventory.json`,
  `imatrix/` (+ `params.txt`, `corpus-manifest.json`), `kld/base-neutral.kld[.NN.part]`
  + manifest (context, chunks, corpus sha: the protocol every measurement repeats),
  `ladder/`, `logs/quantize-*`, `logs/kld-neutral--*`, `logs/verify-*`,
  `results/rows/*.json`, `results.json`, `card/README.draft.md`.

The card: `release.py card` (also the last step of `gguf`) renders the draft
into the metrics repo, and into the main repo only while that has no README.
Everything numeric comes from the metrics repo; the chart, comparisons with
other publishers, findings, speed and the vision demo stay for a person and are
listed when the stage ends.
- `AtomicChat/<Stem>-abliterated` (private first), `AtomicChat/<Stem>-NVFP4`.

## Not done yet

- speed numbers (llama-bench) and the vision check in the card;
- the base reference still runs on the gguf box before its own shards, so with
  `--im-boxes` box 0 finishes its shards ~5 min after the others;
- MoE profile values for a Qwen-style MoE (Flash-Next tensor names were never
  published); the generator will list every uncovered group on the day.
