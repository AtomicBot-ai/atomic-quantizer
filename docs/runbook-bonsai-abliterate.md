# Runbook: refusal ablation on a ternary Bonsai pack

`scripts/abliterate.py` edits the weights of an ordinary transformers model. That path is
closed on PrismML's Ternary Bonsai 2 27B: the weights are ternary with an FP16 scale per
128, the edit `W - r (r^T W)` is dense and lands at about 1.4% of `||W||` against a grid
step of 1.7-2.4x a typical weight, so writing it back rounds it away.

So the edit ships as a rank-1 LoRA instead. llama.cpp keeps a LoRA as two extra matmuls in
the graph and never merges it into the base weights, which carries the projection exactly
at 1.75 bits. The tools here build that adapter, verify it inside the running model, and
measure what it does.

Released from this runbook:
[`AtomicChat/Ternary-Bonsai-2-27B-Abliterate-LoRA-GGUF`](https://huggingface.co/AtomicChat/Ternary-Bonsai-2-27B-Abliterate-LoRA-GGUF)
and its
[metrics dataset](https://huggingface.co/datasets/AtomicChat/Ternary-Bonsai-2-27B-Abliterate-LoRA-GGUF-metrics).

## What is in `scripts/bonsai`

| file | what it is |
| --- | --- |
| `ptq_decode.py` | decodes `PTQ1_0` (143) and `PQ2_0` (142). Stock gguf-py rejects both type ids before it reaches a tensor |
| `compare_f16.py` | checks that decoder against the published F16 pack, fetching only the tensors it needs by HTTP range |
| `make_lora.py` | builds the adapter from a direction, in the basis the fork's LoRA branch actually sees |
| `chain_test.py` | compares an adapter against a reference one and simulates the fork's graph on random input |
| `leak_probe.cpp` | taps every residual write inside the running fork and reports how much signal is left along the direction |
| `collect_acts.cpp` | dumps last-token residual stream activations for a list of prompts |
| `make_directions.py` | estimates directions by difference of means, plain and corrected |
| `ab_run.py` | A/B and sweeps over adapter scale through one `llama-server` |
| `mmlu_run.py` | MMLU with the answer forced to a single letter by a GBNF grammar |

## The box

This one runs on a Mac; the pack is 5.95 GB and decode is memory-bandwidth-bound. It also
runs on a rented CUDA box, but nothing below needs one.

You need PrismML's fork, not upstream. Upstream rejects the private type ids at header
parse, and it has no Hadamard activation runtime, so even the F16 pack would load and emit
nonsense rather than fail.

```bash
git clone --branch prism-b10709-9a9394a https://github.com/PrismML-Eng/llama.cpp prism-llama.cpp
cmake -B prism-llama.cpp/build -S prism-llama.cpp -DGGML_METAL=ON -DGGML_METAL_USE_BF16=ON -DGGML_METAL_EMBED_LIBRARY=ON
cmake --build prism-llama.cpp/build --config Release -j
```

Python side: `numpy`, `safetensors`, `gguf`, `pyarrow`, `huggingface_hub`. The two probes
link against the fork:

```bash
clang++ -std=c++17 -O2 scripts/bonsai/leak_probe.cpp \
    -I prism-llama.cpp/include -I prism-llama.cpp/ggml/include \
    -L prism-llama.cpp/build/bin -lllama -lggml -lggml-base \
    -Wl,-rpath,$PWD/prism-llama.cpp/build/bin -o leak_probe
```

## Two things that silently produce a broken adapter

**The LoRA branch gets the unrotated activation.** The fork rotates the activation for the
base matmul only:

```text
build_lora_mm:   y = W_fold @ H(s * P x)  +  scale * B (A x)
build_inp_embd:  h = s * H(W_fold[tok])   +  scale * B A[tok]
```

so the factors belong in the unfolded input basis:

```text
writers:     A = P^T (s * H(r^T W_fold)),   B = -r
token_embd:  A = W_fold @ H(s * r),         B = -r
```

`H` is the normalized Sylvester WHT over blocks of 1024, `s` the sign vector for that input
width, `P` the tiled-to-grouped V-head permutation the fork applies to `ssm_out` when
`prism.hadamard.gdn_v_grouped` is set. `r` lives on the output side, which is never
rotated, so `r` itself is not transformed. Leave the `ssm_out` factors in the checkpoint's
head order and 48 of the 129 sites do nothing at all - measured, not assumed: 1.6e-2 left
along `r` against 3.8e-6 when the permutation is applied. OrcaRouter's published adapter
has exactly that defect.

**A naive difference of means over-refuses.** The direction carries a component along the
mean harmless activation, so zeroing the projection moves a harmless prompt about 3.5 sigma
toward the harmful cluster and the model starts declining ordinary questions - it answered
a question about the fastest land mammal with "I cannot provide information about the
fastest methods for illegal activities". `make_directions.py` writes a corrected variant
with that component removed; use it.

## Run it

Verify the decoder first. Everything downstream is meaningless if it is wrong, and the
check is cheap because both packs hold the same ternary values in the same folded basis,
so the match must be exact.

```bash
python scripts/bonsai/ptq_decode.py --gguf model/Ternary-Bonsai-2-27B-PTQ1_0.gguf --list
python scripts/bonsai/compare_f16.py --gguf model/Ternary-Bonsai-2-27B-PTQ1_0.gguf \
    blk.0.ffn_down.weight blk.0.ssm_out.weight blk.3.attn_output.weight
```

Expect `rel_err = 0.000e+00`. Then estimate directions on the pack itself and build an
adapter:

```bash
python scripts/bonsai/make_directions.py --out directions/own.safetensors
python scripts/bonsai/make_lora.py --gguf model/Ternary-Bonsai-2-27B-PTQ1_0.gguf \
    --direction directions/own_proj.safetensors --layer 42 --out adapters/own42p-all.gguf
```

Check what it does inside the running model, not on paper:

```bash
./leak_probe -m model/Ternary-Bonsai-2-27B-PTQ1_0.gguf \
    -d directions/own-42p_fp32.bin --lora adapters/own42p-all.gguf
```

At scale 1 every writer should fall to single-digit 1e-6 and the residual stream should
stay below 1.5e-5 across all 64 blocks. Two more controls are worth the minute they cost:
scale 0 must reproduce the base model token for token, and a large scale must visibly break
it. If neither is true the adapter is not in the graph and nothing else you measure means
anything.

Then behaviour and cost:

```bash
python scripts/bonsai/ab_run.py \
    --prompts prompts/test_harmful.jsonl --prompts prompts/test_harmless.jsonl \
    --adapter own42p=adapters/own42p-all.gguf \
    --config base --config own42p:1 --config own42p:1.5 --config own42p:2 --out results/sweep
python scripts/bonsai/mmlu_run.py --adapter own42p=adapters/own42p-all.gguf \
    --config base --config own42p:1.5 --out results/mmlu
```

`ab_run.py` loads every adapter once and sets the scale per request, so all configurations
share the weights, seed, sampling, token budget and system prompt.

## What the numbers looked like

Refusals on 104 held-out harmful prompts: 99% on the base pack, 14% at scale 1, 0% at scale
1.5, and 0% at scale 2 with five empty replies. Over-refusal on 104 harmless prompts stayed
at 0% throughout. MMLU over 500 questions moved from 73.2% to 73.0% at scale 1.5, with the
per-question flips symmetric, so nothing systematic was lost.

Scale 1.5 is the working point. The window is narrow: 1.0 leaves refusals in, 2.0 starts
emptying replies. Restricting the intervention to blocks 15-63 was consistently worse than
covering every site, so the lower blocks and the embedding carry part of the behaviour.

Two traps in the metric, both worth keeping when you reuse `ab_run.py`. An empty or
degenerate reply is damage, not compliance - count it separately or over-projection reads
as a perfect score. And an opening-phrase classifier misses a soft refusal: asked in
Russian to describe traffic with profanity, the base pack answered as if the word meant
mathematics, which scores as compliance.

## What was not covered

The F16 pack was not run. `PTQ1_0` and `PQ2_0` both were, with the same adapter file:
identical metadata, both decoders bit-exact against F16, an adapter rebuilt from `PQ2_0`
bit-identical to the one from `PTQ1_0`, the same leak figures, and all 416 greedy replies of
the refusal evaluation byte-identical across the two packs. Capability was checked with MMLU alone.
Directions come from one prompt-pair source at one token position, so transfer to other
phrasings and languages is unmeasured beyond spot checks.
