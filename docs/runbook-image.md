# Runbook: GGUF releases of Qwen-Image-2.1

`scripts/foundry-image.sh` is self contained. It does not source `foundry.sh`
and it never builds llama.cpp. Read this page once before renting anything,
because the trap on this route is not technical: it is measuring a different
picture instead of a degraded one. Change the seed, the text encoder, the VAE,
the sampler or the sd.cpp commit between two builds, and the distance you
publish is the distance between two unrelated images.

## What is different from GGUF, MLX and NVFP4

**The tool is stable-diffusion.cpp.** llama.cpp cannot touch this model, and
neither can the Atomic fork of it. The converter has no DiT architecture,
`llama-quantize` throws on an architecture it does not know, and there is no
diffusion graph to run the result. The fork's TurboQuant weight types would not
help either: their ids are private to the fork and no image runtime reads them.
stable-diffusion.cpp is ggml underneath and added Qwen-Image-2.1 on 2026-09-20.
Its `sd-cli` does all three jobs: `-M convert` quantizes, an ordinary render
with `--imat-out` collects the importance matrix, and an ordinary render is the
measurement. The file pins it by sha, `IMG_SD_SHA`.

**A render is three models, and one of them is quantized.** The Qwen3-VL-8B
text encoder turns the prompt into conditioning, the 7.1B denoiser runs 20
sampling steps, the VAE decodes. The GGUFs published here are the denoiser
only, the way leejet and unsloth ship theirs. The encoder and the VAE stay at
bf16 for every render this route measures, so the denoiser file is the only
thing that changes between two numbers.

**There is no next token.** No logits, no KL, no top-1. The number is LPIPS,
a learned perceptual distance, between the render of a build and the render
of the bf16 denoiser from the same prompt and the same starting noise, with
SSIM and PSNR beside it. LPIPS answers "how far did this quant move the
picture", which is the question. It does not answer "is the picture good": a
quant can drift and still look fine, and a small LPIPS can hide one broken
glyph. That is why text rendering gets its own columns and why the renders
themselves are published.

**The importance matrix comes from renders.** `sd-cli --imat-out` records the
squared activations at the input of every matrix multiplication in the
denoiser, at every sampling step, for the conditional and the unconditional
pass of cfg alike. 32 calibration prompts, disjoint from the 48 evaluation
prompts, chained through one file.

**The converter decides more than the rules do.** What sd.cpp leaves at the
source type, read off leejet's files, which were made with `--type` alone:

| group | weights | share | with `--type q4_K` |
| --- | --- | --- | --- |
| attention `to_q to_k to_v to_out.0`, 128 tensors | 2147.5 M | 30.2 % | q4_K |
| MLP in, fused `img_mlp.gate_up`, 32 tensors | 3221.2 M | 45.3 % | q4_K |
| MLP out, `img_mlp.out`, 32 tensors | 1610.6 M | 22.6 % | q4_K |
| `modulation.1`, shared by every block | 67.1 M | 0.9 % | bf16 |
| `time_text_embed`, `norm_out`, `proj_out` | 34.9 M | 0.5 % | bf16 |
| `img_in`, `txt_in` | 33.8 M | 0.5 % | bf16, never converted by name |
| 65 norms, 1-D | 0.01 M | 0 % | bf16 |

`img_bits` predicts a layout's size from the bf16 file's safetensors header on
that basis, and reproduces the tensor bytes of leejet's Q2_K, Q4_K, Q6_K and
Q8_0 to the last digit printed. Whether a rule can move a tensor outside the
blocks, the modulation for one, has not been seen yet; `img_layout` on the
first AD file answers it (see what is not verified).

The input file is Comfy-Org's single file, `qwen_image_2.1_bf16.safetensors`,
which fuses the MLP input into `img_mlp.gate_up` where the diffusers shards have
`proj` and `gate_layer`. Every regex in this route is written against it.

## What the others shipped

Read from the file headers with range requests on 2026-09-24, nothing
downloaded. `img_layout` prints the same for any file under `gguf/external`.

| file | tensor GB | what it does |
| --- | --- | --- |
| leejet Q2_K, Q4_K, Q6_K, Q8_0 | 2.5617, 4.1975, 5.9968, 7.6871 | the tool's default: every block tensor at one type, everything else bf16, no prefix on names |
| unsloth Q2_K | 2.4661 | blocks and modulation at q2_K; `norm_out` f32, time embedding and `proj_out` q8_0, norms f32 |
| unsloth Q3_K_XL | 3.6125 | a per-tensor patchwork, q2_K to q6_K: key projections down to q2_K in seven blocks, value projections up to q5_K in most |
| unsloth Q4_K_M | 4.1995 | modulation q5_K; 28 attention tensors up: the value projection in 17 blocks, the attention output in 8 of the first 12, the query in 3 |
| unsloth Q5_K_M | 5.3902 | q5_K base, attention q5_K to q8_0, some MLP at q6_K |
| unsloth Q6_K_XL | 6.7185 | q6_K and q8_0 mixed through every block |

At 4.20 GB the two files are the same size to within 2 MB and spend it
differently. leejet keeps everything outside the blocks at bf16. unsloth takes
88 MB out of the modulation (bf16 to q5_K) and 17 MB out of the time embedding
and `proj_out` (bf16 to q8_0), puts 34 MB back into `norm_out` (bf16 to f32),
and spends the remaining 73 MB on those 28 attention tensors and one MLP
output. That is the whole disagreement, and it is measurable. Across their
rungs the same preference shows: value projections up, key projections down.

## What "better" can honestly mean

Every AD rung is sized to a rival file, so a comparison is between two ways of
spending the same bytes. The claims, in this order, fixed before the first
number exists:

1. **Floors.** Floor (a): a second bf16 render against the first, same seed.
   If it is 0.0000 the kernels are deterministic and any difference is signal;
   if it is not, nothing below it is a difference. Floor (b): seed 42 against
   seed 43 of the same prompt, what a different picture costs. Every build is
   also reported as a fraction of floor (b).
2. **Size-matched distance.** Each AD rung against its rival, paired per
   prompt, with the interval. `AD-Q4_K` against unsloth `Q4_K_M` and leejet
   `Q4_K`; `AD-Q3_K` against `Q3_K_XL`; `AD-Q5_K` against `Q5_K_M`; `AD-Q6_K`
   against `Q6_K_XL`; `AD-Q2_K` against leejet `Q2_K`. The card says which
   intervals clear zero and which do not, in those words.
3. **The two levers apart.** `Q4_K` against `Q4_K-imat` is the importance
   matrix alone, same bytes. `AD-Q4_K-noimat` against `Q4_K` is the layout
   alone. "Our calibration helps" is published only if its interval clears
   zero.
4. **Tails and text.** p95 and max LPIPS, and the `text-en` and `text-zh`
   means. Glyphs break before photographs do, and a mean over photographs
   hides it.
5. **Speed.** Seconds per render and seconds of sampling, same binary, same
   GPU, one `IMG_OFFLOAD` mode. A secondary column with the GPU in the
   caption. Unlike the NVFP4 route, the builds here run different kernels per
   type, so speed is a real difference, but a small one next to the load.

Not measured, and the card says so: aesthetics or prompt adherence (no CLIP
score, no judge), editing with reference images, other samplers, step counts,
cfg values or resolutions, and any runtime other than sd.cpp.

## The AD layouts

The rival's tensor bytes are the budget. The budget is spent in a fixed order
of priority: the modulation first, because every block reads it; then
attention; then the output projection of the MLP in the outer blocks. Each rung
takes as much of that list as fits. `IMG_AD_TABLE` holds it:

| rung | base | modulation | attention one step up | `img_mlp.out` one step up | predicted GB | rival | |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `AD-Q6_K` | q6_K | bf16 | all 32 blocks, q8_0 | blocks 0-7 and 24-31, q8_0 | 6.7120 | unsloth Q6_K_XL 6.7185 | −0.10 % |
| `AD-Q5_K` | q5_K | q8_0 | all, q6_K | 0-7 and 24-31, q6_K | 5.3991 | unsloth Q5_K_M 5.3902 | +0.17 % |
| `AD-Q4_K` | q4_K | q8_0 | 0-3 and 28-31, q5_K | none | 4.2017 | unsloth Q4_K_M 4.1995, leejet Q4_K 4.1975 | +0.05 % |
| `AD-Q3_K` | q3_K | q8_0 | all, q4_K | 0-7 and 24-31, q4_K | 3.5998 | unsloth Q3_K_XL 3.6125 | −0.35 % |
| `AD-Q2_K` | q2_K | q8_0 | 0-3 and 28-31, q3_K | none | 2.5533 | leejet Q2_K 2.5617, unsloth Q2_K 2.4661 | −0.33 % |

In every AD rung the time embedding, `norm_out`, `proj_out` and the norms stay
bf16 by rule, and `img_in` and `txt_in` stay bf16 because sd.cpp never
converts them. The sizes are `img_ladder --dry` output.

**The bands are a prior.** Early and late blocks mattering more holds for
language models and has not been measured on this model; unsloth's layouts
point somewhere else, at the value projection in every block. `img_scan`
measures both: the tool's Q8_0 as a base, one tensor group at a time dropped to
q2_K, cut once by block range and once by attention projection (q and k, v,
out), ranked by LPIPS added per GB saved. If the ranking disagrees with the
prior, `IMG_BAND_NARROW`, `IMG_BAND_WIDE`, `IMG_ATTN_PROJ` (for example
`'(to_v|to_out)'`) or `IMG_AD_TABLE` change before the ladder runs, the new
sizes are checked with `img_ladder --dry`, and the change is committed with
the scan that justified it.

**Naming.** `AD-<type>` names the type of the fused MLP input, the largest
group at 45 % of the weights. Nothing is named after a type it does not
contain.

**Uniform rungs.** `Q8_0 Q6_K Q4_K Q3_K Q2_K` are converted with `--type`
alone. They are the baselines, the other half of the ablation, and a check:
`img_same Q4_K leejet--qwen_image_2.1-Q4_K` tells whether the pin produces
leejet's weights byte for byte. If it does, the leejet entries can come out of
`IMG_EXT` and the uniform rows speak for them, which saves three renders of 96
images. Only `BF16`, `Q8_0` and the AD rungs are published (`IMG_PUBLISH`); the
rest are measured and their logs and renders published.

## The box

| | |
| --- | --- |
| GPU | one of 80 to 96 GB: H100 80, A100 80, RTX PRO 6000. The bf16 denoiser (14.2 GB) and the bf16 encoder (17.5 GB) stay resident. 48 GB works with `IMG_OFFLOAD=1`, slower, and its timings compare only with each other |
| RAM | 64 GB or more, so the 32 GB of inputs stay in page cache between renders |
| disk | 250 GB. Inputs 32 GB, ours about 70 GB (ten rungs, the ablation pair, BF16), others 43 GB, the scan 8 GB at a time, renders 150 MB per build |
| image | the CUDA 13 devel image, as everywhere in this pipeline; `img_build` installs cmake if it is missing |
| arch | read off the card by `img_build`; `IMG_CUDA_ARCH` overrides |

One box does everything. A second box can take `img_ext` off the first: it
pulls the reference renders with `img_get ref` instead of rendering them, and
must first render one image itself and measure it against the pulled one,
because floor (a) is a property of a GPU and a build of sd.cpp, not of a
prompt.

## The command list

`img_box` prints it. Paste one line at a time.

```bash
export HF_TOKEN=hf_...
git clone https://github.com/AtomicBot-ai/atomic-quantizer /quantizer
source /quantizer/scripts/foundry-image.sh
img_setup                              # 5-10 min
img_build                              # 5-15 min, sd-cli at the pin
img_persist ; img_repos ; img_env ; img_check
img_get src                            # 32 GB, then 2 min of sha256
img_prompts
# smoke test, 15 min, before anything long
img_convert Q4_K                       # the layout it prints is what the pin keeps in bf16
IMG_EVAL_N=1 IMG_SEEDS=42 img_gen ref
IMG_EVAL_N=1 IMG_SEEDS=42 img_gen Q4_K
IMG_EVAL_N=1 IMG_SEEDS=42 img_metrics Q4_K
IMG_EVAL_N=1 IMG_SEEDS=42 img_gen ref-repeat ; IMG_EVAL_N=1 IMG_SEEDS=42 img_metrics ref-repeat
# the reference and the floors
img_ref                                # 60-90 min, resumes from the one render above
img_repeat                             # 60-90 min, floor (a)
img_floors                             # floor (b), seconds
# the evidence for the layouts
img_imatrix                            # 30-45 min
img_imatrix_check
img_scan                               # about an hour; read the ranking before going on
img_ladder --dry
# the ladder
img_ladder                             # 10 rungs, about 70 min each
img_ablate                             # 3-4 h
img_convert BF16 ; img_push BF16
# the other publishers
img_get ext                            # 43 GB
img_same Q4_K leejet--qwen_image_2.1-Q4_K
img_ext                                # about 70 min per file
img_table ; img_grid ; img_chart
img_push_logs
```

Things the smoke test is for, in the order it shows them: the layout of the
first real conversion, which settles what the pin leaves alone and whether the
predicted size was right; whether a render works at all on this GPU and how
long the model load takes against the sampling (`gen-<build>.jsonl` has both);
whether torch, lpips and the pairing produce a number; and whether two bf16
renders of the same seed are identical. If the load is most of the time,
batching two seeds per invocation is the fix, and it changes nothing about the
images.

> [!IMPORTANT]
> `img_imatrix` with fewer than `IMG_CALIB_N` prompts leaves a partial matrix,
> and `img_convert` refuses AD rungs until it is complete. A smoke test of the
> AD path goes under a `smoke-` label, which converts into `gguf/scan/` and is
> never published:
> `IMG_IMAT_USE=1 img_convert smoke-AD-Q4_K q4_K "$(img_rules AD-Q4_K --rules-only)"`.

## The protocol

**Prompts.** 48 for evaluation, 32 for calibration, written into the script
itself and versioned with it; their sha256 goes to the metrics dataset. Six
categories, eight each for evaluation: `photo`, `illus`, `text-en`, `text-zh`
(Chinese text in the image, where this model is strongest and quantization
hurts first), `multi` (several objects with attributes, where attention
matters), and `alpha` (transparent RGBA, in Qwen's recommended wording). The
categories are interleaved, so the first N of either file is a balanced subset;
that is what the scan and the smoke test render. No prompt appears in both
files.

**Settings.** 1024x1024, 20 steps, cfg 6.0, euler, `--rng cpu`,
`--diffusion-fa`, seeds 42 and 43: unsloth's recommended sd.cpp settings, so a
reader who follows their guide gets our images. `--rng cpu` is ComfyUI's
noise, so a seed means the same starting latent in every build. The protocol
string, sd.cpp commit included, is stored in every metrics file, and
`img_compare` refuses two files whose strings differ.

**Numbers.** Per image LPIPS (VGG), SSIM over the three channels, PSNR; RGBA
composited over grey 128 first, alpha difference reported beside it. Per
prompt, the mean over its two seeds. The interval is a bootstrap over the 48
prompts, not the 96 images, because the two seeds of a prompt share it. A
paired comparison takes the per-prompt difference of two builds and
bootstraps that. An interval that contains zero means no convincing
difference, not equivalence.

**The scan** renders 8 prompts at one seed, 512x512, 10 steps, against a bf16
render at the same reduced settings, for 13 groups: attention and MLP in four
block ranges each, attention by projection, the modulation with the other
conditioning tensors, and `proj_out`. Its numbers rank groups and never go in
a table.

## How long it takes

Estimates for one H100, to be replaced by what the first run measures.

| step | hours | notes |
| --- | --- | --- |
| setup, build, inputs | 0.75 | the CUDA build of sd-cli is most of it |
| reference and floor (a) | 2-3 | 192 renders |
| imatrix | 0.75 | 32 renders |
| scan | 1.5 | 14 conversions, 112 small renders |
| ladder | 12-14 | 10 rungs x 96 renders, plus conversions |
| ablation | 2.5 | two more builds |
| other publishers | 6-11 | 6 to 9 files |
| total | 25-33 | about 60 dollars of H100 time; a second box running `img_ext` halves the wall clock |

## Publishing

`AtomicChat/Qwen-Image-2.1-GGUF` gets `BF16`, `Q8_0`, the five AD rungs, the
three buttons, `samples-grid.png`, `quality-vs-size.png` and the card.
`AtomicChat/Qwen-Image-2.1-GGUF-metrics` gets everything else: every render of
every build under `images/`, the prompts, the imatrix and its checkpoints,
every layout, every log, the scan, the table. `img_push_logs` writes its README.

The card is written by hand from `logs/results.md`, in the shape of the
Qwen3.8-27B card: a file table with the measured distance and the size, the
size-matched comparison with the other publishers, what the scan found, how it
was measured. Numbers from other publishers' cards are never copied; theirs
are measured here, like ours. The card's "how to run" must name all three
files:

```bash
sd-cli --diffusion-model Qwen-Image-2.1-AD-Q4_K.gguf \
  --vae qwen_image_2.1_vae_bf16.safetensors \
  --llm Qwen3VL-8B-Instruct-Q4_K_M.gguf \
  -p "a neon sign that reads \"OPEN LATE\", rainy night" \
  --steps 20 --cfg-scale 6.0 --sampling-method euler -W 1024 -H 1024 --diffusion-fa -o out.png
```

The VAE comes from `Comfy-Org/Qwen-Image-2.1` (`vae/`), the encoder from
`Qwen/Qwen3-VL-8B-Instruct-GGUF`; editing needs its
`mmproj-Qwen3VL-8B-Instruct-F16.gguf` as `--llm_vision` and the input image
as `-r`. ComfyUI-GGUF and Unsloth Desktop load files like these, but none of
ours has been tried in either, and the card says so.

## What the first run measured

Nothing yet. This section gets the table, the floors, the scan ranking and the
timings, dated, when the box has run.

## What is not verified

Nothing on this route has run on a GPU. What was checked on a Mac before the
first commit: the script under macOS bash 3.2 with `IMG_ROOT` redirected, the
help and every function it names, the prompt files, the size predictor against
the real header of the bf16 input and against leejet's four files, the GGUF
reader against the headers of nine published files, the imatrix reader against
a synthetic file in sd.cpp's format, and the metrics, comparison, table, grid
and chart on synthetic renders. Not checked:

- **Rules outside the blocks.** leejet's files show the tool leaving the
  modulation, time embedding, `norm_out` and `proj_out` at bf16 under `--type`.
  The source read for this route would have converted them, so something else
  keeps them, and whether a `--tensor-type-rules` entry overrides it is
  unknown. If it does not, every AD rung is 63 MB heavier than predicted (the
  modulation stays bf16). `img_layout` after the first AD conversion shows it.
- **The importance matrix.** That `--imat-in` reaches the quantizer in convert
  mode, that the runtime tensor names in the matrix match the names the
  converter quantizes, and that chaining `--imat-in` and `--imat-out` through
  one file accumulates. `img_imatrix_check` and the first check in
  `img_ablate`, which refuses to go on if `Q4_K` and `Q4_K-imat` carry the same
  weights, are the tests.
- **Determinism.** Whether two bf16 renders of one seed are identical on the
  GPU with `--diffusion-fa`. `img_repeat` measures it.
- **RGBA through sd-cli.** Whether the `alpha` prompts come out as four-channel
  PNGs without a flag. `img_metrics` reports how many renders carry alpha.
- **The timing line.** `sampling completed, taking N s` is read from the source
  at the pin; the column is empty if the line differs.
- **Names.** Whether our files carry the `model.diffusion_model.` prefix
  (unsloth's do, leejet's do not). `img_names` shows it; both load in sd.cpp.
- **The other files** loading and rendering in the pinned sd.cpp.
- **The bands.** Every AD layout is a prior until `img_scan` has run.
- **LPIPS as the measure.** It is the standard proxy and the one unsloth
  publishes, which makes comparisons possible. It is not a judgement of image
  quality, and nothing on the card should read as one.
