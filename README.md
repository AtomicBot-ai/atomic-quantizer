# atomic-quantizer

The quantization pipeline behind every GGUF and MLX release under the
[AtomicChat](https://huggingface.co/AtomicChat) organisation on Hugging Face.

Two shell files. You copy one onto a rented machine, source it, and call
functions one at a time in the foreground. Nothing runs in the background,
nothing is hidden, and every function says what it is doing and stops loudly
when something is missing.

| file | what it is |
| --- | --- |
| `foundry.sh` | the GGUF side. Box setup, llama.cpp build, corpus, imatrix, quantize ladder, KLD, uploads |
| `foundry-mlx3.sh` | the MLX side. Self contained, does not need `foundry.sh` |
| `foundry-nvfp4.sh` | the NVFP4 side, DeepSeek-V4.1-Flash only. Calibration by NVIDIA's recipe on one box, measurement on a Blackwell box. See [docs/runbook-nvfp4.md](docs/runbook-nvfp4.md) |
| `auto_fmt.py` | generic chat renderer. The canonical copy lives in `calib-corpora/tools/` |
| `scripts/bonsai/` | refusal ablation on the ternary Bonsai pack, where a weight edit cannot survive requantization. See [docs/runbook-bonsai-abliterate.md](docs/runbook-bonsai-abliterate.md) |

Details on each file: [docs/scripts.md](docs/scripts.md).

## Install on a rented box

```bash
# 1. connect. TERM=xterm avoids a terminfo mismatch inside the container.
#    The instance template drops you straight into tmux.
vastai ssh-url <instance-id>
TERM=xterm ssh root@<ip> -p <port>

# 2. the token. An ssh login shell does not inherit the container environment,
#    so export it by hand once per box.
export HF_TOKEN=hf_...

# 3. get the toolbox and load it
curl -sL https://raw.githubusercontent.com/AtomicBot-ai/atomic-quantizer/main/foundry.sh -o /foundry.sh
source /foundry.sh

# 4. make it survive a new tmux pane
persist_shell
save_token

# 5. check nothing is missing from the file
selfcheck
```

```
all functions present, version 2026-08-21.01
```

> [!NOTE]
> `raw.githubusercontent.com` sits behind a CDN and can hand back a file that is
> minutes old. If a push you just made is not showing up, clone instead, which is
> also what `reload` does for exactly this reason:
>
> ```bash
> git clone https://github.com/AtomicBot-ai/atomic-quantizer /quantizer
> source /quantizer/foundry.sh
> ```

> [!WARNING]
> `save_token` writes `HF_TOKEN` into `~/.bashrc` on the box, which is what
> makes a new tmux pane usable. The box is disposable, the token is not. Revoke
> it when the job is done, and never snapshot an instance into an image.

## The four commands that orient you

```bash
status        # a checklist of THIS box, with the command for every unchecked line
plan          # what already exists in the Hugging Face repositories
help_me       # every function, grouped
menu          # the same commands picked by number instead of typed
```

`status` is the one you will use constantly. It looks like this:

```
  [x] preset: AtomicChat/Qwen3.8-27B-GGUF
  [x] cuda checked
  [x] tools installed
  [x] llama.cpp built
  [x] eval corpus
  [ ] original weights in /src  ->  run:  get_upstream
  [x] bf16 on disk
  [x] quants on disk: 14 files
  [ ] reference built        ->  run:  base
  [ ] kld logs: 0       ->  run:  kld_all
  [ ] bench logs: 0     ->  run:  bench_all
  [ ] results.json        ->  run:  results

  imatrix:
    [x] calibration corpus
    [x] IM_MODEL: Qwen3.8-27B-BF16.gguf
    [ ] no imatrix          ->  run:  im_plan N, then im_shard I N on each box

  full command list: help_me      pick by number: menu
  remote view: plan               integrity: selfcheck
```

Every unchecked line tells you the command. You never have to remember the
order.

## Set up the machine

```bash
check              # disk, GPUs, cores, RAM
cuda_arch          # reads compute capability off the card
cuda_check         # toolkit against host driver, removes broken compat libs
setup              # build tools and python packages, creates the directories
build 120          # the number cuda_arch printed
gpu_test           # every GPU has to appear in the list
```

`cuda_arch` prints the exact build line, because the number is not guessable
from the card name (H100 and H200 are both 90, RTX 5090 is 120, B200 is 100):

```
compute capability 12.0 on NVIDIA GeForce RTX 5090
build with:  build 120
```

Two things `setup` does not install and you will want on every box:

```bash
apt-get install -y vim
fix_tmux           # mouse scrolling and 200k lines of scrollback
```

## Pick the model

```bash
use_model qwen3.8-27b Qwen/Qwen3.8-27B
```

```
recipe name   : qwen3.8-27b
our gguf repo : AtomicChat/Qwen3.8-27B-GGUF
metrics repo  : AtomicChat/Qwen3.8-27B-GGUF-metrics  (type: dataset)
upstream      : Qwen/Qwen3.8-27B
eval corpus   : /eval/neutral.txt
context       : 4096

Run plan to see what exists where and what to do next.
```

The repository names are derived from the upstream id, not typed. An id that
does not exist on the hub is refused, so a typo cannot propagate into every
later step. If you are not sure of the id: `find_repo qwen3.8`.

If this box was used for another model before, `status` warns about it and:

```bash
clean_run          # removes /logs /kld /imatrix, leaves the weights alone
clean_gguf         # offers to delete weights belonging to a different model
```

## A full GGUF release

Run these in order, one at a time, reading each one's output.

```bash
# --- weights -------------------------------------------------------------
get_bf16                        # pull our published BF16 file
# for a model nobody has done yet, instead:
#   get_upstream ; setup_convert ; make_bf16 ; check_blocks

# --- calibration ---------------------------------------------------------
ls_corpora                      # what builds exist in calib-corpora
get_calib qwen3.8-27b           # pull builds/<name>/calib_train.txt
pick_model                      # interactive: choose the BF16 file to collect on
im_size                         # how many chunks that is

# --- importance matrix, one box ------------------------------------------
im_shard 0 1
# across six boxes, see the section below

# --- the reference -------------------------------------------------------
get_eval                        # eval/neutral
eval_size
base                            # writes the reference logits, 4 to 8 minutes
push_base

# --- the ladder ----------------------------------------------------------
bits                            # what the model is made of, by tensor role
write_ladder                    # writes /ladder.txt
ladder /ladder.txt --dry        # sizes only, nothing is built
ladder /ladder.txt              # build, measure and publish every rung

# --- measure -------------------------------------------------------------
use_gpus 0,1                    # hold quant runs to a setup readers can relate to
kld_all
bench_all
results

# --- publish -------------------------------------------------------------
audit                           # published but never measured, and vice versa
push_quants
push_logs
push_results
```

`eval_size` and `results` are the two you read most carefully:

```
corpus : /eval/neutral.txt
size   : 1414908 bytes, roughly 353727 tokens
at ctx 4096 that is about 86 chunks
```

```
file                                               GB   mean KLD    top-1     pp512     tg128
Qwen3.8-27B-AD-IQ2_XS                            8.71   0.201443    89.12     1840.2      41.7
Qwen3.8-27B-AD-IQ3_XXS                          10.44   0.118207    91.88     1795.6      38.9
Qwen3.8-27B-AD-IQ4_XS                           16.81   0.015799    97.31     1702.4      31.2
...
written to /logs/results.json
```

`kld_all` skips anything already measured, so it is also the catch up command
after a box built quants before its reference existed. `KLD_FORCE=1 kld_all`
redoes them.

Other publishers' builds get measured the same way, against our reference:

```bash
get_external unsloth/Qwen3.8-27B-GGUF '*UD-IQ3_XXS*'
kld_ext
```

### Ablation: choosing the layout, not the rung

`ladder` decides which size classes get published. `ablate` decides what layout
a rung uses, which is a different question: every candidate is built at the same
size, measured against the same reference, and then deleted.

```bash
write_ablation                  # writes /ablate.txt, eight candidates
ablate --dry                    # predicted sizes only, check they land in one band
ablate                          # build, measure, delete, repeat
ablate_report                   # the table, readable after the files are gone
```

```
candidate                GB    mean KLD    vs base      median        99%    top-1
A-uniform             16.81    0.015799         --    0.002100    0.31000    97.31
B-edge4               17.08    0.014492     -8.3 %    0.001980    0.29500    97.44
C-edge16              17.82    0.009811    -37.9 %    0.001510    0.24100    97.98
```

Sizes outside a 2 percent band around the baseline are flagged, because a
candidate that is larger and better may only be telling you that bits help.
`ABL_KEEP=1` keeps the files, `ABL_FORCE=1` remeasures.

> [!WARNING]
> The block ranges in `write_ablation` assume a sixty four block model. On a
> different depth they produce a working file that lifts the wrong blocks, which
> is the hardest kind of wrong to notice. Check them against `bits` first.

### Against the community, at their sizes

```bash
get_external unsloth/Qwen3.8-27B-GGUF '*UD-IQ3_XXS*'
kld_ext
results
vs_community
```

```
their build                                      GB   their KLD  ours there     we are    top-1
unsloth--Qwen3.8-27B-UD-IQ3_XXS               13.70    0.154161    0.042203    +72.6 %    89.90
bartowski--Qwen3.8-27B-IQ4_XS                 16.05    0.053775    0.020087    +62.6 %    95.10
```

Sizes never line up, so this interpolates our ladder to their exact size rather
than comparing a 17.6 GB build of ours against a 16.1 GB build of theirs. Builds
off the end of our ladder are reported as having no honest comparison instead of
being extrapolated.

## The importance matrix across N boxes

This is the trick that turns a six hour imatrix into a thirty minute one, and it
is exact rather than approximate: the matrix is a sum of squared activations,
and a sum splits. Every node reads the **same corpus file** and takes its own
slice by chunk range, so the union is bit for bit the chunking a single node
would have produced.

On every box: install, `use_model`, `get_calib`, `get_bf16`, `pick_model`.

On one box:

```bash
im_plan 6
```

```
6 nodes, about 1620 chunks each
run get_calib and set IM_MODEL to the same file on every box, then:

  node 0:   im_shard 0 6
  node 1:   im_shard 1 6
  node 2:   im_shard 2 6
  node 3:   im_shard 3 6
  node 4:   im_shard 4 6
  node 5:   im_shard 5 6

then on any one box:   im_merge 6
```

Paste one line into each box. Each shard uploads itself to the metrics
repository when it finishes, so nobody copies files around. Then anywhere:

```bash
im_status                       # what is here, what is in the repo, what ranges were covered
im_merge_all
```

Rented machines are never equal, so when one node is slower than the rest, cut
its range and hand the pieces to whoever is free:

```bash
im_range 7275 700 5a            # chunks 7275 to 7974
im_range 7975 0   5b            # 7975 to the end
IM_SKIP="shard-5-of-6" im_merge_all
```

Full detail: [docs/imatrix-sharding.md](docs/imatrix-sharding.md).

## A full MLX release

`foundry-mlx3.sh` is self contained and does not build llama.cpp, which saves
fifteen minutes on a box that only does MLX.

> [!IMPORTANT]
> Its defaults point at Qwen3.8-27B and are read when the file is sourced.
> For any other model, export first, then source:
>
> ```bash
> export MLX3_UP=Qwen/Qwen3.8-27B
> export MLX3_STEM=Qwen3.8-27B
> export MLX3_REF=/mlx/$MLX3_STEM-MLX-bf16
> export MLX3_TEACHER=/mlx/$MLX3_STEM-MLX-8bit
> export MLX3_METRICS=AtomicChat/$MLX3_STEM-MLX-metrics
> ```

```bash
curl -sL https://raw.githubusercontent.com/AtomicBot-ai/atomic-quantizer/main/foundry-mlx3.sh -o /mlx3.sh
source /mlx3.sh

mlx3_setup
mlx3_check                      # READ THIS. A cpu-only wheel says "cpu" and nothing below works
mlx3_bench                      # tens of TFLOP/s instead of hundreds means no tensor cores
mlx3_persist

mlx3_get ref                    # the bf16 checkpoint, the zero point. About 51 GB
mlx3_get eval
mlx3_cache                      # one reference forward pass, reused by every later measurement

mlx3_get src
mlx3_quant 4 64                 # a plain rung
mlx3_clip /mlx/Qwen3.8-27B-MLX-4bit    # better rounding, identical size to the byte
mlx3_kld /mlx/Qwen3.8-27B-MLX-4bit-CLIP
mlx3_table
mlx3_push /mlx/Qwen3.8-27B-MLX-4bit-CLIP 4bit-CLIP
```

`mlx3_table` puts your builds next to the numbers to beat:

```
build                                            GB    mean KLD   top-1 %       ppl
Qwen3.8-27B-MLX-3bit-CLIP                     12.70    0.180912     92.44    4.9871
Qwen3.8-27B-MLX-4bit-CLIP                     16.05    0.051004     97.02    4.5983

targets, and the best of ours at or below each size:
  13.70 GB  0.154161  maglun 3.80bpw      ours 0.180912 at 12.70 GB -> behind by 17.4 %
  16.05 GB  0.053775  WaveCut 4bit-DWQ    ours 0.051004 at 16.05 GB -> BEATEN by 5.2 %
```

Two more levers, both documented in [docs/runbook-mlx.md](docs/runbook-mlx.md):
`mlx3_gen` / `mlx3_build` for a per tensor bit layout, which is worth about
twenty percent where clipping is worth two to five, and `mlx3_dwq` for
distillation, which pays between 2 and 4 bits.

`mlx3_box hub` and `mlx3_box dwq` print the exact command list for a two box
plan, with the checkpoint to verify at every stage.

## When something breaks

```bash
lastlog                         # open the newest log in a pager you can search
lastlog kld                     # or the newest log matching a pattern
reload                          # pull the current file from git and re-source it
selfcheck                       # verify no function went missing
```

`reload` reports the commit it landed on. It uses git rather than the raw file
URL because both http paths to GitHub are cached and can hand back a copy that
is minutes old:

```
commit: 4f2a1c9 2026-08-21 fix: pin the MTP block before low bit rungs
here: 2026-08-21.01    at that commit: 2026-08-21.02
now running 2026-08-21.02
```

Everything that has cost hours at least once is written down in
[docs/troubleshooting.md](docs/troubleshooting.md). Read it before debugging
anything.

## Documents

| document | when to read it |
| --- | --- |
| [docs/pipeline-map.md](docs/pipeline-map.md) | first. What lives in which repository and which directory |
| [docs/runbook-gguf.md](docs/runbook-gguf.md) | a full GGUF release, with the reasoning behind each step |
| [docs/runbook-mlx.md](docs/runbook-mlx.md) | a full MLX release |
| [docs/imatrix-sharding.md](docs/imatrix-sharding.md) | the sharding scheme in detail |
| [docs/renting-boxes.md](docs/renting-boxes.md) | picking hardware, and what the money goes on |
| [docs/troubleshooting.md](docs/troubleshooting.md) | the failures that cost hours the first time |
| [docs/security-audit.md](docs/security-audit.md) | credentials, what leaves the box, what to change |
| [docs/rough-edges.md](docs/rough-edges.md) | known bugs and unfinished work |
| [docs/scripts.md](docs/scripts.md) | what each file in the root is |

## The one rule the numbers depend on

A KL divergence figure only means something relative to the reference it was
measured against. Numbers taken against different references, different corpora
or different context lengths cannot go in the same table. Every published
comparison in this pipeline was produced by downloading the other publisher's
files and re-measuring them against our own reference, never by copying their
published figures.
