# The files in the root

Five files, all at the repository root. Nothing here is a program to run: they
are toolboxes to work from.

```bash
curl -sL https://raw.githubusercontent.com/AtomicBot-ai/atomic-quantizer/main/foundry.sh -o /foundry.sh
source /foundry.sh
status
```

## `foundry.sh`

The GGUF side, and the larger of the two. Box setup, llama.cpp build, corpus
building, imatrix collection and merging, the quantize ladder, KLD and speed
measurement, uploads, and a Mac section for measurement only.

It also contains an older MLX block, superseded by `foundry-mlx3.sh`. Three
generations of the MLX measurement exist across the two files and only
`mlx3_kld` should be used; see [rough-edges.md](rough-edges.md).

Two mechanisms in it are worth knowing before reading the rest:

**State across tmux panes.** Bash functions live in one shell process, so a new
pane knows nothing. `save_state` writes the current selection to `/state.sh` and
the bottom of the file reads it back, so every pane agrees on which model is
selected. `persist_shell` appends `source /foundry.sh` to `~/.bashrc` so a new
pane loads the toolbox by itself.

**Autopush.** With `AUTOPUSH=1`, every log goes to the metrics repository the
moment it is written. A box that dies mid run does not take its results with it.
Set `AUTOPUSH=0` to keep things local while experimenting.

## `foundry-mlx3.sh`

The MLX side. Self contained: it does not source `foundry.sh` and it does not
build llama.cpp.

It writes its own python helpers to disk with `mlx3_write_py` rather than
keeping them as heredocs inside a pipeline, because a heredoc attached to the
wrong end of a pipe feeds the script to `tee` instead of to python. The four it
writes are `/mlx3_refcache.py`, `/mlx3_kld.py`, `/mlx3_clip.py` and
`/mlx3_layout.py`.

Its model defaults (`MLX3_UP`, `MLX3_STEM`, `MLX3_REF`, `MLX3_TEACHER`,
`MLX3_METRICS`) are read at source time, so for a model other than Qwen3.8-27B
they have to be exported before `source`.

Three environment settings in it are defaults for a reason, and all three cost a
day to find. They are explained in the file's own comments and in
[troubleshooting.md](troubleshooting.md):

| setting                     | value | why                                                                                                                                                        |
| --------------------------- | ----- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `MLX_USE_CUDA_GRAPHS`       | 0     | a compiled graph remembers buffer addresses the allocator later reuses. Turning capture off removes a whole class of failures for about two seconds a step |
| `MLX_CUDA_GRAPH_CACHE_SIZE` | 2048  | 1 aborts with cache thrashing, 6144 runs out of memory                                                                                                     |
| `MLX3_SEQ`                  | 512   | collapses the long tail of tensor shapes into one, which is what makes memory flat rather than climbing                                                    |

## `foundry-nvfp4.sh`

The NVFP4 side, written for DeepSeek-V4.1-Flash and nothing else yet. Self
contained: it does not source `foundry.sh` and never builds llama.cpp.

It is a different shape of work from the other two files, because the model
already ships its experts in MXFP4 and NVFP4 is a lossless re-encoding of them.
What NVIDIA's recipe adds is one calibrated scalar per expert projection, and
that scalar is only consumed on Blackwell. The file therefore describes two
boxes, a calibration box that runs DeepSeek's reference model under modelopt,
and a Blackwell measurement box that runs the vLLM branch which knows
`deepseek_v41`. `nvfp4_box calib` and `nvfp4_box stand` print the command list
for each. [runbook-nvfp4.md](../docs/runbook-nvfp4.md) explains why the
measurement has to be on Blackwell and what the table can honestly claim.

Like `foundry-mlx3.sh` it writes its python helpers to disk with
`nvfp4_write_py`, under `/tools`: a patcher for modelopt's `ptq.py`, the corpus
windower, the flat-amax generator that stands in for "no calibration", the
coverage counter, the vLLM logprob dump and the KLD bracket. Two pins live at
the top of the file and matter: the modelopt commit the patcher was written
against, and the sha of the vLLM branch, which is force pushed upstream.

## `auto_fmt.py`

The generic chat renderer. It drives a model's own `chat_template.jinja` through
`transformers.apply_chat_template`, which is the reference implementation, so
byte equality with what the model sees at inference is true by construction
rather than by assertion. Registered in `calib-corpora/tools/build.py` as
`chat.format: auto`, which means a new model needs a recipe and no new code.

It needs the original weights on disk:

```bash
export FOUNDRY_MODEL_DIR=/src
```

If the template calls `strftime_now`, it refuses to render unless a date is
pinned, rather than quietly producing a corpus nobody can rebuild:

```bash
export FOUNDRY_PIN_DATE=2026-08-14
```

> [!NOTE] The canonical copy lives in `calib-corpora/tools/auto_fmt.py`. The
> copy here exists only so `install_auto_fmt` can curl it onto a box. Since that
> box has already cloned calib-corpora at that point, the function could read
> from the clone and this duplicate could go.

## ABLITERATION

1. curl foundry
2. load model:

```bash
use_model qwen3.8-flash-next Qwen/Qwen3.8-Flash-Next
```

3. convert and convert to gguf:

```bash
abliterate_and_convert_to_gguf
```
-> /abliterated-Qwen/Qwen3.8-Flash-Next

4. use test_chat e.t.c. to ensure abliterated model works

## How they are meant to be used

Source, then call one function at a time in the foreground. Every function
checks its inputs, says what it is doing, and stops loudly when something is
missing. Nothing is backgrounded, so a tmux pane per process plus `ps` is the
whole process monitor.

`reload` updates the file in place from git and re-sources it, reporting the
commit it landed on. It uses git rather than the raw file URL because both http
paths to GitHub are cached and can hand back a copy that is minutes old, and
`reload` would then report success on a file that has not changed.

`selfcheck` verifies that every function the help text advertises actually
exists. Editing a large shell file by cutting between two function names quietly
eats whatever sat between them, and this is what catches that.

## `bonsai/`

Refusal ablation for PrismML's Ternary Bonsai 2 27B. `abliterate.py` cannot be used there:
the pack is ternary, and a dense weight edit rounds away when written back. These tools
build the same projection as a rank-1 LoRA that llama.cpp keeps in the graph, verify it
inside the running model, and measure what it costs. Nine files, all of them run from the
repository root against a local pack; see
[runbook-bonsai-abliterate.md](../docs/runbook-bonsai-abliterate.md).
