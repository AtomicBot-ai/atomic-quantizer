# Runbook: a GGUF release

The complete sequence for a model nobody has quantized yet. Every command runs
in the foreground in its own tmux pane. Nothing here is meant to be pasted as a
block: run one line, read what it prints, then run the next.

## 0. Before renting anything

Two checks cost nothing and save an hour.

```bash
# does this model even have a chat template we can render, and are the
# markers it emits real special tokens in the vocabulary
corpus_check <upstream-repo>

# what the ladder will cost on disk
disk_plan
```

`corpus_check` reads the tokenizer files straight off the hub. If it says not
one marker is a real token, the tokenizer files do not match the weights and
building a corpus on that is pointless.

## 1. The box

```bash
# on your machine
vastai ssh-url <instance-id>
TERM=xterm ssh root@<ip> -p <port>

# inside
export HF_TOKEN=hf_...
curl -sL https://raw.githubusercontent.com/<ORG>/<REPO>/main/foundry.sh -o /foundry.sh
source /foundry.sh
persist_shell
save_token
```

`persist_shell` appends `source /foundry.sh` to `~/.bashrc`, so a new tmux pane
loads the toolbox by itself. `save_token` does the same for `HF_TOKEN`. Both are
per box and both are the reason a second pane is not amnesiac.

Then the environment:

```bash
check            # disk, GPUs, cores, RAM
cuda_arch        # reads compute capability off the card, prints the build line
cuda_check       # toolkit against host driver, removes broken compat libs
setup            # build tools and python packages, creates the directories
build 120        # the number cuda_arch printed
gpu_test         # every GPU has to appear in the list
```

Two things `setup` does not install and you will want:

```bash
apt-get install -y vim
fix_tmux         # mouse scrolling and 200k lines of scrollback
```

`fix_tmux` is worth running on every box. The default tmux keeps almost no
history and scrolling it needs a key sequence, which makes reading a forty
minute log painful.

## 2. Pick the model

```bash
use_model qwen3.8-27b Qwen/Qwen3.8-27B
status
plan
```

`use_model` refuses an upstream id that does not exist on the hub, so a typo
cannot propagate into every later step. `status` is the local checklist. `plan`
goes to the hub and says what already exists there, so you do not rebuild
something that has been published for a week.

If the box was used for another model before:

```bash
clean_run        # removes /logs /kld /imatrix, leaves the weights alone
clean_gguf       # offers to delete weights belonging to a different model
```

`status` warns about this by itself. Stale logs from a previous model are a real
problem: `results` would parse them into this model's table and `push_logs`
would upload them.

## 3. The BF16 reference file

If we already published one, pull it. If not, build it.

```bash
get_bf16                 # if it exists in <Model>-GGUF

# or, for a model nobody has done yet
get_upstream             # original weights into /src
setup_convert            # torch and friends, about 3 GB, only needed to convert
make_bf16                # convert_hf_to_gguf.py --no-nextn
check_blocks             # declared block count against tensors actually present
```

> [!IMPORTANT]
> `make_bf16` passes `--no-nextn` on purpose. A multi token prediction head
> inside the main file is never executed by a plain forward pass, so it only
> adds weight and forces `llama-quantize` to special case it. Worse, when the
> config promises a head that the weights do not carry, the converter writes a
> block count it cannot fill: the export succeeds and the file refuses to load.
> `check_blocks` catches exactly that.

For a vision model:

```bash
make_mmproj              # writes the projector, f16 and bf16
push_mmproj
```

One projector serves the whole ladder and is under a gigabyte, so it is not
worth quantizing. `status` notices `preprocessor_config.json` in `/src` and
tells you when a model needs one.

## 4. The calibration corpus

Usually it already exists and you just pull it:

```bash
ls_corpora               # what builds are in calib-corpora
get_calib qwen3.8-27b    # pulls builds/<name>/calib_train.txt
im_size                  # how many chunks that is at the imatrix context
```

For a model with no build yet, see the corpus section of
`calib-corpora/RECIPE.md`. The short path from this toolbox:

```bash
new_model qwen3.8-27b Qwen/Qwen3.8-27B   # corpus_check, then make_recipe
cat /recipes/qwen3.8-27b.yaml            # read the shares, they are the one judgement call
get_tools                                # clone calib-corpora, with git-lfs
check_pool                               # every jsonl in the pool has to parse
fix_pool                                 # repairs files that came down as LFS pointers
install_auto_fmt                         # the generic renderer
setup_corpus
cp /recipes/qwen3.8-27b.yaml /calib-corpora/recipes/
build_corpus qwen3.8-27b
push_corpus qwen3.8-27b                  # so every other box can pull the same bytes
```

Also worth running once per model, on a box that already has a quant:

```bash
check_template           # our stored template against upstream's current one
```

If upstream changed the template after we converted, the corpus was rendered
through the older one and the calibration no longer matches what the model reads
at inference.

## 5. The importance matrix

This is the step that gets fanned out across machines. It has its own document:
[imatrix-sharding.md](imatrix-sharding.md).

On a single box the whole thing is:

```bash
pick_model               # interactive, choose the bf16 file to collect on
im_size
im_shard 0 1             # one node, the whole corpus
im_stats /imatrix/shard-0-of-1.gguf
```

## 6. The reference measurement

```bash
get_eval                 # eval/neutral by default
eval_size                # sanity: how many chunks the corpus is at this context
base                     # writes the reference logits
```

`base` deliberately uses every GPU regardless of `use_gpus`, because BF16 does
not fit on fewer. Say so in the model card. It takes four to eight minutes and
produces a very large file that is not worth copying anywhere, only reproducing.

Publish the log immediately, and the blob when the run is finished:

```bash
push_base                # log, env, llama.cpp commit, then the blob or its parts
```

If a second box needs the same reference, sending it directly is much faster
than a round trip through the hub:

```bash
send_base root@<ip> <port>
```

On the receiving box, `get_base` reassembles published parts and checks the byte
count against the manifest.

## 7. The ladder

```bash
bits                              # what the model is made of, by tensor role
bits 'ffn_down_exps=iq3_s' '*=q8_0'   # predict a size before spending ten minutes
write_ladder                      # writes /ladder.txt
ladder /ladder.txt --dry          # sizes only, nothing is built
ladder /ladder.txt                # build, measure and publish every rung
```

`ladder` does the whole loop per rung: predict the size, quantize, measure
against the reference, upload the file, move on. Nothing in a rung depends on
another rung, so the file can be split across boxes:

```bash
ladder_part 2 4                   # regenerates /ladder.txt, cuts the second quarter into /my.txt
ladder /my.txt --dry
ladder /my.txt
```

> [!NOTE]
> `ladder_part` regenerates the ladder before slicing on purpose. Cutting a
> slice by hand with `sed` leaves a stale copy behind after every ladder change,
> and the stale copy is what then runs.

Two things about the rules. `--tensor-type` takes a ggml type, not an ftype
mix: `IQ2_M`, `IQ3_M`, `Q3_K_S` and friends are recipes for which tensor gets
which type and cannot be assigned to a tensor. Passing one aborts the run after
the model has been read, which is why `check_types` runs first. And a k or i
quant needs rows divisible by 256; `bits` prints that per group and warns when a
rule would silently fall back to a block-32 type.

After building:

```bash
check_meta                        # what llama.cpp recorded as general.file_type
```

Hugging Face builds its hardware compatibility table from `general.file_type`,
not from the filename. A four bit build whose declared type says Q8_0 gets
grouped with Q8_0 and disappears from its own row. Fix it at quantize time with
`--override-kv general.file_type=int:15`.

## 8. Measuring

```bash
use_gpus 0,1                      # hold the quant runs to a setup readers can relate to
kld_all                           # everything not yet measured
bench_all                         # llama-bench, GPU 0 only
results                           # parses every log into a table and results.json
```

`kld_all` skips what is already measured, so it is also the catch up command
after a box built quants before its reference existed. `KLD_FORCE=1` redoes them.

Two builds a few thousandths of KLD apart may not differ at all. The ± that
llama-perplexity prints belongs to one build and is taken over tokens, so it
says nothing about the gap between two. `kld_diff` pairs two logs chunk by chunk
over the same tokens and puts a bootstrap interval, over chunks, on the
difference:

```bash
kld_diff Q5_K_S AD-Q5_K_S 38.8    # bare names read /logs/kld-$EVALSET--NAME.log
```

```
chunks 30  base 0.007950  variant 0.007140
dKLD -0.000810  95% CI [-0.000900, -0.000717]  rel -10.2% [-11.2, -9.2]  variant lower in 30/30 chunks
d top-1 +0.121 pp  95% CI [+0.027, +0.220]
per 100 MiB: -0.002088  [-0.002320, -0.001847]
```

The third argument is how many MiB the variant adds over the base; with it the
last line says what those MiB bought. An interval that holds zero is not a
result, however the means read. Both logs have to come from the same eval set,
context and reference, and `kld_diff` refuses when the chunk counts or the
reference perplexity disagree. The intervals err slightly wide, because the
log rounds to five decimals; `kld_diff` with no arguments says by how much.

Other publishers' builds get measured the same way, against our reference:

```bash
get_external unsloth/Qwen3.8-27B-GGUF '*UD-IQ3_XXS*'
kld_ext
```

Each publisher lands in `/gguf/external/<org>/` and is logged as
`<org>--<file>`, because everyone names their files identically and otherwise
the measurement belongs to whoever downloaded last.

Behaviour checks, per rung or across the ladder:

```bash
test_chat /gguf/<quant>.gguf      # thinking block opens and closes, no literal markup
test_tools /gguf/<quant>.gguf     # prints the two pane server and curl recipe
test_longctx /gguf/<quant>.gguf 32768
test_vision /gguf/<quant>.gguf    # for a vision model
test_vision_all
```

## 9. Publishing

```bash
audit                             # published but never measured, measured but not uploaded
push_quants                       # idempotent, uploads whatever is missing
push_logs
push_results
merge_results                     # stitches every box's results-<hostname>.json into one
```

`audit` before writing the card. A quant in the repository with no measurement
is a row of the table that cannot be filled in.

Files over 50 GB do not go up whole:

```bash
push_model_split /gguf/<big>.gguf   # llama-gguf-split into 45 GB shards, uploaded as a folder
```

Then the card:

```bash
push_card /path/to/README.md
```

## Two shortcuts

For a completely fresh box, `newbox` chains the setup:

```bash
newbox qwen3.8-27b Qwen/Qwen3.8-27B 120
```

And for a box that built quants before its reference existed:

```bash
catch_up                          # builds a reference if missing, measures, publishes
```
