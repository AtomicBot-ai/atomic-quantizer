#!/bin/bash
# ================================================================= FOUNDRY NVFP4
#
# The NVFP4 side, for DeepSeek-V4.1-Flash. Self contained: it does not source
# foundry.sh and it never builds llama.cpp. On a bare rented box:
#
#   export HF_TOKEN=hf_...
#   git clone https://github.com/AtomicBot-ai/atomic-quantizer /quantizer
#   source /quantizer/scripts/foundry-nvfp4.sh
#   nvfp4_setup
#   nvfp4_persist
#   nvfp4_box calib      the exact command list for the calibration box
#   nvfp4_box stand      the exact command list for the measurement box
#
# WHAT THIS FILE IS FOR
#
# V4.1-Flash ships its routed experts in MXFP4 already: E2M1 nibbles with one
# E8M0 scale per 32 weights. NVFP4 keeps the nibbles bit for bit and rewrites
# the scales as E4M3 per 16 plus one fp32 per tensor. The cast is lossless and
# needs no GPU. What NVIDIA's recipe adds on top is ONE calibrated scalar per
# expert projection, input_scale, which places the E4M3 window for the FP4
# activations. That scalar is consumed only on Blackwell, only on the generic
# FlashInfer/CUTLASS NVFP4 MoE path. On Hopper it is dropped (Marlin runs
# W4A16) and every NVFP4 build produces the same bits.
#
# So there are two boxes and they must not be confused:
#
#   calib box   8xH200 or 8xB200. Runs DeepSeek's reference model.py under
#               modelopt to collect input amax, then exports the checkpoint.
#   stand box   Blackwell: 8xB200, 8xB300 or 8xRTX PRO 6000. Runs the vLLM branch
#               that knows deepseek_v41 and measures. A stand on Hopper measures
#               nothing about the recipe.
#
# Three checkpoints get measured against one reference, the native MXFP4
# checkpoint served by the same vLLM build on the same GPUs:
#
#   cast        input_scale = 1.0 everywhere. No calibration. What s-zaizen and
#               LibertAI published. Made here with nvfp4_amax_flat so the
#               provenance is ours and identical to the other two.
#   nvidia      NVIDIA's own datasets and budget: cnn_dailymail plus
#               nemotron-post-training-dataset-v2, 64 samples of 512 tokens.
#   atomic      our calib-corpora build for this model, chat markup of the
#               model itself, windows of 4096, sixty times more tokens per
#               expert.
#
# Read docs/runbook-nvfp4.md before renting anything. In particular the
# section on what "better" can and cannot mean here: theory says the three
# will sit within a few percent of each other on KLD, and the one number that
# is guaranteed to differ is calibration coverage, which nvfp4_coverage prints.

NVFP4_VERSION=2026-09-10.02

NVFP4_UP=${NVFP4_UP:-deepseek-ai/DeepSeek-V4.1-Flash}
NVFP4_ORG=${NVFP4_ORG:-AtomicChat}
NVFP4_STEM=${NVFP4_STEM:-DeepSeek-V4.1-Flash}
NVFP4_REPO=${NVFP4_REPO:-$NVFP4_ORG/$NVFP4_STEM-NVFP4}
NVFP4_METRICS=${NVFP4_METRICS:-$NVFP4_ORG/$NVFP4_STEM-NVFP4-metrics}
NVFP4_CALIB_BUILD=${NVFP4_CALIB_BUILD:-dsv41-flash}

# Pins. The modelopt commit is the one both community casts used and the one
# whose ptq.py the patcher below was written against. The vLLM sha is the head
# of vllm-project/vllm:dsv41-feat, PR #56214, a branch that can be force pushed
# at any time, which is why it is a sha and not a branch name.
NVFP4_MODELOPT_SHA=${NVFP4_MODELOPT_SHA:-079078de9d3cd26fa5baafdb754bba29dde24f7d}
NVFP4_VLLM_SHA=${NVFP4_VLLM_SHA:-e47aa780bccf59f59dfa2cbb18e17a10b4fe69ba}
NVFP4_VLLM_IMAGE=${NVFP4_VLLM_IMAGE:-vllm/vllm-openai:nightly}

# Everything at the filesystem root, same as foundry.sh. On macOS nothing can
# be created at /, so NVFP4_ROOT redirects the layout for a dry run.
NVFP4_ROOT=${NVFP4_ROOT:-}
NVFP4_SRC=${NVFP4_SRC:-$NVFP4_ROOT/src}          # the HF checkpoint as released
NVFP4_MP=${NVFP4_MP:-$NVFP4_ROOT/mp}             # resharded for the reference runtime
NVFP4_AMAX=${NVFP4_AMAX:-$NVFP4_ROOT/amax}       # one subdir per calibration name
NVFP4_OUT=${NVFP4_OUT:-$NVFP4_ROOT/nvfp4}        # exported checkpoints, one dir each
NVFP4_EVAL=${NVFP4_EVAL:-$NVFP4_ROOT/eval}
NVFP4_LOGS=${NVFP4_LOGS:-$NVFP4_ROOT/logs}
NVFP4_TOOLS=${NVFP4_TOOLS:-$NVFP4_ROOT/tools}    # modelopt clone and the helpers
# The calibration python. modelopt pins transformers below 5.15 and the vLLM
# image ships a newer one, so on a box that does both, calibration lives in
# its own venv and vLLM stays in the image python. nvfp4_calib_env builds it;
# every calibration function uses it when it exists and python3 otherwise.
NVFP4_VENV=${NVFP4_VENV:-$NVFP4_ROOT/calib-venv}
NVFP4_HF=${NVFP4_HF:-$NVFP4_ROOT/hf}

# Calibration protocol.
NVFP4_TP=${NVFP4_TP:-8}
NVFP4_CALIB_SEQ=${NVFP4_CALIB_SEQ:-4096}
NVFP4_CALIB_SIZE=${NVFP4_CALIB_SIZE:-512}        # windows for the atomic run
NVFP4_CALIB_BATCH=${NVFP4_CALIB_BATCH:-2}

# Measurement protocol. Same as the published GGUF and MLX tables so the
# numbers can sit side by side: chunks of 4096, only the second half of each
# chunk scored, reference first in the divergence. One difference that must be
# printed beside every number: vLLM returns top-K log probabilities, not the
# whole vocabulary, so the KLD here is a bracket, see nvfp4_kld.py.
NVFP4_CTX=${NVFP4_CTX:-4096}
NVFP4_CHUNKS=${NVFP4_CHUNKS:-24}
NVFP4_TOPK=${NVFP4_TOPK:-512}
# Every corpus in this list is scored by one model load. Loading V4.1 into
# vLLM is the slow step on the stand, so the corpora are never looped outside.
NVFP4_CORPORA=${NVFP4_CORPORA:-neutral code agentic}

export HF_HOME=${HF_HOME:-$NVFP4_HF}
export HF_HUB_ENABLE_HF_TRANSFER=${HF_HUB_ENABLE_HF_TRANSFER:-1}
export TOKENIZERS_PARALLELISM=false


# ================================================================== orientation

nvfp4_help() {
    cat << EOF
foundry-nvfp4 $NVFP4_VERSION      upstream $NVFP4_UP

  box
    nvfp4_setup              modelopt at the pinned commit, patched, plus the helpers
    nvfp4_calib_env          the calibration venv: torch, tilelang, modelopt, transformers<5.15
    nvfp4_persist            source this file from every new tmux pane
    nvfp4_check              disk, GPUs, RAM, what is on disk already
    nvfp4_box calib|stand    the command list for that box, paste one at a time

  get
    nvfp4_get src            the released checkpoint, 475 GiB
    nvfp4_get eval           the measurement corpora
    nvfp4_get calib          builds/$NVFP4_CALIB_BUILD/calib_train.txt
    nvfp4_get cast           s-zaizen's community cast, for a cross check only

  calibration box
    nvfp4_reshard            DeepSeek convert.py into $NVFP4_TP files, CPU, needs RAM = checkpoint
    nvfp4_calib_jsonl        calib_train.txt into windows of $NVFP4_CALIB_SEQ tokens
    nvfp4_calib nvidia       NVIDIA's datasets and budget
    nvfp4_calib atomic       our corpus
    nvfp4_amax_flat          the no-calibration amax, input_scale = 1.0
    nvfp4_coverage NAME      how many experts a calibration actually saw
    nvfp4_export NAME        lossless cast plus that calibration -> $NVFP4_OUT/$NVFP4_STEM-NVFP4-NAME

  stand box
    nvfp4_stand              the vast template settings and the vLLM branch build, printed
    nvfp4_ref                reference logprobs from the native checkpoint
    nvfp4_score DIR NAME     logprobs from one NVFP4 checkpoint
    nvfp4_kld NAME           the bracket, top-1 and ppl against the reference
    nvfp4_repeat             the noise floor: the reference against itself
    nvfp4_table              everything measured on this box

  publish
    nvfp4_push NAME          checkpoint to $NVFP4_REPO, logs to $NVFP4_METRICS
EOF
}

nvfp4_persist() {
    local me
    me="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
    grep -q "foundry-nvfp4.sh" ~/.bashrc 2>/dev/null || echo "source $me" >> ~/.bashrc
    if [ -n "$HF_TOKEN" ]; then
        grep -q "HF_TOKEN=" ~/.bashrc 2>/dev/null || echo "export HF_TOKEN=$HF_TOKEN" >> ~/.bashrc
        echo "HF_TOKEN written to ~/.bashrc. The box is disposable, the token is not: revoke it after."
    fi
    echo "new panes will source $me"
}

nvfp4_box() {
    case "${1:-}" in
    calib) cat << EOF
# calibration box: 8xH200 or 8xB200, 2 TB NVMe (src + reshard + three exports of ~300 GiB), host RAM >= 600 GB
nvfp4_setup
nvfp4_calib_env
nvfp4_persist
nvfp4_check
nvfp4_get src
nvfp4_get calib
nvfp4_reshard                          # hours, CPU only, watch RAM
nvfp4_calib_jsonl
nvfp4_amax_flat                        # seconds
nvfp4_calib nvidia                     # about an hour incl. tilelang compile
nvfp4_calib atomic                     # a few hours
nvfp4_coverage nvidia ; nvfp4_coverage atomic ; nvfp4_coverage flat
nvfp4_export flat
nvfp4_export nvidia
nvfp4_export atomic
nvfp4_push flat ; nvfp4_push nvidia ; nvfp4_push atomic
EOF
    ;;
    stand) cat << EOF
# stand box: Blackwell, 8xB200 / 8xB300 / 8xRTX PRO 6000, 2 TB NVMe, host RAM >= 300 GB (Engram tables live in pinned host memory)
nvfp4_setup
nvfp4_persist
nvfp4_get src
nvfp4_get eval
nvfp4_stand                            # the template settings and the build, read it first
nvfp4_ref                              # native checkpoint, the reference; also the SM120 smoke test
nvfp4_repeat                           # noise floor, reference against itself
# the three NVFP4 checkpoints, once the calib box pushed them:
hf download $NVFP4_REPO-flat   --local-dir $NVFP4_OUT/$NVFP4_STEM-NVFP4-flat
hf download $NVFP4_REPO-nvidia --local-dir $NVFP4_OUT/$NVFP4_STEM-NVFP4-nvidia
hf download $NVFP4_REPO-atomic --local-dir $NVFP4_OUT/$NVFP4_STEM-NVFP4-atomic
nvfp4_score $NVFP4_OUT/$NVFP4_STEM-NVFP4-flat   flat
nvfp4_score $NVFP4_OUT/$NVFP4_STEM-NVFP4-nvidia nvidia
nvfp4_score $NVFP4_OUT/$NVFP4_STEM-NVFP4-atomic atomic
nvfp4_kld flat ; nvfp4_kld nvidia ; nvfp4_kld atomic
nvfp4_table
EOF
    ;;
    *) echo "nvfp4_box calib|stand" ;;
    esac
}

nvfp4_check() {
    echo "--- disk ---";  df -h "${NVFP4_ROOT:-/}" 2>/dev/null | tail -1
    echo "--- ram ---";   free -g 2>/dev/null | head -2 || vm_stat | head -2
    echo "--- gpus ---";  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "no nvidia-smi"
    echo "--- on disk ---"
    [ -f "$NVFP4_SRC/config.json" ] && echo "  [x] src        $NVFP4_SRC" || echo "  [ ] src        -> nvfp4_get src"
    [ -f "$NVFP4_MP/model0-mp$NVFP4_TP.safetensors" ] && echo "  [x] reshard    $NVFP4_MP" || echo "  [ ] reshard    -> nvfp4_reshard"
    [ -f "$NVFP4_EVAL/calib_train.txt" ] && echo "  [x] calib txt" || echo "  [ ] calib txt  -> nvfp4_get calib"
    [ -f "$NVFP4_EVAL/calib.jsonl" ] && echo "  [x] calib jsonl" || echo "  [ ] calib jsonl -> nvfp4_calib_jsonl"
    local d; for d in "$NVFP4_AMAX"/*/; do [ -d "$d" ] && echo "  [x] amax       $(basename "$d")"; done
    for d in "$NVFP4_OUT"/*/; do [ -f "$d/config.json" ] && echo "  [x] export     $(basename "$d")"; done
    local c; for c in $(echo $NVFP4_CORPORA); do
        [ -f "$NVFP4_EVAL/$c.txt" ] && echo "  [x] eval       $c" || echo "  [ ] eval       $c -> nvfp4_get eval"
    done
    ls "$NVFP4_LOGS"/lp-*.npz 2>/dev/null | sed 's/^/  [x] logprobs   /'
    ls "$NVFP4_LOGS"/kld-*.json 2>/dev/null | sed 's/^/  [x] kld        /'
}


# ================================================================== setup

nvfp4_setup() {
    mkdir -p "$NVFP4_SRC" "$NVFP4_MP" "$NVFP4_AMAX" "$NVFP4_OUT" "$NVFP4_EVAL" \
             "$NVFP4_LOGS" "$NVFP4_TOOLS" "$NVFP4_HF"
    if ! command -v hf > /dev/null; then
        python3 -m pip install -q -U "huggingface_hub[cli,hf_transfer]"
    fi
    if [ ! -d "$NVFP4_TOOLS/modelopt/.git" ]; then
        git clone -q https://github.com/NVIDIA/Model-Optimizer "$NVFP4_TOOLS/modelopt"
    fi
    git -C "$NVFP4_TOOLS/modelopt" fetch -q origin "$NVFP4_MODELOPT_SHA" 2>/dev/null || true
    git -C "$NVFP4_TOOLS/modelopt" checkout -q "$NVFP4_MODELOPT_SHA" \
        || { echo "modelopt: could not check out $NVFP4_MODELOPT_SHA"; return 1; }
    nvfp4_write_py > /dev/null
    python3 "$NVFP4_TOOLS/nvfp4_patch_ptq.py" "$NVFP4_TOOLS/modelopt/examples/deepseek/deepseek_v4/ptq.py" || return 1
    echo
    echo "calibration box: run  nvfp4_calib_env  next. Stand box: nothing more to install."
    echo
    echo "helpers written to $NVFP4_TOOLS, modelopt at $NVFP4_MODELOPT_SHA with the V4.1 patch applied"
}



# The calibration venv. torch from the cu128 index because modelopt pulls
# cupy-cuda12x, which wants a CUDA 12 runtime; a 13.x driver runs it fine.
# tilelang compiles its kernels with whatever nvcc is on PATH, the image's.
nvfp4_calib_env() {
    if [ -x "$NVFP4_VENV/bin/python3" ] && "$NVFP4_VENV/bin/python3" -c "import modelopt, tilelang" 2>/dev/null; then
        echo "already here: $NVFP4_VENV"; return 0
    fi
    [ -d "$NVFP4_TOOLS/modelopt/.git" ] || { echo "no modelopt clone. Run:  nvfp4_setup"; return 1; }
    command -v nvcc > /dev/null || echo "WARNING: no nvcc on PATH, tilelang cannot compile kernels without it"
    python3 -m venv "$NVFP4_VENV" || return 1
    "$NVFP4_VENV/bin/pip" install -q -U pip wheel setuptools || return 1
    "$NVFP4_VENV/bin/pip" install -q torch --index-url https://download.pytorch.org/whl/cu128 || return 1
    "$NVFP4_VENV/bin/pip" install -q "transformers>=4.57,<5.15" tokenizers "safetensors>=0.7" numpy sympy Pillow tqdm \
        tilelang==0.1.8 "huggingface_hub[cli,hf_transfer]" || return 1
    "$NVFP4_VENV/bin/pip" install -q -e "$NVFP4_TOOLS/modelopt" || return 1
    "$NVFP4_VENV/bin/python3" - << 'VENVEOF'
import torch, tilelang, modelopt, transformers
print("torch %s cuda %s   tilelang %s   modelopt %s   transformers %s" % (
    torch.__version__, torch.version.cuda, tilelang.__version__, modelopt.__version__, transformers.__version__))
print("gpus:", torch.cuda.device_count())
VENVEOF
    echo "calibration venv ready at $NVFP4_VENV"
}

nvfp4_py() {
    if [ -x "$NVFP4_VENV/bin/python3" ]; then echo "$NVFP4_VENV/bin/python3"; else echo python3; fi
}

nvfp4_torchrun() {
    if [ -x "$NVFP4_VENV/bin/torchrun" ]; then echo "$NVFP4_VENV/bin/torchrun"; else echo torchrun; fi
}

# ================================================================== get

nvfp4_get() {
    case "${1:-}" in
    src)
        hf download "$NVFP4_UP" --local-dir "$NVFP4_SRC" || return 1
        du -sh "$NVFP4_SRC"
        ;;
    eval)
        hf download "$NVFP4_ORG/calib-corpora" --repo-type dataset --local-dir "$NVFP4_EVAL/corpora" \
            --include "eval/neutral/eval_neutral.txt" --include "eval/code/eval_code_full.txt" \
            --include "eval/agentic/eval_agentic.txt" || return 1
        cp "$NVFP4_EVAL/corpora/eval/neutral/eval_neutral.txt" "$NVFP4_EVAL/neutral.txt"
        cp "$NVFP4_EVAL/corpora/eval/code/eval_code_full.txt" "$NVFP4_EVAL/code.txt"
        cp "$NVFP4_EVAL/corpora/eval/agentic/eval_agentic.txt" "$NVFP4_EVAL/agentic.txt"
        ls -la "$NVFP4_EVAL"/*.txt
        ;;
    calib)
        hf download "$NVFP4_ORG/calib-corpora" --repo-type dataset --local-dir "$NVFP4_EVAL/corpora" \
            --include "builds/$NVFP4_CALIB_BUILD/calib_train.txt" || return 1
        if [ ! -f "$NVFP4_EVAL/corpora/builds/$NVFP4_CALIB_BUILD/calib_train.txt" ]; then
            echo "no build named $NVFP4_CALIB_BUILD in calib-corpora."
            echo "V4.1 changed the chat markup relative to V4 (DSML tags with a leading space,"
            echo "numeric reasoning effort), so the dsv4-flash-0731 build does not apply."
            echo "Build one with calib-corpora/tools/build.py and a dsv41-flash recipe first."
            return 1
        fi
        cp "$NVFP4_EVAL/corpora/builds/$NVFP4_CALIB_BUILD/calib_train.txt" "$NVFP4_EVAL/calib_train.txt"
        wc -c "$NVFP4_EVAL/calib_train.txt"
        ;;
    cast)
        hf download s-zaizen/DeepSeek-V4.1-Flash-NVFP4 --local-dir "$NVFP4_OUT/community-cast" || return 1
        echo "cross check only: score it with  nvfp4_score $NVFP4_OUT/community-cast community-cast"
        ;;
    *) echo "nvfp4_get src|eval|calib|cast" ;;
    esac
}


# ================================================================== calibration box

# DeepSeek's own convert.py, experts kept in fp4. It accumulates every shard of
# every rank in host memory and writes at the end, so the box needs RAM at
# least the size of the checkpoint. There is no streaming variant upstream.
nvfp4_reshard() {
    [ -f "$NVFP4_SRC/config.json" ] || { echo "no checkpoint at $NVFP4_SRC. Run:  nvfp4_get src"; return 1; }
    if [ -f "$NVFP4_MP/model0-mp$NVFP4_TP.safetensors" ]; then
        echo "already here:"; ls -lh "$NVFP4_MP"; return 0
    fi
    local ram
    ram=$(free -g 2>/dev/null | awk '/Mem:/ {print $2}')
    [ -n "$ram" ] && [ "$ram" -lt 550 ] && echo "WARNING: ${ram} GB RAM, convert.py holds ~480 GB before writing"
    date
    "$(nvfp4_py)" "$NVFP4_SRC/inference/convert.py" \
        --hf-ckpt-path "$NVFP4_SRC" --save-path "$NVFP4_MP" \
        --model-parallel "$NVFP4_TP" --expert-dtype fp4 --tokenizer-path "$NVFP4_SRC" \
        2>&1 | tee "$NVFP4_LOGS/reshard.log"
    date
    ls -lh "$NVFP4_MP"
}

nvfp4_calib_jsonl() {
    [ -f "$NVFP4_EVAL/calib_train.txt" ] || { echo "no corpus. Run:  nvfp4_get calib"; return 1; }
    nvfp4_write_py > /dev/null
    "$(nvfp4_py)" "$NVFP4_TOOLS/nvfp4_calib_jsonl.py" "$NVFP4_SRC" "$NVFP4_EVAL/calib_train.txt" \
        "$NVFP4_EVAL/calib.jsonl" --window "$NVFP4_CALIB_SEQ" 2>&1 | tee "$NVFP4_LOGS/calib-jsonl.log"
}

# nvfp4_calib NAME
#   nvidia    cnn_dailymail + nemotron-post-training-dataset-v2, 64 x 512, the
#             published budget. Reproduces what NVIDIA does for V4.
#   atomic    our windows, NVFP4_CALIB_SIZE x NVFP4_CALIB_SEQ.
# Both go through the patched ptq.py, torchrun over NVFP4_TP ranks, and leave
# amax_dict_rank*-mp*.pt in $NVFP4_AMAX/NAME. Only routed experts collect
# anything; shared experts and MTP are disabled in the quant config.
nvfp4_calib() {
    local name="${1:-}" ds size seq
    [ -f "$NVFP4_MP/model0-mp$NVFP4_TP.safetensors" ] || { echo "no resharded checkpoint. Run:  nvfp4_reshard"; return 1; }
    case "$name" in
    nvidia) ds="cnn_dailymail nemotron-post-training-dataset-v2"; size=64; seq=512 ;;
    atomic)
        [ -f "$NVFP4_EVAL/calib.jsonl" ] || { echo "no calib.jsonl. Run:  nvfp4_calib_jsonl"; return 1; }
        ds="$NVFP4_EVAL/calib.jsonl"; size=$NVFP4_CALIB_SIZE; seq=$NVFP4_CALIB_SEQ ;;
    *) echo "nvfp4_calib nvidia|atomic"; return 1 ;;
    esac
    local out="$NVFP4_AMAX/$name"
    if ls "$out"/amax_dict_rank*.pt > /dev/null 2>&1; then
        echo "already here: $out"; ls "$out"; return 0
    fi
    mkdir -p "$out"
    echo "calibration '$name': $ds, $size samples of $seq tokens, $NVFP4_TP ranks"
    echo "first run compiles the tilelang kernels, that alone is many minutes"
    date
    "$(nvfp4_torchrun)" --nproc-per-node "$NVFP4_TP" --master_port 12346 \
        "$NVFP4_TOOLS/modelopt/examples/deepseek/deepseek_v4/ptq.py" \
        --model_path "$NVFP4_MP" \
        --config "$NVFP4_SRC/inference/config.json" \
        --dsv4_inference_dir "$NVFP4_SRC/inference" \
        --output_path "$out" \
        --batch_size "$NVFP4_CALIB_BATCH" \
        --calib_size "$size" --calib_seq "$seq" \
        --calib_dataset $(echo $ds) \
        2>&1 | tee "$NVFP4_LOGS/calib-$name.log"
    date
    ls -la "$out"
}

# The no-calibration variant. input amax = 6 * 448 = 2688 for every routed
# expert, so input_scale = amax / (6 * 448) = 1.0 exactly. This is what a cast
# without a calibration pass amounts to, and it goes through the same export as
# the other two, so the only thing that differs between the three checkpoints
# is the value of that one scalar.
nvfp4_amax_flat() {
    nvfp4_write_py > /dev/null
    local out="$NVFP4_AMAX/flat"
    mkdir -p "$out"
    "$(nvfp4_py)" "$NVFP4_TOOLS/nvfp4_amax_flat.py" "$NVFP4_SRC/config.json" "$out"
}

nvfp4_coverage() {
    local name="${1:-}"
    [ -d "$NVFP4_AMAX/$name" ] || { echo "nvfp4_coverage NAME   (one of: $(ls "$NVFP4_AMAX" 2>/dev/null | tr '\n' ' '))"; return 1; }
    nvfp4_write_py > /dev/null
    "$(nvfp4_py)" "$NVFP4_TOOLS/nvfp4_coverage.py" "$NVFP4_AMAX/$name" "$NVFP4_SRC/config.json" \
        "$NVFP4_LOGS/coverage-$name.json" 2>&1 | tee "$NVFP4_LOGS/coverage-$name.log"
    nvfp4_upload "$NVFP4_LOGS/coverage-$name.json" "logs/coverage-$name.json"
}

# Lossless MXFP4 -> NVFP4 cast on the routed experts, input_scale from the
# named calibration, everything else copied or hard linked from the source.
# Operates shard by shard on the released layout, so it needs no big RAM; the
# GPU is only for dequantizing experts that have no weight amax, which with
# --cast_mxfp4_to_nvfp4 is every expert, so give it one.
nvfp4_export() {
    local name="${1:-}"
    [ -d "$NVFP4_AMAX/$name" ] || { echo "nvfp4_export NAME   (one of: $(ls "$NVFP4_AMAX" 2>/dev/null | tr '\n' ' '))"; return 1; }
    local out="$NVFP4_OUT/$NVFP4_STEM-NVFP4-$name"
    if [ -f "$out/config.json" ]; then echo "already here: $out"; return 0; fi
    local dev=cpu; command -v nvidia-smi > /dev/null && dev=cuda
    date
    "$(nvfp4_py)" "$NVFP4_TOOLS/modelopt/examples/deepseek/deepseek_v4/quantize_to_nvfp4.py" \
        --amax_path "$NVFP4_AMAX/$name" \
        --source_ckpt "$NVFP4_SRC" \
        --output_ckpt "$out" \
        --cast_mxfp4_to_nvfp4 --device "$dev" \
        2>&1 | tee "$NVFP4_LOGS/export-$name.log"
    date
    grep -E '\[cast\]|lossless|synthes' "$NVFP4_LOGS/export-$name.log" | tail -5
    python3 - "$out" << 'CHKEOF'
import json, sys
d = sys.argv[1]
cfg = json.load(open(d + "/config.json"))["quantization_config"]
assert cfg.get("moe_quant_algo") == "NVFP4", cfg
assert cfg.get("weight_block_size") == [32, 32], cfg.get("weight_block_size")
assert cfg.get("expert_dtype") == "fp4", cfg.get("expert_dtype")
print("config.json: moe_quant_algo=NVFP4, weight_block_size=[32,32], expert_dtype=fp4  ok")
CHKEOF
    du -sh "$out"
}


# ================================================================== stand box

# The vLLM branch changes csrc (a new bool on the fused qnorm/rope/kv-insert
# op, group size 32 in per_token_group_quant), so the precompiled wheel of its
# merge base cannot be used: it has to be built. The nightly image has every
# dependency compiled already, vLLM itself is rebuilt on top of it.
#
# On vast.ai the image IS the instance: make a template with this image and
# the SSH launch mode. vast overrides the image ENTRYPOINT in that mode, so
# "vllm serve" never fires and you land in a shell. Set the container disk in
# the search page before renting, it cannot be changed after.
nvfp4_stand() {
    cat << EOF
# vast.ai template for the stand
#   image        $NVFP4_VLLM_IMAGE      (cu129-nightly instead if the host driver reports Max CUDA 12.x)
#   launch mode  SSH
#   disk         >= 2500 GB, set on the search page
#   on-start     nothing; everything below is typed in

# inside the instance:
git clone https://github.com/AtomicBot-ai/atomic-quantizer /quantizer
git clone https://github.com/vllm-project/vllm /vllm-src && cd /vllm-src
git checkout $NVFP4_VLLM_SHA
pip uninstall -y vllm
MAX_JOBS=\$(nproc) pip install -e . --no-build-isolation 2>&1 | tail -3     # about an hour
python3 -c "import vllm, vllm.models.deepseek_v4_1 as m; print(vllm.__version__, m.__file__)"
source /quantizer/scripts/foundry-nvfp4.sh
nvfp4_setup ; nvfp4_persist
# doing the calibration on this same box too: nvfp4_calib_env, it does not touch the image python

# on a bare metal host instead of vast, the same inside a container:
#   docker run --gpus all --ipc=host --shm-size 64g -it --entrypoint bash -v /:/host $NVFP4_VLLM_IMAGE
#   then export NVFP4_ROOT=/host before sourcing, so the layout lives on the host disk
EOF
}

# nvfp4_ref / nvfp4_score DIR NAME
# One vLLM offline run, one model load, every corpus in NVFP4_CORPORA: for
# each, NVFP4_CHUNKS windows of NVFP4_CTX tokens. Every window starts with BOS,
# the second half is scored, top-K log probabilities and the scored token's own
# log probability are written to $NVFP4_LOGS/lp-NAME-CORPUS.npz. No speculative
# decoding, no chat template: raw token ids go in, so the deepseek_v41
# tokenizer mode is never exercised.
nvfp4_score() {
    local dir="${1:-}" name="${2:-}" c files=""
    [ -f "$dir/config.json" ] && [ -n "$name" ] || { echo "nvfp4_score /path/to/checkpoint NAME"; return 1; }
    for c in $(echo $NVFP4_CORPORA); do
        [ -f "$NVFP4_EVAL/$c.txt" ] || { echo "no corpus $NVFP4_EVAL/$c.txt. Run:  nvfp4_get eval"; return 1; }
        files="$files $NVFP4_EVAL/$c.txt"
    done
    nvfp4_write_py > /dev/null
    local have=1
    for c in $(echo $NVFP4_CORPORA); do [ -f "$NVFP4_LOGS/lp-$name-$c.npz" ] || have=0; done
    if [ "$have" = 1 ] && [ "${NVFP4_FORCE:-0}" != "1" ]; then
        echo "already here:"; ls "$NVFP4_LOGS"/lp-"$name"-*.npz; return 0
    fi
    date
    python3 "$NVFP4_TOOLS/nvfp4_logprobs.py" "$dir" "$NVFP4_LOGS/lp-$name" \
        --corpora $(echo $files) \
        --tp "$NVFP4_TP" --ctx "$NVFP4_CTX" --chunks "$NVFP4_CHUNKS" --topk "$NVFP4_TOPK" \
        2>&1 | tee "$NVFP4_LOGS/lp-$name.log"
    date
    nvfp4_upload "$NVFP4_LOGS/lp-$name.log" "logs/lp-$name.log"
}

nvfp4_ref() {
    nvfp4_score "$NVFP4_SRC" ref
}

nvfp4_repeat() {
    NVFP4_FORCE=1 nvfp4_score "$NVFP4_SRC" ref-repeat
    nvfp4_kld ref-repeat
}

nvfp4_kld() {
    local name="${1:-}" c
    [ -n "$name" ] || { echo "nvfp4_kld NAME   (scored: $(ls "$NVFP4_LOGS"/lp-*.log 2>/dev/null | xargs -n1 basename | sed 's/lp-//;s/.log//' | tr '\n' ' '))"; return 1; }
    nvfp4_write_py > /dev/null
    for c in $(echo $NVFP4_CORPORA); do
        [ -f "$NVFP4_LOGS/lp-$name-$c.npz" ] || { echo "no $NVFP4_LOGS/lp-$name-$c.npz. Run:  nvfp4_score DIR $name"; return 1; }
        [ -f "$NVFP4_LOGS/lp-ref-$c.npz" ] || { echo "no reference for $c. Run:  nvfp4_ref"; return 1; }
        echo "--- $name on $c ---"
        python3 "$NVFP4_TOOLS/nvfp4_kld.py" "$NVFP4_LOGS/lp-ref-$c.npz" "$NVFP4_LOGS/lp-$name-$c.npz" \
            "$NVFP4_LOGS/kld-$name-$c.json" 2>&1 | tee "$NVFP4_LOGS/kld-$name-$c.log"
        nvfp4_upload "$NVFP4_LOGS/kld-$name-$c.json" "logs/kld-$name-$c.json"
        nvfp4_upload "$NVFP4_LOGS/kld-$name-$c.log" "logs/kld-$name-$c.log"
    done
}

nvfp4_table() {
    python3 - "$NVFP4_LOGS" "$NVFP4_CORPORA" << 'TBLEOF'
import glob, json, os, sys
log, corpora = sys.argv[1], sys.argv[2].split()
rows = {}
for p in sorted(glob.glob(os.path.join(log, "kld-*.json"))):
    try:
        r = json.load(open(p))
    except Exception:
        continue
    stem = os.path.basename(p)[4:-5]
    for c in corpora:
        if stem.endswith("-" + c):
            r["name"] = stem[: -len(c) - 1]
            rows.setdefault(c, []).append(r)
if not rows:
    print("nothing measured on this box yet"); sys.exit(0)
topk = None
for c in corpora:
    if c not in rows:
        continue
    print()
    print("corpus: %s" % c)
    print("%-22s %10s %10s %10s %9s %9s %9s" % ("build", "KLD low", "KLD high", "median", "p99", "top-1 %", "ppl"))
    for r in rows[c]:
        topk = r["topk"]
        print("%-22s %10.6f %10.6f %10.6f %9.5f %9.2f %9.4f" % (
            r["name"], r["mean_kld_low"], r["mean_kld_high"], r["median_kld"],
            r["p99_kld"], r["top1_agree_pct"], r["quant_ppl"]))
    rep = [r for r in rows[c] if r["name"] == "ref-repeat"]
    if rep:
        print("noise floor: reference against itself, KLD low %.6f, top-1 %.2f %%. Three times it is the"
              % (rep[0]["mean_kld_low"], rep[0]["top1_agree_pct"]))
        print("smallest defensible gap on this corpus.")
    else:
        print("noise floor not measured on this corpus yet: nvfp4_repeat")
print()
print("KLD is reference || candidate over the reference's top-%d tokens plus a rest bucket." % topk)
print("low  puts every reference token the candidate did not rank at the candidate's K-th probability,")
print("high spreads the candidate's leftover mass uniformly over the rest of the vocabulary.")
print("The truth is between the two. If they disagree by more than the gap you are trying to show,")
print("raise NVFP4_TOPK and measure again; the exact columns are top-1 and ppl.")
cov = sorted(glob.glob(os.path.join(log, "coverage-*.json")))
if cov:
    print()
    print("%-22s %12s %14s %12s" % ("calibration", "experts seen", "of routed", "median amax"))
    for p in cov:
        c = json.load(open(p))
        print("%-22s %12d %14d %12.2f" % (os.path.basename(p)[9:-5], c["experts_with_input_amax"],
                                          c["routed_experts_total"], c["median_input_amax_w13"]))
print()
TBLEOF
}


# ================================================================== publish

nvfp4_upload() {
    [ -f "$1" ] || return 0
    [ -z "$HF_TOKEN" ] && return 0
    python3 - "$1" "$2" "$NVFP4_METRICS" << 'UPEOF'
import sys
from huggingface_hub import HfApi
local, remote, repo = sys.argv[1:4]
try:
    HfApi().upload_file(path_or_fileobj=local, path_in_repo=remote, repo_id=repo, repo_type="dataset")
    print("  uploaded -> %s :: %s" % (repo, remote))
except Exception as e:
    print("  upload failed: %s" % str(e).splitlines()[0][:70])
UPEOF
}

# nvfp4_push NAME
# The checkpoint goes to one repository per calibration so a reader can tell
# them apart by id, the way the MLX rungs do. Logs go to the metrics dataset.
nvfp4_push() {
    local name="${1:-}"
    local dir="$NVFP4_OUT/$NVFP4_STEM-NVFP4-$name"
    [ -f "$dir/config.json" ] || { echo "nvfp4_push NAME   (exported: $(ls "$NVFP4_OUT" 2>/dev/null | tr '\n' ' '))"; return 1; }
    [ -n "$HF_TOKEN" ] || { echo "no HF_TOKEN"; return 1; }
    local repo="$NVFP4_REPO-$name"
    echo "pushing $dir -> $repo"
    hf repo create "$repo" --type model 2>/dev/null || true
    hf upload-large-folder "$repo" "$dir" --repo-type model --num-workers 8 || return 1
    hf repo create "$NVFP4_METRICS" --type dataset 2>/dev/null || true
    local f; for f in "$NVFP4_LOGS"/*-"$name".*; do
        [ -f "$f" ] && nvfp4_upload "$f" "logs/$(basename "$f")"
    done
}


# ================================================================== python helpers
#
# Written to disk rather than kept as heredocs inside a pipeline, for the same
# reason foundry-mlx3.sh does it: a heredoc attached to the wrong end of a pipe
# feeds the script to tee instead of to python.

nvfp4_write_py() {
mkdir -p "$NVFP4_TOOLS"

cat > "$NVFP4_TOOLS/nvfp4_patch_ptq.py" << 'PATCHEOF'
"""Make modelopt's DeepSeek-V4 ptq.py run DeepSeek-V4.1-Flash.

    python3 nvfp4_patch_ptq.py path/to/examples/deepseek/deepseek_v4/ptq.py

Two changes, both exact string replacements that fail loudly when the file is
not the one the patch was written against (commit 079078de):

  1. V4.1's Transformer takes the tokenizer: its Engram layers hash n-grams
     over a compressed token map built from it. ptq.py builds Transformer(margs)
     and loads the tokenizer afterwards, so the constructor would fail.
  2. ptq.py dequantizes FP8 linears with a hard coded 128x128 block. V4.1 stores
     them in 32x32 blocks (config quantization_config.weight_block_size), and
     the shared expert, which is an FP8 Expert and gets wrapped like the routed
     ones, hits that assert on the first forward. The block is now read off the
     scale shape.
"""
import sys

path = sys.argv[1]
src = open(path).read()
if "nvfp4-v41-patched" in src:
    print("ptq.py already patched")
    sys.exit(0)

edits = [
    (
        "        model = deekseep_v4_model.Transformer(margs)\n",
        "        model = deekseep_v4_model.Transformer(\n"
        "            margs, AutoTokenizer.from_pretrained(model_path)\n"
        "        )  # nvfp4-v41-patched: V4.1 hashes engram n-grams over the tokenizer\n",
    ),
    (
        "        return _fp8_ue8m0_blockwise_to_bf16(w, w.scale, block=128)\n",
        "        return _fp8_ue8m0_blockwise_to_bf16(\n"
        "            w, w.scale, block=w.shape[0] // w.scale.shape[0]\n"
        "        )  # nvfp4-v41-patched: V4.1 FP8 linears use 32x32 blocks\n",
    ),
]
for old, new in edits:
    n = src.count(old)
    if n != 1:
        print("patch target found %d times, expected 1:\n%s" % (n, old))
        print("ptq.py is not the file this patch was written against. Pin NVFP4_MODELOPT_SHA.")
        sys.exit(1)
    src = src.replace(old, new)
open(path, "w").write(src)
print("ptq.py patched for V4.1: tokenizer into Transformer, FP8 block from scale shape")
PATCHEOF

cat > "$NVFP4_TOOLS/nvfp4_calib_jsonl.py" << 'JSONLEOF'
"""calib_train.txt into windows of N tokens, one JSON object per line.

    python3 nvfp4_calib_jsonl.py TOKENIZER_DIR calib_train.txt OUT.jsonl [--window 4096] [--max N]

modelopt's loader takes a local .jsonl with a "text" field, tokenizes every
sample and pads or truncates it to --calib_seq. A calib_train.txt is one long
stream with documents separated by a blank line, so handed over as is it would
be one sample, truncated. Cut into token windows instead: every window is
exactly --calib_seq tokens long after re-tokenization, or one or two tokens
off where a window boundary split a merge, which for a max calibration does
not matter.

Windows are cut on the token stream and decoded back to text, because the
loader wants text. The decode is not byte exact around special tokens; the
chat markup survives because these tokenizers round trip their own specials.
"""
import argparse, json, sys
from transformers import AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("tokenizer_dir"); ap.add_argument("corpus"); ap.add_argument("out")
ap.add_argument("--window", type=int, default=4096)
ap.add_argument("--max", type=int, default=0, help="stop after this many windows, 0 = all")
a = ap.parse_args()

tok = AutoTokenizer.from_pretrained(a.tokenizer_dir)
text = open(a.corpus, encoding="utf-8").read()
ids = tok.encode(text, add_special_tokens=False)
print("corpus: %d chars, %d tokens" % (len(text), len(ids)))
n = 0
with open(a.out, "w", encoding="utf-8") as f:
    for i in range(0, len(ids) - a.window + 1, a.window):
        f.write(json.dumps({"text": tok.decode(ids[i:i + a.window])}, ensure_ascii=False) + "\n")
        n += 1
        if a.max and n >= a.max:
            break
print("wrote %d windows of %d tokens to %s" % (n, a.window, a.out))
print("that is %d tokens, about %d per routed expert at 6 of 384" % (n * a.window, n * a.window * 6 // 384))
JSONLEOF

cat > "$NVFP4_TOOLS/nvfp4_amax_flat.py" << 'FLATEOF'
"""An amax dump that means "no calibration": input_scale = 1.0 for every expert.

    python3 nvfp4_amax_flat.py config.json OUT_DIR

quantize_to_nvfp4.py turns an input amax into input_scale = amax / (6 * 448).
Writing amax = 2688 for every routed expert projection therefore yields exactly
1.0, which is what a checkpoint cast without a calibration pass carries. The
dump has the same key layout ptq.py produces, one rank, so the export cannot
tell the difference and the three checkpoints share one code path.

Weight amax is left out on purpose: with --cast_mxfp4_to_nvfp4 the weight
scales come from the source E8M0 scales, and the export synthesizes a weight
amax by dequantizing when it needs one.
"""
import json, os, sys
import torch

cfg = json.load(open(sys.argv[1]))
inner = cfg.get("text_config") or cfg
layers, experts = inner["num_hidden_layers"], inner["n_routed_experts"]
out = sys.argv[2]
os.makedirs(out, exist_ok=True)
amax = torch.tensor(6.0 * 448.0, dtype=torch.float32)
state = {}
for l in range(layers):
    for e in range(experts):
        for p in ("w1", "w2", "w3"):
            state["layers.%d.ffn.experts.%d.%s_input_quantizer._amax" % (l, e, p)] = amax.clone()
torch.save(state, os.path.join(out, "amax_dict_rank0-mp1.pt"))
print("wrote %d input amax entries (%d layers x %d experts x 3), all 2688 -> input_scale 1.0" % (len(state), layers, experts))
FLATEOF

cat > "$NVFP4_TOOLS/nvfp4_coverage.py" << 'COVEOF'
"""What a calibration actually saw.

    python3 nvfp4_coverage.py AMAX_DIR config.json OUT.json

Reads every amax_dict_rank*-mp*.pt in the directory, counts the routed experts
that received an input amax at all, and summarizes the amax values. An expert
with no amax got no token during calibration; the export then falls back to
the per projection maximum over the experts that did, which is a guess, and
the count of guesses is the one number guaranteed to separate a 64 x 512
calibration from a bigger one. The amax percentiles say where the E4M3 window
ended up: input_scale = amax / 2688, so amax 2688 is the flat cast.
"""
import glob, json, re, sys
import numpy as np
import torch

amax_dir, cfg_path, out = sys.argv[1:4]
cfg = json.load(open(cfg_path))
inner = cfg.get("text_config") or cfg
layers, experts = inner["num_hidden_layers"], inner["n_routed_experts"]
key_re = re.compile(r"^layers\.(\d+)\.ffn\.experts\.(\d+)\.(w[123])_(input|weight)_quantizer\._amax$")

seen = {}
for p in sorted(glob.glob(amax_dir + "/amax_dict_rank*-mp*.pt")):
    for k, v in torch.load(p, map_location="cpu", weights_only=True).items():
        m = key_re.match(k)
        if m and m.group(4) == "input":
            seen[(int(m.group(1)), int(m.group(2)), m.group(3))] = float(v.reshape(-1).max())

total = layers * experts
per_layer = []
for l in range(layers):
    n = sum(1 for e in range(experts) if (l, e, "w1") in seen)
    per_layer.append(n)
w13 = np.array([v for (l, e, p), v in seen.items() if p in ("w1", "w3")]) if seen else np.array([0.0])
w2 = np.array([v for (l, e, p), v in seen.items() if p == "w2"]) if seen else np.array([0.0])
res = {
    "amax_dir": amax_dir,
    "routed_experts_total": total,
    "experts_with_input_amax": sum(per_layer),
    "experts_without_input_amax": total - sum(per_layer),
    "per_layer_seen": per_layer,
    "median_input_amax_w13": float(np.median(w13)), "p99_input_amax_w13": float(np.percentile(w13, 99)), "max_input_amax_w13": float(w13.max()),
    "median_input_amax_w2": float(np.median(w2)), "p99_input_amax_w2": float(np.percentile(w2, 99)), "max_input_amax_w2": float(w2.max()),
}
json.dump(res, open(out, "w"), indent=2)
print("routed experts     : %d" % total)
print("with input amax    : %d   (%.1f %%)" % (res["experts_with_input_amax"], 100.0 * res["experts_with_input_amax"] / total))
print("without, guessed   : %d" % res["experts_without_input_amax"])
print("per layer seen     : min %d  max %d" % (min(per_layer), max(per_layer)))
print("w1/w3 input amax   : median %.2f  p99 %.2f  max %.2f   (2688 = flat cast, window ceiling)" % (res["median_input_amax_w13"], res["p99_input_amax_w13"], res["max_input_amax_w13"]))
print("w2 input amax      : median %.2f  p99 %.2f  max %.2f   (physical bound 100 from the SwiGLU clamp)" % (res["median_input_amax_w2"], res["p99_input_amax_w2"], res["max_input_amax_w2"]))
print("written to %s" % out)
COVEOF

cat > "$NVFP4_TOOLS/nvfp4_logprobs.py" << 'LPEOF'
"""Top-K log probabilities over the measurement corpora, from vLLM offline.

    python3 nvfp4_logprobs.py MODEL_DIR OUT_PREFIX --corpora a.txt b.txt --tp 8 --ctx 4096 --chunks 24 --topk 512

One model load, every corpus, one file per corpus: OUT_PREFIX-<corpus>.npz
where <corpus> is the file name without extension. Loading V4.1 is minutes,
scoring a corpus is seconds, so the corpora are never looped outside.

Protocol, matching the GGUF and MLX tables: each corpus is tokenized once with
the model's tokenizer, cut into windows of ctx tokens whose first token is BOS,
and only the second half of every window is scored. For every scored position
the file keeps the top-K token ids and their log probabilities, plus the id and
log probability of the token that was actually there. Nothing else: no chat
template, no speculative decoding, raw token ids in.

Why top-K and not the whole vocabulary: vLLM can return all 129,280 entries
(prompt_logprobs=-1) but materializes them as Python objects per position,
which for fifty thousand positions is billions of objects. K = 512 keeps the
file at a few hundred megabytes and, on a language model's distributions,
leaves less than a tenth of a percent of probability mass outside the list at
almost every position. nvfp4_kld.py reports the mass it did not see.
"""
import argparse, json, os, sys, time
import numpy as np

def main():
    # vLLM starts its workers with multiprocessing spawn, which re-imports __main__;
    # without this guard every worker would re-run the script from the top.
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("out_prefix")
    ap.add_argument("--corpora", nargs="+", required=True)
    ap.add_argument("--tp", type=int, default=8)
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--chunks", type=int, default=24)
    ap.add_argument("--topk", type=int, default=512)
    ap.add_argument("--batch", type=int, default=4, help="windows per generate() call")
    ap.add_argument("--gpu-mem", type=float, default=0.90)
    ap.add_argument("--kv-cache-dtype", default="auto")
    a = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(a.model)
    bos = tok.bos_token_id
    assert bos is not None, "tokenizer has no BOS id"
    body = a.ctx - 1
    first = a.ctx // 2  # first scored position inside a window
    K = a.topk

    # Tokenize everything before touching the GPUs, so a bad corpus path fails in
    # seconds and not after a twenty minute model load.
    jobs = []
    for path in a.corpora:
        name = os.path.splitext(os.path.basename(path))[0]
        ids = tok.encode(open(path, encoding="utf-8").read(), add_special_tokens=False)
        n_avail = len(ids) // body
        n = min(a.chunks, n_avail)
        assert n > 0, "%s: fewer than %d tokens" % (path, a.ctx)
        print("%s: %d tokens, %d windows of %d available, scoring %d" % (name, len(ids), n_avail, a.ctx, n), flush=True)
        jobs.append((name, path, [[bos] + ids[i * body:(i + 1) * body] for i in range(n)]))

    t_load = time.time()
    llm = LLM(model=a.model, tensor_parallel_size=a.tp, max_model_len=a.ctx,
              max_logprobs=K, gpu_memory_utilization=a.gpu_mem,
              kv_cache_dtype=a.kv_cache_dtype, enable_prefix_caching=False)
    print("model loaded in %.0f s" % (time.time() - t_load), flush=True)
    sp = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=K, detokenize=False)

    for name, path, windows in jobs:
        n = len(windows)
        n_scored = n * (a.ctx - first)
        top_ids = np.zeros((n_scored, K), dtype=np.int32)
        top_lps = np.full((n_scored, K), -np.inf, dtype=np.float32)
        tok_id = np.zeros(n_scored, dtype=np.int32)
        tok_lp = np.zeros(n_scored, dtype=np.float32)
        row = 0
        t0 = time.time()
        for b in range(0, n, a.batch):
            batch = windows[b:b + a.batch]
            outs = llm.generate([{"prompt_token_ids": w} for w in batch], sp, use_tqdm=False)
            for w, o in zip(batch, outs):
                pl = o.prompt_logprobs
                assert pl is not None and len(pl) == len(w), "prompt_logprobs missing or short"
                for pos in range(first, len(w)):
                    d = pl[pos]
                    actual = w[pos]
                    items = sorted(((lp.logprob, tid) for tid, lp in d.items()), reverse=True)
                    m = min(K, len(items))
                    top_lps[row, :m] = [x[0] for x in items[:m]]
                    top_ids[row, :m] = [x[1] for x in items[:m]]
                    tok_id[row] = actual
                    tok_lp[row] = d[actual].logprob
                    row += 1
            print("  %s: %d / %d windows, %.0f s" % (name, min(b + a.batch, n), n, time.time() - t0), flush=True)
        assert row == n_scored, (row, n_scored)
        out = "%s-%s.npz" % (a.out_prefix, name)
        meta = {"model": os.path.abspath(a.model), "corpus": os.path.abspath(path), "ctx": a.ctx,
                "chunks": n, "topk": K, "first_scored": first, "vocab": len(tok), "scored_positions": n_scored}
        np.savez_compressed(out, top_ids=top_ids, top_lps=top_lps, tok_id=tok_id, tok_lp=tok_lp, meta=json.dumps(meta))
        mass = np.exp(top_lps).sum(axis=1)
        print("written %s: %d positions, top-%d mass median %.5f min %.5f, ppl %.4f"
              % (out, n_scored, K, float(np.median(mass)), float(mass.min()), float(np.exp(-tok_lp.mean()))), flush=True)


if __name__ == "__main__":
    main()
LPEOF

cat > "$NVFP4_TOOLS/nvfp4_kld.py" << 'KLDEOF'
"""Divergence of a candidate from the reference, from two top-K dumps.

    python3 nvfp4_kld.py REF.npz CAND.npz OUT.json

Both files come from nvfp4_logprobs.py on the same corpus, ctx and chunk count.
Per scored position, with p the reference and q the candidate:

    KLD(p || q) = sum over the reference's top-K of p_i * (log p_i - log q_i)
                  + p_rest * (log p_rest - log q_rest)

p_i and p_rest are known exactly. q_i is known when the candidate ranked token
i in its own top-K. When it did not, q_i is somewhere below the candidate's
K-th probability, and the file gives two numbers instead of pretending:

    low   q_i = the candidate's K-th probability, the largest it can be
    high  q_i = the candidate's leftover mass spread evenly over the rest of
          the vocabulary, about the smallest plausible value

The truth lies between. On a normal run the two agree to the third decimal,
and the gap is reported so a reader can see when they do not. The exact
columns, which need no bracket, are top-1 agreement and the perplexity of the
tokens that were actually there.
"""
import json, sys
import numpy as np

ref_p, cand_p, out = sys.argv[1:4]
R, C = np.load(ref_p), np.load(cand_p)
mr, mc = json.loads(str(R["meta"])), json.loads(str(C["meta"]))
for k in ("corpus", "ctx", "chunks", "first_scored", "scored_positions"):
    assert mr[k] == mc[k], "protocol mismatch on %s: %r vs %r" % (k, mr[k], mc[k])
V, K = mr["vocab"], mr["topk"]
n = mr["scored_positions"]

rid, rlp = R["top_ids"], R["top_lps"].astype(np.float64)
cid, clp = C["top_ids"], C["top_lps"].astype(np.float64)
tok, rtok_lp, ctok_lp = R["tok_id"], R["tok_lp"].astype(np.float64), C["tok_lp"].astype(np.float64)
assert (tok == C["tok_id"]).all(), "the two dumps scored different tokens"

kl_low = np.zeros(n); kl_high = np.zeros(n); missing_mass = np.zeros(n)
top1 = 0
for i in range(n):
    p = np.exp(rlp[i]); p_rest = max(1.0 - p.sum(), 1e-12)
    q_of = dict(zip(cid[i].tolist(), clp[i].tolist()))
    q_k = clp[i][np.isfinite(clp[i])].min()          # candidate's K-th log prob
    q_rest = max(1.0 - np.exp(clp[i]).sum(), 1e-12)
    q_tail = np.log(q_rest / max(V - K, 1))
    lq_low = np.empty(K); lq_high = np.empty(K); miss = 0.0
    for j in range(K):
        t = int(rid[i, j])
        if t in q_of:
            lq_low[j] = lq_high[j] = q_of[t]
        else:
            lq_low[j] = q_k; lq_high[j] = q_tail; miss += p[j]
    term = p * (rlp[i] - lq_low); kl_low[i] = term.sum() + p_rest * (np.log(p_rest) - np.log(q_rest))
    term = p * (rlp[i] - lq_high); kl_high[i] = term.sum() + p_rest * (np.log(p_rest) - np.log(q_rest))
    missing_mass[i] = miss
    top1 += int(rid[i, 0] == cid[i, 0])

kl = np.sort(kl_low)
q = lambda f: float(kl[min(n - 1, int(n * f))])
res = {
    "reference": ref_p, "candidate": cand_p, "topk": K, "scored_positions": n,
    "mean_kld_low": float(kl_low.mean()), "mean_kld_high": float(kl_high.mean()),
    "median_kld": q(.50), "p90_kld": q(.90), "p95_kld": q(.95), "p99_kld": q(.99), "max_kld": float(kl[-1]),
    "top1_agree_pct": 100.0 * top1 / n,
    "ref_ppl": float(np.exp(-rtok_lp.mean())), "quant_ppl": float(np.exp(-ctok_lp.mean())),
    "mean_abs_dlogprob_actual": float(np.abs(rtok_lp - ctok_lp).mean()),
    "missing_mass_mean": float(missing_mass.mean()), "missing_mass_max": float(missing_mass.max()),
    "ref_topk_mass_mean": float(np.exp(rlp).sum(axis=1).mean()),
}
json.dump(res, open(out, "w"), indent=2)
print("mean KLD      : %.6f .. %.6f   (low .. high bracket, see docstring)" % (res["mean_kld_low"], res["mean_kld_high"]))
print("median KLD    : %.6f     p90 %.6f  p95 %.6f  p99 %.6f  max %.4f" % (res["median_kld"], res["p90_kld"], res["p95_kld"], res["p99_kld"], res["max_kld"]))
print("same top-1    : %.3f %%" % res["top1_agree_pct"])
print("ppl           : reference %.4f   candidate %.4f" % (res["ref_ppl"], res["quant_ppl"]))
print("|d logprob|   : %.5f on the actual tokens" % res["mean_abs_dlogprob_actual"])
print("unseen mass   : mean %.2e  max %.2e of reference top-%d not ranked by the candidate" % (res["missing_mass_mean"], res["missing_mass_max"], K))
print("written to %s" % out)
KLDEOF

echo "helpers written to $NVFP4_TOOLS"
}


# ================================================================== on source

if [ -n "$NVFP4_ROOT" ] && [ ! -d "$NVFP4_ROOT" ]; then
    mkdir -p "$NVFP4_ROOT" 2>/dev/null
fi
echo "foundry-nvfp4 $NVFP4_VERSION loaded. nvfp4_help for the list, nvfp4_box calib|stand for the plan."
