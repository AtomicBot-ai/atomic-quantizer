#!/bin/bash
# ================================================================= FOUNDRY IMAGE
#
# The image side, for Qwen-Image-2.1. Self contained: it does not source
# foundry.sh and it never builds llama.cpp. On a bare rented box:
#
#   export HF_TOKEN=hf_...
#   git clone https://github.com/AtomicBot-ai/atomic-quantizer /quantizer
#   source /quantizer/scripts/foundry-image.sh
#   img_setup
#   img_persist
#   img_box              the exact command list, paste one line at a time
#
# WHAT THIS FILE IS FOR
#
# Qwen-Image-2.1 renders an image from three models: the Qwen3-VL-8B text
# encoder, a 7.1B single-stream DiT denoiser, and a VAE. The GGUFs published
# from here are the denoiser only, the way everyone else ships them, and the
# denoiser is the only variable in every measurement: the encoder and the VAE
# stay at bf16 for every number.
#
# llama.cpp cannot touch this model. Its converter has no DiT architecture,
# llama-quantize throws on an architecture it does not know, and there is no
# diffusion graph to run the result. The tool is stable-diffusion.cpp, ggml
# underneath, pinned by sha below. One binary, sd-cli, does all three jobs:
#
#   sd-cli -M convert --type T --tensor-type-rules R --imat-in F     quantize
#   an ordinary render with --imat-out F                               importance matrix
#   an ordinary render                                                 measurement
#
# There is no next token, so there is no KL. The number is a perceptual
# distance, LPIPS with SSIM and PSNR beside it, between the image a build
# renders and the image the bf16 denoiser renders from the same prompt and the
# same starting noise. Two floors give it a scale: a second bf16 render against
# the first (what the GPU itself contributes), and seed 42 against seed 43
# (what a different picture costs).
#
# Two publishers were first. leejet's files are sd.cpp's defaults: every
# transformer block at one type, everything outside the blocks left in bf16.
# unsloth's move bits between tensors from a sensitivity scan. Both are
# measured here with the same protocol, the same reference and the same
# binary, and every AD rung is sized to the rival it is compared with.
#
# Read docs/runbook-image.md before renting anything.

IMG_VERSION=2026-09-24.01

IMG_UP=${IMG_UP:-Qwen/Qwen-Image-2.1}
IMG_ORG=${IMG_ORG:-AtomicChat}
IMG_STEM=${IMG_STEM:-Qwen-Image-2.1}
IMG_REPO=${IMG_REPO:-$IMG_ORG/$IMG_STEM-GGUF}
IMG_METRICS=${IMG_METRICS:-$IMG_ORG/$IMG_STEM-GGUF-metrics}
IMG_BTN_SRC=${IMG_BTN_SRC:-$IMG_ORG/Qwen3.8-27B-GGUF}   # the three btn_*.png every card carries

# The pin. Qwen-Image-2.1 landed in stable-diffusion.cpp on 2026-09-20
# (137f7409, #1994), the alpha-channel input fix two days later (e112ab5a),
# and master moves daily. A sha and not a branch, because the layouts this file
# publishes depend on which tensors the converter leaves alone, and that policy
# lives in src/model_loader.cpp and changes without notice. Release tag
# master-898-2bb7294.
IMG_SD_REPO=${IMG_SD_REPO:-https://github.com/leejet/stable-diffusion.cpp}
IMG_SD_SHA=${IMG_SD_SHA:-2bb72947cb129962f350452148658a32f4d3c057}

# The inputs, all public. Comfy-Org's single files are the standard sd.cpp
# input; the diffusers shards under Qwen/Qwen-Image-2.1 name the MLP
# differently (proj + gate_layer where this file has one fused gate_up) and
# nobody else measures from them. The VAE is new in 2.1: the Qwen-Image 1.x and
# Wan VAEs do not fit.
IMG_COMFY=${IMG_COMFY:-Comfy-Org/Qwen-Image-2.1}
IMG_DIT=${IMG_DIT:-diffusion_models/qwen_image_2.1_bf16.safetensors}     # 14.23 GB
IMG_VAE=${IMG_VAE:-vae/qwen_image_2.1_vae_bf16.safetensors}              # 0.68 GB
IMG_TE=${IMG_TE:-text_encoders/qwen3vl_8b_bf16.safetensors}              # 17.53 GB
IMG_DIT_SHA256=${IMG_DIT_SHA256:-89f4158d066cc33906a199fca85634f766892dd78f49b6698dabf187ac86c4bc}
IMG_VAE_SHA256=${IMG_VAE_SHA256:-bb21f7473051e1ac368515dd3f2e15cd44d7a11748ee8823e1ddca3e4876b7c9}
IMG_TE_SHA256=${IMG_TE_SHA256:-68bdc82bc1b66851162ae656225e7e2068166b603db19bd5d5a3b90eb12669a9}
IMG_TE_GGUF=${IMG_TE_GGUF:-Qwen/Qwen3-VL-8B-Instruct-GGUF}              # the card points users here

# Other people's files, measured with the same protocol. REPO:FILE. One rival
# per AD rung, at or near its size; the leejet files double as the check that
# the uniform rungs made here are the tool's defaults (img_same).
IMG_EXT=${IMG_EXT:-"
unsloth/Qwen-Image-2.1-GGUF:qwen-image-2.1-Q2_K.gguf
unsloth/Qwen-Image-2.1-GGUF:qwen-image-2.1-Q3_K_XL.gguf
unsloth/Qwen-Image-2.1-GGUF:qwen-image-2.1-Q4_K_M.gguf
unsloth/Qwen-Image-2.1-GGUF:qwen-image-2.1-Q5_K_M.gguf
unsloth/Qwen-Image-2.1-GGUF:qwen-image-2.1-Q6_K_XL.gguf
unsloth/Qwen-Image-2.1-GGUF:qwen-image-2.1-Q8_0.gguf
leejet/Qwen-Image-2.1-GGUF:qwen_image_2.1-Q2_K.gguf
leejet/Qwen-Image-2.1-GGUF:qwen_image_2.1-Q4_K.gguf
leejet/Qwen-Image-2.1-GGUF:qwen_image_2.1-Q6_K.gguf
"}

# Everything at the filesystem root, same as the other routes. On macOS
# nothing can be created at /, so IMG_ROOT redirects the layout for a dry run.
IMG_ROOT=${IMG_ROOT:-}
IMG_SRC=${IMG_SRC:-$IMG_ROOT/src}              # the three inputs, hub layout kept
IMG_GGUF=${IMG_GGUF:-$IMG_ROOT/gguf}           # ours; external/<org>/ for theirs, scan/ for the scan
IMG_IMAT=${IMG_IMAT:-$IMG_ROOT/imatrix}        # imat.dat plus one checkpoint per calibration prompt
IMG_GEN=${IMG_GEN:-$IMG_ROOT/gen}              # gen/<build>/<pid>-s<seed>.png
IMG_EVAL=${IMG_EVAL:-$IMG_ROOT/eval}           # the two prompt files
IMG_LOGS=${IMG_LOGS:-$IMG_ROOT/logs}
IMG_LAYOUTS=${IMG_LAYOUTS:-$IMG_ROOT/layouts}  # per-tensor type map of every GGUF
IMG_TOOLS=${IMG_TOOLS:-$IMG_ROOT/tools}        # the sd.cpp clone and the python helpers
IMG_HF=${IMG_HF:-$IMG_ROOT/hf}
IMG_SD=${IMG_SD:-$IMG_TOOLS/stable-diffusion.cpp}
IMG_BIN=${IMG_BIN:-$IMG_SD/build/bin/sd-cli}
IMG_DIT_FILE=${IMG_DIT_FILE:-$IMG_SRC/$IMG_DIT}
IMG_VAE_FILE=${IMG_VAE_FILE:-$IMG_SRC/$IMG_VAE}
IMG_TE_FILE=${IMG_TE_FILE:-$IMG_SRC/$IMG_TE}
IMG_DIT_HEADER=${IMG_DIT_HEADER:-$IMG_SRC/dit-header.safetensors}   # 31 KB, enough for img_bits

_img_cores() {
    local n
    n=$(lscpu -p=core,socket 2>/dev/null | grep -v '^#' | sort -u | wc -l | tr -d ' ')
    if [ -n "$n" ] && [ "$n" -gt 0 ] 2>/dev/null; then echo "$n"; return; fi
    n=$(sysctl -n hw.physicalcpu 2>/dev/null)
    if [ -n "$n" ]; then echo "$n"; return; fi
    nproc 2>/dev/null || echo 8
}

# The measurement protocol. Unsloth's recommended sd.cpp settings, so a reader
# who follows their guide gets our images: 20 steps, cfg 6.0, euler, 1024x1024.
# --rng cpu is ComfyUI's noise, so a seed means the same starting latent in
# every build and on every GPU. Two seeds per prompt, 48 prompts.
IMG_W=${IMG_W:-1024}
IMG_H=${IMG_H:-1024}
IMG_STEPS=${IMG_STEPS:-20}
IMG_CFG=${IMG_CFG:-6.0}
IMG_SAMPLER=${IMG_SAMPLER:-euler}
IMG_RNG=${IMG_RNG:-cpu}
IMG_SEEDS=${IMG_SEEDS:-"42 43"}
IMG_EVAL_N=${IMG_EVAL_N:-48}          # prompts of eval_prompts.txt to render; 1 for the smoke test
IMG_OFFLOAD=${IMG_OFFLOAD:-0}         # 1 on a 48 GB card: --offload-to-cpu; timings then compare only with each other
IMG_THREADS=${IMG_THREADS:-$(_img_cores)}
IMG_CUDA_ARCH=${IMG_CUDA_ARCH:-}      # empty: read off the card. 90 H100/H200, 100 B200, 120 RTX PRO 6000
# Calibration: its own prompts, disjoint from the eval set, its own seed, the
# full resolution so the imatrix sees the same 4096 image tokens the eval does.
IMG_CALIB_N=${IMG_CALIB_N:-32}
IMG_CALIB_SEED=${IMG_CALIB_SEED:-7}
# The sensitivity scan: cheap on purpose. A ranking, never a published number.
IMG_SCAN_N=${IMG_SCAN_N:-8}
IMG_SCAN_W=${IMG_SCAN_W:-512}
IMG_SCAN_H=${IMG_SCAN_H:-512}
IMG_SCAN_STEPS=${IMG_SCAN_STEPS:-10}
IMG_SCAN_SEED=${IMG_SCAN_SEED:-42}
IMG_BOOT=${IMG_BOOT:-20000}           # bootstrap resamples, rng seed 0

# The AD layouts. Each rung is sized to the rival in the last column: the
# rival's tensor bytes are the budget, and the budget goes, in this order, to
# the modulation (shared by every block), then attention, then the MLP output
# projection of the outer blocks. That order is a prior. img_scan measures
# where the damage actually is, and if it disagrees, the bands change before
# the ladder runs, not after. img_ladder --dry prints every size first.
#
#   narrow  blocks 0-3 and 28-31        wide  blocks 0-7 and 24-31
#   all     every block                 -     none
IMG_HI=${IMG_HI:-bf16}                # "keep": the source type, what sd.cpp leaves alone by default
IMG_ATTN_PROJ=${IMG_ATTN_PROJ:-'to_'}  # which attention projections the band moves: to_ is all four; (to_v|to_out) two
IMG_BAND_NARROW=${IMG_BAND_NARROW:-'[0-3]|2[89]|3[01]'}
IMG_BAND_WIDE=${IMG_BAND_WIDE:-'[0-7]|2[4-9]|3[01]'}
IMG_AD_TABLE=${IMG_AD_TABLE:-"
# label    base  modulation  attention  mlp.out  rival GB  rival
AD-Q6_K    q6_K  bf16        all        wide     6.7185    unsloth Q6_K_XL
AD-Q5_K    q5_K  q8_0        all        wide     5.3902    unsloth Q5_K_M
AD-Q4_K    q4_K  q8_0        narrow     -        4.1995    unsloth Q4_K_M, leejet Q4_K 4.1975
AD-Q3_K    q3_K  q8_0        all        wide     3.6125    unsloth Q3_K_XL
AD-Q2_K    q2_K  q8_0        narrow     -        2.5617    leejet Q2_K, unsloth Q2_K 2.4661
"}
IMG_GRID_UNIFORM=${IMG_GRID_UNIFORM:-"Q8_0 Q6_K Q4_K Q3_K Q2_K"}      # the tool's defaults, rendered as baselines
IMG_GRID_AD=${IMG_GRID_AD:-"AD-Q6_K AD-Q5_K AD-Q4_K AD-Q3_K AD-Q2_K"}
IMG_PUBLISH=${IMG_PUBLISH:-"BF16 Q8_0 AD-Q6_K AD-Q5_K AD-Q4_K AD-Q3_K AD-Q2_K"}   # what goes into $IMG_REPO

# Every log, layout and measurement goes to $IMG_METRICS the moment it exists,
# so a box that dies keeps what it measured. IMG_UPLOAD=0 keeps everything
# local, for a dry run or an experiment that should not be published.
IMG_UPLOAD=${IMG_UPLOAD:-1}

IMG_SECRET_RE='hf_[A-Za-z0-9]{30,}|sk-[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{30,}|AKIA[0-9A-Z]{16}'

export HF_HOME=${HF_HOME:-$IMG_HF}
export TOKENIZERS_PARALLELISM=false


# ================================================================== orientation

img_help() {
    cat << EOF
foundry-image $IMG_VERSION      upstream $IMG_UP      sd.cpp ${IMG_SD_SHA:0:8}

  box
    img_setup                dirs, hf, torch, lpips, scikit-image, matplotlib, VGG weights, the helpers   5-10 min
    img_build                stable-diffusion.cpp at the pin -> sd-cli (CUDA, Metal on a Mac)            5-15 min
    img_persist              source this file from every new tmux pane, token to \$HF_HOME
    img_env                  sd sha, GPU, driver, versions, protocol -> logs/env.txt
    img_check                disk, GPU, inputs, files, renders per build, what is measured
    img_selfcheck            every function this help names exists
    img_box                  the command list for the one box, with durations

  get
    img_get header           the bf16 file's safetensors header, 31 KB, enough for img_bits
    img_get src              denoiser, VAE, text encoder from $IMG_COMFY, sha256 checked   32 GB
    img_get ext              every file in IMG_EXT -> gguf/external/<org>/                  43 GB
    img_get ref              the reference renders from $IMG_METRICS, for a second box
    img_prompts              eval_prompts.txt (48) and calib_prompts.txt (32), sha256 logged

  calibrate
    img_imatrix              one render per calibration prompt with --imat-out, chained   30-45 min
    img_imatrix_check        which tensors the matrix covers, and how often

  quantize
    img_bits LABEL|TYPE [RULES]   predicted size of a layout from the header, nothing built
    img_rules LABEL          the --type and --tensor-type-rules of a label
    img_convert LABEL [TYPE [RULES]]   sd-cli -M convert -> gguf/$IMG_STEM-LABEL.gguf   2-6 min
    img_layout BUILD         per-tensor types -> layouts/layout-BUILD.json, checked against the request
    img_names [BUILD [OTHER]]     how two files name their tensors
    img_same A B             whether two GGUFs carry the same weights, tensor by tensor
    img_ladder [--dry]       every rung: convert, render, measure, push. --dry: sizes only

  measure
    img_gen BUILD [MODEL]    the eval prompts x IMG_SEEDS -> gen/BUILD/, timings -> logs/gen-BUILD.jsonl   60-90 min
    img_ref                  img_gen ref: the bf16 denoiser, the reference                  60-90 min
    img_repeat               a second bf16 render, measured: floor (a)                      60-90 min
    img_metrics BUILD [REF]  LPIPS, SSIM, PSNR against the reference -> logs/metrics-BUILD.json
    img_floors               floor (a) and floor (b), seed 42 against seed 43, side by side
    img_compare A B          paired per-prompt difference with an interval
    img_scan                 q2_K on one tensor group at a time, the rest q8_0, 512x512   about 1.5 h
    img_ablate               the imatrix and the layout, one at a time, at Q4_K            3-4 h
    img_ext                  render and measure every file under gguf/external             about 70 min each
    img_table                everything measured -> logs/results.json, logs/results.md
    img_grid [BUILDS]        four prompts side by side -> logs/samples-grid.png
    img_chart                LPIPS against size, ours and theirs -> logs/quality-vs-size.png

  publish
    img_repos                create $IMG_REPO and $IMG_METRICS if missing (public)
    img_upload FILE REMOTE   one file to the metrics dataset
    img_push BUILD           the GGUF if BUILD is in IMG_PUBLISH; its logs, layout and renders always
    img_push_images BUILD    gen/BUILD -> images/BUILD in the metrics dataset
    img_push_logs            logs, layouts, imatrix, prompts, results, a dataset README
    img_push_card FILE       buttons, samples grid, chart, then FILE as the model card
    img_audit                published but unmeasured, measured but unpublished
EOF
}

img_persist() {
    local me
    me="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
    grep -q "foundry-image.sh" ~/.bashrc 2>/dev/null || echo "source $me" >> ~/.bashrc
    if [ -n "${HF_TOKEN:-}" ]; then
        grep -q "HF_TOKEN=" ~/.bashrc 2>/dev/null || echo "export HF_TOKEN=$HF_TOKEN" >> ~/.bashrc
        echo "HF_TOKEN written to ~/.bashrc. The box is disposable, the token is not: revoke it after."
        mkdir -p "$HF_HOME" && printf '%s' "$HF_TOKEN" > "$HF_HOME/token"
        echo "token also written to $HF_HOME/token"
    fi
    echo "new panes will source $me"
}

img_box() {
    cat << 'EOF'
# one box: one GPU of 80-96 GB (H100 80, A100 80, RTX PRO 6000), RAM >= 64 GB, 250 GB disk,
# the CUDA 13 devel image. A 48 GB card works with IMG_OFFLOAD=1, slower.
img_setup                              # 5-10 min
img_build                              # 5-15 min, sd-cli at the pin
img_persist ; img_repos ; img_env ; img_check
img_get src                            # 32 GB, then 2 min of sha256
img_prompts
# smoke test, 15 min, before anything long
img_convert Q4_K                       # the layout it prints is what the pin keeps in bf16
IMG_EVAL_N=1 IMG_SEEDS=42 img_gen ref
IMG_EVAL_N=1 IMG_SEEDS=42 img_gen Q4_K
IMG_EVAL_N=1 IMG_SEEDS=42 img_metrics Q4_K      # one LPIPS number: torch, lpips and the pairing work
IMG_EVAL_N=1 IMG_SEEDS=42 img_gen ref-repeat ; IMG_EVAL_N=1 IMG_SEEDS=42 img_metrics ref-repeat   # 0.000 means deterministic
# the reference and the floors
img_ref                                # 60-90 min, 96 renders, resumes from the one above
img_repeat                             # 60-90 min, floor (a)
img_floors                             # floor (b), seconds
# the evidence for the layouts
img_imatrix                            # 30-45 min
img_imatrix_check
img_scan                               # about an hour; read the ranking before the next line
img_ladder --dry                       # every rung's size against its rival
# the ladder: convert, render, measure, push, one rung at a time
img_ladder                             # 10 rungs, about 70 min each
img_ablate                             # the two levers apart, 3-4 h
img_convert BF16 ; img_push BF16       # the lossless base, published, not rendered
# the other publishers, same protocol
img_get ext                            # 43 GB
img_same Q4_K leejet--qwen_image_2.1-Q4_K    # identical weights: drop leejet from IMG_EXT, the uniform rows are theirs
img_ext                                # about 70 min per file
img_table ; img_grid ; img_chart
img_push_logs
EOF
}

_img_nseeds() { set -- $IMG_SEEDS; echo $#; }
_img_gb() { [ -f "$1" ] && python3 -c 'import os,sys; print("%.3f" % (os.path.getsize(sys.argv[1]) / 1e9))' "$1"; }
_img_sd_sha() { git -C "$IMG_SD" rev-parse HEAD 2>/dev/null || echo none; }
_img_now() {
    if [ -n "${EPOCHREALTIME:-}" ]; then echo "$EPOCHREALTIME"
    else python3 -c 'import time; print("%.3f" % time.time())'; fi
}
_img_sha256() {
    if command -v sha256sum > /dev/null; then sha256sum "$1" | cut -d' ' -f1
    else shasum -a 256 "$1" | cut -d' ' -f1; fi
}
# Only a token this box was given: HF_TOKEN, or the copy img_persist wrote.
# A token cached in the home directory of a workstation does not count, so a
# dry run there cannot publish anything by accident.
_img_token() {
    [ -n "${HF_TOKEN:-}" ] || [ -f "$HF_HOME/token" ]
}
_img_protocol() {
    echo "${IMG_W}x${IMG_H} steps=$IMG_STEPS cfg=$IMG_CFG $IMG_SAMPLER rng=$IMG_RNG seeds=[$IMG_SEEDS] n=$IMG_EVAL_N offload=$IMG_OFFLOAD te=bf16 vae=bf16 sd=$(_img_sd_sha | cut -c1-8)"
}

img_check() {
    local d n want b
    echo "--- disk ---"; df -h "${IMG_ROOT:-/}" 2>/dev/null | tail -1
    echo "--- ram ---"
    if command -v free > /dev/null; then free -g | head -2
    else sysctl -n hw.memsize 2> /dev/null | awk '{ printf "%.0f GB\n", $1 / 1e9 }'; fi
    echo "--- gpus ---"; nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader 2>/dev/null || echo "no nvidia-smi"
    echo "--- on disk ---"
    if [ -x "$IMG_BIN" ]; then
        [ "$(_img_sd_sha)" = "$IMG_SD_SHA" ] && echo "  [x] sd-cli     at the pin" || echo "  [!] sd-cli     at $(_img_sd_sha | cut -c1-8), the pin is ${IMG_SD_SHA:0:8}  -> img_build"
    else echo "  [ ] sd-cli     -> img_build"; fi
    [ -f "$IMG_DIT_FILE" ] && echo "  [x] denoiser   $(_img_gb "$IMG_DIT_FILE") GB" || echo "  [ ] denoiser   -> img_get src"
    [ -f "$IMG_VAE_FILE" ] && echo "  [x] vae" || echo "  [ ] vae        -> img_get src"
    [ -f "$IMG_TE_FILE" ]  && echo "  [x] encoder" || echo "  [ ] encoder    -> img_get src"
    [ -f "$IMG_DIT_HEADER" ] || [ -f "$IMG_DIT_FILE" ] && echo "  [x] header     for img_bits" || echo "  [ ] header     -> img_get header"
    [ -f "$IMG_EVAL/eval_prompts.txt" ] && echo "  [x] prompts" || echo "  [ ] prompts    -> img_prompts"
    n=$(ls "$IMG_IMAT"/imat-c*.dat 2>/dev/null | wc -l | tr -d ' ')
    [ "$n" -ge "$IMG_CALIB_N" ] && echo "  [x] imatrix    $n of $IMG_CALIB_N prompts" || echo "  [ ] imatrix    $n of $IMG_CALIB_N prompts -> img_imatrix"
    for d in "$IMG_GGUF"/*.gguf "$IMG_GGUF"/external/*/*.gguf; do
        [ -f "$d" ] && echo "  [x] gguf       $(_img_gb "$d") GB  ${d#$IMG_GGUF/}"
    done
    want=$((IMG_EVAL_N * $(_img_nseeds)))
    for d in "$IMG_GEN"/*/; do
        [ -d "$d" ] || continue
        b=$(basename "$d"); n=$(ls "$d"*.png 2>/dev/null | wc -l | tr -d ' ')
        case "$b" in scan-*|calib) echo "  [.] renders    $b: $n" ;;
        *) [ "$n" -ge "$want" ] && echo "  [x] renders    $b: $n/$want" || echo "  [ ] renders    $b: $n/$want -> img_gen $b" ;;
        esac
    done
    for d in "$IMG_LOGS"/metrics-*.json; do [ -f "$d" ] && echo "  [x] measured   $(basename "$d" .json | cut -c9-)"; done
}

img_selfcheck() {
    local f missing=""
    for f in $(img_help | grep -oE '^ +img_[a-z_]+' | tr -d ' ' | sort -u) img_write_py; do
        declare -F "$f" > /dev/null || missing="$missing $f"
    done
    if [ -n "$missing" ]; then echo "MISSING:$missing"; return 1; fi
    echo "every function the help names exists"
}

img_env() {
    mkdir -p "$IMG_LOGS"
    {
        echo "date        $(date -u +%FT%TZ)"
        echo "foundry     foundry-image $IMG_VERSION"
        echo "sd.cpp      $(_img_sd_sha)  (pin $IMG_SD_SHA)"
        echo "os          $(uname -srm)"
        nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader 2>/dev/null | sed 's/^/gpu         /'
        nvcc --version 2>/dev/null | grep release | sed 's/^/nvcc        /'
        python3 - << 'ENVEOF' 2>/dev/null
import importlib
for m in ("torch", "torchvision", "lpips", "skimage", "PIL", "numpy", "huggingface_hub"):
    try:
        v = importlib.import_module(m).__version__
    except Exception:
        v = "missing"
    print("%-11s %s" % (m, v))
try:
    import torch
    print("cuda        %s, %d device(s)" % (torch.version.cuda, torch.cuda.device_count()))
except Exception:
    pass
ENVEOF
        echo "threads     $IMG_THREADS"
        echo "protocol    $(_img_protocol)"
        set | grep -E '^IMG_(W|H|STEPS|CFG|SAMPLER|RNG|SEEDS|EVAL_N|OFFLOAD|CALIB_N|CALIB_SEED|SCAN_[A-Z]+|BOOT|HI|BAND_[A-Z]+|GRID_[A-Z]+|SD_SHA)='
    } > "$IMG_LOGS/env.txt"
    cat "$IMG_LOGS/env.txt"
    img_upload "$IMG_LOGS/env.txt" "logs/env.txt"
}


# ================================================================== setup

_img_pip() { python3 -m pip install --break-system-packages -q -U "$@"; }

img_setup() {
    mkdir -p "$IMG_SRC" "$IMG_GGUF/external" "$IMG_GGUF/scan" "$IMG_IMAT" "$IMG_GEN" "$IMG_EVAL" \
             "$IMG_LOGS" "$IMG_LAYOUTS" "$IMG_TOOLS/tmp" "$HF_HOME" || return 1
    _img_pip "huggingface_hub[cli,hf_xet]" numpy pillow scikit-image matplotlib scipy tqdm || return 1
    if ! python3 -c "import torch, torchvision" 2> /dev/null; then
        if [ "$(uname)" = Linux ] && command -v nvidia-smi > /dev/null; then
            _img_pip torch torchvision --index-url https://download.pytorch.org/whl/cu128 || return 1
        else
            _img_pip torch torchvision || return 1
        fi
    fi
    # lpips pins nothing it needs; --no-deps keeps pip from swapping the torch above
    _img_pip --no-deps lpips || return 1
    echo "fetching the VGG weights LPIPS uses, so the first measurement does not stall on the network"
    python3 -c "import lpips; lpips.LPIPS(net='vgg', verbose=False); print('lpips ready')" || return 1
    img_write_py
    echo
    echo "next: img_build"
}

img_build() {
    if [ -x "$IMG_BIN" ] && [ "$(_img_sd_sha)" = "$IMG_SD_SHA" ] && [ "${IMG_FORCE:-0}" != 1 ]; then
        echo "already here: $IMG_BIN at ${IMG_SD_SHA:0:8}"; return 0
    fi
    if ! command -v cmake > /dev/null && command -v apt-get > /dev/null; then
        apt-get update -qq && apt-get install -y -qq cmake git build-essential > /dev/null
    fi
    command -v cmake > /dev/null && command -v git > /dev/null || { echo "need cmake and git"; return 1; }
    mkdir -p "$IMG_TOOLS" "$IMG_LOGS"
    [ -d "$IMG_SD/.git" ] || git clone -q "$IMG_SD_REPO" "$IMG_SD" || return 1
    git -C "$IMG_SD" fetch -q origin "$IMG_SD_SHA" 2> /dev/null || true
    git -C "$IMG_SD" checkout -q "$IMG_SD_SHA" || { echo "could not check out $IMG_SD_SHA"; return 1; }
    # ggml is a submodule, and sd.cpp's is patched (int8 convrot); the sha pins it too
    git -C "$IMG_SD" submodule update --init --recursive -q || return 1
    local backend="" arch="$IMG_CUDA_ARCH"
    if [ "$(uname)" = Darwin ]; then
        backend="-DSD_METAL=ON"
    elif command -v nvcc > /dev/null || [ -x /usr/local/cuda/bin/nvcc ]; then
        export PATH="/usr/local/cuda/bin:$PATH"
        [ -z "$arch" ] && arch=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2> /dev/null | head -1 | tr -d '. ')
        backend="-DSD_CUDA=ON${arch:+ -DCMAKE_CUDA_ARCHITECTURES=$arch}"
    else
        echo "WARNING: no nvcc and not a Mac: a CPU build. Fine for convert, far too slow to render."
    fi
    echo "building sd-cli at ${IMG_SD_SHA:0:8} with ${backend:-CPU only}"
    date
    cmake -S "$IMG_SD" -B "$IMG_SD/build" -DCMAKE_BUILD_TYPE=Release $backend 2>&1 | tee "$IMG_LOGS/build.log" | tail -3
    [ "${PIPESTATUS[0]}" = 0 ] || { echo "cmake failed, see $IMG_LOGS/build.log"; return 1; }
    cmake --build "$IMG_SD/build" --config Release -j "$IMG_THREADS" --target sd-cli 2>&1 | tee -a "$IMG_LOGS/build.log" | grep -E 'error|warning: unused|Linking|Built target' | tail -5
    [ "${PIPESTATUS[0]}" = 0 ] && [ -x "$IMG_BIN" ] || { echo "build failed, see $IMG_LOGS/build.log"; return 1; }
    date
    "$IMG_BIN" --help 2>&1 | head -2
    echo "sd-cli at $IMG_BIN, $(_img_sd_sha | cut -c1-8)"
}


# ================================================================== get

# Hashing 32 GB takes a couple of minutes, so a verified file is remembered in
# logs/inputs-sha256.txt and not hashed again.
_img_verify() {
    local f="$1" want="$2" got rec="$IMG_LOGS/inputs-sha256.txt"
    [ -f "$f" ] || { echo "missing: $f"; return 1; }
    if grep -q "^$want  ${f#$IMG_SRC/}\$" "$rec" 2> /dev/null; then echo "  sha256 ok (recorded)  ${f#$IMG_SRC/}"; return 0; fi
    echo "  hashing ${f#$IMG_SRC/}, $(_img_gb "$f") GB"
    got=$(_img_sha256 "$f")
    if [ "$got" != "$want" ]; then
        echo "!! sha256 mismatch on $f"
        echo "   expected $want"
        echo "   got      $got"
        echo "   the hub file changed or the download is damaged: delete it and img_get src again"
        return 1
    fi
    echo "$got  ${f#$IMG_SRC/}" >> "$rec"
    echo "  sha256 ok  ${f#$IMG_SRC/}"
}

img_get() {
    mkdir -p "$IMG_SRC" "$IMG_LOGS" "$IMG_TOOLS/tmp"
    case "${1:-}" in
    header)
        local url="https://huggingface.co/$IMG_COMFY/resolve/main/$IMG_DIT" n
        curl -sfL -r 0-7 "$url" -o "$IMG_TOOLS/tmp/st8" || { echo "could not reach $url"; return 1; }
        n=$(python3 -c 'import struct,sys; print(struct.unpack("<Q", open(sys.argv[1], "rb").read(8))[0])' "$IMG_TOOLS/tmp/st8")
        curl -sfL -r "0-$((n + 7))" "$url" -o "$IMG_DIT_HEADER" || return 1
        echo "header of $IMG_DIT: $n bytes of JSON -> $IMG_DIT_HEADER"
        ;;
    src)
        local inc=() what="${IMG_GET_ONLY:-all}"
        case "$what" in
        all) inc=(--include "$IMG_DIT" --include "$IMG_VAE" --include "$IMG_TE") ;;
        dit) inc=(--include "$IMG_DIT") ;;
        *) echo "IMG_GET_ONLY=all|dit"; return 1 ;;
        esac
        hf download "$IMG_COMFY" "${inc[@]}" --local-dir "$IMG_SRC" || return 1
        _img_verify "$IMG_DIT_FILE" "$IMG_DIT_SHA256" || return 1
        [ "$what" = dit ] && return 0
        _img_verify "$IMG_VAE_FILE" "$IMG_VAE_SHA256" || return 1
        _img_verify "$IMG_TE_FILE" "$IMG_TE_SHA256" || return 1
        img_upload "$IMG_LOGS/inputs-sha256.txt" "logs/inputs-sha256.txt"
        ;;
    ext)
        local e repo file org
        for e in $IMG_EXT; do
            repo=${e%%:*}; file=${e#*:}; org=${repo%%/*}
            if [ -f "$IMG_GGUF/external/$org/$file" ]; then echo "already here: $org/$file"; continue; fi
            hf download "$repo" --include "$file" --local-dir "$IMG_GGUF/external/$org" || return 1
        done
        ls -la "$IMG_GGUF"/external/*/*.gguf
        echo "measured as <org>--<file stem>, so the publisher is in every log name"
        ;;
    ref)
        hf download "$IMG_METRICS" --repo-type dataset --include "images/ref/*" --local-dir "$IMG_TOOLS/pull" || return 1
        mkdir -p "$IMG_GEN/ref"
        cp -n "$IMG_TOOLS/pull/images/ref/"*.png "$IMG_GEN/ref/" 2> /dev/null
        echo "$(ls "$IMG_GEN/ref"/*.png | wc -l | tr -d ' ') reference renders in $IMG_GEN/ref"
        echo "they were made on another GPU: render one here with IMG_FORCE=1 IMG_EVAL_N=1 IMG_SEEDS=42 img_gen ref-check"
        echo "and measure it against them before trusting anything measured on this box"
        ;;
    *) echo "img_get header|src|ext|ref" ;;
    esac
}

# The prompts live in this file, versioned with it, and their sha256 goes to
# the metrics dataset beside every number. Format: id|category|prompt. The
# categories are interleaved, so the first N of either file is always a
# balanced subset: that is what IMG_EVAL_N=8 or the scan renders.
# Calibration and evaluation share no prompt.
img_prompts() {
    mkdir -p "$IMG_EVAL" "$IMG_LOGS"
    cat > "$IMG_EVAL/eval_prompts.txt" << 'EVALEOF'
e01|photo|A weathered fisherman mending a net on a wooden pier at dawn, soft golden light, shallow depth of field, 85mm photograph
e02|illus|A cozy treehouse library at dusk in a flat vector illustration style, warm lanterns, muted teal and orange palette
e03|text-en|A vintage bakery storefront with a hand-painted sign that reads "FRESH BREAD DAILY", morning street scene
e04|text-zh|一家老式茶馆的木质招牌，上面用金色书法写着"清风茶社"，傍晚的灯笼光
e05|multi|Three fruits on a marble table: a red apple on the left, a halved lemon in the middle and a bunch of purple grapes on the right, studio lighting
e06|alpha|This is an RGBA image with transparency. A glossy red sports car seen from the side. The image has alpha channel and the background is transparent.
e07|photo|Macro photograph of dew drops on a spider web in a misty forest, backlit, bokeh
e08|illus|A watercolor painting of a Venetian canal with gondolas, loose brushstrokes, soft pastel colors
e09|text-en|A movie poster for a science fiction film titled "ORBITAL DRIFT" with the tagline "No signal. No way home.", astronaut silhouette against a planet
e10|text-zh|地铁站的指示牌，白底蓝字写着"出口 A 人民广场"，现代简洁的设计
e11|multi|A golden retriever wearing a blue scarf sitting next to a black cat wearing a yellow raincoat on a wooden porch
e12|alpha|This is an RGBA image with transparency. A ceramic coffee mug with steam rising, isometric 3D render. The image has alpha channel and the background is transparent.
e13|photo|Portrait of an elderly woman with silver hair laughing, window light, natural skin texture, 50mm photograph
e14|illus|A pixel art scene of a small village in winter with smoke rising from chimneys, 16-bit style
e15|text-en|A chalkboard menu in a cafe listing "ESPRESSO 3.50", "LATTE 4.20" and "MATCHA 4.80" in neat handwriting
e16|text-zh|一本书的封面，标题是"星河之旅"，下方小字"第一卷"，深蓝色背景配银色星空
e17|multi|A red bicycle leaning against a yellow wall under a green window with white shutters, bright midday sun
e18|alpha|This is an RGBA image with transparency. A cute cartoon owl mascot waving, flat vector style. The image has alpha channel and the background is transparent.
e19|photo|Aerial photograph of terraced rice fields after rain, reflections of clouds in the water, high detail
e20|illus|A Japanese ukiyo-e style woodblock print of a giant wave crashing over a lighthouse
e21|text-en|A neon sign on a brick wall that says "OPEN LATE" in pink cursive letters, night, wet reflections
e22|text-zh|春节贴在红色木门上的对联，左联"春回大地千山秀"，右联"日照神州万里明"，横批"万象更新"
e23|multi|Four toy blocks stacked in a tower, red, blue, green and yellow from bottom to top, showing the letters A, B, C and D
e24|alpha|This is an RGBA image with transparency. A potted monstera plant in a white ceramic pot. The image has alpha channel and the background is transparent.
e25|photo|A bowl of ramen with a soft-boiled egg, chashu pork and scallions, steam rising, overhead food photograph on a dark wooden table
e26|illus|A children's book illustration of a fox and a rabbit sharing an umbrella in the rain, gouache texture
e27|text-en|A label on a glass jar of honey that reads "WILDFLOWER HONEY" and "NET WT 12 OZ", rustic kitchen background
e28|text-zh|夜市摊位上的手写价格牌，写着"烤红薯 每斤十二元"，暖黄色灯光
e29|multi|A chess board mid-game with a white queen, a black knight and a white pawn in the foreground, the rest of the board out of focus
e30|alpha|This is an RGBA image with transparency. A stack of three macarons, pink, green and yellow. The image has alpha channel and the background is transparent.
e31|photo|A snowy mountain cabin at night under the aurora borealis, long exposure, stars visible
e32|illus|An isometric illustration of a tiny office with a desk, plants, a bookshelf and a cat asleep on a chair, soft pastel colors
e33|text-en|A retro video game title screen with the words "PRESS START" under a logo reading "STAR RUNNER"
e34|text-zh|一张毕业典礼的横幅，红底白字写着"热烈祝贺二零二六届毕业生"
e35|multi|A kitchen counter with a blue kettle, a white teapot, two green cups and a plate of cookies, arranged from left to right
e36|alpha|This is an RGBA image with transparency. A pair of white sneakers with orange laces, product photo. The image has alpha channel and the background is transparent.
e37|photo|Street photograph of a busy crosswalk in Tokyo at night, motion blur, neon reflections on wet asphalt
e38|illus|An oil painting of a lighthouse keeper reading by candlelight, dramatic chiaroscuro, visible brush texture
e39|text-en|A road sign in a desert that reads "LAST GAS 120 MILES", heat haze, blue sky
e40|text-zh|餐厅菜单的一页，标题"今日推荐"，列出"宫保鸡丁"、"麻婆豆腐"、"清蒸鲈鱼"
e41|multi|Two astronauts in white suits planting a small tree on the surface of the moon, the Earth rising behind them
e42|alpha|This is an RGBA image with transparency. A vintage brass pocket watch with its lid open. The image has alpha channel and the background is transparent.
e43|photo|A hummingbird feeding from a red flower, frozen wings, high shutter speed, green background
e44|illus|A cyberpunk city skyline in a synthwave illustration style, magenta and cyan, a large sun setting behind skyscrapers
e45|text-en|A handwritten birthday card that says "Happy 30th, Maya!" with balloons drawn around the text
e46|text-zh|快递包裹上的标签，写着"易碎物品 小心轻放"，纸箱纹理清晰
e47|multi|A bookshelf with exactly five books: red, orange, green, blue and purple, lined up in that order
e48|alpha|This is an RGBA image with transparency. A glass bottle of olive oil with a cork stopper and a sprig of rosemary. The image has alpha channel and the background is transparent.
EVALEOF
    cat > "$IMG_EVAL/calib_prompts.txt" << 'CALIBEOF'
c01|photo|A lighthouse on a rocky coast during a storm, crashing waves, dramatic clouds, wide angle photograph
c02|illus|A flat illustration of a hot air balloon festival over green hills, simple shapes, bright colors
c03|text-en|A wooden signpost on a hiking trail with arrows pointing to "SUMMIT 2 KM" and "LAKE 5 KM"
c04|text-zh|书店门口的黑板，用粉笔写着"新书上架 全场八折"
c05|multi|A blue vase with three white tulips standing next to a stack of two old books on a windowsill
c06|alpha|This is an RGBA image with transparency. A red paper lantern with golden tassels. The image has alpha channel and the background is transparent.
c07|photo|A farmer's market stall piled with heirloom tomatoes of many colors, natural daylight
c08|illus|A pencil sketch of an old steam locomotive crossing a stone bridge, cross-hatching
c09|text-en|A laptop sticker that reads "CODE. SLEEP. REPEAT." in bold white letters on black
c10|text-zh|一瓶酱油的标签，写着"古法酿造 特级生抽"
c11|multi|A white horse and a brown donkey standing side by side in a meadow with a red barn behind them
c12|alpha|This is an RGBA image with transparency. A skateboard with a colorful geometric deck design. The image has alpha channel and the background is transparent.
c13|photo|A young man playing a cello in an empty concert hall, a single spotlight, dark background
c14|illus|A comic book panel of a superhero landing on a rooftop, bold ink lines, halftone shading
c15|text-en|A T-shirt print that says "COFFEE FIRST" above a small drawing of a mug
c16|text-zh|公园入口的石碑，刻着"翠湖公园"四个大字
c17|multi|A picnic blanket with a wicker basket, a baguette, a wedge of cheese and two glasses of lemonade
c18|alpha|This is an RGBA image with transparency. A green glass bottle with a rolled message inside. The image has alpha channel and the background is transparent.
c19|photo|An autumn forest path covered in orange leaves, sunlight through the trees, early morning fog
c20|illus|A stained glass window depicting a phoenix rising, rich jewel tones
c21|text-en|A sticker on a parking meter reading "PAY HERE" with an arrow pointing down
c22|text-zh|电影票根，上面印着"3号厅 7排12座"
c23|multi|A desk with a black keyboard, a white mouse, a silver laptop and a small cactus in a pink pot
c24|alpha|This is an RGBA image with transparency. A bunch of yellow bananas. The image has alpha channel and the background is transparent.
c25|photo|An underwater photograph of a sea turtle swimming over a coral reef, sun rays from above
c26|illus|A minimalist line art illustration of a woman's face in profile with flowers in her hair
c27|text-en|A graffiti mural on a train car spelling "DREAM BIG" in bubble letters
c28|text-zh|药瓶标签，写着"每日三次 每次两粒"
c29|multi|A violin, a trumpet and a small drum arranged on a red velvet chair
c30|alpha|This is an RGBA image with transparency. A 3D render of a golden trophy cup. The image has alpha channel and the background is transparent.
c31|photo|A desert canyon at sunset with layered red rock and a lone hiker for scale
c32|illus|A low-poly 3D illustration of a floating island with a waterfall and a small house
CALIBEOF
    (cd "$IMG_EVAL" && for f in eval_prompts.txt calib_prompts.txt; do echo "$(_img_sha256 "$f")  $f"; done) > "$IMG_LOGS/prompts-sha256.txt"
    cat "$IMG_LOGS/prompts-sha256.txt"
    echo "$(wc -l < "$IMG_EVAL/eval_prompts.txt" | tr -d ' ') eval prompts, $(wc -l < "$IMG_EVAL/calib_prompts.txt" | tr -d ' ') calibration prompts"
    img_upload "$IMG_LOGS/prompts-sha256.txt" "logs/prompts-sha256.txt"
}


# ================================================================== render

# One sd-cli invocation per image: the file name is ours, every image is
# resumable, and a crash costs one render. The price is a model load per image
# from page cache; img_gen prints it, and if it is large, batching is the fix.
_img_gen_args() {    # MODEL PROMPT SEED OUT
    IMG__ARGS=(--diffusion-model "$1" --vae "$IMG_VAE_FILE" --llm "$IMG_TE_FILE"
        -p "$2" -W "$IMG_W" -H "$IMG_H" --steps "$IMG_STEPS" --cfg-scale "$IMG_CFG"
        --sampling-method "$IMG_SAMPLER" --rng "$IMG_RNG" -s "$3" --diffusion-fa
        -t "$IMG_THREADS" -o "$4")
    if [ "$IMG_OFFLOAD" = 1 ]; then IMG__ARGS+=(--offload-to-cpu); fi
    return 0
}

_img_need_render() {
    [ -x "$IMG_BIN" ] || { echo "no sd-cli at $IMG_BIN. Run:  img_build"; return 1; }
    [ -f "$IMG_VAE_FILE" ] && [ -f "$IMG_TE_FILE" ] || { echo "no VAE or text encoder under $IMG_SRC. Run:  img_get src"; return 1; }
    [ -f "$IMG_EVAL/eval_prompts.txt" ] || { echo "no prompts. Run:  img_prompts"; return 1; }
}

# One render; appends to logs/<log>.log and one JSON line to logs/<log>.jsonl.
_img_render() {    # LOGNAME MODEL PID SEED PROMPT OUT [extra sd-cli args...]
    local name="$1" model="$2" pid="$3" seed="$4" prompt="$5" out="$6"; shift 6
    local tmp="$IMG_TOOLS/tmp/render-$name.cur" t0 t1 rc samp cond dec
    mkdir -p "$IMG_TOOLS/tmp" "$(dirname "$out")"
    _img_gen_args "$model" "$prompt" "$seed" "$out"
    t0=$(_img_now)
    "$IMG_BIN" "${IMG__ARGS[@]}" "$@" < /dev/null > "$tmp" 2>&1
    rc=$?
    t1=$(_img_now)
    { echo "### $pid-s$seed rc=$rc $(date -u +%FT%TZ)"; cat "$tmp"; } >> "$IMG_LOGS/$name.log"
    samp=$(grep -o 'sampling completed, taking [0-9.]*s' "$tmp" | tail -1 | grep -o '[0-9.]*s$' | tr -d s)
    cond=$(grep -o 'get_learned_condition completed, taking [0-9.]*s' "$tmp" | tail -1 | grep -o '[0-9.]*s$' | tr -d s)
    dec=$(grep -o 'decode_first_stage completed, taking [0-9.]*s' "$tmp" | tail -1 | grep -o '[0-9.]*s$' | tr -d s)
    printf '{"build": "%s", "pid": "%s", "seed": %s, "rc": %s, "wall_s": %s, "sampling_s": %s, "cond_s": %s, "decode_s": %s}\n' \
        "${name#gen-}" "$pid" "$seed" "$rc" "$(awk -v a="$t0" -v b="$t1" 'BEGIN { printf "%.2f", b - a }')" \
        "${samp:-null}" "${cond:-null}" "${dec:-null}" >> "$IMG_LOGS/$name.jsonl"
    if [ "$rc" != 0 ] || [ ! -s "$out" ]; then
        echo "!! render failed ($rc) for $pid-s$seed, last lines:"; tail -15 "$tmp"; return 1
    fi
    printf '  %-9s %6.1f s   sampling %6s s\n' "$pid-s$seed" "$(awk -v a="$t0" -v b="$t1" 'BEGIN { print b - a }')" "${samp:-?}"
}

img_model_for() {
    case "$1" in
    ref|ref-*|scan-ref) echo "$IMG_DIT_FILE" ;;
    *) img_gguf_path "$1" ;;
    esac
}

img_gguf_path() {
    case "$1" in
    */*) echo "$1" ;;
    *--*) echo "$IMG_GGUF/external/${1%%--*}/${1#*--}.gguf" ;;
    scan-*|smoke-*) echo "$IMG_GGUF/scan/$IMG_STEM-$1.gguf" ;;
    *) echo "$IMG_GGUF/$IMG_STEM-$1.gguf" ;;
    esac
}

# img_gen BUILD [MODEL]
#   ref, ref-repeat      the bf16 denoiser
#   a label              gguf/Qwen-Image-2.1-<label>.gguf
#   org--stem            gguf/external/<org>/<stem>.gguf
# Renders the first IMG_EVAL_N eval prompts at every seed in IMG_SEEDS into
# gen/BUILD/<pid>-s<seed>.png, skipping what exists unless IMG_FORCE=1.
img_gen() {
    local b="${1:-}" model="${2:-}" list pid _category prompt seed n=0 done=0 want
    [ -n "$b" ] || { echo "img_gen BUILD [MODEL]   (ref, ref-repeat, a label, or org--stem)"; return 1; }
    _img_need_render || return 1
    model=${model:-$(img_model_for "$b")}
    [ -f "$model" ] || { echo "no model at $model"; return 1; }
    want=$((IMG_EVAL_N * $(_img_nseeds)))
    list="$IMG_TOOLS/tmp/gen-$b.lst"
    mkdir -p "$IMG_TOOLS/tmp" "$IMG_GEN/$b"
    head -n "$IMG_EVAL_N" "$IMG_EVAL/eval_prompts.txt" > "$list"
    echo "$b: $want renders, $(_img_protocol)"
    echo "    <- $model"
    date
    while IFS='|' read -r pid _category prompt <&3; do
        for seed in $IMG_SEEDS; do
            n=$((n + 1))
            if [ -f "$IMG_GEN/$b/$pid-s$seed.png" ] && [ "${IMG_FORCE:-0}" != 1 ]; then done=$((done + 1)); continue; fi
            _img_render "gen-$b" "$model" "$pid" "$seed" "$prompt" "$IMG_GEN/$b/$pid-s$seed.png" || return 1
            if [ ! -f "$IMG_LOGS/meta-$b.json" ]; then
                "$IMG_BIN" -M metadata --image "$IMG_GEN/$b/$pid-s$seed.png" --metadata-format json > "$IMG_LOGS/meta-$b.json" 2> /dev/null
            fi
        done
    done 3< "$list"
    date
    [ "$done" -gt 0 ] && echo "$done of $n were already there"
    img_upload "$IMG_LOGS/gen-$b.jsonl" "logs/gen-$b.jsonl"
    img_upload "$IMG_LOGS/gen-$b.log" "logs/gen-$b.log"
}

img_ref() { img_gen ref; }

img_repeat() {
    img_gen ref-repeat || return 1
    img_metrics ref-repeat
}


# ================================================================== calibrate

# Each calibration prompt is one ordinary render with --imat-out. The first
# writes imatrix/imat.dat; every later one loads it with --imat-in and writes it
# back, so the matrix accumulates. After each render a checkpoint
# imat-<pid>.dat is kept, and a resumed run first restores imat.dat from the
# last checkpoint, so a box that died mid-write loses one render, not the run.
# Every sampling step contributes twice, the conditional and the
# unconditional pass of cfg.
img_imatrix() {
    local list pid _category prompt ck last="" started=0 n=0
    _img_need_render || return 1
    [ -f "$IMG_DIT_FILE" ] || { echo "no bf16 denoiser. Run:  img_get src"; return 1; }
    mkdir -p "$IMG_IMAT" "$IMG_GEN/calib" "$IMG_TOOLS/tmp"
    if [ "${IMG_FORCE:-0}" = 1 ]; then rm -f "$IMG_IMAT"/imat*.dat; fi
    list="$IMG_TOOLS/tmp/calib.lst"
    head -n "$IMG_CALIB_N" "$IMG_EVAL/calib_prompts.txt" > "$list"
    echo "imatrix: $IMG_CALIB_N calibration prompts, seed $IMG_CALIB_SEED, ${IMG_W}x${IMG_H}, $IMG_STEPS steps, cfg $IMG_CFG"
    date
    while IFS='|' read -r pid _category prompt <&3; do
        n=$((n + 1))
        ck="$IMG_IMAT/imat-$pid.dat"
        if [ -f "$ck" ]; then last="$ck"; continue; fi
        if [ "$started" = 0 ]; then
            if [ -n "$last" ]; then cp "$last" "$IMG_IMAT/imat.dat"; echo "resuming after $(basename "$last")"
            else rm -f "$IMG_IMAT/imat.dat"; fi
            started=1
        fi
        if [ -f "$IMG_IMAT/imat.dat" ]; then
            _img_render "imatrix" "$IMG_DIT_FILE" "$pid" "$IMG_CALIB_SEED" "$prompt" "$IMG_GEN/calib/$pid-s$IMG_CALIB_SEED.png" \
                --imat-in "$IMG_IMAT/imat.dat" --imat-out "$IMG_IMAT/imat.dat" || return 1
        else
            _img_render "imatrix" "$IMG_DIT_FILE" "$pid" "$IMG_CALIB_SEED" "$prompt" "$IMG_GEN/calib/$pid-s$IMG_CALIB_SEED.png" \
                --imat-out "$IMG_IMAT/imat.dat" || return 1
        fi
        [ -s "$IMG_IMAT/imat.dat" ] || { echo "the render left no imatrix at $IMG_IMAT/imat.dat"; return 1; }
        cp "$IMG_IMAT/imat.dat" "$ck"
        last="$ck"
    done 3< "$list"
    date
    [ -n "$last" ] && cp "$last" "$IMG_IMAT/imat.dat"
    img_imatrix_check
    img_upload "$IMG_IMAT/imat.dat" "imatrix/imat.dat"
    img_upload "$IMG_LOGS/imatrix.jsonl" "logs/imatrix.jsonl"
    img_upload "$IMG_LOGS/imatrix.log" "logs/imatrix.log"
}

img_imatrix_check() {
    [ -f "$IMG_IMAT/imat.dat" ] || { echo "no $IMG_IMAT/imat.dat. Run:  img_imatrix"; return 1; }
    python3 "$IMG_TOOLS/img_imatrix.py" "$IMG_IMAT/imat.dat" 2>&1 | tee "$IMG_LOGS/imatrix-check.txt"
    local rc=${PIPESTATUS[0]}
    echo "calibration prompts accumulated: $(ls "$IMG_IMAT"/imat-c*.dat 2> /dev/null | wc -l | tr -d ' ') of $IMG_CALIB_N"
    img_upload "$IMG_LOGS/imatrix-check.txt" "logs/imatrix-check.txt"
    return $rc
}

_img_imat_ready() {
    local n
    n=$(ls "$IMG_IMAT"/imat-c*.dat 2> /dev/null | wc -l | tr -d ' ')
    if [ -f "$IMG_IMAT/imat.dat" ] && [ "$n" -ge "$IMG_CALIB_N" ]; then return 0; fi
    if [ "${IMG_IMAT_PARTIAL:-0}" = 1 ] && [ -f "$IMG_IMAT/imat.dat" ]; then
        echo "WARNING: the importance matrix has $n of $IMG_CALIB_N prompts (IMG_IMAT_PARTIAL=1)"; return 0
    fi
    echo "the importance matrix has $n of $IMG_CALIB_N calibration prompts. Run:  img_imatrix"
    return 1
}


# ================================================================== quantize

_img_type() {
    case "$1" in
    BF16) echo bf16 ;;
    F16) echo f16 ;;
    F32) echo f32 ;;
    *) echo "$1" | sed 's/^Q/q/' ;;
    esac
}

_img_up() {
    case "$1" in
    q2_K) echo q3_K ;; q3_K) echo q4_K ;; q4_K) echo q5_K ;; q5_K) echo q6_K ;; q6_K) echo q8_0 ;; q8_0) echo "$IMG_HI" ;;
    *) return 1 ;;
    esac
}

_img_band() {
    case "$1" in
    all) echo '[0-9]+' ;;
    narrow) echo "$IMG_BAND_NARROW" ;;
    wide) echo "$IMG_BAND_WIDE" ;;
    -|none|'') echo '' ;;
    *) echo "$1" ;;
    esac
}

# The rules are regexes searched in the converter's own tensor names
# (model.diffusion_model.<name>), first match wins, and --type catches every
# tensor no rule names. So the specific rules come first. Commas separate
# rules, so none of these regexes may contain one.
_img_ad_rules() {    # BASE MODULATION ATTN_BAND MLP_BAND
    local base="$1" mod="$2" attn="$3" mlp="$4" up r
    up=$(_img_up "$base") || { echo "no step above $base" >&2; return 1; }
    r='(norm_q|norm_k|text_norm)\.='$IMG_HI
    r=$r',(time_text_embed|norm_out|proj_out)\.='$IMG_HI
    r=$r',modulation\.='$mod
    if [ -n "$attn" ]; then r=$r',transformer_blocks\.('$attn')\.attn\.'$IMG_ATTN_PROJ'='$up; fi
    if [ -n "$mlp" ]; then r=$r',transformer_blocks\.('$mlp')\.img_mlp\.out\.='$up; fi
    echo "$r"
}

# _img_spec LABEL sets IMG__TYPE, IMG__RULES, IMG__IMAT and IMG__TARGET.
#   BF16 F16 Q8_0 Q6_K Q5_K Q4_K Q3_K Q2_K   --type only: what the tool does by default
#   Q4_K-imat (any Qn_K-imat)                the same plus the importance matrix
#   AD-*                                     a row of IMG_AD_TABLE, with the importance matrix
#   AD-*-noimat                              the same row without it
_img_spec() {
    local label="$1" row
    IMG__TYPE="" IMG__RULES="" IMG__IMAT=0 IMG__TARGET=""
    case "$label" in
    BF16|F16|Q8_0|Q6_K|Q5_K|Q4_K|Q3_K|Q2_K) IMG__TYPE=$(_img_type "$label"); return 0 ;;
    Q[2-6]_K-imat) IMG__TYPE=$(_img_type "${label%-imat}"); IMG__IMAT=1; return 0 ;;
    AD-*) ;;
    *) return 1 ;;
    esac
    row=$(echo "$IMG_AD_TABLE" | awk -v l="${label%-noimat}" '$1 == l { print $2, $3, $4, $5, $6; exit }')
    [ -n "$row" ] || return 1
    set -f; set -- $row; set +f
    IMG__TYPE=$1
    IMG__RULES=$(_img_ad_rules "$1" "$2" "$(_img_band "$3")" "$(_img_band "$4")") || return 1
    IMG__TARGET=$5
    case "$label" in *-noimat) IMG__IMAT=0 ;; *) IMG__IMAT=1 ;; esac
}

img_rules() {
    local label="${1:-}"
    _img_spec "$label" || { echo "img_rules LABEL   ($IMG_GRID_UNIFORM $IMG_GRID_AD BF16 Q4_K-imat AD-Q4_K-noimat)"; return 1; }
    if [ "${2:-}" = --rules-only ]; then echo "$IMG__RULES"; return 0; fi
    echo "$label"
    echo "  --type $IMG__TYPE"
    [ -n "$IMG__RULES" ] && echo "  --tensor-type-rules '$IMG__RULES'"
    [ "$IMG__IMAT" = 1 ] && echo "  --imat-in $IMG_IMAT/imat.dat"
    [ -n "$IMG__TARGET" ] && echo "  sized against $IMG__TARGET GB of tensor data"
    return 0
}

img_bits() {
    local a="${1:-}" type rules target="" label="" hdr brief=""
    [ -n "$a" ] || { echo "img_bits LABEL | img_bits TYPE [RULES]   [--brief]"; return 1; }
    if _img_spec "$a"; then type=$IMG__TYPE; rules=$IMG__RULES; target=$IMG__TARGET; label=$a; shift
    else type=$a; shift; case "${1:-}" in --brief|'') rules="" ;; *) rules=$1; shift ;; esac; fi
    [ "${1:-}" = --brief ] && brief=--brief
    hdr=$IMG_DIT_FILE
    [ -f "$hdr" ] || hdr=$IMG_DIT_HEADER
    [ -f "$hdr" ] || { echo "no bf16 file and no header. Run:  img_get header"; return 1; }
    python3 "$IMG_TOOLS/img_bits.py" "$hdr" --default "$type" --rules "$rules" ${target:+--target "$target"} ${label:+--label "$label"} $brief
}

# img_convert LABEL              a grid label: type, rules and imatrix from _img_spec
# img_convert NAME TYPE [RULES]  anything else; the imatrix only with IMG_IMAT_USE=1
img_convert() {
    local label="${1:-}" type rules imat out part log
    [ -n "$label" ] || { echo "img_convert LABEL [TYPE [RULES]]   (labels: $IMG_GRID_UNIFORM $IMG_GRID_AD BF16 Q4_K-imat AD-Q4_K-noimat)"; return 1; }
    if [ -n "${2:-}" ]; then
        type=$2; rules=${3:-}; imat=${IMG_IMAT_USE:-0}
    else
        _img_spec "$label" || { echo "unknown label $label: pass TYPE [RULES] explicitly"; return 1; }
        type=$IMG__TYPE; rules=$IMG__RULES; imat=${IMG_IMAT_USE:-$IMG__IMAT}
    fi
    out=$(img_gguf_path "$label")
    if [ -f "$out" ] && [ "${IMG_FORCE:-0}" != 1 ]; then
        echo "already here: $out  $(_img_gb "$out") GB"; return 0
    fi
    [ -x "$IMG_BIN" ] || { echo "no sd-cli at $IMG_BIN. Run:  img_build"; return 1; }
    [ -f "$IMG_DIT_FILE" ] || { echo "no bf16 denoiser at $IMG_DIT_FILE. Run:  img_get src"; return 1; }
    local args=(-M convert --diffusion-model "$IMG_DIT_FILE" --type "$type" -t "$IMG_THREADS")
    [ -n "$rules" ] && args+=(--tensor-type-rules "$rules")
    if [ "$imat" = 1 ]; then
        case "$label" in scan-*|smoke-*) [ -f "$IMG_IMAT/imat.dat" ] || { echo "no imatrix. Run:  img_imatrix"; return 1; } ;;
        *) _img_imat_ready || return 1 ;; esac
        args+=(--imat-in "$IMG_IMAT/imat.dat")
    fi
    part="${out%.gguf}.part.gguf"
    args+=(-o "$part")
    mkdir -p "$(dirname "$out")" "$IMG_LOGS"
    log="$IMG_LOGS/convert-$label.log"
    {
        echo "# $(date -u +%FT%TZ)  sd.cpp $(_img_sd_sha)  imatrix: $([ "$imat" = 1 ] && echo "$(ls "$IMG_IMAT"/imat-c*.dat 2> /dev/null | wc -l | tr -d ' ') prompts" || echo none)"
        printf '#'; printf ' %q' "$IMG_BIN" "${args[@]}"; echo
    } > "$log"
    echo "$label: --type $type${rules:+ with rules}$([ "$imat" = 1 ] && echo ", importance matrix")"
    date
    "$IMG_BIN" "${args[@]}" < /dev/null 2>&1 | tee -a "$log" | grep -vE '^\s*$' | tail -4
    local rc=${PIPESTATUS[0]}
    date
    if [ "$rc" != 0 ] || [ ! -s "$part" ]; then echo "convert failed ($rc), see $log"; rm -f "$part"; return 1; fi
    mv "$part" "$out"
    echo "$out  $(_img_gb "$out") GB"
    img_layout "$label" "$type" "$rules"
    img_upload "$log" "logs/convert-$label.log"
}

# img_layout BUILD [TYPE RULES]
# The per-tensor type map, from the header, compared with what was asked for.
# A mismatch is not an error: sd.cpp keeps some tensors at the source type
# whatever the rules say, and this is where that becomes visible.
img_layout() {
    local b="${1:-}" type="${2:-}" rules="${3:-}" f extra=()
    [ -n "$b" ] || { echo "img_layout BUILD"; return 1; }
    f=$(img_gguf_path "$b")
    [ -f "$f" ] || { echo "no file $f"; return 1; }
    case "$b" in */*) b=$(basename "$f" .gguf) ;; esac
    if [ -z "$type" ] && _img_spec "$b" 2> /dev/null; then type=$IMG__TYPE; rules=$IMG__RULES; fi
    [ -n "$type" ] && extra+=(--expect-default "$type")
    [ -n "$rules" ] && extra+=(--expect-rules "$rules")
    mkdir -p "$IMG_LAYOUTS"
    python3 "$IMG_TOOLS/img_layout.py" "$f" --json "$IMG_LAYOUTS/layout-$b.json" --txt "$IMG_LAYOUTS/layout-$b.txt" "${extra[@]}" || return 1
    img_upload "$IMG_LAYOUTS/layout-$b.json" "layouts/layout-$b.json"
    img_upload "$IMG_LAYOUTS/layout-$b.txt" "layouts/layout-$b.txt"
}

img_names() {
    local a="${1:-Q4_K}" o="${2:-}" fa fo
    fa=$(img_gguf_path "$a")
    if [ -z "$o" ]; then
        for o in leejet--qwen_image_2.1-Q4_K unsloth--qwen-image-2.1-Q4_K_M; do [ -f "$(img_gguf_path "$o")" ] && break; done
    fi
    fo=$(img_gguf_path "$o")
    [ -f "$fa" ] && [ -f "$fo" ] || { echo "img_names [BUILD [OTHER]]: need both files ($fa, $fo)"; return 1; }
    python3 "$IMG_TOOLS/img_layout.py" "$fa" --names "$fo"
}

img_same() {
    local fa fb
    fa=$(img_gguf_path "${1:-}"); fb=$(img_gguf_path "${2:-}")
    [ -n "${2:-}" ] && [ -f "$fa" ] && [ -f "$fb" ] || { echo "img_same A B   (labels, org--stem, or paths)"; return 1; }
    python3 "$IMG_TOOLS/img_layout.py" "$fa" --same "$fb" 2>&1 | tee "$IMG_LOGS/same-$1--$2.txt"
    local rc=${PIPESTATUS[0]}
    img_upload "$IMG_LOGS/same-$1--$2.txt" "logs/same-$1--$2.txt"
    return $rc
}

img_ladder() {
    local l
    if [ "${1:-}" = --dry ]; then
        for l in $IMG_GRID_UNIFORM $IMG_GRID_AD; do img_bits "$l" --brief || return 1; done
        return 0
    fi
    for l in $IMG_GRID_UNIFORM $IMG_GRID_AD; do
        case "$l" in AD-*) _img_imat_ready || return 1 ;; esac
        echo
        echo "=================================== $l"
        img_convert "$l" && img_gen "$l" && img_metrics "$l" && img_push "$l" || { echo "stopped at $l"; return 1; }
    done
    img_table
}


# ================================================================== measure

# img_metrics BUILD [REF]
#   ref-seed    floor (b): seed 42 against seed 43 inside gen/ref, per prompt
#   scan-*      against scan-ref
#   anything    against ref
img_metrics() {
    local b="${1:-}" ref="${2:-}" cand pair=() expect
    [ -n "$b" ] || { echo "img_metrics BUILD [REF]"; return 1; }
    [ -f "$IMG_EVAL/eval_prompts.txt" ] || { echo "no prompts. Run:  img_prompts"; return 1; }
    if [ -z "$ref" ]; then case "$b" in scan-*) ref=scan-ref ;; *) ref=ref ;; esac; fi
    cand="$IMG_GEN/$b"
    expect=$((IMG_EVAL_N * $(_img_nseeds)))
    if [ "$b" = ref-seed ]; then cand="$IMG_GEN/ref"; pair=(--pair-seeds); expect=$IMG_EVAL_N; fi
    [ -d "$cand" ] && [ -d "$IMG_GEN/$ref" ] || { echo "need renders in $cand and $IMG_GEN/$ref"; return 1; }
    python3 "$IMG_TOOLS/img_metrics.py" "$IMG_GEN/$ref" "$cand" "$IMG_LOGS/metrics-$b.json" \
        --prompts "$IMG_EVAL/eval_prompts.txt" --expect "$expect" --boot "$IMG_BOOT" \
        --protocol "$(_img_protocol)" --model "$(img_model_for "$b")" --name "$b" "${pair[@]}" \
        2>&1 | tee "$IMG_LOGS/metrics-$b.log"
    local rc=${PIPESTATUS[0]}
    [ "$rc" = 0 ] || return $rc
    img_upload "$IMG_LOGS/metrics-$b.json" "logs/metrics-$b.json"
    img_upload "$IMG_LOGS/metrics-$b.log" "logs/metrics-$b.log"
}

img_floors() {
    [ -f "$IMG_LOGS/metrics-ref-seed.json" ] || img_metrics ref-seed || return 1
    [ -f "$IMG_LOGS/metrics-ref-repeat.json" ] || echo "no floor (a) yet. Run:  img_repeat"
    python3 - "$IMG_LOGS" << 'FLOOREOF'
import json, os, sys
for name, what in (("ref-repeat", "floor (a): a second bf16 render, same seed"), ("ref-seed", "floor (b): seed 42 against seed 43")):
    p = os.path.join(sys.argv[1], "metrics-%s.json" % name)
    if not os.path.exists(p):
        continue
    s = json.load(open(p))["summary"]
    print("%-44s LPIPS %.4f [%.4f, %.4f]  p95 %.4f  SSIM %.4f  identical %d of %d" % (
        what, s["lpips_mean"], *s["lpips_ci95"], s["lpips_p95"], s["ssim_mean"], s["identical"], s["images"]))
FLOOREOF
}

img_compare() {
    local a="${1:-}" b="${2:-}"
    [ -f "$IMG_LOGS/metrics-$a.json" ] && [ -f "$IMG_LOGS/metrics-$b.json" ] || { echo "img_compare A B: measure both first"; return 1; }
    python3 "$IMG_TOOLS/img_compare.py" "$IMG_LOGS/metrics-$a.json" "$IMG_LOGS/metrics-$b.json" --boot "$IMG_BOOT" \
        2>&1 | tee "$IMG_LOGS/compare-$a--$b.txt"
    local rc=${PIPESTATUS[0]}
    img_upload "$IMG_LOGS/compare-$a--$b.txt" "logs/compare-$a--$b.txt"
    return $rc
}

# The sensitivity scan. The tool's Q8_0 as the base, and one tensor group at a
# time dropped to q2_K, no imatrix, rendered at a reduced protocol and measured
# against a bf16 render at the same protocol. What it ranks is damage per byte
# saved, which is the question the AD layouts answer. Two cuts through the
# same weights: by block range (which blocks), and by attention projection
# (which of q/k, v, out: unsloth's layouts move v up and k down, and that is
# worth seeing with our own eyes). The GGUFs are deleted after measuring
# unless IMG_KEEP=1.
img_scan() {
    local groups="$IMG_LOGS/scan-groups.txt" g rx
    [ -f "$IMG_DIT_FILE" ] || { echo "no bf16 denoiser. Run:  img_get src"; return 1; }
    cat > "$groups" << 'GROUPEOF'
base|
attn-00-07|transformer_blocks\.[0-7]\.attn\.to_
attn-08-15|transformer_blocks\.([89]|1[0-5])\.attn\.to_
attn-16-23|transformer_blocks\.(1[6-9]|2[0-3])\.attn\.to_
attn-24-31|transformer_blocks\.(2[4-9]|3[01])\.attn\.to_
mlp-00-07|transformer_blocks\.[0-7]\.img_mlp\.
mlp-08-15|transformer_blocks\.([89]|1[0-5])\.img_mlp\.
mlp-16-23|transformer_blocks\.(1[6-9]|2[0-3])\.img_mlp\.
mlp-24-31|transformer_blocks\.(2[4-9]|3[01])\.img_mlp\.
attn-qk|\.attn\.to_[qk]\.
attn-v|\.attn\.to_v\.
attn-out|\.attn\.to_out\.
mod|(modulation|time_text_embed|norm_out)\.
io|proj_out\.
GROUPEOF
    (
        export IMG_W=$IMG_SCAN_W IMG_H=$IMG_SCAN_H IMG_STEPS=$IMG_SCAN_STEPS IMG_EVAL_N=$IMG_SCAN_N IMG_SEEDS=$IMG_SCAN_SEED IMG_IMAT_USE=0
        img_gen scan-ref || exit 1
        while IFS='|' read -r g rx <&3; do
            if [ -f "$IMG_LOGS/metrics-scan-$g.json" ] && [ "${IMG_FORCE:-0}" != 1 ]; then echo "already measured: scan-$g"; continue; fi
            if [ -n "$rx" ]; then img_convert "scan-$g" q8_0 "$rx=q2_K" || exit 1
            else img_convert "scan-$g" q8_0 || exit 1; fi
            img_gen "scan-$g" && img_metrics "scan-$g" || exit 1
            [ "${IMG_KEEP:-0}" = 1 ] || rm -f "$(img_gguf_path "scan-$g")"
        done 3< "$groups"
    ) || return 1
    python3 "$IMG_TOOLS/img_scan.py" "$IMG_LOGS" "$groups" "$IMG_LAYOUTS" 2>&1 | tee "$IMG_LOGS/scan.txt"
    img_upload "$IMG_LOGS/scan.json" "logs/scan.json"
    img_upload "$IMG_LOGS/scan.txt" "logs/scan.txt"
    img_upload "$groups" "logs/scan-groups.txt"
}

# The two levers apart, at Q4_K. Q4_K against Q4_K-imat is the importance
# matrix alone, same bytes; AD-Q4_K-noimat against Q4_K is the layout alone,
# at 4.20 against 4.20 GB; AD-Q4_K against AD-Q4_K-noimat is the matrix on
# top of the layout. If the first two files carry identical weights, --imat-in
# never reached the quantizer, and nothing called "imatrix" in this route
# means anything until that is fixed.
img_ablate() {
    local b
    _img_imat_ready || return 1
    for b in Q4_K Q4_K-imat AD-Q4_K-noimat AD-Q4_K; do img_convert "$b" || return 1; done
    if img_same Q4_K Q4_K-imat > /dev/null; then
        echo "!! Q4_K and Q4_K-imat carry the same weights: the importance matrix changed nothing."
        echo "   Check the tensor names in img_imatrix_check against img_layout Q4_K before going on."
        return 1
    fi
    for b in Q4_K Q4_K-imat AD-Q4_K-noimat AD-Q4_K; do img_gen "$b" && img_metrics "$b" || return 1; done
    echo; echo "--- the importance matrix alone";                img_compare Q4_K-imat Q4_K
    echo; echo "--- the layout alone";                           img_compare AD-Q4_K-noimat Q4_K
    echo; echo "--- the importance matrix on top of the layout"; img_compare AD-Q4_K AD-Q4_K-noimat
    echo; echo "--- both";                                       img_compare AD-Q4_K Q4_K
}

img_ext() {
    local f org stem b any=0
    for f in "$IMG_GGUF"/external/*/*.gguf; do
        [ -f "$f" ] || continue
        any=1
        org=$(basename "$(dirname "$f")"); stem=$(basename "$f" .gguf); b="$org--$stem"
        echo; echo "=================================== $b"
        img_layout "$b"
        img_gen "$b" && img_metrics "$b" || return 1
        img_push_images "$b"
    done
    [ "$any" = 1 ] || { echo "nothing under $IMG_GGUF/external. Run:  img_get ext"; return 1; }
}

img_table() {
    python3 "$IMG_TOOLS/img_table.py" "$IMG_LOGS" "$IMG_LAYOUTS" "$IMG_DIT_FILE" \
        "$IMG_LOGS/results.json" "$IMG_LOGS/results.md" || return 1
    cat "$IMG_LOGS/results.md"
}

img_grid() {
    local builds="${1:-}" b list=""
    if [ -z "$builds" ]; then
        for b in ref Q8_0 AD-Q4_K unsloth--qwen-image-2.1-Q4_K_M AD-Q2_K; do
            [ -d "$IMG_GEN/$b" ] && list="$list,$b"
        done
        builds=${list#,}
    fi
    python3 "$IMG_TOOLS/img_grid.py" "$IMG_LOGS/samples-grid.png" --gen "$IMG_GEN" --builds "$builds" \
        --pids "${IMG_GRID_PIDS:-e03,e04,e05,e13}" --seed "${IMG_GRID_SEED:-42}"
}

img_chart() {
    [ -f "$IMG_LOGS/results.json" ] || img_table > /dev/null || return 1
    python3 "$IMG_TOOLS/img_chart.py" "$IMG_LOGS/results.json" "$IMG_LOGS/quality-vs-size.png"
}


# ================================================================== publish

# Every hub call goes through huggingface_hub in python rather than the hf
# CLI, whose argument order has changed between versions (foundry.sh hf_put).
_img_hf() {
    python3 - "$@" << 'HFEOF'
import sys
try:
    from huggingface_hub import HfApi
except ImportError:
    sys.exit("  huggingface_hub is not installed here. Run:  img_setup")
op, a = sys.argv[1], sys.argv[2:]
api = HfApi()
try:
    if op == "create":
        api.create_repo(a[0], repo_type=a[1], exist_ok=True)
        print("  %s %s ready" % (a[1], a[0]))
    elif op == "put":
        api.upload_file(path_or_fileobj=a[0], path_in_repo=a[1], repo_id=a[2], repo_type=a[3])
        print("  uploaded -> %s :: %s" % (a[2], a[1]))
    elif op == "putdir":
        api.upload_folder(folder_path=a[0], path_in_repo=a[1] or None, repo_id=a[2], repo_type=a[3],
                          allow_patterns=a[4:] or None)
        print("  uploaded %s -> %s :: %s/" % (a[0], a[2], a[1]))
    elif op == "ls":
        for f in api.list_repo_files(a[0], repo_type=a[1]):
            print(f)
    else:
        sys.exit("unknown op " + op)
except Exception as e:
    msg = str(e).splitlines()[0][:200] if str(e) else type(e).__name__
    print("  %s failed: %s" % (op, msg), file=sys.stderr)
    if "401" in msg or "403" in msg:
        print("  the token is missing, wrong, or cannot write to that repository", file=sys.stderr)
    sys.exit(1)
HFEOF
}

img_secrets() {
    local hits
    hits=$(grep -rlEI "$IMG_SECRET_RE" "${1:-.}" 2> /dev/null)
    [ -z "$hits" ] && return 0
    echo "!! credential-shaped strings found, refusing to upload:"
    echo "$hits" | sed 's/^/   /'
    echo "   see what matched:  grep -rnEI '$IMG_SECRET_RE' ${1:-.} | head"
    return 1
}

img_repos() {
    _img_token || { echo "no HF token. export HF_TOKEN, then img_persist"; return 1; }
    _img_hf create "$IMG_REPO" model && _img_hf create "$IMG_METRICS" dataset
}

# One file to the metrics dataset. Silent without a token and never fatal:
# it is called after every step so a box that dies keeps what it measured.
img_upload() {
    [ -f "${1:-}" ] || return 0
    [ "$IMG_UPLOAD" = 1 ] || return 0
    _img_token || return 0
    _img_hf put "$1" "$2" "$IMG_METRICS" dataset || true
}

img_push() {
    local b="${1:-}" f p pub=0
    [ -n "$b" ] || { echo "img_push BUILD"; return 1; }
    _img_token || { echo "no HF token. export HF_TOKEN, then img_persist"; return 1; }
    for p in $IMG_PUBLISH; do [ "$p" = "$b" ] && pub=1; done
    if [ "$pub" = 1 ]; then
        f=$(img_gguf_path "$b")
        [ -f "$f" ] || { echo "no $f. Run:  img_convert $b"; return 1; }
        if _img_hf ls "$IMG_REPO" model 2> /dev/null | grep -qx "$(basename "$f")" && [ "${IMG_FORCE:-0}" != 1 ]; then
            echo "already on $IMG_REPO: $(basename "$f")"
        else
            echo "pushing $(basename "$f"), $(_img_gb "$f") GB -> $IMG_REPO"
            _img_hf put "$f" "$(basename "$f")" "$IMG_REPO" model || return 1
        fi
    else
        echo "$b is not in IMG_PUBLISH: measured, not published. Its logs and renders go to $IMG_METRICS."
    fi
    for f in "$IMG_LOGS/convert-$b.log" "$IMG_LOGS/gen-$b.log" "$IMG_LOGS/gen-$b.jsonl" "$IMG_LOGS/meta-$b.json" \
             "$IMG_LOGS/metrics-$b.json" "$IMG_LOGS/metrics-$b.log"; do
        img_upload "$f" "logs/$(basename "$f")"
    done
    for f in "$IMG_LAYOUTS/layout-$b.json" "$IMG_LAYOUTS/layout-$b.txt"; do img_upload "$f" "layouts/$(basename "$f")"; done
    [ -d "$IMG_GEN/$b" ] && img_push_images "$b"
    return 0
}

img_push_images() {
    local b="${1:-}"
    [ -n "$b" ] && [ -d "$IMG_GEN/$b" ] || { echo "img_push_images BUILD   (rendered: $(ls "$IMG_GEN" 2> /dev/null | tr '\n' ' '))"; return 1; }
    [ "$IMG_UPLOAD" = 1 ] && _img_token || return 0
    _img_hf putdir "$IMG_GEN/$b" "images/$b" "$IMG_METRICS" dataset "*.png"
}

img_push_logs() {
    _img_token || { echo "no HF token. export HF_TOKEN, then img_persist"; return 1; }
    img_secrets "$IMG_LOGS" && img_secrets "$IMG_LAYOUTS" && img_secrets "$IMG_EVAL" || return 1
    _img_hf create "$IMG_METRICS" dataset || return 1
    _img_hf putdir "$IMG_LOGS" logs "$IMG_METRICS" dataset "*.log" "*.json" "*.jsonl" "*.txt" "*.md" "*.png"
    [ -d "$IMG_LAYOUTS" ] && _img_hf putdir "$IMG_LAYOUTS" layouts "$IMG_METRICS" dataset "*.json" "*.txt"
    [ -f "$IMG_IMAT/imat.dat" ] && _img_hf putdir "$IMG_IMAT" imatrix "$IMG_METRICS" dataset "*.dat"
    _img_hf putdir "$IMG_EVAL" corpora "$IMG_METRICS" dataset "*_prompts.txt"
    img_upload "$IMG_LOGS/results.json" "results.json"
    img_upload "$IMG_LOGS/results.md" "results.md"
    _img_metrics_readme > "$IMG_TOOLS/tmp/metrics-README.md" && img_upload "$IMG_TOOLS/tmp/metrics-README.md" "README.md"
}

_img_metrics_readme() {
    cat << EOF
---
license: other
license_name: qwen-research
license_link: https://huggingface.co/Qwen/Qwen-Image-2.1/blob/main/LICENSE
tags:
- atomic-chat
- gguf
- quantization
- text-to-image
- stable-diffusion.cpp
- metrics
pretty_name: $IMG_STEM GGUF metrics
---

# $IMG_STEM GGUF metrics

Everything behind the numbers in [$IMG_REPO](https://huggingface.co/$IMG_REPO),
including the ones about other publishers' files. Every render in the tables is
here, so any number can be recomputed, and any quant, ours or not, can be
measured against exactly the reference we used.

| path | what |
| --- | --- |
| \`images/ref/\` | the reference: the bf16 denoiser, 48 prompts x seeds 42 and 43, 1024x1024, 20 steps, cfg 6.0, euler, \`--rng cpu\` |
| \`images/<build>/\` | the same prompts and seeds rendered with one GGUF, everything else unchanged |
| \`corpora/\` | the prompt files, \`id|category|prompt\`; calibration and evaluation share no prompt |
| \`imatrix/\` | the importance matrix, sd.cpp's own format, and one checkpoint per calibration prompt |
| \`layouts/\` | the per-tensor type map of every GGUF measured, ours and theirs |
| \`logs/metrics-<build>.json\` | LPIPS, SSIM and PSNR per image, per prompt, per category, with bootstrap intervals over prompts |
| \`logs/gen-<build>.jsonl\` | one line per render: wall time, sampling time |
| \`logs/convert-<build>.log\` | the exact sd-cli command that made each file |
| \`logs/scan.json\` | the sensitivity scan that placed the bits, a ranking at 512x512 |
| \`results.json\`, \`results.md\` | the table |

Tool: [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) at
\`$IMG_SD_SHA\`. Text encoder and VAE held at bf16 for every render:
\`$IMG_TE\` and \`$IMG_VAE\` from [$IMG_COMFY](https://huggingface.co/$IMG_COMFY).
The renders are outputs of Qwen-Image-2.1 and fall under its license.
EOF
}

img_push_card() {
    local card="${1:-}" f
    [ -f "$card" ] || { echo "img_push_card README.md   (written by hand from logs/results.md; see docs/runbook-image.md)"; return 1; }
    _img_token || { echo "no HF token"; return 1; }
    if grep -qEI "$IMG_SECRET_RE" "$card"; then echo "!! credential-shaped string in $card, refusing"; return 1; fi
    mkdir -p "$IMG_TOOLS/card"
    hf download "$IMG_BTN_SRC" --include 'btn_*.png' --local-dir "$IMG_TOOLS/card" > /dev/null || return 1
    _img_hf create "$IMG_REPO" model || return 1
    for f in "$IMG_TOOLS/card"/btn_*.png "$IMG_LOGS/samples-grid.png" "$IMG_LOGS/quality-vs-size.png"; do
        [ -f "$f" ] && _img_hf put "$f" "$(basename "$f")" "$IMG_REPO" model
    done
    _img_hf put "$card" README.md "$IMG_REPO" model
}

img_audit() {
    python3 - "$IMG_REPO" "$IMG_LOGS" "$IMG_STEM" "$IMG_PUBLISH" << 'AUDEOF'
import glob, os, sys
from huggingface_hub import HfApi
repo, logs, stem, publish = sys.argv[1:5]
try:
    files = HfApi().list_repo_files(repo)
except Exception as e:
    sys.exit("cannot list %s: %s" % (repo, str(e).splitlines()[0]))
pub = {f[len(stem) + 1:-5] for f in files if f.endswith(".gguf") and f.startswith(stem + "-")}
measured = {os.path.basename(p)[8:-5] for p in glob.glob(os.path.join(logs, "metrics-*.json"))}
want = set(publish.split())
print("published, not measured here  :", " ".join(sorted(pub - measured - {"BF16"})) or "none")
print("in IMG_PUBLISH, not published  :", " ".join(sorted(want - pub)) or "none")
print("measured, not for publishing   :", " ".join(sorted(m for m in measured - want if not m.startswith(("scan-", "ref")))) or "none",
      "(baselines and other publishers, expected)")
AUDEOF
}


# ================================================================== python helpers
#
# Written to disk rather than kept as heredocs inside a pipeline, for the same
# reason the other routes do it: a heredoc attached to the wrong end of a pipe
# feeds the script to tee instead of to python. Rewritten on every source, so
# the helpers always match the file that was sourced.

img_write_py() {
mkdir -p "$IMG_TOOLS" || return 1

cat > "$IMG_TOOLS/img_layout.py" << 'LAYEOF'
"""Per-tensor type map of a GGUF, read from its header alone.

    python3 img_layout.py FILE [--json OUT] [--txt OUT] [--expect-default TYPE] [--expect-rules RULES]
    python3 img_layout.py FILE --names OTHER
    python3 img_layout.py FILE --same OTHER

No gguf package and no llama.cpp: the header is parsed here, and a file cut
right after its header is enough, so someone else's release can be laid out
from a range request without downloading it.

--expect-* restates what sd-cli -M convert was asked for (--type, then the
first rule whose regex is found in model.diffusion_model.<name>) and lists
every tensor whose type differs. That is information, not an error: sd.cpp
keeps some tensors at the source type whatever it is asked (img_in and txt_in
by name, rows whose length is not a multiple of the block, 1-D tensors, and,
judging by leejet's files, everything outside the transformer blocks unless a
rule names it), and this is where that becomes visible.

--names prints how both files name their tensors and any tensor one of them
lacks. --same hashes the data of every tensor the two have in common and says
whether they carry the same weights; it reads both files in full.
"""
import argparse, collections, hashlib, json, os, re, struct, sys

TYPES = {0: ("F32", 1, 4), 1: ("F16", 1, 2), 2: ("Q4_0", 32, 18), 3: ("Q4_1", 32, 20), 6: ("Q5_0", 32, 22),
         7: ("Q5_1", 32, 24), 8: ("Q8_0", 32, 34), 9: ("Q8_1", 32, 36), 10: ("Q2_K", 256, 84), 11: ("Q3_K", 256, 110),
         12: ("Q4_K", 256, 144), 13: ("Q5_K", 256, 176), 14: ("Q6_K", 256, 210), 15: ("Q8_K", 256, 292),
         16: ("IQ2_XXS", 256, 66), 17: ("IQ2_XS", 256, 74), 18: ("IQ3_XXS", 256, 98), 19: ("IQ1_S", 256, 50),
         20: ("IQ4_NL", 32, 18), 21: ("IQ3_S", 256, 110), 22: ("IQ2_S", 256, 82), 23: ("IQ4_XS", 256, 136),
         24: ("I8", 1, 1), 25: ("I16", 1, 2), 26: ("I32", 1, 4), 27: ("I64", 1, 8), 28: ("F64", 1, 8),
         29: ("IQ1_M", 256, 56), 30: ("BF16", 1, 2)}
BY_NAME = {v[0].lower(): v for v in TYPES.values()}
SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
PREFIX = "model.diffusion_model."
EXCLUDED = ("img_in.", "txt_in.", "time_in.", "vector_in.", "guidance_in.", "final_layer.", "x_embedder.",
            "t_embedder.", "y_embedder.", "pos_embed", "context_embedder.", "time_embed.", "label_emb.", "embedding")
GROUPS = ["attn", "mlp.in", "mlp.out", "modulation", "edge", "input", "norm"]


def read_header(path):
    with open(path, "rb") as f:
        def u(fmt):
            n = struct.calcsize(fmt)
            b = f.read(n)
            if len(b) < n:
                raise EOFError("%s: the header is cut short" % path)
            return struct.unpack(fmt, b)[0]

        def s():
            return f.read(u("<Q")).decode("utf-8", "replace")

        def val(t):
            if t in SCALAR:
                return u(SCALAR[t])
            if t == 8:
                return s()
            if t == 9:
                et, n = u("<I"), u("<Q")
                return [val(et) for _ in range(n)]
            raise ValueError("unknown metadata type %d" % t)

        if f.read(4) != b"GGUF":
            raise ValueError("%s is not a GGUF file" % path)
        version, nt, nkv = u("<I"), u("<Q"), u("<Q")
        kv = {}
        for _ in range(nkv):
            k = s()
            kv[k] = val(u("<I"))
        tensors = []
        for _ in range(nt):
            name = s()
            nd = u("<I")
            ne = [u("<Q") for _ in range(nd)]
            tensors.append({"name": name, "ne": ne, "type": u("<I"), "offset": u("<Q")})
        align = kv.get("general.alignment", 32)
        start = (f.tell() + align - 1) // align * align
    return version, kv, tensors, start


def prod(ne):
    p = 1
    for d in ne:
        p *= d
    return p


def strip(n):
    return n[len(PREFIX):] if n.startswith(PREFIX) else n


def group(n):
    n = strip(n)
    if ".attn.to_" in n:
        return "attn"
    if ".img_mlp.out" in n:
        return "mlp.out"
    if ".img_mlp." in n:
        return "mlp.in"
    if re.search(r"(norm_q|norm_k|text_norm)\.", n):
        return "norm"
    if n.startswith("modulation"):
        return "modulation"
    if n.startswith(("img_in", "txt_in")):
        return "input"
    return "edge"


def nbytes(t):
    name, blk, sz = TYPES.get(t["type"], ("T%d" % t["type"], 1, 0))
    return prod(t["ne"]) // blk * sz if sz else None


def ranges(ids):
    out, run = [], []
    for i in sorted(ids):
        if run and i == run[-1] + 1:
            run.append(i)
        else:
            if run:
                out.append(str(run[0]) if len(run) == 1 else "%d-%d" % (run[0], run[-1]))
            run = [i]
    if run:
        out.append(str(run[0]) if len(run) == 1 else "%d-%d" % (run[0], run[-1]))
    return ",".join(out)


def parse_rules(text):
    rules = []
    for r in (text or "").split(","):
        if r.strip():
            rx, t = r.rsplit("=", 1)
            rules.append((re.compile(rx), t.strip()))
    return rules


def reason(row, req):
    s = strip(row["name"])
    if any(e in s for e in EXCLUDED):
        return "sd.cpp never converts this name"
    if len(row["shape"]) == 1:
        return "1-D"
    blk = BY_NAME.get(req.lower(), (None, 1, 0))[1]
    if row["shape"][0] % blk:
        return "row of %d is not a multiple of %d" % (row["shape"][0], blk)
    if not s.startswith("transformer_blocks."):
        return "outside the transformer blocks"
    return "unexplained: check the rule"


def layout(a):
    version, kv, ts, start = read_header(a.file)
    rows = []
    for t in ts:
        rows.append({"name": t["name"], "shape": t["ne"], "type": TYPES.get(t["type"], ("T%d" % t["type"],))[0],
                     "params": prod(t["ne"]), "bytes": nbytes(t), "group": group(t["name"])})
    params = sum(r["params"] for r in rows)
    tbytes = sum(r["bytes"] or 0 for r in rows)
    prefixed = sum(1 for r in rows if r["name"].startswith(PREFIX))
    L = []
    L.append("%s  GGUF v%d, %d tensors, %s, %d metadata keys" % (
        os.path.basename(a.file), version, len(rows),
        "names prefixed model.diffusion_model." if prefixed == len(rows) else
        ("names without a prefix" if prefixed == 0 else "%d of %d names prefixed" % (prefixed, len(rows))), len(kv)))
    L.append("tensor data %.4f GB, %.1f M weights, %.3f bits per weight" % (tbytes / 1e9, params / 1e6, 8.0 * tbytes / params))
    L.append("")
    L.append("%-11s %10s %7s %9s %7s   %s" % ("group", "weights", "share", "GB", "bpw", "types"))
    groups = collections.OrderedDict((g, [0, 0, collections.Counter()]) for g in GROUPS)
    for r in rows:
        e = groups[r["group"]]
        e[0] += r["params"]; e[1] += r["bytes"] or 0; e[2][r["type"]] += 1
    gjson = {}
    for g, (p, b, c) in groups.items():
        if not p:
            continue
        gjson[g] = {"params": p, "bytes": b, "bpw": 8.0 * b / p, "types": dict(c)}
        L.append("%-11s %9.1fM %6.1f%% %9.4f %7.3f   %s" % (g, p / 1e6, 100.0 * p / params, b / 1e9, 8.0 * b / p,
                                                    ", ".join("%s x%d" % kv_ for kv_ in c.most_common())))
    blocks = collections.defaultdict(dict)
    for r in rows:
        m = re.search(r"transformer_blocks\.(\d+)\.(attn\.to_q|attn\.to_k|attn\.to_v|attn\.to_out\.0|img_mlp\.gate_up|img_mlp\.proj|img_mlp\.gate_layer|img_mlp\.out)\.", r["name"])
        if m:
            blocks[int(m.group(1))][m.group(2)] = r["type"]
    if blocks:
        cols = [c for c in ("attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0", "img_mlp.gate_up", "img_mlp.proj",
                            "img_mlp.gate_layer", "img_mlp.out") if any(c in v for v in blocks.values())]
        pats = collections.defaultdict(list)
        for b, v in blocks.items():
            pats[tuple(v.get(c, "-") for c in cols)].append(b)
        L.append("")
        L.append("per block      " + " ".join("%-10s" % c.split(".", 1)[1] for c in cols))
        for pat, ids in sorted(pats.items(), key=lambda kv_: min(kv_[1])):
            L.append("  %2d blocks    %s   %s" % (len(ids), " ".join("%-10s" % x for x in pat), ranges(ids)))
    L.append("")
    L.append("outside the blocks")
    for r in rows:
        if not strip(r["name"]).startswith("transformer_blocks."):
            L.append("  %-52s %-16s %s" % (strip(r["name"]), "x".join(str(d) for d in r["shape"]), r["type"]))
    expect = None
    if a.expect_default:
        rules = parse_rules(a.expect_rules)
        mism = []
        for r in rows:
            req, via = a.expect_default, "--type"
            for rx, rt in rules:
                if rx.search(PREFIX + strip(r["name"])):
                    req, via = rt, "rule " + rx.pattern
                    break
            if r["type"].lower() != req.lower():
                mism.append({"name": r["name"], "requested": req, "via": via, "got": r["type"], "why": reason(r, req)})
        expect = {"default": a.expect_default, "rules": a.expect_rules or "", "mismatches": mism}
        L.append("")
        L.append("requested: --type %s%s" % (a.expect_default, " and %d rule(s)" % len(rules) if rules else ""))
        if not mism:
            L.append("  every tensor has the type it was asked for")
        else:
            agg = collections.OrderedDict()
            for m in mism:
                agg.setdefault((m["requested"], m["via"], m["got"], m["why"]), []).append(strip(m["name"]))
            for (req, via, got, why), names in agg.items():
                L.append("  %3d asked %s (%s), got %s: %s   e.g. %s" % (len(names), req, via, got, why, names[0]))
    text = "\n".join(L)
    print(text)
    if a.txt:
        open(a.txt, "w").write(text + "\n")
    if a.json:
        json.dump({"file": os.path.abspath(a.file), "file_bytes": os.path.getsize(a.file), "gguf_version": version,
                   "tensor_bytes": tbytes, "params": params, "bpw": 8.0 * tbytes / params,
                   "prefixed_names": prefixed, "metadata": {k: (v if not isinstance(v, list) else "[%d]" % len(v)) for k, v in kv.items()},
                   "groups": gjson, "tensors": rows, "expect": expect}, open(a.json, "w"), indent=1)


def names(a):
    _, _, ta, _ = read_header(a.file)
    _, _, tb, _ = read_header(a.names)
    for path, ts in ((a.file, ta), (a.names, tb)):
        p = sum(1 for t in ts if t["name"].startswith(PREFIX))
        print("%-40s %d tensors, %s, first: %s" % (os.path.basename(path), len(ts),
              "all prefixed" if p == len(ts) else ("no prefix" if p == 0 else "%d prefixed" % p), ts[0]["name"] if ts else "-"))
    na, nb = {strip(t["name"]) for t in ta}, {strip(t["name"]) for t in tb}
    only_a, only_b = sorted(na - nb), sorted(nb - na)
    if not only_a and not only_b:
        print("same tensor names once the prefix is set aside")
    for label, lst in (("only in " + os.path.basename(a.file), only_a), ("only in " + os.path.basename(a.names), only_b)):
        if lst:
            print("%s: %d, e.g. %s" % (label, len(lst), ", ".join(lst[:5])))


def same(a):
    _, _, ta, sa = read_header(a.file)
    _, _, tb, sb = read_header(a.same)
    ma, mb = {strip(t["name"]): t for t in ta}, {strip(t["name"]): t for t in tb}
    common = sorted(set(ma) & set(mb))

    def digest(path, start, t):
        n = nbytes(t)
        h = hashlib.sha1()
        with open(path, "rb") as f:
            f.seek(start + t["offset"])
            left = n
            while left:
                b = f.read(min(left, 1 << 24))
                if not b:
                    raise EOFError("%s ends inside %s" % (path, t["name"]))
                h.update(b)
                left -= len(b)
        return h.hexdigest()

    diffs = []
    try:
        pairs = [(name, ma[name], mb[name]) for name in common]
        for name, x, y in pairs:
            if x["type"] != y["type"] or x["ne"] != y["ne"]:
                diffs.append((name, "%s against %s" % (TYPES.get(x["type"], ("?",))[0], TYPES.get(y["type"], ("?",))[0])))
            elif digest(a.file, sa, x) != digest(a.same, sb, y):
                diffs.append((name, "same type, different bytes"))
    except EOFError as e:
        sys.exit("cannot compare the weights: %s (a header-only file?)" % e)
    print("%d tensors in common, %d identical, %d differ; %d only in the first, %d only in the second" % (
        len(common), len(common) - len(diffs), len(diffs), len(set(ma) - set(mb)), len(set(mb) - set(ma))))
    for name, why in diffs[:10]:
        print("  %-52s %s" % (name, why))
    ok = not diffs and len(common) == len(ma) == len(mb)
    print("the same weights" if ok else "not the same weights")
    sys.exit(0 if ok else 3)


ap = argparse.ArgumentParser()
ap.add_argument("file")
ap.add_argument("--json"); ap.add_argument("--txt")
ap.add_argument("--expect-default"); ap.add_argument("--expect-rules", default="")
ap.add_argument("--names"); ap.add_argument("--same")
a = ap.parse_args()
if a.names:
    names(a)
elif a.same:
    same(a)
else:
    layout(a)
LAYEOF

cat > "$IMG_TOOLS/img_bits.py" << 'BITSEOF'
"""Predicted size of a layout from the bf16 safetensors header alone. Nothing is built.

    python3 img_bits.py DIT.safetensors --default TYPE [--rules RULES] [--target GB] [--label NAME] [--brief]

Works on the full 14 GB file or on the 31 KB header img_get header fetches.
It predicts what sd-cli -M convert writes, and the prediction rests on one
observation and one reading of the source:

  - leejet's files, made with --type alone, quantize every 2-D tensor inside
    transformer_blocks and leave everything else at bf16. This script
    reproduces the tensor bytes of their Q2_K, Q4_K, Q6_K and Q8_0 exactly;
  - src/model_loader.cpp keeps a tensor at its source type when its name holds
    img_in. or txt_in. (and a few other names), or when its row length is not
    a multiple of the target type's block.

A rule is assumed to override the first and not the second. img_layout on the
real file is the check. Sizes are tensor data in GB of 10^9 bytes, which is
what the rival column of IMG_AD_TABLE holds too.
"""
import argparse, collections, json, re, struct, sys

Q = {"f32": (1, 4), "f16": (1, 2), "bf16": (1, 2), "q8_0": (32, 34), "q6_k": (256, 210), "q5_k": (256, 176),
     "q4_k": (256, 144), "q3_k": (256, 110), "q2_k": (256, 84), "q5_0": (32, 22), "q5_1": (32, 24),
     "q4_0": (32, 18), "q4_1": (32, 20)}
SRC = {"BF16": "bf16", "F16": "f16", "F32": "f32"}
PREFIX = "model.diffusion_model."
EXCLUDED = ("img_in.", "txt_in.", "time_in.", "vector_in.", "guidance_in.", "final_layer.", "x_embedder.",
            "t_embedder.", "y_embedder.", "pos_embed", "context_embedder.", "time_embed.", "label_emb.", "embedding")
GROUPS = ["attn", "mlp.in", "mlp.out", "modulation", "edge", "input", "norm"]
SHOW = {"q6_k": "q6_K", "q5_k": "q5_K", "q4_k": "q4_K", "q3_k": "q3_K", "q2_k": "q2_K"}


def group(n):
    if ".attn.to_" in n:
        return "attn"
    if ".img_mlp.out" in n:
        return "mlp.out"
    if ".img_mlp." in n:
        return "mlp.in"
    if re.search(r"(norm_q|norm_k|text_norm)\.", n):
        return "norm"
    if n.startswith("modulation"):
        return "modulation"
    if n.startswith(("img_in", "txt_in")):
        return "input"
    return "edge"


ap = argparse.ArgumentParser()
ap.add_argument("file")
ap.add_argument("--default", required=True)
ap.add_argument("--rules", default="")
ap.add_argument("--target", type=float)
ap.add_argument("--label", default="")
ap.add_argument("--brief", action="store_true")
a = ap.parse_args()

raw = open(a.file, "rb").read(8)
n = struct.unpack("<Q", raw)[0]
with open(a.file, "rb") as f:
    f.seek(8)
    h = json.loads(f.read(n))
h.pop("__metadata__", None)
default = a.default.lower()
rules = []
for r in a.rules.split(","):
    if r.strip():
        rx, t = r.rsplit("=", 1)
        rules.append((re.compile(rx), t.strip().lower()))
for t in [default] + [t for _, t in rules]:
    if t not in Q:
        sys.exit("unknown type %s (known: %s)" % (t, " ".join(sorted(Q))))

groups = collections.OrderedDict((g, [0, 0, collections.Counter()]) for g in GROUPS)
fallback = collections.Counter()
for name, info in sorted(h.items()):
    shape = info["shape"]
    p = 1
    for d in shape:
        p *= d
    src = SRC[info["dtype"]]
    t = None
    for rx, rt in rules:
        if rx.search(PREFIX + name):
            t = rt
            break
    if t is None:
        t = default if name.startswith("transformer_blocks.") and len(shape) >= 2 else src
    if any(e in name for e in EXCLUDED) and t != src:
        fallback["%s: name excluded" % name.split(".")[0]] += 1
        t = src
    blk, sz = Q[t]
    if blk > 1 and shape[-1] % blk:
        fallback["row of %d, %s block %d" % (shape[-1], t, blk)] += 1
        t = src
        blk, sz = Q[t]
    e = groups[group(name)]
    e[0] += p
    e[1] += p // blk * sz
    e[2][SHOW.get(t, t)] += 1

params = sum(v[0] for v in groups.values())
total = sum(v[1] for v in groups.values())
gb = total / 1e9
delta = "" if not a.target else "  rival %.4f GB  %+.2f %%" % (a.target, 100.0 * (gb / a.target - 1))
if a.brief:
    print("%-16s %8.4f GB  %6.3f bpw%s" % (a.label or a.default, gb, 8.0 * total / params, delta))
    sys.exit(0)
print("%s: --type %s%s, predicted from %s" % (a.label or a.default, a.default,
      " and %d rule(s)" % len(rules) if rules else "", a.file))
print("%-11s %10s %7s %9s %7s   %s" % ("group", "weights", "share", "GB", "bpw", "types"))
for g, (p, b, c) in groups.items():
    if p:
        print("%-11s %9.1fM %6.1f%% %9.4f %7.3f   %s" % (g, p / 1e6, 100.0 * p / params, b / 1e9, 8.0 * b / p,
                                                   ", ".join("%s x%d" % kv for kv in c.most_common())))
print("%-11s %9.1fM %7s %9.4f %7.3f%s" % ("total", params / 1e6, "", gb, 8.0 * total / params, delta))
for why, k in fallback.items():
    print("  kept at the source type: %d tensor(s), %s" % (k, why))
BITSEOF

cat > "$IMG_TOOLS/img_imatrix.py" << 'IMATEOF'
"""What an sd.cpp importance matrix covers.

    python3 img_imatrix.py imat.dat

sd.cpp's own format (src/runtime/imatrix.cpp): an int32 entry count, then per
entry an int32 name length, the name, an int32 call count, an int32 value count
and that many float32 (the mean squared activation of each input column, times
the call count), and at the end an int32 with the number of calls. sd.cpp
drops entries with partial data when it saves, so a block tensor missing here
is one the renders never exercised completely, and a quant that asks for it
gets no importance weighting there. Exits 1 when a block linear is missing.
"""
import collections, re, struct, sys

PREFIX = "model.diffusion_model."
data = open(sys.argv[1], "rb").read()
pos = 0


def take(fmt):
    global pos
    v = struct.unpack_from(fmt, data, pos)[0]
    pos += struct.calcsize(fmt)
    return v


entries = {}
for _ in range(take("<i")):
    ln = take("<i")
    name = data[pos:pos + ln].decode("utf-8", "replace")
    pos += ln
    ncall, nval = take("<i"), take("<i")
    vals = struct.unpack_from("<%df" % nval, data, pos)
    pos += 4 * nval
    entries[name] = (ncall, vals)
last = take("<i") if pos + 4 <= len(data) else None


def short(n):
    n = n[len(PREFIX):] if n.startswith(PREFIX) else n
    return n[:-7] if n.endswith(".weight") else n


calls = [v[0] for v in entries.values()]
print("%s: %d entries, calls per entry %d to %d, last call count %s" % (
    sys.argv[1], len(entries), min(calls or [0]), max(calls or [0]), last))
if entries:
    print("first name: %s" % next(iter(entries)))
groups = collections.defaultdict(list)
for name, (ncall, vals) in entries.items():
    s = short(name)
    g = "attn" if ".attn.to_" in s else ("mlp.out" if ".img_mlp.out" in s else ("mlp.in" if ".img_mlp." in s else "other"))
    groups[g].append(sum(vals) / len(vals) / max(ncall, 1))
for g in ("attn", "mlp.in", "mlp.out", "other"):
    if groups[g]:
        v = sorted(groups[g])
        print("  %-8s %4d entries   mean activation^2 per column: median %.4g, min %.4g, max %.4g" % (
            g, len(v), v[len(v) // 2], v[0], v[-1]))
have = {short(n) for n in entries}
missing = []
for b in range(32):
    p = "transformer_blocks.%d." % b
    for t in ("attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0", "img_mlp.out"):
        if p + t not in have:
            missing.append(p + t)
    if p + "img_mlp.gate_up" not in have and not (p + "img_mlp.proj" in have and p + "img_mlp.gate_layer" in have):
        missing.append(p + "img_mlp.gate_up")
outside = sorted(s for s in have if not s.startswith("transformer_blocks."))
print("outside the blocks: %s" % (", ".join(outside) or "none"))
if missing:
    print("MISSING %d block linear(s), e.g. %s" % (len(missing), ", ".join(missing[:5])))
    sys.exit(1)
print("every block linear of the 32 blocks is covered")
IMATEOF

cat > "$IMG_TOOLS/img_metrics.py" << 'METEOF'
"""Perceptual distance of one build's renders from the reference renders.

    python3 img_metrics.py REF_DIR CAND_DIR OUT.json --prompts eval_prompts.txt
                           [--expect N] [--pair-seeds] [--boot 20000] [--device auto]
                           [--protocol STR] [--model PATH] [--name BUILD]

Renders pair by file name, <pid>-s<seed>.png. Every candidate needs its
reference, and with --expect the candidate must have exactly that many, so a
half-rendered build cannot produce a number. --pair-seeds pairs instead, in
REF_DIR alone, the lowest seed of every prompt with the next: floor (b), what a
different picture costs.

Per pair: LPIPS (VGG, the lpips package, inputs scaled to [-1, 1]), SSIM
(scikit-image, over the three channels, data range 255) and PSNR (100 dB for
identical images). RGBA renders are composited over grey 128 first, and the
mean absolute alpha difference is reported beside them. Byte-identical pairs
are counted and scored 0 without running the network.

Uncertainty is a bootstrap over prompts, not images: the two seeds of a prompt
share the prompt, they are not two independent draws. Each prompt's value is
the mean over its seeds; the interval is the 95 % percentile interval of the
mean over prompts drawn with replacement.
"""
import argparse, glob, json, os, re, sys
import numpy as np
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("ref_dir"); ap.add_argument("cand_dir"); ap.add_argument("out")
ap.add_argument("--prompts", required=True)
ap.add_argument("--expect", type=int, default=0)
ap.add_argument("--pair-seeds", action="store_true")
ap.add_argument("--boot", type=int, default=20000)
ap.add_argument("--device", default="auto")
ap.add_argument("--protocol", default="")
ap.add_argument("--model", default="")
ap.add_argument("--name", default="")
a = ap.parse_args()

cats = {}
for line in open(a.prompts, encoding="utf-8"):
    line = line.rstrip("\n")
    if line and not line.startswith("#"):
        pid, cat, _ = line.split("|", 2)
        cats[pid] = cat

NAME = re.compile(r"^([a-z]\d+)-s(\d+)\.png$")


def listing(d):
    out = {}
    for p in glob.glob(os.path.join(d, "*.png")):
        m = NAME.match(os.path.basename(p))
        if m:
            out[(m.group(1), int(m.group(2)))] = p
    return out


ref = listing(a.ref_dir)
if a.pair_seeds:
    by = {}
    for (pid, seed), p in ref.items():
        by.setdefault(pid, {})[seed] = p
    pairs = []
    for pid in sorted(by):
        s = sorted(by[pid])
        if len(s) >= 2:
            pairs.append((pid, s[0], s[1], by[pid][s[0]], by[pid][s[1]]))
else:
    cand = listing(a.cand_dir)
    missing = sorted(k for k in cand if k not in ref)
    if missing:
        sys.exit("%d render(s) without a reference, first %s-s%d: render the reference first" % (len(missing), missing[0][0], missing[0][1]))
    pairs = [(pid, seed, seed, ref[(pid, seed)], p) for (pid, seed), p in sorted(cand.items())]
if not pairs:
    sys.exit("nothing to compare in %s" % a.cand_dir)
if a.expect and len(pairs) != a.expect:
    sys.exit("%d pairs where the protocol expects %d: finish the renders, or set IMG_EVAL_N and IMG_SEEDS to what was rendered" % (len(pairs), a.expect))
unknown = sorted({p[0] for p in pairs} - set(cats))
if unknown:
    sys.exit("prompt ids not in %s: %s" % (a.prompts, ", ".join(unknown[:5])))

import torch
import lpips
from skimage.metrics import structural_similarity

dev = a.device
if dev == "auto":
    if torch.cuda.is_available():
        dev = "cuda"
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        dev = "mps"
    else:
        dev = "cpu"
net = lpips.LPIPS(net="vgg", verbose=False).to(dev).eval()


def load(path):
    im = Image.open(path)
    im.load()
    alpha = None
    if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
        rgba = im.convert("RGBA")
        alpha = np.asarray(rgba)[..., 3].astype(np.float64)
        im = Image.alpha_composite(Image.new("RGBA", rgba.size, (128, 128, 128, 255)), rgba)
    return np.asarray(im.convert("RGB")), alpha


def tensor(x):
    return torch.from_numpy(np.ascontiguousarray(x)).permute(2, 0, 1)[None].float().div(127.5).sub(1.0).to(dev)


rows = []
with torch.no_grad():
    for pid, sa, sb, pa, pb in pairs:
        x, ax = load(pa)
        y, ay = load(pb)
        if x.shape != y.shape:
            sys.exit("%s: %s against %s, different sizes" % (os.path.basename(pb), x.shape, y.shape))
        same_alpha = (ax is None and ay is None) or (ax is not None and ay is not None and np.array_equal(ax, ay))
        ident = bool(np.array_equal(x, y) and same_alpha)
        if ident:
            lp, ss = 0.0, 1.0
        else:
            lp = float(net(tensor(x), tensor(y)).item())
            ss = float(structural_similarity(x, y, channel_axis=2, data_range=255))
        mse = float(np.mean((x.astype(np.float64) - y.astype(np.float64)) ** 2))
        ps = 100.0 if mse == 0 else float(min(100.0, 10.0 * np.log10(255.0 ** 2 / mse)))
        row = {"pid": pid, "seed": sa, "seed_b": sb, "category": cats[pid], "lpips": lp, "ssim": ss, "psnr": ps, "identical": ident}
        if ax is not None or ay is not None:
            row["alpha"] = "both" if (ax is not None and ay is not None) else ("reference only" if ax is not None else "candidate only")
            if ax is not None and ay is not None:
                row["alpha_mae"] = float(np.mean(np.abs(ax - ay)) / 255.0)
        rows.append(row)
        print("  %-10s LPIPS %.4f  SSIM %.4f  PSNR %6.2f%s" % ("%s-s%d" % (pid, sb), lp, ss, ps, "  identical" if ident else ""), flush=True)

pids = sorted({r["pid"] for r in rows})
per = {}
for pid in pids:
    rs = [r for r in rows if r["pid"] == pid]
    per[pid] = {"category": cats[pid], "images": len(rs)}
    for k in ("lpips", "ssim", "psnr"):
        per[pid][k] = float(np.mean([r[k] for r in rs]))
L = np.array([per[p]["lpips"] for p in pids])
S = np.array([per[p]["ssim"] for p in pids])
P = np.array([per[p]["psnr"] for p in pids])
rng = np.random.default_rng(0)
idx = rng.integers(0, len(pids), (a.boot, len(pids)))


def ci(v):
    b = v[idx].mean(1)
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


img = np.array([r["lpips"] for r in rows])
worst = max(rows, key=lambda r: r["lpips"])
am = [r["alpha_mae"] for r in rows if "alpha_mae" in r]
summary = {
    "images": len(rows), "prompts": len(pids), "identical": int(sum(r["identical"] for r in rows)),
    "lpips_mean": float(L.mean()), "lpips_ci95": ci(L), "lpips_median": float(np.median(img)),
    "lpips_p95": float(np.percentile(img, 95)), "lpips_max": float(img.max()),
    "lpips_max_image": "%s-s%d" % (worst["pid"], worst["seed_b"]),
    "ssim_mean": float(S.mean()), "ssim_ci95": ci(S), "psnr_mean": float(P.mean()),
    "alpha_images": sum(1 for r in rows if "alpha" in r),
    "alpha_mismatch": sum(1 for r in rows if r.get("alpha") in ("reference only", "candidate only")),
    "alpha_mae_mean": float(np.mean(am)) if am else None,
}
by_cat = {}
for c in sorted({cats[p] for p in pids}):
    cp = [p for p in pids if cats[p] == c]
    by_cat[c] = {"prompts": len(cp), "lpips": float(np.mean([per[p]["lpips"] for p in cp])),
                 "ssim": float(np.mean([per[p]["ssim"] for p in cp]))}
res = {"name": a.name or os.path.basename(os.path.normpath(a.cand_dir)),
       "reference": os.path.basename(os.path.normpath(a.ref_dir)), "pair_seeds": a.pair_seeds,
       "protocol": a.protocol, "model": a.model,
       "model_bytes": os.path.getsize(a.model) if a.model and os.path.exists(a.model) else None,
       "lpips_net": "vgg", "device": dev, "bootstrap": {"unit": "prompt", "resamples": a.boot, "seed": 0},
       "summary": summary, "by_category": by_cat, "per_prompt": per, "per_image": rows}
json.dump(res, open(a.out, "w"), indent=1)
s = summary
print()
print("%s against %s: %d images, %d prompts%s" % (res["name"], res["reference"], s["images"], s["prompts"],
      ", seeds paired" if a.pair_seeds else ""))
print("LPIPS  mean %.4f  95%% CI [%.4f, %.4f]   median %.4f  p95 %.4f  max %.4f (%s)" % (
    s["lpips_mean"], s["lpips_ci95"][0], s["lpips_ci95"][1], s["lpips_median"], s["lpips_p95"], s["lpips_max"], s["lpips_max_image"]))
print("SSIM   mean %.4f  95%% CI [%.4f, %.4f]" % (s["ssim_mean"], s["ssim_ci95"][0], s["ssim_ci95"][1]))
print("PSNR   mean %.2f dB" % s["psnr_mean"])
print("identical %d of %d" % (s["identical"], s["images"]))
if s["alpha_images"]:
    print("alpha: %d renders carry it, %d pairs where only one side does, mean |d alpha| %s" % (
        s["alpha_images"], s["alpha_mismatch"], "%.4f" % s["alpha_mae_mean"] if s["alpha_mae_mean"] is not None else "-"))
print("by category, LPIPS: " + "  ".join("%s %.4f" % (c, v["lpips"]) for c, v in by_cat.items()))
print("written to %s" % a.out)
METEOF

cat > "$IMG_TOOLS/img_compare.py" << 'CMPEOF'
"""Paired difference of two builds measured against the same reference.

    python3 img_compare.py A.json B.json [--boot 20000]

Both from img_metrics.py: same reference, same protocol, same prompts. Per
prompt d = A - B (each prompt's value is already the mean over its seeds),
bootstrap over prompts, 95 % percentile interval, and the share of resamples
in which A is the closer one. An interval that contains zero means no
convincing difference was found, not that the two are equivalent.
"""
import argparse, json, sys
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("a"); ap.add_argument("b")
ap.add_argument("--boot", type=int, default=20000)
args = ap.parse_args()
A, B = json.load(open(args.a)), json.load(open(args.b))
problems = []
if A.get("pair_seeds") != B.get("pair_seeds"):
    problems.append("one of them is a seed floor (seed against seed inside the reference), the other a build")
if A["reference"] != B["reference"]:
    problems.append("references differ: %s and %s" % (A["reference"], B["reference"]))
if A.get("protocol") and B.get("protocol") and A["protocol"] != B["protocol"]:
    problems.append("protocols differ:\n    A %s\n    B %s" % (A["protocol"], B["protocol"]))
pa, pb = A["per_prompt"], B["per_prompt"]
if set(pa) != set(pb):
    problems.append("prompt sets differ (%d and %d prompts)" % (len(pa), len(pb)))
if problems:
    print("not comparable:")
    for p in problems:
        print("  " + p)
    sys.exit(1)
pids = sorted(pa)
rng = np.random.default_rng(0)
idx = rng.integers(0, len(pids), (args.boot, len(pids)))
print("A = %s   B = %s   against %s, %d prompts" % (A["name"], B["name"], A["reference"], len(pids)))
verdict = None
for key, label, lower in (("lpips", "LPIPS", True), ("ssim", "SSIM", False), ("psnr", "PSNR dB", False)):
    d = np.array([pa[p][key] - pb[p][key] for p in pids])
    s = d[idx].mean(1)
    lo, hi = np.percentile(s, [2.5, 97.5])
    closer = (s < 0).mean() if lower else (s > 0).mean()
    print("%-8s A %.4f  B %.4f   A - B %+.5f  95%% CI [%+.5f, %+.5f]   P(A closer) %.3f" % (
        label, np.mean([pa[p][key] for p in pids]), np.mean([pb[p][key] for p in pids]), d.mean(), lo, hi, closer))
    if key == "lpips":
        verdict = ("A is closer to the reference" if hi < 0 else
                   "B is closer to the reference" if lo > 0 else
                   "no convincing difference: the interval contains zero")
print("LPIPS A - B by category:")
for c in sorted({v["category"] for v in pa.values()}):
    cp = [p for p in pids if pa[p]["category"] == c]
    d = np.array([pa[p]["lpips"] - pb[p]["lpips"] for p in cp])
    print("  %-8s %+.5f over %d prompts" % (c, d.mean(), len(cp)))
print(verdict)
CMPEOF

cat > "$IMG_TOOLS/img_table.py" << 'TBLEOF'
"""Everything measured on this box, one row per build.

    python3 img_table.py LOGS LAYOUTS DIT.safetensors OUT.json OUT.md

Rows come from logs/metrics-*.json; the scan is left out, it is a ranking at a
reduced protocol. The "by" column: ours (AD layouts), defaults (--type alone,
the tool's layout, made here), ablation (one lever pulled), floor, or the org
that published the file. Size and bits per weight are tensor data from
layouts/layout-<build>.json; the reference rows use the bf16 file. Timings are
medians over the last successful render of every image in logs/gen-<build>.jsonl.
A row measured on a different number of images than the full protocol is a
check, listed apart and never mixed into the table.
"""
import glob, json, os, statistics, sys

logs, layouts, src, out_json, out_md = sys.argv[1:6]


def load(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


seed = load(os.path.join(logs, "metrics-ref-seed.json"))
seed_floor = seed["summary"]["lpips_mean"] if seed else None
rows = []
for p in sorted(glob.glob(os.path.join(logs, "metrics-*.json"))):
    b = os.path.basename(p)[len("metrics-"):-len(".json")]
    if b.startswith("scan-"):
        continue
    m = load(p)
    if not m or "summary" not in m:
        continue
    s = m["summary"]
    r = {"build": b, "images": s["images"], "prompts": s["prompts"], "protocol": m.get("protocol", ""),
         "pair_seeds": m.get("pair_seeds", False), "identical": s["identical"],
         "lpips_mean": s["lpips_mean"], "lpips_ci95": s["lpips_ci95"], "lpips_p95": s["lpips_p95"],
         "lpips_max": s["lpips_max"], "lpips_max_image": s.get("lpips_max_image"),
         "ssim_mean": s["ssim_mean"], "psnr_mean": s["psnr_mean"],
         "by_category": {c: v["lpips"] for c, v in m.get("by_category", {}).items()}}
    if b in ("ref-repeat", "ref-seed"):
        r["publisher"] = "floor"
    elif "--" in b:
        r["publisher"] = b.split("--")[0]
    elif b.endswith(("-imat", "-noimat")):
        r["publisher"] = "ablation"
    elif b.startswith("AD-"):
        r["publisher"] = "ours"
    else:
        r["publisher"] = "defaults"   # --type alone: the tool's layout, made here
    lay = load(os.path.join(layouts, "layout-%s.json" % b))
    if lay:
        r["gb"], r["bpw"] = lay["tensor_bytes"] / 1e9, lay["bpw"]
    elif b.startswith("ref") and os.path.exists(src):
        r["gb"], r["bpw"] = os.path.getsize(src) / 1e9, 16.0
    else:
        r["gb"] = r["bpw"] = None
    last = {}
    jl = os.path.join(logs, "gen-%s.jsonl" % b)
    if os.path.exists(jl):
        for line in open(jl):
            try:
                e = json.loads(line)
            except ValueError:
                continue
            if e.get("rc") == 0:
                last[(e["pid"], e["seed"])] = e
    wall = [e["wall_s"] for e in last.values() if e.get("wall_s") is not None]
    samp = [e["sampling_s"] for e in last.values() if e.get("sampling_s") is not None]
    r["s_per_image"] = statistics.median(wall) if wall else None
    r["sampling_s"] = statistics.median(samp) if samp else None
    r["rel_seed_floor"] = r["lpips_mean"] / seed_floor if seed_floor else None
    rows.append(r)

if not rows:
    print("nothing measured on this box yet")
    sys.exit(0)
full = max([r["images"] for r in rows if not r["pair_seeds"]] or [0])
for r in rows:
    r["check"] = (not r["pair_seeds"]) and r["images"] != full
main = [r for r in rows if not r["check"]]
checks = [r for r in rows if r["check"]]
order = {"ref-repeat": 0, "ref-seed": 1}
main.sort(key=lambda r: (order.get(r["build"], 2), -(r["gb"] or 0), r["build"]))
json.dump({"seed_floor": seed_floor, "full_protocol_images": full, "rows": main, "checks": checks},
          open(out_json, "w"), indent=1)


def f(x, fmt):
    return "" if x is None else fmt % x


head = ("| build | by | GB | bpw | LPIPS [95 % CI] | p95 | max | × seed floor | SSIM | PSNR dB | text-en | text-zh | s/image | sampling s |\n"
        "|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
lines = [head]
for r in main:
    lines.append("| `%s` | %s | %s | %s | %.4f [%.4f, %.4f] | %.4f | %.4f | %s | %.4f | %.2f | %s | %s | %s | %s |" % (
        r["build"], r["publisher"], f(r["gb"], "%.2f"), f(r["bpw"], "%.2f"), r["lpips_mean"], r["lpips_ci95"][0],
        r["lpips_ci95"][1], r["lpips_p95"], r["lpips_max"], f(r["rel_seed_floor"], "%.3f"), r["ssim_mean"],
        r["psnr_mean"], f(r["by_category"].get("text-en"), "%.4f"), f(r["by_category"].get("text-zh"), "%.4f"),
        f(r["s_per_image"], "%.1f"), f(r["sampling_s"], "%.1f")))
md = ["LPIPS, SSIM and PSNR of every render against the bf16 render of the same prompt and seed; lower LPIPS "
      "is closer. Intervals: bootstrap over %d prompts. `ref-repeat` is a second bf16 render (floor a), "
      "`ref-seed` is seed 42 against seed 43 (floor b, a different picture)." % (main[0]["prompts"]), ""] + lines
protos = sorted({r["protocol"] for r in main if r["protocol"]})
if len(protos) > 1:
    md += ["", "> [!WARNING]", "> Rows measured under different protocols. Do not compare across them:"]
    md += ["> - `%s`" % p for p in protos]
elif protos:
    md += ["", "Protocol: `%s`" % protos[0]]
if checks:
    md += ["", "Checks at a partial protocol, not comparable with the table:", ""]
    md += ["- `%s`: %d images, LPIPS %.4f, identical %d" % (r["build"], r["images"], r["lpips_mean"], r["identical"]) for r in checks]
open(out_md, "w").write("\n".join(md) + "\n")
TBLEOF

cat > "$IMG_TOOLS/img_scan.py" << 'SCANEOF'
"""The sensitivity scan, ranked.

    python3 img_scan.py LOGS scan-groups.txt LAYOUTS

For every group: the bytes q2_K saved against the q8_0 base, measured from the
two layouts rather than predicted, the LPIPS it added over the base, and the
ratio, LPIPS added per GB saved. Ranked by that ratio, most sensitive first,
which is the order the AD layouts should spend bytes in. A ranking at a
reduced protocol, never a published number: the intervals are wide and the
resolution is a quarter of the real one.
"""
import json, os, sys

logs, groups_file, layouts = sys.argv[1:4]


def load(p):
    return json.load(open(p)) if os.path.exists(p) else None


groups = []
for line in open(groups_file):
    line = line.strip()
    if line:
        g, rx = line.split("|", 1)
        groups.append((g, rx))
base = load(os.path.join(logs, "metrics-scan-base.json"))
lbase = load(os.path.join(layouts, "layout-scan-base.json"))
if base is None or lbase is None:
    sys.exit("no scan base measured yet: run img_scan")
b0 = base["summary"]["lpips_mean"]
rows = []
for g, rx in groups:
    if not rx:
        continue
    m = load(os.path.join(logs, "metrics-scan-%s.json" % g))
    lay = load(os.path.join(layouts, "layout-scan-%s.json" % g))
    if m is None or lay is None:
        print("  %-12s not measured" % g)
        continue
    saved = (lbase["tensor_bytes"] - lay["tensor_bytes"]) / 1e9
    d = m["summary"]["lpips_mean"] - b0
    rows.append({"group": g, "regex": rx, "gb_saved": saved, "lpips": m["summary"]["lpips_mean"],
                 "lpips_ci95": m["summary"]["lpips_ci95"], "lpips_added": d,
                 "added_per_gb": d / saved if saved > 1e-4 else None})
rows.sort(key=lambda r: -(r["added_per_gb"] if r["added_per_gb"] is not None else -1e9))
print("base: the tool's q8_0, LPIPS %.4f against the bf16 render; %s" % (b0, base.get("protocol", "")))
print("%-12s %9s %9s %20s %9s %11s" % ("group", "GB saved", "LPIPS", "95 % CI", "added", "added / GB"))
for r in rows:
    print("%-12s %9.3f %9.4f   [%.4f, %.4f] %+9.4f %11s" % (
        r["group"], r["gb_saved"], r["lpips"], r["lpips_ci95"][0], r["lpips_ci95"][1], r["lpips_added"],
        "%.4f" % r["added_per_gb"] if r["added_per_gb"] is not None else "nothing saved"))
print("a group that saved nothing was not converted: check layouts/layout-scan-<group>.txt")
json.dump({"base_lpips": b0, "protocol": base.get("protocol"), "groups": rows},
          open(os.path.join(logs, "scan.json"), "w"), indent=1)
SCANEOF

cat > "$IMG_TOOLS/img_grid.py" << 'GRIDEOF'
"""Renders side by side, for the model card.

    python3 img_grid.py OUT.png --gen GEN_DIR --builds ref,Q8_0,AD-Q4_K --pids e03,e04 [--seed 42] [--size 320]
"""
import argparse, os
from PIL import Image, ImageDraw, ImageFont

ap = argparse.ArgumentParser()
ap.add_argument("out"); ap.add_argument("--gen", required=True)
ap.add_argument("--builds", required=True); ap.add_argument("--pids", required=True)
ap.add_argument("--seed", type=int, default=42); ap.add_argument("--size", type=int, default=320)
a = ap.parse_args()
builds = [b for b in a.builds.split(",") if b]
pids = [p for p in a.pids.split(",") if p]
W, top, gap = a.size, 36, 6
try:
    font = ImageFont.load_default(size=16)
except TypeError:
    font = ImageFont.load_default()


def label(b):
    if b == "ref":
        return "bf16 (reference)"
    if "--" in b:
        org, stem = b.split("--", 1)
        return "%s %s" % (org, stem.replace("qwen-image-2.1-", "").replace("qwen_image_2.1-", ""))
    return b


canvas = Image.new("RGB", (gap + len(builds) * (W + gap), top + len(pids) * (W + gap) + gap), (252, 252, 251))
d = ImageDraw.Draw(canvas)
for c, b in enumerate(builds):
    d.text((gap + c * (W + gap) + 4, 10), label(b), fill=(11, 11, 11), font=font)
for r, pid in enumerate(pids):
    for c, b in enumerate(builds):
        x, y = gap + c * (W + gap), top + r * (W + gap)
        p = os.path.join(a.gen, b, "%s-s%d.png" % (pid, a.seed))
        if not os.path.exists(p):
            d.rectangle([x, y, x + W, y + W], fill=(229, 228, 224))
            d.text((x + 8, y + 8), "missing", fill=(82, 81, 78), font=font)
            continue
        im = Image.open(p).convert("RGBA")
        im = Image.alpha_composite(Image.new("RGBA", im.size, (128, 128, 128, 255)), im).convert("RGB")
        canvas.paste(im.resize((W, W), Image.LANCZOS), (x, y))
canvas.save(a.out)
print("written %s: %d prompts x %d builds, seed %d" % (a.out, len(pids), len(builds), a.seed))
GRIDEOF

cat > "$IMG_TOOLS/img_chart.py" << 'CHARTEOF'
"""LPIPS against size, every file measured, for the model card.

    python3 img_chart.py results.json OUT.png

Colors are the first four slots of the validated categorical palette, in
fixed order: ours, unsloth, the tool's defaults made here, leejet. Ours filled
and joined, the rest hollow; our labels to the right of a point, everyone
else's to the left, because at 4.20 GB three files sit on one x.
The dashed line is floor (b), what a different seed costs; bars are the 95 %
interval over prompts. Log scale, because the ladder spans two decades.
"""
import json, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

res = json.load(open(sys.argv[1]))
out = sys.argv[2]
rows = [r for r in res["rows"] if r["publisher"] not in ("floor", "ablation") and r.get("gb") and r["lpips_mean"] > 0]
if not rows:
    sys.exit("nothing to plot yet")
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e5e4e0"
COLORS = {"ours": "#2a78d6", "unsloth": "#eb6834", "defaults": "#1baf7a", "leejet": "#eda100"}
NAMES = {"ours": "AtomicChat AD", "unsloth": "unsloth", "defaults": "sd.cpp --type alone, made here", "leejet": "leejet"}


def short(b):
    s = b.split("--", 1)[-1]
    return s.replace("qwen-image-2.1-", "").replace("qwen_image_2.1-", "")


fig, ax = plt.subplots(figsize=(9, 5.4), dpi=160)
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)
pubs = [p for p in ("ours", "unsloth", "defaults", "leejet") if any(r["publisher"] == p for r in rows)]
pubs += sorted({r["publisher"] for r in rows} - set(pubs))
for pub in pubs:
    rs = sorted([r for r in rows if r["publisher"] == pub], key=lambda r: r["gb"])
    col = COLORS.get(pub, INK2)
    xs = [r["gb"] for r in rs]
    ys = [r["lpips_mean"] for r in rs]
    err = [[y - r["lpips_ci95"][0] for y, r in zip(ys, rs)], [r["lpips_ci95"][1] - y for y, r in zip(ys, rs)]]
    if pub == "ours":
        ax.plot(xs, ys, color=col, lw=2, zorder=2)
    ax.errorbar(xs, ys, yerr=err, fmt="o", ms=8, color=col, mfc=col if pub == "ours" else SURFACE, mec=col,
                mew=2, elinewidth=1.2, capsize=0, zorder=3, label=NAMES.get(pub, pub))
    right = pub == "ours"
    for x, y, r in zip(xs, ys, rs):
        ax.annotate(short(r["build"]), (x, y), xytext=(7 if right else -7, 4), textcoords="offset points",
                    fontsize=7, color=INK2, ha="left" if right else "right")
floor = res.get("seed_floor")
if floor:
    ax.axhline(floor, color=INK2, lw=1, ls=(0, (4, 3)), zorder=1)
    ax.text(0.01, floor, " a different seed", transform=ax.get_yaxis_transform(), fontsize=7, color=INK2, va="bottom")
ax.set_yscale("log")
ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: "%g" % v))
ax.yaxis.set_minor_formatter(NullFormatter())
ax.set_xlabel("tensor data, GB", color=INK2, fontsize=9)
ax.set_ylabel("LPIPS to the bf16 render, lower is closer", color=INK2, fontsize=9)
ax.grid(True, which="major", color=GRID, lw=0.8)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color(GRID)
ax.tick_params(colors=INK2, labelsize=8)
leg = ax.legend(frameon=False, fontsize=8, loc="upper right")
for t in leg.get_texts():
    t.set_color(INK)
ax.set_title("Qwen-Image-2.1 denoiser GGUFs, every file measured the same way", loc="left", fontsize=11, color=INK)
fig.text(0.01, 0.01, "%s; bars: 95 %% interval over prompts" % rows[0].get("protocol", ""), fontsize=6, color=INK2)
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(out, facecolor=SURFACE)
print("written %s" % out)
CHARTEOF

echo "helpers written to $IMG_TOOLS"
}


# ================================================================== on source

if [ -n "$IMG_ROOT" ] && [ ! -d "$IMG_ROOT" ]; then
    mkdir -p "$IMG_ROOT" 2> /dev/null
fi
if mkdir -p "$IMG_TOOLS" 2> /dev/null && [ -w "$IMG_TOOLS" ]; then
    img_write_py > /dev/null
fi
echo "foundry-image $IMG_VERSION loaded. img_help for the list, img_box for the plan."
