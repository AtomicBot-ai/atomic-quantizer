# shellcheck shell=bash
# Sourced by every pipeline node. Runs on a rented box as root.
#
# Contract (see pipeline/README.md):
#   - input is environment variables, the HF token comes only from /root/.hf_env
#   - outputs go to Hugging Face; the hub is the only state between boxes
#   - a node whose outputs already exist exits at once unless FORCE=1
#   - the last line is "NODE_DONE <node> {json}" or "NODE_FAIL <node> <reason>"
#   - exit codes: 1 error, 2 model or toolchain not supported, 3 a check refused the result
#
# The commands are the ones scripts/foundry.sh used for the August releases;
# where one is lifted, the foundry function is named next to it.
set -euo pipefail

[ -f /root/.hf_env ] && . /root/.hf_env
: "${HF_TOKEN:?no HF_TOKEN: put export HF_TOKEN=... into /root/.hf_env}"
export HF_TOKEN HF_HUB_DISABLE_PROGRESS_BARS=1 HF_XET_HIGH_PERFORMANCE=1

NODE=${NODE:-$(basename "$0" .sh)}
PIPE=${PIPE:-/opt/pipeline}
: "${STEM:?STEM, e.g. Qwen3.8-27B}"
MAIN=${MAIN:-AtomicChat/$STEM-GGUF${REPO_SUFFIX:-}}
METRICS=${METRICS:-AtomicChat/$STEM-GGUF-metrics${REPO_SUFFIX:-}}
LLAMA_REPO=${LLAMA_REPO:-https://github.com/ggml-org/llama.cpp}
LLAMA_COMMIT=${LLAMA_COMMIT:?LLAMA_COMMIT, the llama.cpp commit every node of a run builds}
CALIB_REPO=${CALIB_REPO:-AtomicChat/calib-corpora}
CALIB_REV=${CALIB_REV:-814d662f6c94d207fc8f38545a1b4abea11484b5}
WORK=${WORK:-/work/$STEM}
FORCE=${FORCE:-0}
LOGS=$WORK/logs
mkdir -p "$WORK" "$LOGS"

T0=$(date +%s)
say() { echo "[$(date -u +%H:%M:%S) +$(( $(date +%s) - T0 ))s] $*"; }
FAILED=""
fail() { local code=$1; shift; FAILED=1; echo "NODE_FAIL $NODE $*"; exit "$code"; }

# the llama tools write to their own logs; every 5 minutes say which one moved and how far,
# so the driver can tell a long imatrix run from a hung box
heartbeat() {
    while sleep 300; do
        local f
        f=$(ls -t "$LOGS"/* 2>/dev/null | head -1 || true)
        [ -n "$f" ] && say "alive, $(basename "$f"): $(tail -c 300 "$f" | tr '\r' '\n' | grep -v '^\s*$' | tail -1 | cut -c1-140)"
    done
}
heartbeat & HEARTBEAT=$!
trap 'rc=$?; kill $HEARTBEAT 2>/dev/null; if [ $rc -ne 0 ] && [ -z "$FAILED" ]; then echo "NODE_FAIL $NODE unexpected exit $rc near line $LINENO"; fi' EXIT

VENV=/opt/venv
PY=$VENV/bin/python

# ------------------------------------------------------------------ hub
hf_py() { "$PY" - "$@"; }

# hf_has REPO TYPE PATH_OR_GLOB  -> 0 when at least one file matches
hf_has() {
    hf_py "$@" <<'PY'
import fnmatch, sys
from huggingface_hub import HfApi
repo, kind, pat = sys.argv[1:4]
try:
    files = HfApi().list_repo_files(repo, repo_type=kind)
except Exception:
    sys.exit(1)
sys.exit(0 if any(fnmatch.fnmatch(f, pat) for f in files) else 1)
PY
}

# hf_ensure REPO TYPE  -> create the repo private if it is not there
hf_ensure() {
    hf_py "$@" <<'PY'
import sys
from huggingface_hub import HfApi
repo, kind = sys.argv[1:3]
HfApi().create_repo(repo, repo_type=kind, private=True, exist_ok=True)
PY
}

# hf_up LOCAL REMOTE REPO TYPE  -> one file, three tries (foundry hf_put)
hf_up() {
    local i
    for i in 1 2 3; do
        if hf_py "$@" <<'PY'
import sys
from huggingface_hub import HfApi
local, remote, repo, kind = sys.argv[1:5]
HfApi().upload_file(path_or_fileobj=local, path_in_repo=remote, repo_id=repo, repo_type=kind,
                    commit_message=f"pipeline: {remote}")
PY
        then return 0; fi
        say "upload of $2 failed, try $i of 3"; sleep $(( i * 20 ))
    done
    return 1
}

# hf_up_dir LOCAL_DIR REMOTE_DIR REPO TYPE  -> a folder in one commit (foundry hf_put_dir)
hf_up_dir() {
    scan_secrets "$1"
    local i
    for i in 1 2 3; do
        if hf_py "$@" <<'PY'
import sys
from huggingface_hub import HfApi
local, remote, repo, kind = sys.argv[1:5]
HfApi().upload_folder(folder_path=local, path_in_repo=remote, repo_id=repo, repo_type=kind,
                      commit_message=f"pipeline: {remote}/")
PY
        then return 0; fi
        say "upload of $2/ failed, try $i of 3"; sleep $(( i * 20 ))
    done
    return 1
}

# hf_get REPO TYPE REVISION PATTERN DEST_DIR
hf_get() {
    hf_py "$@" <<'PY'
import sys
from huggingface_hub import snapshot_download
repo, kind, rev, pat, dest = sys.argv[1:6]
snapshot_download(repo, repo_type=kind, revision=None if rev == "-" else rev,
                  allow_patterns=[pat], local_dir=dest)
PY
}

# a log should never carry the token (foundry scan_secrets)
scan_secrets() {
    if grep -rIl "hf_[A-Za-z0-9]\{30,\}" "$1" 2>/dev/null | grep -q .; then
        fail 1 "a file under $1 contains something that looks like an HF token, refusing to upload"
    fi
}

push_log() { [ -f "$1" ] && hf_up "$1" "logs/$(basename "$1")" "$METRICS" dataset || true; }

# ------------------------------------------------------------------ tools
ensure_tools() {
    if [ ! -x "$PY" ]; then
        say "installing system packages and a venv"
        export DEBIAN_FRONTEND=noninteractive
        apt-get update -qq >/dev/null
        apt-get install -y -qq build-essential cmake ninja-build git curl ccache tmux \
            libcurl4-openssl-dev libssl-dev python3-venv python3-pip >/dev/null
        python3 -m venv $VENV
        $VENV/bin/pip install -q -U pip "huggingface_hub>=1.0" pyyaml numpy
    fi
}

# build the pinned llama.cpp once per box (foundry build, plus the pin it lacked)
LLAMA=/opt/llama-${LLAMA_COMMIT:0:12}
BIN=$LLAMA/build/bin
ensure_llama() {
    ensure_tools
    if [ ! -x "$BIN/llama-quantize" ]; then
        say "building llama.cpp $LLAMA_COMMIT from $LLAMA_REPO"
        rm -rf "$LLAMA"
        git clone -q "$LLAMA_REPO" "$LLAMA"
        git -C "$LLAMA" checkout -q "$LLAMA_COMMIT"
        local cuda="-DGGML_CUDA=OFF"
        if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
            local arch
            arch=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')
            cuda="-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=$arch"
        fi
        cmake -S "$LLAMA" -B "$LLAMA/build" -G Ninja -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF $cuda \
            > "$LOGS/build.log" 2>&1
        cmake --build "$LLAMA/build" -j "$(nproc)" --target llama-quantize llama-imatrix \
            llama-perplexity llama-gguf-split llama-cli llama-mtmd-cli >> "$LOGS/build.log" 2>&1 \
            || { tail -30 "$LOGS/build.log"; fail 1 "llama.cpp build failed"; }
        $VENV/bin/pip install -q -r "$LLAMA/requirements/requirements-convert_hf_to_gguf.txt" \
            --extra-index-url https://download.pytorch.org/whl/cpu >> "$LOGS/build.log" 2>&1
        $VENV/bin/pip install -q -e "$LLAMA/gguf-py" >> "$LOGS/build.log" 2>&1
    fi
    git -C "$LLAMA" rev-parse HEAD > "$LOGS/llama-commit.txt"
    "$BIN/llama-quantize" --version > "$LOGS/llama-version.txt" 2>&1 || true
    {
        echo "node $NODE  box $(hostname)  $(date -u +%FT%TZ)"
        nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || echo "no GPU"
        echo "cores $(nproc)  ram $(free -g | awk '/Mem:/{print $2}') GB  disk free $(df -h "$WORK" | awk 'NR==2{print $4}')"
    } > "$LOGS/env-$NODE.txt"
}

# ------------------------------------------------------------------ shared inputs
BF16=$WORK/gguf/$STEM-BF16.gguf
# the BF16 file on this box, from the hub if another box converted it
ensure_bf16() {
    [ -f "$BF16" ] && return 0
    local first
    first=$(ls "$WORK"/gguf/bf16/$STEM-BF16-00001-of-*.gguf 2>/dev/null | head -1 || true)
    if [ -z "$first" ]; then
        say "fetching the BF16 GGUF from $METRICS"
        hf_get "$METRICS" dataset - "bf16/*" "$WORK/gguf"
        first=$(ls "$WORK"/gguf/bf16/$STEM-BF16*.gguf | head -1)
    fi
    BF16=$first
}

INVENTORY=$WORK/inventory.json
ensure_inventory() {
    [ -f "$INVENTORY" ] || hf_get "$METRICS" dataset - "inventory.json" "$WORK"
}

EVALSET=${EVALSET:-neutral}
EVAL=$WORK/eval/eval_$EVALSET.txt
CTX=${CTX:-4096}
ensure_eval() {
    [ -f "$EVAL" ] && return 0
    local remote
    case "$EVALSET" in
        neutral) remote=eval/neutral/eval_neutral.txt ;;   # foundry get_eval_set
        agentic) remote=eval/agentic/eval_agentic.txt ;;
        code)    remote=eval/code/eval_code_full.txt ;;
        *) fail 1 "unknown EVALSET $EVALSET" ;;
    esac
    hf_get "$CALIB_REPO" dataset "$CALIB_REV" "$remote" "$WORK/eval/dl"
    mv "$WORK/eval/dl/$remote" "$EVAL"
}

done_node() { echo "NODE_DONE $NODE {\"seconds\": $(( $(date +%s) - T0 )), $1}"; exit 0; }
