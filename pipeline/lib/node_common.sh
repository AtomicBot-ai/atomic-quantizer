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
#
# LOCAL_HUB=/dir turns the AtomicChat/ repos into folders (lib/hub.py), for a
# run in a local container without a token.
set -euo pipefail

[ -f /root/.hf_env ] && . /root/.hf_env
if [ -z "${LOCAL_HUB:-}" ]; then
    : "${HF_TOKEN:?no HF_TOKEN: put export HF_TOKEN=... into /root/.hf_env}"
    export HF_TOKEN
fi
export HF_HUB_DISABLE_PROGRESS_BARS=1 HF_XET_HIGH_PERFORMANCE=1 HF_HUB_DOWNLOAD_TIMEOUT=60

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
# (the sleep must not hold the node's stdout: the driver's tee, and so NODE_EXIT, waits for every
# writer to close, and an orphaned `sleep 300` kept each node alive up to five minutes past its end)
heartbeat() {
    local f s
    trap 'kill $s 2>/dev/null; exit 0' TERM
    while :; do
        sleep 300 </dev/null >/dev/null 2>&1 & s=$!
        wait $s
        f=$(ls -t "$LOGS"/* 2>/dev/null | head -1 || true)
        [ -n "$f" ] && say "alive, $(basename "$f"): $(tail -c 300 "$f" | tr '\r' '\n' | grep -v '^\s*$' | tail -1 | cut -c1-140)"
    done
}
heartbeat & HEARTBEAT=$!
# set -e stops at the first failing command; remember which one, the EXIT trap alone only knows "line 1"
set -E
LAST_ERR=""
trap 'LAST_ERR="${BASH_SOURCE[0]##*/}:$LINENO: $BASH_COMMAND"' ERR
trap 'rc=$?; kill $HEARTBEAT 2>/dev/null; if [ $rc -ne 0 ] && [ -z "$FAILED" ]; then echo "NODE_FAIL $NODE unexpected exit $rc at ${LAST_ERR:-?}"; fi' EXIT

VENV=/opt/venv
PY=$VENV/bin/python

# ------------------------------------------------------------------ hub (lib/hub.py)
hub() { "$PY" "$PIPE/lib/hub.py" "$@"; }
hf_has() { hub has "$@"; }                    # REPO TYPE GLOB
hf_ensure() { hub ensure "$@"; }              # REPO TYPE
hf_up() { hub up "$@"; }                      # LOCAL REMOTE REPO TYPE
hf_up_dir() { scan_secrets "$1"; hub updir "$@"; }   # LOCAL_DIR REMOTE_DIR REPO TYPE
hf_get() { hub get "$@"; }                    # REPO TYPE REV GLOB DEST

# a file about to be published must never carry the token (foundry scan_secrets)
# (no `| grep -q`: under pipefail a SIGPIPE in the writer would read as "nothing found")
scan_secrets() {
    local hits
    hits=$(grep -rIl "hf_[A-Za-z0-9]\{30,\}" "$1" 2>/dev/null || true)
    [ -z "$hits" ] || fail 1 "a file under $1 contains something that looks like an HF token, refusing to upload"
}

push_log() {
    [ -f "$1" ] || return 0
    scan_secrets "$1"
    hf_up "$1" "logs/$(basename "$1")" "$METRICS" dataset || say "could not upload $(basename "$1")"
}

# ------------------------------------------------------------------ box
PIP=(--retries 8 --timeout 60)   # box networks drop; a read timeout must not end a node
retry() {  # CMD...: three tries, 20 and 40 s apart
    local n
    for n in 1 2 3; do
        "$@" && return 0
        [ $n -lt 3 ] && { say "retrying ($n): $*"; sleep $(( 20 * n )); }
    done
    return 1
}
ensure_tools() {
    if [ ! -f "$VENV/.ready" ]; then
        say "installing system packages and a venv"
        export DEBIAN_FRONTEND=noninteractive
        retry apt-get -o Acquire::Retries=5 update -qq >/dev/null || fail 1 "apt-get update failed"
        retry apt-get -o Acquire::Retries=5 install -y -qq build-essential cmake ninja-build git curl ccache tmux procps \
            libcurl4-openssl-dev libssl-dev python3-venv python3-pip >/dev/null || fail 1 "apt-get install failed"
        python3 -m venv $VENV
        retry $VENV/bin/pip install -q "${PIP[@]}" -U pip "huggingface_hub>=1.0" pyyaml numpy || fail 1 "pip failed"
        touch "$VENV/.ready"   # only now: a half made venv is rebuilt, not trusted
    fi
}

NGPU=0
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    NGPU=$(nvidia-smi -L | wc -l)
fi
NGL=99   # on a box without GPUs llama.cpp ignores it

# build the pinned llama.cpp once per box (foundry build, plus the pin it lacked)
LLAMA=/opt/llama-${LLAMA_COMMIT:0:12}
BIN=$LLAMA/build/bin
ensure_llama() {
    ensure_tools
    if [ ! -f "$LLAMA/.ready" ]; then
        say "building llama.cpp $LLAMA_COMMIT from $LLAMA_REPO"
        retry bash -c "rm -rf '$LLAMA' && git clone -q '$LLAMA_REPO' '$LLAMA'" || fail 1 "could not clone $LLAMA_REPO"
        git -C "$LLAMA" checkout -q "$LLAMA_COMMIT" || fail 1 "no commit $LLAMA_COMMIT in $LLAMA_REPO"
        local cuda="-DGGML_CUDA=OFF"
        if [ "$NGPU" -gt 0 ]; then
            local arch
            arch=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')
            cuda="-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=$arch"
        fi
        cmake -S "$LLAMA" -B "$LLAMA/build" -G Ninja -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF $cuda \
            > "$LOGS/build.log" 2>&1
        cmake --build "$LLAMA/build" -j "$(nproc)" --target llama-quantize llama-imatrix \
            llama-perplexity llama-gguf-split llama-cli llama-mtmd-cli >> "$LOGS/build.log" 2>&1 \
            || { tail -30 "$LOGS/build.log"; fail 1 "llama.cpp build failed"; }
        retry $VENV/bin/pip install -q "${PIP[@]}" -r "$LLAMA/requirements/requirements-convert_hf_to_gguf.txt" \
            --extra-index-url https://download.pytorch.org/whl/cpu >> "$LOGS/build.log" 2>&1 \
            || { tail -30 "$LOGS/build.log"; fail 1 "converter requirements failed"; }
        retry $VENV/bin/pip install -q "${PIP[@]}" -e "$LLAMA/gguf-py" >> "$LOGS/build.log" 2>&1 \
            || { tail -30 "$LOGS/build.log"; fail 1 "gguf-py install failed"; }
        touch "$LLAMA/.ready"
    fi
    git -C "$LLAMA" rev-parse HEAD > "$LOGS/llama-commit.txt"
    "$BIN/llama-cli" --version > "$LOGS/llama-version.txt" 2>&1 || true   # llama-quantize has no --version
    {
        echo "node $NODE  box $(hostname)  $(date -u +%FT%TZ)"
        if [ "$NGPU" -gt 0 ]; then
            nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
        else
            echo "no GPU"
        fi
        echo "cores $(nproc)  ram $(free -g | awk '/Mem:/{print $2}') GB  disk free $(df -h "$WORK" | awk 'NR==2{print $4}')"
    } > "$LOGS/env-$NODE.txt"
}

# ------------------------------------------------------------------ shared inputs
BF16=$WORK/gguf/$STEM-BF16.gguf
# first file of the BF16 as published (single or split); ls would exit 2 on the missing variant
bf16_first() {
    local f
    for f in "$WORK"/gguf/bf16/$STEM-BF16-00001-of-*.gguf "$WORK"/gguf/bf16/$STEM-BF16.gguf; do
        [ -f "$f" ] && { echo "$f"; return 0; }
    done
    return 1
}
# the BF16 file on this box, from the hub if another box converted it
ensure_bf16() {
    [ -f "$BF16" ] && return 0
    local first
    if ! first=$(bf16_first); then
        say "fetching the BF16 GGUF from $METRICS"
        hf_get "$METRICS" dataset - "bf16/*" "$WORK/gguf"
        first=$(bf16_first) || fail 1 "no $STEM-BF16 GGUF under bf16/ in $METRICS"
    fi
    BF16=$first
}

INVENTORY=$WORK/inventory.json
ensure_inventory() {
    [ -f "$INVENTORY" ] || hf_get "$METRICS" dataset - "inventory.json" "$WORK"
}

# KLD protocol: the reference and every measurement must use the same corpus,
# context and chunk count. node_base writes them into the manifest, the others read them back.
EVALSET=${EVALSET:-neutral}
EVAL=$WORK/eval/eval_$EVALSET.txt
CTX=${CTX:-4096}
KLD_CHUNKS=${KLD_CHUNKS:-0}   # 0 = the whole corpus (87 chunks of 4096 for neutral)
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
    mkdir -p "$(dirname "$EVAL")"
    mv "$WORK/eval/dl/$remote" "$EVAL"
}
kld_chunk_args() { [ "$KLD_CHUNKS" -gt 0 ] && echo "--chunks $KLD_CHUNKS" || true; }

done_node() { echo "NODE_DONE $NODE {\"seconds\": $(( $(date +%s) - T0 )), $1}"; exit 0; }

# every node talks to the hub through the venv, so it has to exist before the first check
ensure_tools
