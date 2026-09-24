#!/bin/bash
# Foundry toolbox. Source it, then call one function at a time, in the foreground.
#
#   source /foundry.sh
#   use_nemotron
#   status          <- always tells you what to do next
#   menu            <- pick a command by number instead of typing it
#
# Nothing runs in the background. Every function says what it is doing,
# checks its inputs, and stops loudly when something is missing.

# Bump this on every change. reload compares it against what is on github.
FOUNDRY_VERSION=2026-08-21.03

# ------------------------------------------------------------------ settings
# These live here and nowhere else. An earlier edit lost them, which left BIN,
# STATE and CTX empty: status reported a build that existed as missing, and
# save_state wrote to an empty path.
BIN=/llama.cpp/build/bin
STATE=/state.sh
CTX=4096

EVALSET=${EVALSET:-neutral}
EVAL=${EVAL:-/eval/neutral.txt}
BASE=${BASE:-/kld/base-neutral.kld}

IM_CTX=${IM_CTX:-512}
IM_BATCH=${IM_BATCH:-4096}
IM_CORPUS=${IM_CORPUS:-/eval/calib_train.txt}
IM_NO_PPL=${IM_NO_PPL:-0}
IM_OFREQ=${IM_OFREQ:-20}
MTP_TYPE=${MTP_TYPE:-q5_k}
AUTOPUSH=${AUTOPUSH:-1}
GPUS=${GPUS:-all}
INCLUDE_EXPERIMENTAL=${INCLUDE_EXPERIMENTAL:-0}

export HF_XET_HIGH_PERFORMANCE=1
export HF_HOME=/hf

# The container gets its environment from the server template, but an ssh
# login shell does not inherit it: PID 1 has it and we do not. Reading it back
# out of /proc/1/environ is the only way to see what the template actually set,
# and it is why HF_TOKEN kept looking absent in every new pane.
load_env() {
    local v
    for v in HF_TOKEN HF_XET_HIGH_PERFORMANCE HF_HOME; do
        if [ -z "${!v}" ] && [ -r /proc/1/environ ]; then
            local val
            val=$(tr '\0' '\n' < /proc/1/environ | grep "^$v=" | head -1 | cut -d= -f2-)
            [ -n "$val" ] && export "$v=$val"
        fi
    done
    if [ -n "$HF_TOKEN" ]; then
        echo "HF_TOKEN present (${#HF_TOKEN} chars)"
    else
        echo "no HF_TOKEN anywhere, not in this shell and not in the container"
        echo "environment. Set it in the template, or export it here."
        return 1
    fi
}

load_env > /dev/null 2>&1

# Both http paths to github are cached: raw.githubusercontent behind a CDN, and
# the contents API behind its own layer. Either can serve a file that is minutes
# old, and reload then reports success on a copy that has not changed. git does
# not go through any of that, so a clone is the only version of this that can be
# trusted, and it can also show which commit it is actually on.
reload() {
    if [ -d /quantizer/.git ]; then
        git -C /quantizer fetch -q origin && git -C /quantizer reset -q --hard origin/HEAD || return 1
    else
        rm -rf /quantizer
        git clone -q https://github.com/AtomicBot-ai/atomic-quantizer /quantizer || return 1
    fi

    echo "commit: $(git -C /quantizer log -1 --format='%h %ad %s' --date=short)"
    local got
    got=$(grep -m1 "^FOUNDRY_VERSION=" /quantizer/foundry.sh | cut -d= -f2)
    echo "here: ${FOUNDRY_VERSION:-unknown}    at that commit: ${got:-none}"

    if [ "$got" = "$FOUNDRY_VERSION" ]; then
        echo "same version. That commit above is what is on the remote, so if it"
        echo "is not what you pushed, the push did not land on the default branch."
        return 0
    fi
    cp /quantizer/foundry.sh /foundry.sh
    source /foundry.sh
    echo "now running $FOUNDRY_VERSION"
}


# ------------------------------------------------------------------ basics
# Bash functions live in one shell process, so a new tmux pane knows nothing.
# save_state writes the current selection to /state.sh and the bottom of this
# file reads it back, so every pane agrees.
save_state() {
    cat > $STATE << STATEEOF
MAIN=$MAIN
RECIPE=$RECIPE
METRICS=$METRICS
METRICS_KIND=$METRICS_KIND
UPSTREAM=$UPSTREAM
EVALSET=$EVALSET
EVAL=$EVAL
BASE=$BASE
CTX=$CTX
IM_TOKENS_EXACT=$IM_TOKENS_EXACT
IM_MODEL=$IM_MODEL
IM_CORPUS=$IM_CORPUS
IM_CTX=$IM_CTX
GPUS=$GPUS
AUTOPUSH=$AUTOPUSH
INCLUDE_EXPERIMENTAL=$INCLUDE_EXPERIMENTAL
STATEEOF
}

persist_shell() {
    grep -q "source /foundry.sh" ~/.bashrc 2>/dev/null || echo "source /foundry.sh" >> ~/.bashrc
    echo "Added to ~/.bashrc. Every new tmux pane now loads this by itself."
}

save_token() {
    if [ -z "$HF_TOKEN" ]; then
        echo "export HF_TOKEN first, then run this"
        return 1
    fi
    grep -q "export HF_TOKEN=" ~/.bashrc 2>/dev/null \
        && sed -i "s|export HF_TOKEN=.*|export HF_TOKEN=$HF_TOKEN|" ~/.bashrc \
        || echo "export HF_TOKEN=$HF_TOKEN" >> ~/.bashrc
    echo "stored in ~/.bashrc, every new pane will have it"
    echo "this box is disposable, the token is not: revoke it when you are done"
}

# The cli's argument order changed between huggingface_hub versions and boxes
# rented at different times do not have the same one, so uploads go through the
# python API instead.
hf_put() {
    if [ -z "$HF_TOKEN" ] && [ ! -f ~/.cache/huggingface/token ]; then
        echo "no token in this shell. A new tmux pane does not inherit it."
        echo "  export HF_TOKEN=<yours>      and to stop repeating it:  save_token"
        return 1
    fi
    python3 - "$1" "$2" "$3" "$4" << 'PUTEOF'
import sys
from huggingface_hub import HfApi
local, remote, repo, kind = sys.argv[1:5]
try:
    HfApi().upload_file(path_or_fileobj=local, path_in_repo=remote,
                        repo_id=repo, repo_type=kind)
except Exception as e:
    msg = str(e).splitlines()[0]
    print(msg)
    if "401" in msg:
        print("that is an auth failure: the token is missing, wrong, or has no")
        print("write access to " + repo)
    sys.exit(1)
PUTEOF
}

hf_put_dir() {
    if [ -z "$HF_TOKEN" ] && [ ! -f ~/.cache/huggingface/token ]; then
        echo "no token in this shell. export HF_TOKEN, then save_token"
        return 1
    fi
    scan_secrets "$1" || return 1
    python3 - "$1" "$2" "$3" "$4" << 'PUTDEOF'
import sys
from huggingface_hub import HfApi
local, remote, repo, kind = sys.argv[1:5]
try:
    HfApi().upload_folder(folder_path=local, path_in_repo=remote,
                          repo_id=repo, repo_type=kind)
except Exception as e:
    print(str(e).splitlines()[0])
    sys.exit(1)
PUTDEOF
}

# Never guess a repo id.
find_repo() {
    [ -z "$1" ] && { echo "find_repo QUERY"; return 1; }
    python3 - "$1" << 'FINDEOF'
import sys
from huggingface_hub import HfApi
api = HfApi()
try:
    hits = list(api.list_models(search=sys.argv[1], sort="downloads", limit=40))
except TypeError:
    hits = list(api.list_models(search=sys.argv[1], limit=40))
if not hits:
    print("nothing matches %r" % sys.argv[1]); sys.exit(1)
for m in hits:
    print("%-54s %10s" % (m.id, m.downloads or 0))
FINDEOF
}

# Boxes get reused across models and the previous one's weights are still on
# disk, so anything that picks a file filters by this stem.
model_stem() {
    [ -n "$MAIN" ] && basename "$MAIN" | sed "s/-GGUF$//"
}

INCLUDE_EXPERIMENTAL=${INCLUDE_EXPERIMENTAL:-0}

quant_files() {
    if [ "$INCLUDE_EXPERIMENTAL" = "1" ]; then
        DEPTH=2
    else
        DEPTH=1
    fi
    local stem
    stem=$(model_stem)
    find /gguf -maxdepth $DEPTH -name "*.gguf" 2>/dev/null \
        | grep -v -i "bf16" \
        | grep -v "/dflash-" \
        | grep -v "/dspark-" \
        | grep -v "/mmproj-" \
        | grep -v -i "imatrix" \
        | { [ -n "$stem" ] && grep -F "$stem" || cat; } \
        | sort
}

autopush() {
    local src_path="$1" dest_path="$2" err
    if [ "$AUTOPUSH" != "1" ] || [ -z "$METRICS" ] || [ ! -f "$src_path" ]; then
        return 0
    fi
    if err=$(hf_put "$src_path" "$dest_path" "$METRICS" "$METRICS_KIND" 2>&1); then
        echo "  uploaded -> $METRICS :: $dest_path"
    else
        echo "  UPLOAD FAILED for $dest_path:"
        echo "$err" | sed "s/^/    /"
    fi
}

setup_convert() {
    echo "This installs torch and friends so convert_hf_to_gguf.py can run."
    echo "Around 3 GB of downloads. Only needed when we have to BUILD a bf16 gguf."
    ask "install?" || return 1
    pip install --break-system-packages -q -U -r /llama.cpp/requirements/requirements-convert_hf_to_gguf.txt
    echo "done. If the converter complains about RoPE, pin transformers below 5."
}

# Every function this file advertises has to exist. Editing a big shell file by
# cutting between two function names quietly eats whatever sat between them.
selfcheck() {
    local f missing=""
    for f in save_state persist_shell save_token hf_put hf_put_dir reload \
             find_repo selfcheck model_stem quant_files autopush use_nemotron \
             use_qwen use_model show_preset need_preset ask mark status menu \
             help_me use_evalset set_ctx check cuda_check fix_cuda_compat \
             clean_run clean_gguf repo_ok token_check make_metrics make_repos \
             setup setup_convert build gpu_test get_tools check_pool fix_pool \
             install_auto_fmt setup_corpus build_corpus push_corpus \
             corpus_check make_recipe new_model pick_model get_eval \
             get_eval_set eval_size get_bf16 get_quants get_one find_bf16 \
             get_upstream make_bf16 base kld kld_all kld_ext bench bench_all \
             gen results bits quantize ladder write_ladder get_external \
             push_base push_logs push_results push_model push_model_split \
             push_card pull_logs get_imatrix get_calib wait_calib im_size \
             im_plan im_shard im_range im_merge im_merge_all im_stats \
             im_status push_shards push_quants catch_up audit del_model \
             del_old_ad plan ls_main ls_metrics ls_corpora \
             get_recipe use_gpus apply_gpus list_repo fetch_one \
             write_kld_readme send_base get_base \
             is_mac mac_setup mac_info mac_memory mac_build mac_get \
             speed_note speed_gguf speed_mlx speed_all speed_report \
             kld_install mlx_kld_selftest mlx_kld2 mlx_kld2_all \
             mlx_results2 ppl_compare help_measure \
             cuda_arch disk_plan mlx_reference mlx_get_ours \
             mlx_ext_list mlx_get_external mlx_audit; do
        type -t $f > /dev/null 2>&1 || missing="$missing $f"
    done
    if [ -n "$missing" ]; then
        echo "MISSING functions:$missing"
        echo "the file on disk is incomplete. Pull it again."
        return 1
    fi
    echo "all functions present, version $FOUNDRY_VERSION"
}

help_me() {
cat << 'HELP_EOF'

ORIENTATION   status | plan | menu | selfcheck | help_me | reload

PRESET        use_model NAME REPO | use_nemotron | use_qwen | find_repo QUERY

BOX SETUP     check | cuda_check | setup | build 120 | gpu_test
              persist_shell | save_token | clean_run | clean_gguf

REPOS         token_check | repo_ok | make_repos | ls_main | ls_metrics

NEW MODEL     new_model NAME REPO | make_recipe NAME REPO | corpus_check REPO
              get_upstream | setup_convert | make_bf16

CORPUS        get_tools | check_pool | fix_pool | install_auto_fmt
              setup_corpus | build_corpus NAME | push_corpus NAME
              get_calib NAME | wait_calib NAME | ls_corpora | get_recipe NAME

IMATRIX       pick_model | im_size | im_plan N | im_shard I N
              im_range FROM COUNT LABEL | im_merge_all | im_status
              push_shards | get_imatrix | im_stats FILE
              IM_TOKENS_EXACT=N  IM_BATCH=4096  IM_SKIP="name"  IM_NO_PPL=1

QUANTIZE      bits | bits 'RULE' 'RULE' | quantize LABEL TYPE ...
              write_ladder | ladder /ladder.txt --dry | ladder /ladder.txt
              get_external REPO PATTERN

MEASURE       get_eval | get_eval_set NAME | eval_size | set_ctx N | base
              use_gpus 0,1 | kld MODEL | kld_all | kld_ext
              bench MODEL | bench_all | gen MODEL NGL "EXTRA"
              results | pull_logs

UPLOAD        push_base | push_logs | push_results | push_quants | push_model FILE
              push_model_split FILE | push_card FILE | send_base user@host PORT

MLX           mlx_help       the whole MLX side
MEASURE v2    help_measure   mac setup, matched speed, llama.cpp style kld

HELP_EOF
}


# ------------------------------------------------------------------ safety

# Anything that looks like a credential must not leave the box. Checked before
# every upload rather than after: a secret in a public repo is not something a
# later commit takes back.
SECRET_RE='hf_[A-Za-z0-9]{30,}|sk-[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{30,}|AKIA[0-9A-Z]{16}'

scan_secrets() {
    local target="${1:-.}" hits
    hits=$(grep -rlEI "$SECRET_RE" "$target" 2>/dev/null)
    if [ -n "$hits" ]; then
        echo
        echo "!! credential-shaped strings found, refusing to upload:"
        echo "$hits" | sed "s/^/   /"
        echo
        echo "see what matched:"
        echo "   grep -rnEI '$SECRET_RE' $target | head"
        return 1
    fi
    return 0
}

# ------------------------------------------------------------------ vision

# This model takes images and video, not only text. convert_hf_to_gguf.py
# writes the language half by default, so a repo without this file is text only
# however many quants it holds. One file serves the whole ladder and is not
# worth quantizing at under a gigabyte.
make_mmproj() {
    need_preset || return 1
    if [ ! -d /src ]; then
        echo "no original weights in /src. Run get_upstream."
        return 1
    fi
    if [ ! -f /src/preprocessor_config.json ]; then
        echo "no preprocessor_config.json in /src, so this model has no vision"
        echo "tower and needs no mmproj file."
        return 1
    fi
    if ! python3 /llama.cpp/convert_hf_to_gguf.py --help 2>&1 | grep -q -- "--mmproj"; then
        echo "this llama.cpp build has no --mmproj flag. Update it."
        return 1
    fi

    local stem out t
    stem=$(basename $MAIN | sed "s/-GGUF$//")
    for t in f16 bf16; do
        out=/gguf/mmproj-$stem-$(echo $t | tr a-z A-Z).gguf
        echo
        echo "=== $t ==="
        date
        python3 /llama.cpp/convert_hf_to_gguf.py /src --mmproj --outtype $t \
            --outfile $out 2>&1 | tee /logs/mmproj-$t.log
        ls -lh $out 2>/dev/null
        autopush /logs/mmproj-$t.log logs/mmproj-$t.log
    done
    echo
    echo "publish with:  push_mmproj"
}

push_mmproj() {
    need_preset || return 1
    local f n
    for f in /gguf/mmproj-*.gguf; do
        [ -f "$f" ] || continue
        n=$(basename "$f")
        echo "$n  $(du -h "$f" | cut -f1)"
        hf_put "$f" "$n" "$MAIN" model && echo "  up" || echo "  failed"
    done
}

# ------------------------------------------------------------------ checks

# What the file says about itself against what it actually holds. Cheap, and
# it catches the whole class of bug where metadata and tensors disagree.
check_blocks() {
    local f="${1:-}"
    if [ -z "$f" ]; then
        find_bf16 || return 1
        f=$BF16_FIRST
    fi
    python3 - "$f" << 'BLKEOF'
import re, sys
sys.path.insert(0, "/llama.cpp/gguf-py")
from gguf import GGUFReader
r = GGUFReader(sys.argv[1])

def s(key):
    fld = r.fields.get(key)
    return bytes(fld.parts[fld.data[0]]).decode() if fld else None

def i(key):
    fld = r.fields.get(key)
    return int(fld.parts[fld.data[0]][0]) if fld else None

arch = s("general.architecture")
declared = i("%s.block_count" % arch) if arch else None
seen = {int(m.group(1)) for t in r.tensors
        for m in [re.match(r"blk\.(\d+)\.", t.name)] if m}
actual = max(seen) + 1 if seen else 0

print("architecture : %s" % arch)
print("declared     : %s blocks" % declared)
print("present      : %d blocks, blk.0 to blk.%d" % (len(seen), actual - 1))
if declared is not None and declared != actual:
    print()
    print("MISMATCH. llama.cpp will ask for blk.%d and there is nothing" % actual)
    print("there, so this file will not load. If the extra block is an MTP")
    print("head the weights never shipped, reconvert with --no-nextn.")
    sys.exit(1)
print("ok, they agree")
BLKEOF
}

# What llama.cpp recorded inside each file. Hugging Face builds its hardware
# compatibility table from general.file_type, not from the filename, so a build
# whose rules made it four bit while its declared type says otherwise will not
# appear in the row users look at.
check_meta() {
    python3 - << 'METAEOF'
import glob, sys
sys.path.insert(0, "/llama.cpp/gguf-py")
from gguf import GGUFReader

files = sorted(glob.glob("/gguf/*.gguf"))
if not files:
    print("nothing in /gguf")
    sys.exit(0)

print("%-46s %10s  %s" % ("file", "declared", "template"))
for p in files:
    try:
        r = GGUFReader(p)
    except Exception as e:
        print("%-46s  unreadable: %s" % (p.split("/")[-1][:46], str(e)[:40]))
        continue
    ft = r.fields.get("general.file_type")
    ftv = int(ft.parts[ft.data[0]][0]) if ft else None
    tpl = "yes" if "tokenizer.chat_template" in r.fields else "NO"
    print("%-46s %10s  %s" % (p.split("/")[-1][:46], ftv, tpl))

print()
print("file_type numbers: 7 Q8_0, 15 Q4_K_M, 17 Q5_K_M, 18 Q6_K, 30 IQ4_XS,")
print("26 IQ3_S, 23 IQ3_XXS, 28 IQ2_S, 20 IQ2_XS, 19 IQ2_XXS, 31 IQ1_M")
print()
print("A four bit build that declares 7 gets grouped with Q8_0 and vanishes")
print("from its own row. Set it at quantize time:")
print("  --override-kv general.file_type=int:15")
METAEOF
}

# The corpus was rendered through whichever template was current when we
# converted. If upstream has changed it since, the calibration no longer
# matches what the model reads at inference.
check_template() {
    need_preset || return 1
    local mine=/tmp/template-mine.jinja up=/tmp/template-up.jinja f
    f=$(ls /gguf/*.gguf 2>/dev/null | grep -v -i bf16 | grep -v "/mmproj-" | head -1)
    if [ -z "$f" ]; then
        echo "no gguf on this box to read a template from"
        return 1
    fi
    echo "reading from $(basename $f)"
    python3 - "$f" > $mine << 'TPLEOF'
import sys
sys.path.insert(0, "/llama.cpp/gguf-py")
from gguf import GGUFReader
r = GGUFReader(sys.argv[1])
fld = r.fields.get("tokenizer.chat_template")
if not fld:
    print("NO TEMPLATE IN THIS FILE")
    sys.exit(0)
print(bytes(fld.parts[fld.data[0]]).decode("utf-8"))
TPLEOF

    rm -rf /tmp/up
    hf download $UPSTREAM --include "chat_template.jinja" --local-dir /tmp/up > /dev/null 2>&1
    if [ ! -f /tmp/up/chat_template.jinja ]; then
        echo "upstream ships no chat_template.jinja; it may live inside"
        echo "tokenizer_config.json instead. Compare by hand."
        return 1
    fi
    cp /tmp/up/chat_template.jinja $up

    # A missing newline at end of file is not a difference in the template.
    if diff -q <(sed -e '$a\' $mine) <(sed -e '$a\' $up) > /dev/null; then
        echo "identical to $UPSTREAM"
    else
        echo "DIFFERENT from $UPSTREAM. Upstream changed it after we converted,"
        echo "so the corpus was rendered through the older one."
        echo
        diff $mine $up | head -40
    fi
}

# One command for a box that has nothing on it.
newbox() {
    if [ -z "$2" ]; then
        echo "newbox SHORTNAME UPSTREAM_REPO [cuda_arch]"
        echo "  newbox qwen3.8-27b Qwen/Qwen3.8-27B 120"
        return 1
    fi
    check
    cuda_check
    setup
    build "${3:-120}"
    gpu_test
    use_model "$1" "$2" || return 1
    persist_shell
    echo
    echo "box ready. Next, depending on what you came for:"
    echo "  get_upstream ; setup_convert ; make_mmproj ; push_mmproj"
    echo "  get_quants ; check_meta ; check_template"
    echo "  get_eval ; base ; kld_all"
}

# ------------------------------------------------------------------ vision test

# Does the vision tower actually work with our quants. Downloads a small test
# image, runs one quant with the projector, and prints what the model saw.
# test_vision MODEL [IMAGE] [PROMPT]
# A picture with text on it is the better test: a description of a diagram is
# hard to score, but a misread word is obvious. That makes it possible to say
# where on the ladder vision stops being reliable, rather than only that it
# works somewhere.
test_vision() {
    local model="$1" img="${2:-}" prompt="${3:-}" proj
    if [ -z "$model" ]; then
        echo "test_vision /gguf/<quant>.gguf"
        echo "available:"
        quant_files | sed "s/^/   /"
        return 1
    fi
    # "BF16" contains "F16", so a glob on F16 matches both and sorts BF16
    # first. Ask for the one that is not BF16 unless MMPROJ says otherwise.
    proj=${MMPROJ:-$(ls /gguf/mmproj-*.gguf 2>/dev/null | grep -v -i bf16 | head -1)}
    [ -z "$proj" ] && proj=$(ls /gguf/mmproj-*.gguf 2>/dev/null | head -1)
    if [ -z "$proj" ]; then
        echo "no mmproj on this box. Run make_mmproj, or pull it from the repo."
        return 1
    fi
    if [ ! -x $BIN/llama-mtmd-cli ]; then
        echo "llama-mtmd-cli is not built. Rebuild with the current build function."
        return 1
    fi
    if [ -z "$img" ]; then
        img=/tmp/test.jpg
        if [ ! -f "$img" ]; then
            curl -sL "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/ai2d-demo.jpg" \
                -o "$img" || { echo "could not fetch a test image"; return 1; }
        fi
    fi
    if [ ! -f "$img" ]; then
        echo "no such image: $img"
        return 1
    fi
    [ -z "$prompt" ] && prompt="Describe this image in two sentences."
    echo "model: $(basename $model)"
    echo "proj : $(basename $proj)"
    echo "image: $img"
    echo
    # llama.cpp warns that this family needs at least 1024 image tokens or
    # grounding suffers. Pass it rather than ignore the warning.
    # Two known sources of noise are filtered from what you see, not from the
    # log: unused tensor lines are the MTP head the plain graph never runs, and
    # find_slot lines are how this family numbers image patches. Real errors
    # still come through.
    stdbuf -oL -eL $BIN/llama-mtmd-cli -m "$model" --mmproj "$proj" \
        --image "$img" -ngl 99 -c 8192 --image-min-tokens 1024 \
        -p "$prompt" \
        2>&1 | tee /logs/vision-$(basename $model .gguf).log \
        | grep -vE "unused tensor|find_slot|does not match expectation"
    autopush /logs/vision-$(basename $model .gguf).log \
        logs/vision-$(basename $model .gguf).log
}

# Vision across the ladder. The projector is shared, so what changes from file
# to file is how much of the image the language half can still use. Low bit
# builds are where it breaks first, and that is exactly what a reader wants to
# know before downloading one.
# test_vision_all [IMAGE] [PROMPT]
test_vision_all() {
    local f n=0 total img="${1:-}" prompt="${2:-}"
    total=$(quant_files | wc -l)
    for f in $(quant_files); do
        n=$(( n + 1 ))
        echo
        echo "########## $n of $total ##########"
        test_vision "$f" "$img" "$prompt" | tail -25
    done
    echo
    echo "the full transcripts are in /logs/vision-*.log"
}

# Does the chat template survive quantization intact. --jinja makes llama.cpp
# use the template stored in the file rather than a built in guess, which is
# the whole point: it tests what a user will actually get.
test_chat() {
    local model="$1"
    if [ -z "$model" ]; then
        echo "test_chat /gguf/<quant>.gguf"
        return 1
    fi
    stdbuf -oL -eL $BIN/llama-cli -m "$model" -ngl 99 -c 8192 --jinja -no-cnv \
        -p "Think briefly, then answer: what is 17 times 23?" -n 400 \
        2>&1 | tee /logs/chat-$(basename $model .gguf).log | tail -30

    echo
    echo "what to look for: a thinking block that opens and closes, a correct"
    echo "answer after it, and no template markup showing up as literal text."
    echo "Tool calling is a server feature, not a cli one: run test_tools."
    autopush /logs/chat-$(basename $model .gguf).log logs/chat-$(basename $model .gguf).log
}

# Tool calling needs the server, because the cli has nowhere to put a function
# list. Two panes rather than a background process, so you can watch both.
test_tools() {
    local model="${1:-}"
    if [ -z "$model" ]; then
        echo "test_tools /gguf/<quant>.gguf"
        return 1
    fi
    cat << TOOLSEOF

In one pane:

  $BIN/llama-server -m $model -ngl 99 -c 8192 --jinja --port 8080

In another:

  curl -s localhost:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
    "messages": [{"role":"user","content":"What is the weather in Paris?"}],
    "tools": [{"type":"function","function":{
      "name":"get_weather",
      "description":"Get the current weather for a city",
      "parameters":{"type":"object","properties":{"city":{"type":"string"}},
                    "required":["city"]}}}],
    "tool_choice": "auto"
  }' | python3 -m json.tool | tee /logs/tools-$(basename $model .gguf).json

A pass looks like a tool_calls array with get_weather and a parsed arguments
object. A fail looks like the model writing the call as plain text in content,
which means llama.cpp did not recognise the template's tool syntax.

TOOLSEOF
}

# Long context behaves differently from short: rope scaling, cache growth and
# any attention window only show up past a few thousand tokens.
test_longctx() {
    local model="$1" ctx="${2:-32768}"
    if [ -z "$model" ]; then
        echo "test_longctx /gguf/<quant>.gguf [context]"
        return 1
    fi
    if [ ! -f $IM_CORPUS ] && [ ! -f $EVAL ]; then
        echo "no text to feed it. Run get_eval."
        return 1
    fi
    local text=${EVAL}
    [ -f "$IM_CORPUS" ] && text=$IM_CORPUS

    echo "filling $ctx tokens from $(basename $text), then asking for a summary"
    echo "this tests that the model can attend across the whole window and that"
    echo "rope scaling holds. It does not test long range reasoning: the corpus"
    echo "is unrelated passages, so there is nothing far apart to connect."
    head -c $(( ctx * 3 )) "$text" > /tmp/long.txt
    echo "" >> /tmp/long.txt
    echo "Summarise the text above in three sentences." >> /tmp/long.txt

    stdbuf -oL -eL $BIN/llama-cli -m "$model" -ngl 99 -c $ctx --jinja -no-cnv \
        -f /tmp/long.txt -n 800 \
        2>&1 | tee /logs/longctx-$(basename $model .gguf).log | tail -30

    echo
    grep -E "n_ctx|rope|kv cache|KV self|eval time" /logs/longctx-$(basename $model .gguf).log | head -20
    autopush /logs/longctx-$(basename $model .gguf).log \
        logs/longctx-$(basename $model .gguf).log
}


# A demo image in the repo lets anyone reproduce the vision example without
# hunting for a picture, and lets us show a screenshot people can check.
#
# Use something whose licence you know. The image test_vision downloads comes
# from someone else's repository and is fine for a local check, but it is not
# ours to redistribute.
push_demo_image() {
    local f="$1" src="${2:-}"
    if [ -z "$f" ] || [ ! -f "$f" ]; then
        echo "push_demo_image FILE [SOURCE_NOTE]"
        echo "  push_demo_image /root/shot.png 'Atomic Chat UI, our own screenshot'"
        return 1
    fi
    need_preset || return 1
    token_check > /dev/null || return 1

    local ext name
    ext="${f##*.}"
    name="demo.$ext"
    hf_put "$f" "$name" "$MAIN" model || return 1

    if [ -n "$src" ]; then
        printf '%s\n' "$name: $src" > /tmp/demo-source.txt
        hf_put /tmp/demo-source.txt "demo-source.txt" "$MAIN" model
    else
        echo
        echo "no source note given. Add one so nobody has to guess where the"
        echo "image came from:  push_demo_image $f 'where it is from'"
    fi
    echo
    echo "people can now run:"
    echo "  hf download $MAIN --include '$name' --local-dir ."
    echo "  llama-mtmd-cli -m <quant>.gguf --mmproj mmproj-*-F16.gguf \\"
    echo "    --image $name --image-min-tokens 1024 -ngl 99 -c 8192"
}


# ------------------------------------------------------------------ renaming

# Rename a published file and move the metrics log with it, so a build never
# ends up with results filed under a name that no longer exists.
rename_quant() {
    if [ -z "$2" ]; then
        echo "rename_quant OLD NEW      names without .gguf"
        echo "  rename_quant Qwen3.8-27B-AD-Q4_K Qwen3.8-27B-AD-Q4_K_M"
        return 1
    fi
    need_preset || return 1
    token_check > /dev/null || return 1
    local old="$1" new="$2"

    if [ -f "/gguf/$old.gguf" ]; then
        mv "/gguf/$old.gguf" "/gguf/$new.gguf"
        echo "renamed on disk"
    fi
    if [ ! -f "/gguf/$new.gguf" ]; then
        echo "no /gguf/$new.gguf to upload"
        return 1
    fi

    echo "uploading $new.gguf  $(du -h /gguf/$new.gguf | cut -f1)"
    hf_put "/gguf/$new.gguf" "$new.gguf" "$MAIN" model || return 1
    del_model "$old"

    local ol="/logs/kld-$EVALSET--$old.log" nl="/logs/kld-$EVALSET--$new.log"
    if [ -f "$ol" ]; then
        mv "$ol" "$nl"
        autopush "$nl" "logs/kld-$EVALSET--$new.log"
        python3 - "$METRICS" "$METRICS_KIND" "logs/kld-$EVALSET--$old.log" << 'DELLOGEOF'
import sys
from huggingface_hub import HfApi
try:
    HfApi().delete_file(path_in_repo=sys.argv[3], repo_id=sys.argv[1], repo_type=sys.argv[2])
    print("  old log removed from the metrics repo")
except Exception as e:
    print("  old log not removed: %s" % str(e).splitlines()[0])
DELLOGEOF
    fi
}

# ------------------------------------------------------------------ scrollback

# tmux keeps very little history by default and scrolling it needs a key
# sequence. This makes the wheel work and the buffer deep, once per box.
fix_tmux() {
    cat > ~/.tmux.conf << 'TMUXEOF'
set -g mouse on
set -g history-limit 200000
set -g status-interval 5
setw -g mode-keys vi
TMUXEOF
    tmux source-file ~/.tmux.conf 2>/dev/null
    echo "mouse scrolling on, 200k lines of history."
    echo "search inside the buffer: ctrl-b [ then / and a pattern."
}

# Open the newest log, or a named one, in a pager you can search.
lastlog() {
    local f
    if [ -n "$1" ]; then
        f=$(ls -t /logs/*"$1"* 2>/dev/null | head -1)
    else
        f=$(ls -t /logs/* 2>/dev/null | head -1)
    fi
    if [ -z "$f" ]; then
        echo "no log matches"
        ls /logs | sed "s/^/   /"
        return 1
    fi
    echo "$f"
    less -R "$f"
}


# ------------------------------------------------------------------ presets

use_nemotron() {
    MAIN=AtomicChat/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF
    METRICS=AtomicChat/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF-metrics
    METRICS_KIND=dataset
    UPSTREAM=nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16
    RECIPE=nemotron-3.5-lightning
    save_state
    show_preset
}

# For anything without a preset. Names our repos by convention and refuses a
# repo id that does not exist, so a typo cannot propagate into every later step.
use_model() {
    if [ -z "$2" ]; then
        echo "use_model SHORTNAME UPSTREAM_REPO"
        echo "  use_model qwen3.8-27b Qwen/Qwen3.8-27B"
        return 1
    fi
    if ! python3 -c "
import sys
from huggingface_hub import HfApi
sys.exit(0 if HfApi().repo_exists(sys.argv[1]) else 1)
" "$2" 2>/dev/null; then
        echo "no such repo on the hub: $2"
        echo "find the real id with:   find_repo <part of the name>"
        return 1
    fi
    RECIPE=$1
    UPSTREAM=$2
    MAIN=AtomicChat/$(basename $2)-GGUF
    METRICS=AtomicChat/$(basename $2)-GGUF-metrics
    METRICS_KIND=dataset
    save_state
    show_preset
}

use_qwen() {
    use_model qwen3.8-27b Qwen/Qwen3.8-27B
}

show_preset() {
    echo "recipe name   : ${RECIPE:-not set}"
    echo "our gguf repo : $MAIN"
    echo "metrics repo  : $METRICS  (type: $METRICS_KIND)"
    echo "upstream      : $UPSTREAM"
    echo "eval corpus   : $EVAL"
    echo "context       : $CTX"
    echo
    echo "Run plan to see what exists where and what to do next."
}

need_preset() {
    if [ -z "$MAIN" ]; then
        echo "No preset loaded. Run use_model NAME REPO first."
        return 1
    fi
}

ask() {
    read -p "$1 [y/N] " answer
    if [ "$answer" = "y" ] || [ "$answer" = "Y" ]; then
        return 0
    fi
    echo "skipped"
    return 1
}

use_evalset() {
    if [ -z "$1" ]; then
        echo "use_evalset neutral | agentic | code       current: $EVALSET"
        return 1
    fi
    EVALSET=$1
    EVAL=/eval/$1.txt
    BASE=/kld/base-$1.kld
    save_state
    echo "eval set  : $EVALSET"
    echo "corpus    : $EVAL"
    echo "reference : $BASE"
}

set_ctx() {
    if [ -z "$1" ]; then
        echo "set_ctx 4096      current: $CTX"
        return 1
    fi
    CTX=$1
    save_state
    echo "context is now $CTX"
    eval_size
}

# ------------------------------------------------------------------ orientation

mark() {
    if [ -e "$2" ]; then
        echo "  [x] $1"
    else
        echo "  [ ] $1        ->  run:  $3"
    fi
}

status() {
    echo
    if [ -z "$MAIN" ]; then
        echo "  [ ] preset loaded      ->  run:  use_model NAME REPO"
    else
        echo "  [x] preset: $MAIN"
    fi
    mark "cuda checked"       /logs/env.txt              "cuda_check"
    mark "tools installed"    /usr/bin/ninja             "setup"
    mark "llama.cpp built"    $BIN/llama-imatrix         "build 120"
    mark "eval corpus"        $EVAL                      "get_eval"

    if [ -d /src ] && ls /src/*.safetensors > /dev/null 2>&1; then
        echo "  [x] original weights in /src  ($(du -sh /src 2>/dev/null | cut -f1))"
    else
        echo "  [ ] original weights in /src  ->  run:  get_upstream"
    fi

    if ls /gguf/mmproj-*.gguf > /dev/null 2>&1; then
        echo "  [x] mmproj on disk: $(ls /gguf/mmproj-*.gguf | wc -l) files"
    elif [ -f /src/preprocessor_config.json ]; then
        echo "  [ ] mmproj missing      ->  run:  make_mmproj   (this model sees images)"
    fi

    if find /gguf -name "*.gguf" 2>/dev/null | grep -v "/mmproj-" | grep -qi bf16; then
        echo "  [x] bf16 on disk"
    else
        echo "  [ ] bf16 on disk       ->  run:  get_bf16, or make_bf16 for a new model"
    fi

    STALE=$(ls /logs/kld-*.log /logs/bench-*.json 2>/dev/null | grep -v -i "$(basename ${MAIN:-zzz} -GGUF)" | wc -l)
    if [ "$STALE" -gt 0 ]; then
        echo
        echo "  !! $STALE log(s) in /logs belong to a different model."
        echo "     results would mix them in and push_logs would upload them."
        echo "     run:  clean_run"
        echo
    fi

    QN=$(quant_files | wc -l)
    if [ "$QN" -gt 0 ]; then
        echo "  [x] quants on disk: $QN files"
    else
        echo "  [ ] quants on disk     ->  run:  get_quants, or quantize them here"
    fi

    mark "reference built"    $BASE                      "base"
    KLDS=$(ls /logs/kld-*.log 2>/dev/null | wc -l)
    echo "  [$([ $KLDS -gt 0 ] && echo x || echo ' ')] kld logs: $KLDS       ->  run:  kld_all"
    BENCHES=$(ls /logs/bench-*.json 2>/dev/null | wc -l)
    echo "  [$([ $BENCHES -gt 0 ] && echo x || echo ' ')] bench logs: $BENCHES     ->  run:  bench_all"
    mark "results.json"       /logs/results.json         "results"

    echo
    echo "  imatrix:"
    if [ -f /eval/calib_train.txt ]; then
        echo "    [x] calibration corpus"
    else
        echo "    [ ] calibration corpus  ->  run:  get_calib $RECIPE"
    fi
    if [ -n "$IM_MODEL" ] && [ -f "$IM_MODEL" ]; then
        echo "    [x] IM_MODEL: $(basename $IM_MODEL)"
    else
        echo "    [ ] IM_MODEL not set    ->  run:  pick_model"
    fi
    SHARDS=$(ls /imatrix/shard-*.gguf 2>/dev/null | wc -l)
    if [ -f /imatrix/imatrix.gguf ]; then
        echo "    [x] imatrix merged"
    elif [ "$SHARDS" -gt 0 ]; then
        echo "    [~] $SHARDS shard(s) here  ->  run:  im_merge_all once every node is done"
    else
        echo "    [ ] no imatrix          ->  run:  im_plan N, then im_shard I N on each box"
    fi

    echo
    echo "  full command list: help_me      pick by number: menu"
    echo "  remote view: plan               integrity: selfcheck"
    echo
}

menu() {
    echo
    echo "  model    : ${MAIN:-none, pick a preset first}"
    echo "  eval set : $EVALSET   context: $CTX"
    echo
    PS3="
number: "
    select picked in \
        "status" "plan" "help_me" "selfcheck" \
        "cuda_check" "setup" "build 120" "gpu_test" \
        "get_bf16" "get_quants" "find_bf16" "get_eval" "eval_size" \
        "pick_model" "im_size" "bits" \
        "base" "kld_all" "bench_all" "results" \
        "push_base" "push_logs" "push_results" "pull_logs" \
        "ls_main" "ls_metrics" "ls_corpora" \
        "quit"
    do
        if [ "$picked" = "quit" ] || [ -z "$picked" ]; then
            break
        fi
        echo "--> $picked"
        eval "$picked"
        break
    done
}

# ------------------------------------------------------------------ box setup

check() {
    echo "=== free space on / ==="
    df -h /
    echo
    echo "=== GPUs ==="
    nvidia-smi --query-gpu=index,name,memory.total --format=csv
    echo
    echo "=== cores / RAM ==="
    nproc
    free -g | head -2
}

cuda_check() {
    mkdir -p /logs
    echo "=== toolkit in the container ==="
    nvcc --version | tail -2
    echo
    echo "=== host driver ==="
    nvidia-smi --query-gpu=name,driver_version --format=csv
    echo
    echo "=== forward compatibility libraries ==="
    COMPAT_LIB=$(ls /usr/local/cuda/compat/libcuda.so.*.* 2>/dev/null | head -1)
    if [ -z "$COMPAT_LIB" ]; then
        echo "none present, nothing to do"
    else
        COMPAT_MAJOR=$(basename $COMPAT_LIB | sed "s/libcuda.so.//" | cut -d. -f1)
        HOST_MAJOR=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
        echo "compat libs target driver $COMPAT_MAJOR, host driver is $HOST_MAJOR"
        if [ "$HOST_MAJOR" -ge "$COMPAT_MAJOR" ]; then
            echo "Host driver is newer or equal, so these libs can only break CUDA init."
            fix_cuda_compat
        else
            echo "Host driver is older, so these libs are needed. Leaving them alone."
        fi
    fi

    {
        grep PRETTY_NAME /etc/os-release
        nvcc --version | tail -2
        nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
    } > /logs/env.txt
    echo
    echo "written to /logs/env.txt, it ships to the metrics repo with push_logs"
}

fix_cuda_compat() {
    rm -rf /usr/local/cuda*/compat
    ldconfig
    echo "Compat libraries removed. The host driver now wins."
}



# Measurements from a previous model on the same box would be parsed into this
# model's results and uploaded to its metrics repo. Wipe them when switching.
clean_run() {
    echo "about to remove measurements from any earlier model on this box:"
    echo "  /logs      $(ls /logs 2>/dev/null | wc -l) files"
    echo "  /kld       $(ls /kld 2>/dev/null | wc -l) files"
    echo "  /imatrix   $(ls /imatrix 2>/dev/null | wc -l) files"
    echo "the weights in /gguf and /src are left alone."
    ask "remove?" || return 1
    rm -rf /logs/* /kld/* /imatrix/*
    mkdir -p /logs /kld /imatrix
    echo "clean. Re-run cuda_check so env.txt exists again."
}

# Weights from an earlier model on this box. They are large and they confuse
# every file picker, so this offers to remove exactly those.
clean_gguf() {
    need_preset || return 1
    local stem others
    stem=$(model_stem)
    others=$(find /gguf /src -maxdepth 2 -name "*.gguf" 2>/dev/null | grep -v -F "$stem")
    if [ -z "$others" ]; then
        echo "nothing here belongs to another model"
        return 0
    fi
    echo "not part of $stem:"
    echo "$others" | xargs -r du -sh 2>/dev/null | sed "s/^/  /"
    echo
    du -sh /gguf
    ask "delete them?" || return 1
    echo "$others" | xargs -r rm -f
    find /gguf -type d -empty -delete
    du -sh /gguf
}

# Every log that says anything useful goes to the metrics repo the moment it
# is written. Set AUTOPUSH=0 to keep things local while experimenting.
AUTOPUSH=1

# Which GPUs the measurement runs may use. "all" means do not restrict.
# The reference needs all of them (bf16 does not fit on two cards), but the
# quant runs are deliberately held to a setup the community can relate to.
GPUS=all

use_gpus() {
    if [ -z "$1" ]; then
        echo "use_gpus 0,1     or     use_gpus all       current: $GPUS"
        return 1
    fi
    GPUS=$1
    save_state
    echo "quant measurements will run on GPUs: $GPUS"
}

apply_gpus() {
    if [ "$GPUS" = "all" ]; then
        unset CUDA_VISIBLE_DEVICES
    else
        export CUDA_VISIBLE_DEVICES=$GPUS
    fi
}

token_check() {
    if [ -z "$HF_TOKEN" ]; then
        echo "HF_TOKEN is not set. export the real one, then retry."
        return 1
    fi
    case "$HF_TOKEN" in
        PUT_*|hf_xxx*|CHANGE*)
            echo "HF_TOKEN is still a placeholder: $HF_TOKEN"
            echo "This file must never export a token. export it in your shell."
            return 1 ;;
    esac
    hf auth whoami
}

repo_ok() {
    if [ -z "$METRICS" ]; then
        echo "no preset loaded"
        return 1
    fi
    if python3 -c "
import sys
from huggingface_hub import HfApi
sys.exit(0 if HfApi().repo_exists(sys.argv[1], repo_type=sys.argv[2]) else 1)
" "$METRICS" "$METRICS_KIND" 2>/dev/null; then
        return 0
    fi
    echo
    echo "cannot see $METRICS as a $METRICS_KIND"
    echo
    echo "check the token first:   token_check"
    echo "if the token is fine, create the repo:   make_metrics"
    echo "if it exists as a model repo:   METRICS_KIND=model ; save_state"
    return 1
}

make_metrics() {
    if [ -z "$METRICS" ]; then
        echo "no preset loaded"
        return 1
    fi
    token_check || return 1
    echo "creating $METRICS as a $METRICS_KIND"
    ask "create?" || return 1
    hf repos create $METRICS --repo-type $METRICS_KIND
    repo_ok && echo "reachable now"
}

# Both repos for a release: the quants and the metrics beside them.
make_repos() {
    need_preset || return 1
    token_check || return 1
    echo "  $MAIN            (model)"
    echo "  $METRICS   ($METRICS_KIND)"
    ask "create both?" || return 1
    for pair in "$MAIN model" "$METRICS $METRICS_KIND"; do
        set -- $pair
        if python3 -c "
import sys
from huggingface_hub import HfApi
sys.exit(0 if HfApi().repo_exists(sys.argv[1], repo_type=sys.argv[2]) else 1)
" "$1" "$2" 2>/dev/null; then
            echo "  already there: $1"
        else
            hf repos create $1 --repo-type $2 > /dev/null 2>&1 \
                && echo "  created: $1" || echo "  FAILED: $1"
        fi
    done
    repo_ok && echo "metrics repo reachable"
}

# Clone the calibration dataset with its pipeline. Large pool files live in
# LFS, and a clone without git-lfs brings down pointer stubs instead of data,
# which then fail to parse as JSON somewhere deep inside the build.
get_tools() {
    token_check > /dev/null || return 1
    apt-get install -y -qq git-lfs > /dev/null 2>&1
    if git lfs version > /dev/null 2>&1; then
        git lfs install > /dev/null 2>&1
        HAVE_LFS=1
    else
        echo "git-lfs unavailable, large files will arrive as pointers and get"
        echo "fetched from the hub instead"
        HAVE_LFS=0
    fi

    if [ -d /calib-corpora ]; then
        echo "already at /calib-corpora"
        cd /calib-corpora
        git pull --ff-only 2>&1 || echo "local changes present, keeping them and skipping the pull"
        [ "$HAVE_LFS" = "1" ] && git lfs pull
        cd - > /dev/null
    else
        git clone https://huggingface.co/datasets/AtomicChat/calib-corpora /calib-corpora || return 1
        [ "$HAVE_LFS" = "1" ] && (cd /calib-corpora && git lfs pull)
    fi
    check_pool || true
    echo
    echo "if anything above is a pointer, run:  fix_pool"
}

# Pull the files that came down as pointers straight from the hub. Simpler and
# more reliable than getting git-lfs working on a fresh container.
fix_pool() {
    POINTERS=$(grep -l "^version https://git-lfs" /calib-corpora/pool/*/*.jsonl \
               /calib-corpora/pool/*/*/*.jsonl 2>/dev/null)
    if [ -z "$POINTERS" ]; then
        echo "no pointer files, nothing to fix"
        return 0
    fi
    for f in $POINTERS; do
        REL=${f#/calib-corpora/}
        echo "fetching $REL"
        hf download AtomicChat/calib-corpora --repo-type dataset \
            --include "$REL" --local-dir /calib-corpora > /dev/null || return 1
    done
    check_pool
}

# Every jsonl in the pool has to parse, and an LFS pointer does not. Find the
# bad file by name instead of watching build.py die on an anonymous line.
check_pool() {
    python3 - << 'POOLEOF'
import glob, json, os

bad, pointers, ok = [], [], 0
for path in sorted(glob.glob("/calib-corpora/pool/**/*.jsonl", recursive=True)):
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            first = f.readline()
            if first.startswith("version https://git-lfs"):
                pointers.append(path)
                continue
            json.loads(first)
        ok += 1
    except Exception as e:
        bad.append((path, str(e)[:60]))

print("%d pool files parse" % ok)
if pointers:
    print()
    print("%d files are LFS pointers, not data:" % len(pointers))
    for p in pointers[:10]:
        print("   ", p.replace("/calib-corpora/", ""))
    print()
    print("fix:  fix_pool")
if bad:
    print()
    print("%d files do not parse:" % len(bad))
    for p, e in bad[:10]:
        print("    %s   %s" % (p.replace("/calib-corpora/", ""), e))
if not pointers and not bad:
    print("pool is clean")
POOLEOF
}


setup() {
    apt-get update -qq
    apt-get install -y -qq build-essential cmake ninja-build git curl ccache \
        libcurl4-openssl-dev libssl-dev python3-pip
    pip install --break-system-packages -q -U "huggingface_hub[hf_xet]"
    mkdir -p /gguf /kld /eval /logs /hf
    echo
    hf version
    python3 --version
    echo "Missing something? Add it to this function, not to the server template."
}

build() {
    if [ -z "$1" ]; then
        echo "Pass the CUDA arch. build 120"
        return 1
    fi
    cd /
    git clone https://github.com/ggml-org/llama.cpp
    cd /llama.cpp
    git rev-parse --short HEAD > /logs/llama-commit.txt
    cat /logs/llama-commit.txt

    cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
        -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=$1
    cmake --build build -j $(nproc) --target \
        llama-perplexity llama-bench llama-cli llama-gguf-split \
        llama-imatrix llama-quantize llama-mtmd-cli

    ls -la $BIN/
    echo
    echo "Now run gpu_test."
}

gpu_test() {
    if [ ! -x $BIN/llama-cli ]; then
        echo "not built yet. Run build 120."
        return 1
    fi
    $BIN/llama-cli --list-devices
    echo
    echo "Every GPU should be listed above. If the list is empty or short,"
    echo "run fix_cuda_compat and try again."
}


# ================================================================== what exists where

ls_main() {
    need_preset || return 1
    list_repo "$MAIN" model
}

ls_metrics() {
    need_preset || return 1
    list_repo "$METRICS" "$METRICS_KIND"
}

list_repo() {
    python3 - "$1" "$2" << 'LSEOF'
import sys
from huggingface_hub import HfApi
repo, kind = sys.argv[1], sys.argv[2]

def human(n):
    for unit, div in (("GB", 1e9), ("MB", 1e6), ("KB", 1e3)):
        if n >= div:
            return "%.2f %s" % (n / div, unit)
    return "%d B" % n

try:
    tree = list(HfApi().list_repo_tree(repo, repo_type=kind, recursive=True))
except Exception as e:
    print("cannot read %s (%s): %s" % (repo, kind, e))
    sys.exit(1)
total = 0
for f in sorted(tree, key=lambda x: x.path):
    size = getattr(f, "size", None)
    if size is None:
        continue
    total += size
    print("%12s  %s" % (human(size), f.path))
print("---")
print("%12s  total, %d files" % (human(total), len([f for f in tree if getattr(f, "size", None)])))
LSEOF
}

plan() {
    need_preset || return 1
    python3 - "$MAIN" "$METRICS" "$METRICS_KIND" "$UPSTREAM" << 'PLANEOF'
import os, sys, glob
from huggingface_hub import HfApi

main, metrics, kind, upstream = sys.argv[1:5]
api = HfApi()

def files(repo, t):
    try:
        return api.list_repo_files(repo, repo_type=t)
    except Exception:
        return None

main_files = files(main, "model")
metric_files = files(metrics, kind)

def block(title, on_disk, remote_line, todo):
    print()
    print(title)
    print("  on disk         : %s" % on_disk)
    print("  remote          : %s" % remote_line)
    print("  -> %s" % todo)

# --- bf16
local_bf16 = [p for p in glob.glob("/gguf/**/*.gguf", recursive=True) if "bf16" in p.lower()]
if main_files is None:
    remote_bf16 = []
else:
    remote_bf16 = [f for f in main_files if "bf16" in f.lower() and f.endswith(".gguf")]

if local_bf16:
    todo = "nothing, it is here"
elif remote_bf16:
    todo = "run: get_bf16"
else:
    todo = "not published yet. run: get_upstream, then setup_convert, then make_bf16"
block("BF16 reference weights",
      ("%d files" % len(local_bf16)) if local_bf16 else "no",
      ("%d files in %s" % (len(remote_bf16), main)) if remote_bf16
        else ("repo unreadable" if main_files is None else "not in %s" % main),
      todo)

# --- kld base
local_base = bool(glob.glob("/kld/base-*.kld"))
remote_base = [f for f in (metric_files or []) if f.endswith(".kld")]
if local_base:
    todo = "nothing, it is here"
elif remote_base:
    todo = "run: get_base"
elif local_bf16:
    todo = "run: base      (this computes it, 4 to 8 minutes)"
else:
    todo = "get the bf16 first, then run: base"
block("KLD reference (.kld)",
      "yes: %s" % ", ".join(os.path.basename(p) for p in glob.glob("/kld/base-*.kld")) if local_base else "no",
      ("%d in %s" % (len(remote_base), metrics)) if remote_base
        else ("metrics repo unreadable, check METRICS_KIND" if metric_files is None
              else "not in %s" % metrics),
      todo)

# --- quants
local_q = [p for p in glob.glob("/gguf/*.gguf") if "bf16" not in p.lower()]
remote_q = [f for f in (main_files or [])
            if f.endswith(".gguf") and "bf16" not in f.lower()
            and "imatrix" not in f and not f.startswith(("dflash-", "dspark-"))]
block("Quants to measure",
      "%d files" % len(local_q),
      "%d files in %s" % (len(remote_q), main) if main_files is not None else "repo unreadable",
      "nothing, they are here" if len(local_q) >= len(remote_q) and local_q else "run: get_quants")
print()
PLANEOF
}

get_upstream() {
    need_preset || return 1
    echo "This pulls the ORIGINAL weights from $UPSTREAM into /src."
    echo "Only needed when we have not published a bf16 gguf ourselves."
    ask "download?" || return 1
    mkdir -p /src
    hf download $UPSTREAM --local-dir /src
    du -sh /src
    ls /src
}

# The target file, always without the MTP head. A head inside the main file is
# never executed by a plain forward pass, so it only adds weight and forces
# quantize to special case it. And when the config promises a head the weights
# do not have, the converter writes a block count it cannot fill: the export
# succeeds and the file refuses to load.
make_bf16() {
    need_preset || return 1
    if [ ! -d /src ]; then
        echo "no original weights in /src. Run get_upstream."
        return 1
    fi
    if [ ! -f /llama.cpp/convert_hf_to_gguf.py ]; then
        echo "llama.cpp not cloned. Run build first."
        return 1
    fi
    BF16_OUT=/gguf/$(basename $MAIN | sed "s/-GGUF//")-BF16.gguf
    echo "converting /src -> $BF16_OUT"
    echo "If it fails with NaN in token_embd, add --no-lazy and rerun."
    date
    python3 /llama.cpp/convert_hf_to_gguf.py /src --outtype bf16 --no-nextn \
        --outfile $BF16_OUT 2>&1 | tee /logs/convert.log
    date
    ls -lh $BF16_OUT
    echo
    echo "Now:  check_blocks $BF16_OUT"
}

# The speculative draft as its own file. Only possible when the weights carry
# the head. The config is not evidence: Ornith-1.5-9B claims one and ships none.
make_bf16_mtp() {
    need_preset || return 1
    if [ ! -f /src/model.safetensors.index.json ]; then
        echo "no weight index in /src. Run get_upstream."
        return 1
    fi
    if ! grep -qi -e nextn -e mtp /src/model.safetensors.index.json; then
        echo "no MTP tensors in these weights, nothing to export."
        return 1
    fi
    MTP_OUT=/gguf/mtp-$(basename $MAIN | sed "s/-GGUF//")-BF16.gguf
    echo "converting the MTP head only -> $MTP_OUT"
    date
    python3 /llama.cpp/convert_hf_to_gguf.py /src --outtype bf16 --mtp \
        --outfile $MTP_OUT 2>&1 | tee /logs/convert-mtp.log
    date
    ls -lh $MTP_OUT
    echo
    echo "serve it with:  --spec-type draft-mtp --spec-draft-n-max 6"
}


# ================================================================== fetching one file

# fetch_one REPO KIND REMOTE_PATTERN DEST
# Matches against the FULL path inside the repo and against the bare filename,
# prints exactly what it found and where it put it, then places the file itself.
fetch_one() {
    python3 - "$1" "$2" "$3" "$4" << 'FETCHEOF'
import fnmatch, os, shutil, sys
from huggingface_hub import HfApi, hf_hub_download

repo, kind, pattern, dest = sys.argv[1:5]

try:
    files = HfApi().list_repo_files(repo, repo_type=kind)
except Exception as e:
    print("cannot read %s (%s): %s" % (repo, kind, e))
    sys.exit(1)

hits = [f for f in files
        if fnmatch.fnmatch(f, pattern) or fnmatch.fnmatch(os.path.basename(f), pattern)]

if not hits:
    print("nothing in %s matches %r" % (repo, pattern))
    print("the repo contains:")
    for f in sorted(files)[:60]:
        print("   ", f)
    sys.exit(1)

sizes = {}
try:
    for entry in HfApi().list_repo_tree(repo, repo_type=kind, recursive=True):
        s = getattr(entry, "size", None)
        if s is not None:
            sizes[entry.path] = s
except Exception:
    pass

hits.sort(key=lambda f: sizes.get(f, 0), reverse=True)

if len(hits) > 1:
    print("%d files match %r, taking the BIGGEST:" % (len(hits), pattern))
    for f in hits:
        mark = " <-- taking this" if f == hits[0] else ""
        print("   %8.2f MB  %s%s" % (sizes.get(f, 0) / 1e6, f, mark))

src_path = hits[0]
print()
print("  from : %s :: %s" % (repo, src_path))
print("  to   : %s" % dest)
print()

cached = hf_hub_download(repo, src_path, repo_type=kind)
os.makedirs(os.path.dirname(dest), exist_ok=True)
shutil.copyfile(cached, dest)
print("placed, %.2f MB" % (os.path.getsize(dest) / 1e6))
FETCHEOF
}

ls_corpora() {
    list_repo AtomicChat/calib-corpora dataset
}

get_recipe() {
    if [ -z "$1" ]; then
        echo "get_recipe NAME       e.g. get_recipe nemotron-3.5-lightning"
        echo "see what exists with: ls_corpora"
        return 1
    fi
    fetch_one AtomicChat/calib-corpora dataset "*$1*.yaml" /eval/recipe.yaml
    echo
    cat /eval/recipe.yaml
}


# ================================================================== download

# Exact paths inside AtomicChat/calib-corpora. Globs are not safe here:
# every set also has a *.manifest.jsonl next to it, which is not the corpus.
get_eval_set() {
    if [ -z "$1" ]; then
        echo "get_eval_set neutral | agentic | code"
        return 1
    fi
    case "$1" in
        neutral) REMOTE=eval/neutral/eval_neutral.txt ;;
        agentic) REMOTE=eval/agentic/eval_agentic.txt ;;
        code)    REMOTE=eval/code/eval_code_full.txt ;;
        *) echo "unknown set: $1. Run ls_corpora and pass an exact path instead."
           return 1 ;;
    esac

    use_evalset $1
    if [ -f $EVAL ]; then
        echo "already here:"
        eval_size
        return 0
    fi
    ask "download?" || return 1
    fetch_one AtomicChat/calib-corpora dataset "$REMOTE" $EVAL || return 1
    eval_size
}

get_eval() {
    get_eval_set neutral
}

# The corpus is built on one box and pulled by the rest. This polls until it
# lands, so the other boxes can be started at the same time.
wait_calib() {
    if [ -z "$1" ]; then
        echo "wait_calib NAME"
        return 1
    fi
    echo "polling AtomicChat/calib-corpora for builds/$1. Ctrl-C to stop."
    while true; do
        if get_calib $1 2>/dev/null; then
            return 0
        fi
        echo "not built yet, $(date +%H:%M:%S), retry in 30s"
        sleep 30
    done
}

eval_size() {
    if [ ! -f $EVAL ]; then
        echo "no corpus at $EVAL"
        return 1
    fi
    BYTES=$(stat -c %s $EVAL)
    TOKENS=$(( BYTES / 4 ))
    CHUNKS=$(( TOKENS / CTX ))
    echo
    echo "corpus : $EVAL"
    echo "size   : $BYTES bytes, roughly $TOKENS tokens"
    echo "at ctx $CTX that is about $CHUNKS chunks"
    if [ $CHUNKS -lt 2 ]; then
        echo
        echo "TOO SMALL. llama-perplexity needs at least two full contexts."
        echo "Either you grabbed the wrong file (run ls_corpora and look), or"
        echo "lower the context with set_ctx."
        return 1
    fi
    if [ $CHUNKS -lt 20 ]; then
        echo
        echo "That is a thin sample. Error bars on the mean KLD will be wide."
    fi
}

set_ctx() {
    if [ -z "$1" ]; then
        echo "set_ctx 4096      current: $CTX"
        return 1
    fi
    CTX=$1
    save_state
    echo "context is now $CTX"
    eval_size
}

find_bf16() {
    local stem all
    stem=$(model_stem)
    all=$(find /gguf -name "*.gguf" 2>/dev/null | grep -i bf16 \
          | grep -v "/mmproj-" | grep -v "/mtp-" \
          | grep -v "mmproj" | grep -v "^/gguf/mtp-" \
          | grep -v "/dflash-" | grep -v "/dspark-")
    if [ -n "$stem" ]; then
        local mine
        mine=$(echo "$all" | grep -F "$stem")
        if [ -z "$mine" ] && [ -n "$all" ]; then
            echo "there are bf16 files here, but none belongs to $stem:"
            echo "$all" | sed "s/^/   /"
            return 1
        fi
        all=$mine
    fi
    FOUND=$(echo "$all" | grep "00001-of-" | head -1)
    [ -z "$FOUND" ] && FOUND=$(echo "$all" | head -1)
    if [ -z "$FOUND" ]; then
        echo "no bf16 model for $stem in /gguf (mmproj does not count)."
        echo "Build it:  setup_convert ; make_bf16"
        return 1
    fi
    BF16_FIRST=$FOUND
    ls -lh $BF16_FIRST
    echo "bf16 entry point: $BF16_FIRST"
}

get_bf16() {
    need_preset || return 1
    if find /gguf -name "*.gguf" 2>/dev/null | grep -qi bf16; then
        echo "bf16 already on disk:"
        find_bf16
        return 0
    fi
    echo "This pulls the bf16 reference from $MAIN. Around 66 GB for Nemotron."
    ask "download?" || return 1
    hf download $MAIN --include "*bf16*" --include "*BF16*" \
        --exclude "dflash-*" --exclude "dspark-*" --local-dir /gguf
    find_bf16
}

get_quants() {
    need_preset || return 1
    if ls /gguf/*.gguf >/dev/null 2>&1; then
        echo "already on disk:"
        ls -la /gguf/*.gguf
        ask "download anyway, to fill in what is missing?" || return 0
    else
        echo "This pulls every published quant from $MAIN. Around 185 GB for Nemotron."
        ask "download?" || return 1
    fi
    hf download $MAIN --include "*.gguf" \
        --exclude "*bf16*" --exclude "*BF16*" --exclude "*imatrix*" \
        --exclude "dflash-*" --exclude "dspark-*" --local-dir /gguf
    du -sh /gguf
    ls -la /gguf/*.gguf
}

get_one() {
    need_preset || return 1
    echo "pattern: $1"
    ask "download?" || return 1
    hf download $MAIN --include "$1" --local-dir /gguf
    ls -la /gguf
}


# ================================================================== corpus validation

# corpus_check DIR_OR_REPO
# Renders the model's own chat template against a conversation that uses every
# construct we calibrate on, then checks that every marker the template emits
# is a real single token in the vocabulary. If it is not, that part of the
# corpus calibrates nothing and the imatrix is quietly wrong.
corpus_check() {
    if [ -z "$1" ]; then
        echo "corpus_check /src                          local original weights"
        echo "corpus_check nvidia/Some-Model-BF16        straight from the hub"
        return 1
    fi
    pip install --break-system-packages -q -U jinja2 >/dev/null 2>&1
    python3 - "$1" << 'CHECKEOF'
import json, os, re, sys

target = sys.argv[1]

def load(name):
    if os.path.isdir(target):
        p = os.path.join(target, name)
        return open(p, encoding="utf-8").read() if os.path.exists(p) else None
    from huggingface_hub import hf_hub_download
    try:
        return open(hf_hub_download(target, name), encoding="utf-8").read()
    except Exception:
        return None

cfg_raw = load("tokenizer_config.json")
if cfg_raw is None:
    print("FAIL  no tokenizer_config.json at %s" % target)
    sys.exit(1)
cfg = json.loads(cfg_raw)

template = load("chat_template.jinja")
where = "chat_template.jinja"
if template is None:
    template = cfg.get("chat_template")
    where = "tokenizer_config.json"
if isinstance(template, list):
    template = template[0].get("template")

if not template:
    print("FAIL  no chat template anywhere. The corpus cannot be rendered")
    print("      the way the model actually sees text.")
    sys.exit(1)
print("ok    chat template found in %s, %d chars" % (where, len(template)))

# Every construct we calibrate on. Templates disagree about the shape of a tool
# call, so try the common ones and report which one this model accepts: the
# pool records have to match it or the agentic slice renders with no tool
# markup at all.
def convo(args_as_dict, tool_role="tool"):
    args = {"city": "Paris"} if args_as_dict else '{"city": "Paris"}'
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the weather in Paris?"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"type": "function", "function": {"name": "get_weather", "arguments": args}}]},
        {"role": tool_role, "content": '{"temp_c": 19}', "name": "get_weather"},
        {"role": "assistant", "content": "It is 19 degrees in Paris."},
    ]

tools = [{"type": "function", "function": {
    "name": "get_weather",
    "description": "Get the weather for a city",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                   "required": ["city"]}}}]
tools_bare = [t["function"] for t in tools]

from jinja2 import Environment
from jinja2.exceptions import TemplateError
env = Environment(trim_blocks=True, lstrip_blocks=True)
env.policies["json.dumps_kwargs"] = {"ensure_ascii": False}
try:
    tpl = env.from_string(template)
except Exception as e:
    print("FAIL  the template does not even parse as jinja: %s" % e)
    sys.exit(1)

attempts = [
    ("tool_calls arguments as a DICT, tools wrapped in type/function",
     dict(messages=convo(True), tools=tools, add_generation_prompt=True)),
    ("tool_calls arguments as a DICT, bare function schemas",
     dict(messages=convo(True), tools=tools_bare, add_generation_prompt=True)),
    ("tool_calls arguments as a JSON STRING",
     dict(messages=convo(False), tools=tools, add_generation_prompt=True)),
    ("no tools at all",
     dict(messages=convo(True)[:2] + convo(True)[4:], add_generation_prompt=True)),
]

rendered = None
winner = None
for label, kwargs in attempts:
    try:
        out = tpl.render(bos_token="", eos_token="", **kwargs)
    except Exception as e:
        print("  no    %-56s %s" % (label, str(e)[:60]))
        continue
    has_tool = "get_weather" in out
    print("  ok    %-56s tool markup: %s" % (label, "yes" if has_tool else "NO"))
    if rendered is None or (has_tool and "get_weather" not in rendered):
        rendered, winner = out, label
    if has_tool:
        break

if rendered is None:
    print("FAIL  the template will not render any of our conversations")
    sys.exit(1)

print()
print("using: %s" % winner)
if "get_weather" not in rendered:
    print()
    print("WARNING  no shape produced tool markup. The agentic slice of the")
    print("         corpus would render with none of it, and the imatrix would")
    print("         have no statistics for the tool token set at all.")
    print("         Check how the pool stores tool_calls before building.")

# which markers does it emit, and are they real tokens
markers = sorted(set(re.findall(r"<\|[^|>]{1,40}\|>|<[a-z_]{2,20}>|\[/?[A-Z_]{2,20}\]", rendered)))
if not markers:
    print("warn  the template emits no special markers at all, unusual")


known = set()
for v in (cfg.get("added_tokens_decoder") or {}).values():
    if isinstance(v, dict) and "content" in v:
        known.add(v["content"])
tok_raw = load("tokenizer.json")
if tok_raw:
    try:
        for a in json.loads(tok_raw).get("added_tokens", []):
            known.add(a["content"])
    except Exception:
        pass
add_raw = load("added_tokens.json")
if add_raw:
    try:
        known.update(json.loads(add_raw).keys())
    except Exception:
        pass

print()
print("markers the template emits:")
special, plain = [], []
for m in markers:
    (special if m in known else plain).append(m)

for m in special:
    print("  special token   %s" % m)
for m in plain:
    print("  plain text      %s" % m)

print()
if not special:
    print("WARNING  not one marker is a real token. That usually means this is")
    print("         not the model's own template, or the tokenizer files do not")
    print("         match the weights. Do not build a corpus on this.")
else:
    print("RESULT  %d real special tokens, %d plain text delimiters." % (len(special), len(plain)))
    print("        Plain delimiters are fine: the model was trained on this same")
    print("        template, so it reads them as ordinary text at inference too,")
    print("        and the corpus matches. What matters is the special ones:")
    print("        run llama-imatrix with --parse-special or they get tokenized")
    print("        as punctuation and that part of the corpus calibrates nothing.")
print()
print("first 600 chars of the rendered conversation, eyeball it:")
print("-" * 60)
print(rendered[:600])
CHECKEOF
}


# ================================================================== measure

base() {
    need_preset || return 1
    find_bf16 || return 1
    if [ ! -f $EVAL ]; then
        echo "no corpus at $EVAL. Run get_eval."
        return 1
    fi
    eval_size || return 1

    echo
    echo "Writing the reference. This one uses ALL GPUs regardless of use_gpus,"
    echo "because bf16 does not fit on fewer. Say so in the model card."
    echo "Warnings about unused blk.52.nextn tensors are normal: that is the MTP"
    echo "block, which a plain forward pass never executes."
    date
    START=$(date +%s)

    unset CUDA_VISIBLE_DEVICES
    stdbuf -oL -eL $BIN/llama-perplexity -m $BF16_FIRST -f $EVAL \
        --kl-divergence-base $BASE -c $CTX -ngl 99 \
        2>&1 | tee /logs/base-$EVALSET.log

    autopush /logs/base-$EVALSET.log logs/base-$EVALSET.log

    echo "took $(( $(date +%s) - START )) seconds"
    echo
    echo "=== SIZE OF THE REFERENCE ==="
    ls -lh $BASE
    echo
    echo "This file holds fp16 logits over the whole vocabulary for every scored"
    echo "token, so it is large by design and it is NOT worth shipping anywhere."
    echo "It reproduces from the bf16 weights in the time you just watched."
    echo "Measure the quants on this same box, and publish the log, not the blob."
}

kld() {
    if [ ! -f "$1" ]; then echo "no model at $1"; return 1; fi
    if [ ! -f $BASE ]; then echo "no reference at $BASE. Run base, or get_base."; return 1; fi

    NAME=$(basename $1 .gguf)
    case "$1" in
        /gguf/external/*)
            # /gguf/external/<org>/<file>.gguf -> <org>--<file>
            NAME=$(echo "$1" | sed "s|/gguf/external/||; s|/|--|; s|\.gguf$||") ;;
    esac
    apply_gpus
    echo "--------------------------------------------------"
    echo "KLD on $EVALSET: $NAME   (GPUs: $GPUS)"
    START=$(date +%s)

    stdbuf -oL -eL $BIN/llama-perplexity -m $1 -f $EVAL \
        --kl-divergence-base $BASE --kl-divergence -c $CTX -ngl 99 \
        2>&1 | tee /logs/kld-$EVALSET--$NAME.log

    echo "took $(( $(date +%s) - START )) seconds"
    autopush /logs/kld-$EVALSET--$NAME.log logs/kld-$EVALSET--$NAME.log
}

# Skips anything already measured, so it is the catch up command after a box
# built quants before its reference existed. KLD_FORCE=1 redoes them.
kld_all() {
    FILES=$(quant_files)
    if [ -z "$FILES" ]; then
        echo "no quants in /gguf. Run get_quants."
        echo "what is there now:"
        find /gguf -name "*.gguf" | sed "s/^/   /"
        return 1
    fi
    local todo="" f name
    for f in $FILES; do
        name=$(basename "$f" .gguf)
        if [ -f "/logs/kld-$EVALSET--$name.log" ] && [ "$KLD_FORCE" != "1" ]; then
            echo "already measured, skipping: $name"
            continue
        fi
        todo="$todo $f"
    done
    if [ -z "$todo" ]; then
        echo "everything here is measured. KLD_FORCE=1 kld_all to redo."
        results
        return 0
    fi

    TOTAL=$(echo $todo | wc -w)
    N=0
    for f in $todo; do
        N=$(( N + 1 ))
        echo
        echo "########## $N of $TOTAL, roughly $(( (TOTAL - N) * 3 )) minutes left ##########"
        kld $f
    done
    results
}

# Everything a box needs after it has built quants: a reference if it has none,
# then measure whatever is not measured, then publish.
catch_up() {
    need_preset || return 1
    if [ ! -f "$BASE" ]; then
        echo "no reference on this box, building one. Two minutes."
        get_eval || return 1
        base || return 1
    fi
    kld_all
    push_quants
    push_logs
}

# Measure other publishers' builds on OUR reference. Numbers taken against
# different references cannot be put in the same table, so their published
# figures are not usable directly, only their files are.
kld_ext() {
    local f n=0 total
    # mmproj is a vision projector and mtp is a draft head. Neither is a
    # quantization of the model and neither can be measured against it.
    ext_files() {
        find /gguf/external -name "*.gguf" 2>/dev/null \
            | grep -v "/mmproj-" | grep -v "/mtp-" \
            | grep -v -i "bf16" | sort
    }
    total=$(ext_files | wc -l)
    if [ "$total" = "0" ]; then
        echo "nothing measurable in /gguf/external. Pull some with:"
        echo "  get_external unsloth/Qwen3.8-27B-GGUF '*UD-IQ3_XXS*'"
        return 1
    fi
    local todo="" name
    for f in $(ext_files); do
        name=$(echo "$f" | sed "s|/gguf/external/||; s|/|--|; s|\.gguf$||")
        if [ -f "/logs/kld-$EVALSET--$name.log" ] && [ "$KLD_FORCE" != "1" ]; then
            echo "already measured: $name"
            continue
        fi
        todo="$todo $f"
    done
    total=$(echo $todo | wc -w)
    for f in $todo; do
        n=$(( n + 1 ))
        echo
        echo "########## external $n of $total ##########"
        kld "$f"
    done
    results
}

bench() {
    NAME=$(basename $1 .gguf)
    echo "GPU 0 only, so the numbers stay comparable to the published card."
    CUDA_VISIBLE_DEVICES=0 $BIN/llama-bench -m $1 -p 512 -n 128 -ngl 99 -r 5 -o json \
        > /logs/bench-$NAME.json
    head -40 /logs/bench-$NAME.json
    autopush /logs/bench-$NAME.json logs/bench-$NAME.json
}

bench_all() {
    FILES=$(quant_files)
    if [ -z "$FILES" ]; then
        echo "no quants in /gguf. Run get_quants."
        return 1
    fi
    for f in $FILES; do
        bench $f
    done
    results
}

gen() {
    if [ -z "$2" ]; then
        echo 'gen MODEL NGL "EXTRA ARGS"'
        echo 'partial offload:   gen /gguf/x.gguf 20 ""'
        echo 'with MTP:          gen /gguf/x.gguf 20 "--spec-type draft-mtp"'
        return 1
    fi
    NAME=$(basename $1 .gguf)
    TAG=$(echo "ngl$2 $3" | tr -c 'a-zA-Z0-9' '-')

    CUDA_VISIBLE_DEVICES=0 $BIN/llama-cli -m $1 -ngl $2 $3 \
        -n 400 --seed 1234 -c 8192 -no-cnv \
        -p "Write a detailed technical explanation of how a mixture of experts layer routes tokens." \
        2>&1 | tee /logs/gen-$NAME-$TAG.log

    echo
    grep -E "eval time|tokens per second|accept" /logs/gen-$NAME-$TAG.log
}


# ================================================================== new model scaffold

# make_recipe NAME UPSTREAM_REPO
# Reads config.json, tokenizer_config.json and the chat template straight off
# the hub and writes a complete recipe. Nothing is left for you to fill in by
# hand except the domain shares, which are a judgement call about the model's
# use, not a fact you can read out of a config file.
make_recipe() {
    if [ -z "$2" ]; then
        echo "make_recipe NAME UPSTREAM_REPO"
        echo "  make_recipe qwen3.8-27b Qwen/Qwen3.8-27B"
        return 1
    fi
    pip install --break-system-packages -q -U jinja2 pyyaml > /dev/null 2>&1
    mkdir -p /recipes
    python3 - "$1" "$2" << 'RECIPEEOF'
import json, os, re, sys
from huggingface_hub import hf_hub_download

name, repo = sys.argv[1], sys.argv[2]

# build.py resolves chat.format through FORMATTERS. auto_fmt drives the model's
# own template through transformers, so one renderer covers every model and no
# per-model module has to be written.
fmt_key = "auto"

def grab(fn):
    try:
        return open(hf_hub_download(repo, fn), encoding="utf-8").read()
    except Exception:
        return None

cfg_raw = grab("config.json")
if cfg_raw is None:
    print("cannot read config.json from %s" % repo)
    sys.exit(1)
cfg = json.loads(cfg_raw)
tok_cfg = json.loads(grab("tokenizer_config.json") or "{}")

# some repos nest the real config under text_config
inner = cfg.get("text_config") or cfg

def pick(*keys, default=None):
    for k in keys:
        for src_ in (inner, cfg):
            if src_.get(k) is not None:
                return src_[k]
    return default

vocab   = pick("vocab_size", default=0)
layers  = pick("num_hidden_layers", "n_layer", default=0)
ctxlen  = pick("max_position_embeddings", default=0)
window  = pick("sliding_window", default=None)
hidden  = pick("hidden_size", default=0)
inter   = pick("intermediate_size", default=0)
moe     = pick("moe_intermediate_size", default=None)
experts = pick("num_experts", "n_routed_experts", "num_local_experts", default=None)

print("read from %s:" % repo)
for k, v in (("vocab_size", vocab), ("layers", layers), ("context", ctxlen),
             ("sliding_window", window), ("hidden_size", hidden),
             ("intermediate_size", inter), ("moe_intermediate_size", moe),
             ("experts", experts)):
    print("  %-22s %s" % (k, v))

print()
print("superblock check, rows must divide by 256 or every k/i quant below")
print("4.5 bpw silently falls back to a block-32 type:")
shape_lines = []
for k, v in (("hidden_size", hidden), ("intermediate_size", inter),
             ("moe_intermediate_size", moe)):
    if not v:
        continue
    rem = v % 256
    verdict = "ok" if rem == 0 else "BREAKS k/i quants"
    print("  %-22s %6d   %%256 = %3d   %s" % (k, v, rem, verdict))
    shape_lines.append("%s = %d (%%256 = %d)" % (k, v, rem))

# chat template
template = grab("chat_template.jinja")
where = "chat_template.jinja"
if template is None:
    template = tok_cfg.get("chat_template")
    where = "tokenizer_config.json"
if isinstance(template, list):
    template = template[0].get("template")

add_bos = bool(tok_cfg.get("add_bos_token", False))
uses_date = bool(template and "strftime_now" in template)
thinking = bool(template and re.search(r"enable_thinking|<think>", template))

print()
if template:
    print("chat template found in %s, %d chars" % (where, len(template)))
else:
    print("NO CHAT TEMPLATE. The corpus cannot be rendered the way the model")
    print("actually reads text. Run corpus_check and decide before building.")
print("  add_bos_token       : %s" % add_bos)
print("  calls strftime_now  : %s%s" % (uses_date, "  <- pinned below" if uses_date else ""))
print("  mentions thinking   : %s" % thinking)

known_renders = ["dsv4", "nemotron", "muse-glimmer"]
excl = [r for r in known_renders if r not in name]

reasoning_share = 15 if thinking else 10
code_share = 18 if thinking else 23
longctx_share = 12 if window is None else 10

body = """# Calibration recipe for {repo}.
# Generated by foundry make_recipe. Everything under model: was read from the
# hub, not typed. The shares below are the one human judgement call: they say
# what this model is for, which no config file knows.
#
# Architecture facts that matter here:
{shapes}
#
# Shares:
#   agentic 25      tool markup is a dense token set natural text never emits,
#                   so without real weight the imatrix has no statistics for it
#   code {code}         the agentic evals in this class are code shaped
#   reasoning {reason}    {reason_why}
#   multilingual 14 non-Latin embedding rows are only exercised by real
#                   multilingual text
#   longctx {lctx}      {lctx_why}
#   vocab_sweep 10  regenerated against THIS tokenizer, {vocab} rows. A sweep
#                   built for another model is meaningless here
#   structured 5    JSON, YAML, TOML, SQL, as files and inside assistant turns
#   graphics 3      kept small for continuity with the other builds

name: {name}

model:
  repo: {repo}
  tokenizer: /src/tokenizer.json
  gguf: /gguf/{name}-bf16.gguf
  vocab_size: {vocab}
  layers: {layers}
  sliding_window: {window}
  context_length: {ctxlen}
  llama_cpp_commit: FILLED_AT_BUILD_TIME

seed: 20260814

chat:
  # auto_fmt drives this model's own chat_template through transformers, so
  # byte equality with inference holds by construction. Install it once with
  # install_auto_fmt; no per-model renderer is needed.
  format: {fmt}
  add_bos_per_document: {add_bos}
{date_pin}
calib_train:
  target_tokens: 5000000
  max_doc_fraction:
    default: 0.05
    longctx: 0.12
  shares:
    agentic: 25
    code: {code}
    reasoning: {reason}
    multilingual: 14
    longctx: {lctx}
    vocab_sweep: 10
    structured: 5
    graphics: 3

calib_longctx:
  target_tokens: 750000
  doc_tokens_min: 16384
  doc_tokens_max: 32768
  sources: [longctx]

exclude:
  # Anything pre-rendered with another model's markup. Those byte sequences are
  # not special tokens for this tokenizer, so they would calibrate on text this
  # model never sees.
  render: [{excl}]
  provenance_key: excluded_from_builds

notes:
  imatrix: >
    llama-imatrix defaults to parse_special = false. Without --parse-special the
    chat markup in calib_train.txt is tokenised as literal punctuation and the
    agentic and reasoning slices calibrate on text the model never sees.
""".format(
    repo=repo, name=name, fmt=fmt_key, vocab=vocab, layers=layers, ctxlen=ctxlen,
    window="null" if window is None else window,
    add_bos="true" if add_bos else "false",
    code=code_share, reason=reasoning_share, lctx=longctx_share,
    reason_why=("the thinking channel is on the default inference path here"
                if thinking else "no thinking channel found in the template"),
    lctx_why=("no sliding window, so long range behaviour is only exercised by "
              "long documents" if window is None else
              "sliding window present, short documents already exercise it"),
    shapes="\n".join("#   " + s for s in shape_lines) or "#   (no shape info)",
    excl=", ".join(excl),
    date_pin=('  # the template calls strftime_now, so the date is pinned or two\n'
              '  # builds on different days would differ\n'
              '  pin_date: "2026-08-14"\n' if uses_date else ""),
)

out = "/recipes/%s.yaml" % name
open(out, "w").write(body)
print()
if vocab:
    per_100k = vocab * 2 * 100000 / 1e9
    print()
    print("vocabulary is %d rows, so the kld reference costs about %.0f GB per" % (vocab, per_100k))
    print("100k scored tokens. At ctx 4096 with a 400k token corpus that is")
    print("roughly %.0f GB. Plan disk for it before running base." % (per_100k * 2))

print("wrote %s" % out)
print()
print("one thing this file cannot do for you: the domain shares are a claim")
print("about what this model is for. Read them once.")
RECIPEEOF
}

# One command for a brand new model: validate the template, then write a recipe
# with every architecture field already filled in.
new_model() {
    if [ -z "$2" ]; then
        echo "new_model NAME UPSTREAM_REPO"
        echo "  new_model qwen3.8-27b Qwen/Qwen3.8-27B"
        return 1
    fi
    echo "=============== 1. chat template and special tokens ==============="
    if ! corpus_check $2; then
        echo
        echo "Stopping. Nothing below can work without the model's own files."
        echo "If the repo id is wrong, find it with:   find_repo <part of the name>"
        return 1
    fi
    echo
    echo "=============== 2. recipe ==============="
    make_recipe $1 $2 || return 1
    echo
    echo "=============== 3. next ==============="
    echo "  cat /recipes/$1.yaml"
    echo "  get_tools                     # clone calib-corpora"
    echo "  cp /recipes/$1.yaml /calib-corpora/recipes/"
    echo "  then build the corpus with the pipeline in /calib-corpora/tools"
}

# Interactive picker for the model the imatrix is collected on. bf16 first:
# q8 is a memory compromise, not a better choice.
pick_model() {
    FILES=$(find /gguf -maxdepth 2 -name "*.gguf" 2>/dev/null \
            | grep -v "/dflash-" | grep -v "/dspark-" | grep -v -i "imatrix" \
            | grep -v -- "-0000[2-9]-of-" | sort)
    if [ -z "$FILES" ]; then
        echo "nothing in /gguf yet"
        return 1
    fi
    echo
    echo "bf16 is the right choice when it fits. q8 is a compromise for memory."
    echo
    PS3="
number: "
    select f in $FILES "cancel"; do
        if [ "$f" = "cancel" ] || [ -z "$f" ]; then
            break
        fi
        IM_MODEL=$f
        save_state
        echo "IM_MODEL = $IM_MODEL"
        ls -lh $IM_MODEL
        break
    done
}


# ================================================================== corpus build

# One generic renderer instead of one per model. Installs auto_fmt.py into the
# pipeline and registers it in build.py's FORMATTERS.
install_auto_fmt() {
    if [ ! -d /calib-corpora ]; then
        echo "no pipeline yet. Run get_tools."
        return 1
    fi
    curl -sL https://raw.githubusercontent.com/AtomicBot-ai/atomic-quantizer/main/auto_fmt.py \
        -o /calib-corpora/tools/auto_fmt.py || return 1

    if grep -q '"auto"' /calib-corpora/tools/build.py; then
        echo "already registered in FORMATTERS"
    else
        sed -i 's/^FORMATTERS = {/FORMATTERS = {\n    "auto":     "auto_fmt",/' \
            /calib-corpora/tools/build.py
        echo "registered in FORMATTERS"
    fi
    sed -n '/^FORMATTERS = {/,/^}/p' /calib-corpora/tools/build.py
}

setup_corpus() {
    pip install --break-system-packages -q -U transformers pyyaml jinja2 sentencepiece
    python3 -c "import transformers; print('transformers', transformers.__version__)"
    echo "If a converter later complains about RoPE, pin transformers below 5."
}

# build_corpus NAME
# Needs: the original weights in /src, the recipe in /recipes, get_tools done.
build_corpus() {
    if [ -z "$1" ]; then
        echo "build_corpus NAME        e.g. build_corpus qwen3.8-27b"
        return 1
    fi
    NAME=$1

    if [ ! -f /src/tokenizer.json ]; then
        echo "no tokenizer at /src/tokenizer.json. Run get_upstream."
        return 1
    fi
    if [ ! -d /calib-corpora ]; then
        echo "no pipeline. Run get_tools."
        return 1
    fi
    if [ ! -f /calib-corpora/tools/auto_fmt.py ]; then
        echo "generic renderer not installed. Run install_auto_fmt."
        return 1
    fi
    if [ ! -f /recipes/$NAME.yaml ]; then
        echo "no recipe at /recipes/$NAME.yaml. Run make_recipe or new_model."
        return 1
    fi

    export FOUNDRY_MODEL_DIR=/src
    PIN=$(grep -o 'pin_date: *"[^"]*"' /recipes/$NAME.yaml | cut -d'"' -f2)
    if [ -n "$PIN" ]; then
        export FOUNDRY_PIN_DATE=$PIN
        echo "date pinned to $PIN, so this build is reproducible"
    fi

    cd /calib-corpora

    check_pool || return 1

    echo
    echo "=============== vocabulary sweep for THIS tokenizer ==============="
    echo "A sweep built for another model covers a different vocabulary and is"
    echo "worthless here, so this is regenerated every time."

    # Compare before against after. A file can already be dirty for reasons
    # that have nothing to do with the sweep: repairing an LFS pointer replaces
    # the stub with real data, which git reports as a modification.
    git -C /calib-corpora status --porcelain pool/vocab_sweep/synthetic \
        | sort > /tmp/sweep-before.txt

    python3 tools/vocab_sweep.py --tokenizer /src/tokenizer.json --name $NAME \
        2>&1 | tee /logs/vocab-sweep-$NAME.log

    git -C /calib-corpora status --porcelain pool/vocab_sweep/synthetic \
        | sort > /tmp/sweep-after.txt

    echo
    echo "what the sweep itself changed:"
    comm -13 /tmp/sweep-before.txt /tmp/sweep-after.txt | sed "s/^/  /"
    echo "  (already dirty before the sweep, ignored:)"
    comm -12 /tmp/sweep-before.txt /tmp/sweep-after.txt | sed "s/^/    /"

    TOUCHED=$(comm -13 /tmp/sweep-before.txt /tmp/sweep-after.txt \
              | grep "^ M" | awk '{print $2}')
    for f in $TOUCHED; do
        case "$f" in
            *$NAME*) ;;
            *) echo
               echo "!! $f is not named after $NAME."
               echo "   The sweep wrote over another model's file. That file is"
               echo "   built for a different vocabulary and is now gone."
               echo "   Restore it and give vocab_sweep an output name:"
               echo "     git -C /calib-corpora checkout -- $f"
               echo "     python3 tools/vocab_sweep.py --help"
               return 1 ;;
        esac
    done

    echo
    echo "=============== building the corpus ==============="
    date
    python3 tools/build.py --recipe /recipes/$NAME.yaml 2>&1 | tee /logs/build-$NAME.log
    date

    echo
    echo "=============== what appeared ==============="
    ls -la /calib-corpora/builds/$NAME/ 2>/dev/null || find /calib-corpora/builds -newer /calib-corpora/recipes/$NAME.yaml

    if [ -f /calib-corpora/builds/$NAME/calib_train.txt ]; then
        cp /calib-corpora/builds/$NAME/calib_train.txt $IM_CORPUS
        im_size
        echo
        echo "publish it so the other boxes can pull the same bytes:"
        echo "  push_corpus $NAME"
    fi
    cd -
}

push_corpus() {
    if [ -z "$1" ]; then
        echo "push_corpus NAME"
        return 1
    fi
    token_check > /dev/null || return 1
    if [ ! -f /calib-corpora/builds/$1/calib_train.txt ]; then
        echo "no corpus at /calib-corpora/builds/$1/calib_train.txt"
        echo "the directory may exist and be empty from an earlier failed run."
        echo "run build_corpus $1 and read its output before pushing."
        return 1
    fi
    ls -la /calib-corpora/builds/$1
    cd /calib-corpora
    hf_put_dir builds/$1 builds/$1 AtomicChat/calib-corpora dataset
    hf_put /recipes/$1.yaml recipes/$1.yaml AtomicChat/calib-corpora dataset
    cd -
    echo "every box can now: get_calib $1"
}


# ================================================================== imatrix, fanned out
#
# Splitting by chunk range, not by cutting the text file. Every node reads the
# SAME corpus and takes its own slice with --from-chunk / --chunks, so the union
# is bit-for-bit the chunking a single node would have produced. Merging is a
# sum of squared activations, so --in-file gives the same answer as one long run.

IM_CTX=512
IM_BATCH=8192

# --no-ppl skips the perplexity pass, which is a few percent faster. It also
# removes the per-chunk line that was the only sign of progress, leaving a
# forty minute run completely silent. Visibility is worth more than the few
# percent, so it is off by default. Set IM_NO_PPL=1 to get it back.
IM_NO_PPL=${IM_NO_PPL:-0}

# How often llama-imatrix reports that it stored collected data. Also a
# heartbeat.
IM_OFREQ=20
IM_CORPUS=/eval/calib_train.txt
IM_MODEL=""

get_calib() {
    if [ -z "$1" ]; then
        echo "get_calib NAME      e.g. get_calib nemotron-3.5-lightning"
        echo "see what builds exist with: ls_corpora"
        return 1
    fi
    fetch_one AtomicChat/calib-corpora dataset "builds/$1/calib_train.txt" $IM_CORPUS || return 1
    im_size
}

im_size() {
    if [ ! -f $IM_CORPUS ]; then
        echo "no calibration corpus at $IM_CORPUS. Run get_calib."
        return 1
    fi
    BYTES=$(stat -c %s $IM_CORPUS)
    if [ -n "$IM_TOKENS_EXACT" ]; then
        IM_TOKENS=$IM_TOKENS_EXACT
        SOURCE="exact, from the build log"
    else
        IM_TOKENS=$(( BYTES / 4 ))
        SOURCE="estimated at 4 bytes per token, usually low"
    fi
    IM_CHUNKS=$(( IM_TOKENS / IM_CTX ))
    echo "corpus  : $IM_CORPUS"
    echo "size    : $BYTES bytes, $IM_TOKENS tokens ($SOURCE)"
    echo "at ctx $IM_CTX that is about $IM_CHUNKS chunks"
    if [ -z "$IM_TOKENS_EXACT" ]; then
        echo "build.py printed the real token count. Set it so the shards are even:"
        echo "  IM_TOKENS_EXACT=<number> ; save_state"
    fi
}

# im_plan N  ->  prints the exact command for each of N nodes. Paste one per box.
im_plan() {
    if [ -z "$1" ] || [ -z "$IM_MODEL" ]; then
        echo "im_plan N          after setting IM_MODEL=/gguf/<the q8 or bf16>.gguf"
        echo "current IM_MODEL: ${IM_MODEL:-not set}"
        return 1
    fi
    im_size || return 1
    N=$1
    PER=$(( IM_CHUNKS / N + 1 ))
    echo
    echo "$N nodes, about $PER chunks each"
    echo "run get_calib and set IM_MODEL to the same file on every box, then:"
    echo
    I=0
    while [ $I -lt $N ]; do
        echo "  node $I:   im_shard $I $N"
        I=$(( I + 1 ))
    done
    echo
    echo "then on any one box:   im_merge $N"
}

im_shard() {
    if [ -z "$2" ]; then
        echo "im_shard INDEX TOTAL        index is 0 based"
        return 1
    fi
    if [ ! -f "$IM_MODEL" ]; then
        echo "IM_MODEL is not set or missing: ${IM_MODEL:-unset}"
        return 1
    fi
    im_size || return 1

    # The logits tensor is batch x vocab x 4 bytes and lives in VRAM alongside
    # the weights. A batch tuned for a small vocabulary will OOM on a large one.
    VOCAB=$(python3 -c "
import sys
sys.path.insert(0, '/llama.cpp/gguf-py')
from gguf import GGUFReader
r = GGUFReader('$IM_MODEL')
for t in r.tensors:
    if 'token_embd' in t.name or t.name == 'output.weight':
        print(int(t.shape[1]) if len(t.shape) > 1 else 0); break
else:
    print(0)
" 2>/dev/null)
    VOCAB=$(echo "$VOCAB" | head -1 | tr -dc '0-9')
    if [ -n "$VOCAB" ] && [ "$VOCAB" -gt 0 ] 2>/dev/null; then
        LOGIT_GB=$(( VOCAB * IM_BATCH * 4 / 1000000000 ))
        MODEL_GB=$(( $(stat -c %s $IM_MODEL) / 1000000000 ))
        # Only the cards this process can see. Two shards on one box each get
        # their own pair through CUDA_VISIBLE_DEVICES, so counting all of them
        # would say everything fits when it does not.
        PER_GPU=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
        if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
            NGPU=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
        else
            NGPU=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | wc -l)
        fi
        VRAM_GB=$(( PER_GPU * NGPU / 1000 ))
        echo "visible GPUs: ${CUDA_VISIBLE_DEVICES:-all} ($NGPU cards)"
        echo "vocabulary $VOCAB, batch $IM_BATCH"
        echo "weights ~${MODEL_GB} GB + logits ~${LOGIT_GB} GB against ${VRAM_GB} GB of VRAM"
        if [ $(( MODEL_GB + LOGIT_GB + 4 )) -gt $VRAM_GB ]; then
            echo
            echo "That does not fit. Lower the batch and try again:"
            echo "  IM_BATCH=2048 ; im_shard $1 $2"
            return 1
        fi
    fi

    I=$1
    N=$2
    PER=$(( IM_CHUNKS / N + 1 ))
    FROM=$(( I * PER ))
    SHARD_OUT=/imatrix/shard-$I-of-$N.gguf
    mkdir -p /imatrix

    echo "node $I of $N: chunks $FROM onward"
    if [ $I -eq $(( N - 1 )) ]; then
        echo "last node, runs to the end of the corpus"
        LIMIT=""
    else
        echo "taking $PER chunks"
        LIMIT="--chunks $PER"
    fi
    date
    START=$(date +%s)

    PPLFLAG=""
    [ "$IM_NO_PPL" = "1" ] && PPLFLAG="--no-ppl"

    # stdbuf keeps the output line buffered. Without it the pipe into tee
    # switches libc to 4 KB blocks and nothing appears for many minutes.
    stdbuf -oL -eL $BIN/llama-imatrix -m $IM_MODEL -f $IM_CORPUS -o $SHARD_OUT \
        -ngl 99 -c $IM_CTX -b $IM_BATCH -ub $IM_BATCH \
        --parse-special $PPLFLAG --output-frequency $IM_OFREQ \
        --from-chunk $FROM $LIMIT \
        2>&1 | tee /logs/imatrix-shard-$I-of-$N.log

    echo "took $(( $(date +%s) - START )) seconds"
    ls -lh $SHARD_OUT
    autopush /logs/imatrix-shard-$I-of-$N.log logs/imatrix-shard-$I-of-$N.log
    if [ -f "$SHARD_OUT" ]; then
        hf_put "$SHARD_OUT" "imatrix/shard-$I-of-$N.gguf" "$METRICS" "$METRICS_KIND" \
            && echo "  shard uploaded, the merging box can pick it up"
    fi
}

# im_range FROM COUNT LABEL
# im_shard splits evenly, which assumes every box runs at the same speed. They
# do not: rented machines differ by half again on the same cards, and the
# slowest one ends up on the critical path. This takes an explicit range so a
# long shard can be cut up and handed to whoever is free.
#
#   im_range 7275 700 5a
#   im_range 7975 0   5b      count 0 means "to the end of the corpus"
im_range() {
    if [ -z "$2" ] || [ -z "$3" ]; then
        echo "im_range FROM COUNT LABEL      count 0 runs to the end"
        echo "  im_range 7275 700 5a"
        return 1
    fi
    if [ ! -f "$IM_MODEL" ]; then
        echo "IM_MODEL is not set or missing: ${IM_MODEL:-unset}"
        return 1
    fi
    FROM=$1
    COUNT=$2
    SHARD_OUT=/imatrix/shard-$3.gguf
    mkdir -p /imatrix

    if [ "$COUNT" = "0" ]; then
        LIMIT=""
        echo "chunks $FROM to the end of the corpus"
    else
        LIMIT="--chunks $COUNT"
        echo "chunks $FROM to $(( FROM + COUNT - 1 ))"
    fi
    date
    START=$(date +%s)

    PPLFLAG=""
    [ "$IM_NO_PPL" = "1" ] && PPLFLAG="--no-ppl"

    stdbuf -oL -eL $BIN/llama-imatrix -m $IM_MODEL -f $IM_CORPUS -o $SHARD_OUT \
        -ngl 99 -c $IM_CTX -b $IM_BATCH -ub $IM_BATCH \
        --parse-special $PPLFLAG --output-frequency $IM_OFREQ \
        --from-chunk $FROM $LIMIT \
        2>&1 | tee /logs/imatrix-shard-$3.log

    echo "took $(( $(date +%s) - START )) seconds"
    ls -lh $SHARD_OUT 2>/dev/null
    autopush /logs/imatrix-shard-$3.log logs/imatrix-shard-$3.log
    if [ -f "$SHARD_OUT" ]; then
        hf_put "$SHARD_OUT" "imatrix/shard-$3.gguf" "$METRICS" "$METRICS_KIND" \
            && echo "  shard uploaded"
    fi
}

# Merge whatever shards are present, whatever they are called. Use this after
# rebalancing with im_range, when the names no longer follow I-of-N.
# Upload every shard sitting in /imatrix that is not in the repo yet. Safe to
# run as often as you like. Use it whenever a shard finished in a pane that had
# an older copy of this file loaded and never got to its own upload line.
push_shards() {
    need_preset || return 1
    local f name
    for f in /imatrix/shard-*.gguf; do
        [ -f "$f" ] || continue
        name=$(basename "$f")
        echo "$name"
        hf_put "$f" "imatrix/$name" "$METRICS" "$METRICS_KIND" \
            && echo "  up" || echo "  failed"
    done
}

# What is on this box, what is in the repo, and whether the chunk ranges cover
# the corpus without a gap.
im_status() {
    need_preset || return 1
    echo "on this box:"
    ls -lh /imatrix/shard-*.gguf 2>/dev/null | awk '{print "  ", $9, $5}' || echo "   none"
    echo
    echo "in $METRICS:"
    python3 - "$METRICS" "$METRICS_KIND" << 'IMSTEOF'
import sys
from huggingface_hub import HfApi
repo, kind = sys.argv[1], sys.argv[2]
try:
    files = [f for f in HfApi().list_repo_files(repo, repo_type=kind)
             if f.startswith("imatrix/shard-")]
except Exception as e:
    print("   cannot read the repo: %s" % str(e).splitlines()[0])
    sys.exit(0)
if not files:
    print("   none yet")
for f in sorted(files):
    print("  ", f)
print()
print("%d shard(s) published" % len(files))
IMSTEOF
    echo
    echo "chunk ranges are in the logs:"
    grep -h "computing over\|removing initial" /logs/imatrix-shard-*.log 2>/dev/null \
        | sed "s/^/  /" | tail -20
}

# Merge whatever shards are present, whatever they are called.
#
# IM_SKIP holds substrings of shard names to leave out. A shard that was killed
# still left a partial file behind, because llama-imatrix writes snapshots as it
# goes, and merging that alongside the ranges that replaced it counts those
# chunks twice.
#
#   IM_SKIP="shard-5-of-6" im_merge_all
im_merge_all() {
    need_preset || return 1
    repo_ok || return 1
    if [ ! -f "$IM_MODEL" ]; then
        echo "IM_MODEL is not set. Run pick_model."
        return 1
    fi
    mkdir -p /imatrix
    hf download $METRICS --repo-type $METRICS_KIND --include "imatrix/shard-*" --local-dir /tmp/im > /dev/null
    cp -n /tmp/im/imatrix/shard-*.gguf /imatrix/ 2>/dev/null

    local f keep="" skipped="" pat
    for f in $(ls /imatrix/shard-*.gguf 2>/dev/null | sort); do
        for pat in $IM_SKIP; do
            case "$f" in *$pat*) skipped="$skipped $f"; continue 2 ;; esac
        done
        keep="$keep,$f"
    done
    keep=${keep#,}

    if [ -z "$keep" ]; then
        echo "no shards to merge"
        return 1
    fi
    echo "merging:"
    echo "$keep" | tr ',' '\n' | sed "s/^/  /"
    if [ -n "$skipped" ]; then
        echo "skipped (IM_SKIP):"
        for f in $skipped; do echo "  $f"; done
    fi
    echo

    # This build wants one comma separated list, not a repeated flag, and it
    # wants the model even for a pure merge.
    stdbuf -oL -eL $BIN/llama-imatrix -m "$IM_MODEL" --in-file "$keep" \
        -o /imatrix/imatrix.gguf 2>&1 | tee /logs/imatrix-merge.log

    if [ ! -f /imatrix/imatrix.gguf ]; then
        echo "merge produced nothing, read the log above"
        autopush /logs/imatrix-merge.log logs/imatrix-merge.log
        return 1
    fi
    ls -lh /imatrix/imatrix.gguf

    echo
    echo "=== merged statistics ==="
    stdbuf -oL -eL $BIN/llama-imatrix -m "$IM_MODEL" --in-file /imatrix/imatrix.gguf \
        --show-statistics 2>&1 | tee /logs/imatrix-stats.log | head -30

    autopush /logs/imatrix-merge.log logs/imatrix-merge.log
    autopush /logs/imatrix-stats.log logs/imatrix-stats.log
    hf_put /imatrix/imatrix.gguf imatrix/imatrix.gguf "$METRICS" "$METRICS_KIND"
}

im_merge() {
    if [ -z "$1" ]; then
        echo "im_merge N          N is the number of shards"
        return 1
    fi
    need_preset || return 1
    repo_ok || return 1
    N=$1
    mkdir -p /imatrix

    echo "pulling shards from $METRICS"
    hf download $METRICS --repo-type $METRICS_KIND --include "imatrix/*" --local-dir /imatrix
    find /imatrix -name "shard-*-of-$N.gguf" | sort

    GOT=$(find /imatrix -name "shard-*-of-$N.gguf" | wc -l)
    if [ $GOT -ne $N ]; then
        echo "only $GOT of $N shards are here. Wait for the rest, or pass the real count."
        return 1
    fi

    LIST=$(find /imatrix -name "shard-*-of-$N.gguf" | sort | paste -sd,)
    echo "merging: $LIST"

    stdbuf -oL -eL $BIN/llama-imatrix -m "$IM_MODEL" --in-file "$LIST" \
        -o /imatrix/imatrix.gguf 2>&1 | tee /logs/imatrix-merge.log
    ls -lh /imatrix/imatrix.gguf

    echo
    echo "=== merged statistics ==="
    stdbuf -oL -eL $BIN/llama-imatrix -m "$IM_MODEL" --in-file /imatrix/imatrix.gguf \
        --show-statistics 2>&1 | tee /logs/imatrix-stats.log | head -30
    echo
    echo "Sanity check: the sum of squared activations here should equal the sum"
    echo "across the shards. Compare against im_stats on each shard."

    autopush /logs/imatrix-merge.log logs/imatrix-merge.log
    autopush /logs/imatrix-stats.log logs/imatrix-stats.log
    hf_put /imatrix/imatrix.gguf imatrix/imatrix.gguf "$METRICS" "$METRICS_KIND"
}

# Block until the corpus this build needs shows up in the dataset.
wait_calib() {
    if [ -z "$1" ]; then
        echo "wait_calib NAME"
        return 1
    fi
    echo "waiting for builds/$1/calib_train.txt in calib-corpora. Ctrl-C to stop."
    while true; do
        if fetch_one AtomicChat/calib-corpora dataset "builds/$1/calib_train.txt" $IM_CORPUS 2>/dev/null; then
            im_size
            return 0
        fi
        echo "not published yet, $(date +%H:%M:%S), retry in 20s"
        sleep 20
    done
}

# Block until every node has uploaded its shard, then merge.
im_wait_merge() {
    if [ -z "$1" ]; then
        echo "im_wait_merge N"
        return 1
    fi
    need_preset || return 1
    N=$1
    echo "waiting for $N shards in $METRICS. Ctrl-C to stop."
    while true; do
        hf download $METRICS --repo-type $METRICS_KIND --include "imatrix/*" --local-dir /imatrix > /dev/null 2>&1
        GOT=$(find /imatrix -name "shard-*-of-$N.gguf" 2>/dev/null | wc -l)
        echo "  $GOT of $N here, $(date +%H:%M:%S)"
        if [ "$GOT" -ge "$N" ]; then
            break
        fi
        sleep 20
    done
    im_merge $N
}

im_stats() {
    if [ -z "$1" ]; then
        echo "im_stats FILE.gguf"
        return 1
    fi
    $BIN/llama-imatrix -m "$IM_MODEL" --in-file "$1" --show-statistics 2>&1 | head -30
}


# ================================================================== bit accounting

# bits [RULE ...]
# Reads the bf16 gguf, groups the tensors by role, and shows how much of the
# model each group actually is. With rules it also predicts the output size,
# so a target ("must fit 16 GB") can be solved before spending ten minutes
# quantizing to find out.
#
#   bits
#   bits 'ffn_down_exps=iq3_s' 'ffn_up_exps=iq2_m' '*=q8_0'
bits() {
    find_bf16 || return 1
    python3 - "$BF16_FIRST" "$@" << 'BITSEOF'
import fnmatch, sys, os
sys.path.insert(0, "/llama.cpp/gguf-py")
from gguf import GGUFReader

path, rules_raw = sys.argv[1], sys.argv[2:]

# Nominal bits per weight. Real files differ slightly because of per-tensor
# fallbacks, but this is close enough to aim a size at.
BPW = {
    "f32": 32.0, "f16": 16.0, "bf16": 16.0,
    "q8_0": 8.5, "q6_k": 6.5625, "q5_1": 6.0, "q5_k": 5.5, "q5_0": 5.5,
    "q4_k": 4.5, "q4_0": 4.5, "iq4_nl": 4.5, "iq4_xs": 4.25,
    "mxfp4": 4.25, "nvfp4": 4.5,
    "q3_k": 3.4375, "iq3_s": 3.4375, "iq3_xxs": 3.0625,
    "iq2_m": 2.6875, "q2_k": 2.625, "iq2_s": 2.5, "iq2_xs": 2.3125,
    "q2_0": 2.25, "iq2_xxs": 2.0625,
    "iq1_m": 1.75, "iq1_s": 1.5625, "q1_0": 1.125,
}

# Types that never read the importance matrix, whatever you pass.
BLIND = {"q2_0", "mxfp4", "nvfp4", "q8_0", "f16", "bf16", "f32", "q1_0"}

reader = GGUFReader(path)

def group_of(name):
    for key in ("attn_gate", "ssm_out", "ssm_in", "ssm_conv1d", "ssm_norm",
                "ssm_alpha", "ssm_beta", "ssm_dt", "ssm_a", "ssm_d",
                "nextn", "post_attention_norm",
                "ffn_down_exps", "ffn_up_exps", "ffn_gate_exps",
                "ffn_down_shexp", "ffn_up_shexp", "ffn_gate_shexp",
                "ffn_down", "ffn_up", "ffn_gate", "ffn_gate_inp",
                "attn_q", "attn_k", "attn_v", "attn_output", "attn_norm",
                "token_embd", "output"):
        if key in name:
            return key
    return "other"

groups = {}
for t in reader.tensors:
    n = 1
    for d in t.shape:
        n *= int(d)
    g = groups.setdefault(group_of(t.name), {"elems": 0, "count": 0, "ne0": set()})
    g["elems"] += n
    g["count"] += 1
    g["ne0"].add(int(t.shape[0]))

total = sum(g["elems"] for g in groups.values())

print("%-18s %6s %14s %7s   %s" % ("group", "count", "elements", "share", "row length / 256 / 64 / 32"))
for name, g in sorted(groups.items(), key=lambda kv: -kv[1]["elems"]):
    ne0s = sorted(g["ne0"])[:3]
    div = " ".join(
        "%d(%s%s%s)" % (v,
                        "K" if v % 256 == 0 else "-",
                        "6" if v % 64 == 0 else "-",
                        "3" if v % 32 == 0 else "-")
        for v in ne0s)
    print("%-18s %6d %14d %6.1f%%   %s" % (
        name, g["count"], g["elems"], 100.0 * g["elems"] / total, div))
print()
unmatched = [t.name for t in reader.tensors if group_of(t.name) == "other"]
if unmatched:
    big = sorted(((t.name, 1) for t in reader.tensors if group_of(t.name) == "other"))
    sizes = {}
    for t in reader.tensors:
        if group_of(t.name) != "other":
            continue
        n = 1
        for d in t.shape:
            n *= int(d)
        key = t.name.split(".", 2)[-1] if t.name.startswith("blk.") else t.name
        sizes[key] = sizes.get(key, 0) + n
    print()
    print("what fell into 'other', by suffix:")
    for k, v in sorted(sizes.items(), key=lambda kv: -kv[1])[:12]:
        print("  %-40s %14d  %5.1f%%" % (k, v, 100.0 * v / total))
    print()

print("K = row divides by 256, so k and i quants work on it")
print("6 = divides by 64,  3 = divides by 32")
print("A group without K cannot hold any k/i quant: llama.cpp silently swaps in")
print("a block-32 type and keeps the name you asked for.")

if not rules_raw:
    print()
    print("pass rules to predict a size, e.g.")
    print("  bits 'ffn_down_exps=iq3_s' 'ffn_up_exps=iq2_m' '*=q8_0'")
    sys.exit(0)

rules = []
for r in rules_raw:
    pat, _, ty = r.partition("=")
    rules.append((pat.strip(), ty.strip().lower()))

def type_for(name):
    for pat, ty in rules:
        if pat == "*" or fnmatch.fnmatch(name, pat) or pat in name:
            return ty
    return "q8_0"

print()
print("%-18s %-10s %8s %10s" % ("group", "type", "bpw", "GB"))
bytes_total = 0.0
warned = []
for name, g in sorted(groups.items(), key=lambda kv: -kv[1]["elems"]):
    ty = type_for(name)
    bpw = BPW.get(ty)
    if bpw is None:
        print("%-18s %-10s   unknown type" % (name, ty))
        continue
    gb = g["elems"] * bpw / 8 / 1e9
    bytes_total += gb
    print("%-18s %-10s %8.3f %10.2f" % (name, ty, bpw, gb))

    ne0 = sorted(g["ne0"])[0]
    if ty in ("q2_k", "q3_k", "q4_k", "q5_k", "q6_k") or ty.startswith("iq"):
        if ty != "iq4_nl" and ne0 % 256 != 0:
            warned.append("%s: %s needs rows divisible by 256, %d is not. "
                          "llama.cpp will fall back and the file will be bigger "
                          "and worse than this estimate." % (name, ty, ne0))
    if ty in BLIND and g["elems"] > total * 0.1:
        warned.append("%s: %s ignores the importance matrix, and this group is "
                      "%.0f%% of the model. Calibration buys nothing here."
                      % (name, ty, 100.0 * g["elems"] / total))

print("-" * 50)
print("%-18s %-10s %8s %10.2f GB" % ("TOTAL", "", "", bytes_total))
print("%-18s %-10s %8s %10.2f GiB" % ("", "", "", bytes_total * 1e9 / 2**30))
if warned:
    print()
    for w in warned:
        print("!! " + w)
BITSEOF
}


# ================================================================== quantize

# quantize LABEL FALLBACK_TYPE [--tensor-type RULE ...]
# The fallback type covers everything no rule matches. Rules do the real work:
# the expert tensors are most of the model, so they decide the file size, and
# everything else is cheap to keep high.
#
#   quantize AD-IQ4_NL Q8_0 \
#     --tensor-type 'ffn_down_exps=iq4_nl' \
#     --tensor-type 'ffn_up_exps=iq4_nl'
quantize() {
    if [ -z "$2" ]; then
        echo "quantize LABEL FALLBACK_TYPE [--tensor-type RULE ...]"
        echo "  quantize AD-IQ4_NL Q8_0 --tensor-type 'ffn_down_exps=iq4_nl' \\"
        echo "                          --tensor-type 'ffn_up_exps=iq4_nl'"
        return 1
    fi
    need_preset || return 1
    find_bf16 || return 1

    LABEL=$1
    FTYPE=$2
    shift 2

    STEM=$(basename $MAIN | sed "s/-GGUF//")
    QUANT_OUT=/gguf/$STEM-$LABEL.gguf

    # Pin the MTP block unless the caller named that block specifically. The
    # test used to be "does any argument mention blk", which every edge
    # weighted rung does, so the pin was silently skipped and the run aborted
    # partway through on the MTP tensors.
    local mtp
    mtp=$(find_mtp_block)
    if [ -n "$mtp" ] && ! echo "$*" | grep -q "blk[\\.]*[.]$mtp[\\.]*[.]"; then
        echo "pinning the MTP block blk.$mtp to $MTP_TYPE: it collects no"
        echo "imatrix data, so a low bit type would abort the run"
        set -- --tensor-type "blk\.$mtp\.=$MTP_TYPE" "$@"
    fi

    IM=""
    if [ -f /imatrix/imatrix.gguf ]; then
        IM="--imatrix /imatrix/imatrix.gguf"
        echo "using /imatrix/imatrix.gguf"
    else
        echo "NO IMATRIX. This will be an uncalibrated build."
        ask "continue without calibration?" || return 1
    fi

    echo "$BF16_FIRST  ->  $QUANT_OUT   fallback $FTYPE"
    date
    START=$(date +%s)

    stdbuf -oL -eL $BIN/llama-quantize $IM "$@" $BF16_FIRST $QUANT_OUT $FTYPE $(nproc) \
        2>&1 | tee /logs/quantize-$LABEL.log
    local rc=${PIPESTATUS[0]}

    echo "took $(( $(date +%s) - START )) seconds"
    autopush /logs/quantize-$LABEL.log logs/quantize-$LABEL.log

    if [ "$rc" != "0" ]; then
        echo "quantize failed. Removing the partial file so nothing downstream"
        echo "tries to load it:"
        ls -lh $QUANT_OUT 2>/dev/null
        rm -f $QUANT_OUT
        return 1
    fi
    ls -lh $QUANT_OUT
    echo
    echo "publish it with:  push_model $QUANT_OUT"
}

# Exact size for a rule set, in seconds, without quantizing. bits estimates
# from nominal bits per weight and does not understand a regex over block
# numbers, so it cannot price an edge weighted layout. This can.
#
#   dryrun q8_0 --tensor-type 'blk\.([0-3])\.ffn_.*=q6_k' --tensor-type ffn_down=q5_k
dryrun() {
    if [ -z "$1" ]; then
        echo "dryrun FALLBACK [--tensor-type RULE ...]"
        return 1
    fi
    find_bf16 || return 1
    local ftype=$1
    shift
    local mtp
    mtp=$(find_mtp_block)
    if [ -n "$mtp" ]; then
        set -- --tensor-type "blk\.$mtp\.=$MTP_TYPE" "$@"
    fi
    $BIN/llama-quantize --dry-run $IMFLAG "$@" $BF16_FIRST /tmp/dry.gguf $ftype 2>&1 \
        | grep -E "quant size|model size|BPW" 
}

# The list llama-quantize prints under "allowed quantization types" is for the
# positional ftype argument. --tensor-type takes a ggml_type, and the two are
# different sets: IQ2_M, IQ3_M, IQ3_XS, Q2_K_S, Q3_K_*, Q4_K_*, Q5_K_* are
# mixes, recipes for which tensor gets which type, so they cannot be assigned
# to a tensor. Passing one aborts the run after the model has been read.
GGML_TYPES="f32 f16 bf16 q4_0 q4_1 q5_0 q5_1 q8_0 q2_k q3_k q4_k q5_k q6_k \
iq1_s iq1_m iq2_xxs iq2_xs iq2_s iq3_xxs iq3_s iq4_xs iq4_nl tq1_0 tq2_0 \
mxfp4 q1_0 q2_0"

# A multi token prediction head is never executed in a normal forward pass, so
# the imatrix has no statistics for it at any corpus size. Assign it a low bit
# type and llama-quantize refuses outright, halfway through, leaving a
# truncated file behind. Find it and pin it.
find_mtp_block() {
    find_bf16 > /dev/null 2>&1 || return 1
    python3 - "$BF16_FIRST" << 'MTPEOF'
import re, sys
sys.path.insert(0, "/llama.cpp/gguf-py")
from gguf import GGUFReader
blocks = set()
for t in GGUFReader(sys.argv[1]).tensors:
    m = re.match(r"blk\.(\d+)\.nextn\.", t.name)
    if m:
        blocks.add(int(m.group(1)))
print(max(blocks) if blocks else "")
MTPEOF
}

# Type the MTP block gets. It is one block out of many, so keeping it high
# costs almost nothing and it cannot be calibrated anyway.
MTP_TYPE=${MTP_TYPE:-q5_k}

check_types() {
    local r t bad=""
    for r in "$@"; do
        case "$r" in *=*) t=${r#*=} ;; *) continue ;; esac
        case " $GGML_TYPES " in
            *" $t "*) ;;
            *) bad="$bad $t" ;;
        esac
    done
    if [ -n "$bad" ]; then
        echo "not ggml types, cannot go in a --tensor-type rule:$bad"
        echo "those are ftype mixes. Assignable types are:"
        echo "  $GGML_TYPES" | fold -s -w 70 | sed "s/^/    /"
        return 1
    fi
}

# What is published, what is measured, and what is neither. Run it before
# writing the card: a quant in the repo with no measurement is a row of the
# table that cannot be filled in.
audit() {
    need_preset || return 1
    python3 - "$MAIN" "$METRICS" "$METRICS_KIND" "$EVALSET" << 'AUDITEOF'
import os, glob, sys
from huggingface_hub import HfApi

main, metrics, kind, evalset = sys.argv[1:5]
api = HfApi()

def files(repo, t):
    try:
        return api.list_repo_files(repo, repo_type=t)
    except Exception as e:
        print("cannot read %s: %s" % (repo, str(e).splitlines()[0]))
        return []

published = sorted(f[:-5] for f in files(main, "model") if f.endswith(".gguf"))
logs = files(metrics, kind)
measured = set()
for f in logs:
    b = os.path.basename(f)
    if b.startswith("kld-%s--" % evalset) and b.endswith(".log"):
        measured.add(b[len("kld-%s--" % evalset):-4])

local_logs = set()
for p in glob.glob("/logs/kld-%s--*.log" % evalset):
    local_logs.add(os.path.basename(p)[len("kld-%s--" % evalset):-4])

print("published in %s: %d" % (main, len(published)))
print("measured (in the metrics repo): %d" % len(measured))
print("measured (on this box only, not uploaded): %d" % len(local_logs - measured))
print()

missing = [p for p in published if p not in measured and p not in local_logs]
if missing:
    print("PUBLISHED BUT NEVER MEASURED, %d:" % len(missing))
    for p in missing:
        here = os.path.exists("/gguf/%s.gguf" % p)
        print("   %-46s %s" % (p, "on this box" if here else "not on this box"))
    print()
    print("on the box that holds them:  kld_all")
else:
    print("every published quant has a measurement")

only_local = sorted(local_logs - measured)
if only_local:
    print()
    print("measured here but not uploaded, %d:" % len(only_local))
    for p in only_local:
        print("  ", p)
    print("run:  push_logs")

ext = sorted(os.path.basename(p)[:-5] for p in glob.glob("/gguf/external/*.gguf"))
un = [e for e in ext if e not in measured and e not in local_logs]
if un:
    print()
    print("other publishers' files here, not measured, %d:" % len(un))
    for e in un:
        print("  ", e)
    print("run:  kld_ext")
AUDITEOF
}

# ladder_part I N
# Take part I of N from the current /ladder.txt into /my.txt, always from the
# file write_ladder just produced. Slicing by hand with sed leaves a stale copy
# behind after every ladder change, and the stale copy is what then runs.
ladder_part() {
    if [ -z "$2" ]; then
        echo "ladder_part I N       part I of N, 1 based"
        echo "  ladder_part 2 4     second quarter of the ladder"
        return 1
    fi
    # Always regenerate first. A slice cut before an update is the single most
    # common way to spend an hour quantizing rungs that were already fixed.
    write_ladder > /dev/null
    grep -v "^#" /ladder.txt | grep "|" > /tmp/rungs.txt
    local total per from
    total=$(wc -l < /tmp/rungs.txt)
    per=$(( (total + $2 - 1) / $2 ))
    from=$(( ($1 - 1) * per + 1 ))
    sed -n "${from},$(( from + per - 1 ))p" /tmp/rungs.txt > /my.txt
    echo "$total rungs total, part $1 of $2 is $(wc -l < /my.txt) of them:"
    cut -d"|" -f1 /my.txt | sed "s/^/  /"
    echo
    echo "run:  ladder /my.txt --dry     then without --dry"
}

# ladder FILE
# One rung per line:   LABEL | FALLBACK | rule rule rule
# Blank lines and lines starting with # are ignored.
#
# For each rung: predict the size, quantize, measure against the reference,
# publish the file, and move on. Split the file across boxes to run in
# parallel; nothing here depends on another rung.
ladder() {
    if [ -z "$1" ]; then
        echo "ladder FILE       run every rung in it"
        echo "ladder FILE --dry  only predict the sizes"
        return 1
    fi
    if [ ! -f "$1" ]; then
        echo "no such file: $1"
        return 1
    fi
    need_preset || return 1
    find_bf16 || return 1

    local line label fallback rules args r dry="" bad=0
    [ "$2" = "--dry" ] && dry=1

    echo
    echo "this run will build, in order:"
    grep -v "^#" "$1" | grep "|" | cut -d"|" -f1 | tr -d " " | nl -w4 -s". " | sed "s/^/  /"
    echo

    while IFS= read -r line; do
        case "$line" in ""|\#*) continue ;; esac

        # A rung is LABEL | FALLBACK | rules. Anything else is not one: a
        # failed download leaves an html or json error page behind, and every
        # line of it would otherwise be treated as a quantization to run.
        if [ "$(echo "$line" | tr -cd '|' | wc -c)" -lt 2 ]; then
            echo "not a rung, skipping: $line"
            bad=$(( bad + 1 ))
            continue
        fi
        label=$(echo "$line" | cut -d'|' -f1 | tr -d ' ')
        fallback=$(echo "$line" | cut -d'|' -f2 | tr -d ' ')
        rules=$(echo "$line" | cut -d'|' -f3-)
        case "$label" in
            *[!A-Za-z0-9_.-]*|"")
                echo "label is not usable, skipping: $label"
                bad=$(( bad + 1 )); continue ;;
        esac

        echo
        echo "=================================================="
        echo "$label   fallback $fallback"
        echo "  $rules"
        echo

        if ! check_types $rules; then
            bad=$(( bad + 1 ))
            continue
        fi
        bits $rules "*=$fallback" | tail -6

        [ -n "$dry" ] && continue

        args=""
        for r in $rules; do
            args="$args --tensor-type $r"
        done
        quantize "$label" "$fallback" $args || continue

        local stem out
        stem=$(basename $MAIN | sed "s/-GGUF//")
        out=/gguf/$stem-$label.gguf
        [ -f "$out" ] || continue

        kld "$out"
        push_model "$out"
    done < "$1"

    if [ "$bad" -gt 0 ]; then
        echo
        echo "$bad rung(s) were skipped."
        echo "If they were rejected for their types, this file is older than the"
        echo "ladder. Re-cut it from the current one:"
        echo "  write_ladder ; ladder_part I N"
        echo "If every line looked like json or html, the file is a failed"
        echo "download rather than a ladder."
    fi
    results
}

# The ladder lives here rather than in a separate file, so a failed download
# can never leave a json error page standing in for it.
write_ladder() {
    cat > /ladder.txt << 'LADDEREOF'
# Qwen3.8-27B. LABEL | FALLBACK | rules
#
# Built on a layout that was measured against six alternatives at the same size
# class, all on the same bf16 reference:
#
#   uniform layers                     0.015799 @ 16.81 GB
#   4 layers lifted                    0.014492 @ 17.08
#   16 lifted at both ends             0.009811 @ 17.82
#   16 + 8 at the head                 0.007427 @ 18.63
#   ... plus attn_gate up              0.008214 @ 18.39
#   ... plus attn_gate and ssm_out up  0.007296 @ 18.55   <- this recipe
#   wider edges instead                0.008263 @ 18.42
#   output up instead                  0.007996 @ 18.78
#
# What that says. Spending bits on the ends of the network rather than evenly
# halves the divergence. attn_gate and ssm_out are worth a step each: im_stats
# ranks attn_gate first in the whole model by Sum(Act^2), and this model is a
# hybrid so ssm_out carries the recurrent path. token_embd is worth less than
# its 4.7% suggests, since a lookup error stays local to one token, and paying
# for it out of attn_q is a net gain. Widening the edge band further does not
# pay, and neither does lifting the output head.
#
# The pattern per rung, where D is the rung's base type:
#   blocks 0-3 and 52-63 ffn   one step above D
#   blocks 4-11 ffn            D
#   ffn_down                   D, ffn_gate and ffn_up one step below
#   attn_gate, ssm_out         one step above D
#   attn_k, attn_v             q8_0 always, they are 0.3% each
#   output                     high, token_embd low

# ---- above 24 GB
Q8_0             | q8_0 |
AD-Q6_K          | q8_0 | blk\.([0-3]|5[2-9]|6[0-3])\.ffn_.*=q8_0 ffn_down=q6_k ffn_gate=q6_k ffn_up=q6_k attn_q=q6_k attn_gate=q8_0 ssm_out=q8_0

# ---- 32 GB card
AD-Q6_K-Q5_K     | q8_0 | blk\.([0-3]|5[2-9]|6[0-3])\.ffn_.*=q8_0 blk\.([4-9]|1[01])\.ffn_.*=q6_k ffn_down=q6_k ffn_gate=q5_k ffn_up=q5_k attn_q=q5_k attn_gate=q8_0 ssm_out=q8_0 token_embd=q5_k
AD-Q5_K          | q8_0 | blk\.([0-3]|5[2-9]|6[0-3])\.ffn_.*=q6_k blk\.([4-9]|1[01])\.ffn_.*=q6_k ffn_down=q5_k ffn_gate=q5_k ffn_up=q5_k attn_q=q5_k attn_gate=q6_k ssm_out=q6_k output=q6_k token_embd=q4_k

# ---- 24 GB card
AD-Q5_K-Q4_K     | q8_0 | blk\.([0-3]|5[2-9]|6[0-3])\.ffn_.*=q6_k blk\.([4-9]|1[01])\.ffn_.*=q5_k ffn_down=q5_k ffn_gate=q4_k ffn_up=q4_k attn_q=q4_k attn_gate=q6_k ssm_out=q6_k output=q6_k token_embd=iq4_xs
AD-Q4_K          | q8_0 | blk\.([0-3]|5[2-9]|6[0-3])\.ffn_.*=q5_k blk\.([4-9]|1[01])\.ffn_.*=q5_k ffn_down=q4_k ffn_gate=q4_k ffn_up=q4_k attn_q=q4_k attn_gate=q5_k ssm_out=q5_k output=q6_k token_embd=iq4_xs

# ---- 16 GB card
AD-IQ4_XS        | q8_0 | blk\.([0-3]|5[2-9]|6[0-3])\.ffn_.*=q5_k blk\.([4-9]|1[01])\.ffn_.*=q4_k ffn_down=q4_k ffn_gate=iq4_xs ffn_up=iq4_xs attn_q=iq4_xs attn_gate=q5_k ssm_out=q5_k output=q6_k token_embd=iq4_xs
AD-IQ4_XS-IQ3_S  | q8_0 | blk\.([0-3]|5[2-9]|6[0-3])\.ffn_.*=q4_k blk\.([4-9]|1[01])\.ffn_.*=iq4_xs ffn_down=iq4_xs ffn_gate=iq3_s ffn_up=iq3_s attn_q=iq4_xs attn_gate=q4_k ssm_out=q4_k output=q5_k token_embd=iq4_xs
AD-IQ3_S         | q8_0 | blk\.([0-3]|5[2-9]|6[0-3])\.ffn_.*=iq4_xs blk\.([4-9]|1[01])\.ffn_.*=iq4_xs ffn_down=iq3_s ffn_gate=iq3_s ffn_up=iq3_s attn_q=iq4_xs attn_gate=iq4_xs ssm_out=iq4_xs output=q5_k token_embd=iq4_xs

# ---- 12 GB card
AD-IQ3_S-IQ3_XXS | q8_0 | blk\.([0-3]|5[2-9]|6[0-3])\.ffn_.*=iq4_xs blk\.([4-9]|1[01])\.ffn_.*=iq3_s ffn_down=iq3_s ffn_gate=iq3_xxs ffn_up=iq3_xxs attn_q=iq3_s attn_gate=iq4_xs ssm_out=iq4_xs output=q5_k token_embd=iq4_xs
AD-IQ3_XXS       | q8_0 | blk\.([0-3]|5[2-9]|6[0-3])\.ffn_.*=iq3_s blk\.([4-9]|1[01])\.ffn_.*=iq3_s ffn_down=iq3_xxs ffn_gate=iq3_xxs ffn_up=iq3_xxs attn_q=iq3_s attn_gate=iq3_s ssm_out=iq3_s output=q5_k token_embd=iq4_xs
AD-IQ2_S         | q8_0 | blk\.([0-3]|5[2-9]|6[0-3])\.ffn_.*=iq3_s blk\.([4-9]|1[01])\.ffn_.*=iq3_xxs ffn_down=iq2_s ffn_gate=iq2_s ffn_up=iq2_s attn_q=iq3_xxs attn_gate=iq3_s ssm_out=iq3_s output=q5_k token_embd=iq4_xs

# ---- 8 to 10 GB
AD-IQ2_S-IQ2_XS  | q8_0 | blk\.([0-3]|5[2-9]|6[0-3])\.ffn_.*=iq3_xxs blk\.([4-9]|1[01])\.ffn_.*=iq2_s ffn_down=iq2_s ffn_gate=iq2_xs ffn_up=iq2_xs attn_q=iq3_xxs attn_gate=iq3_xxs ssm_out=iq3_xxs output=q4_k token_embd=iq4_xs
AD-IQ2_XS        | q8_0 | blk\.([0-3]|5[2-9]|6[0-3])\.ffn_.*=iq3_xxs blk\.([4-9]|1[01])\.ffn_.*=iq2_s ffn_down=iq2_xs ffn_gate=iq2_xs ffn_up=iq2_xs attn_q=iq2_s attn_gate=iq3_xxs ssm_out=iq3_xxs output=q4_k token_embd=iq4_xs
AD-IQ2_XXS       | q8_0 | blk\.([0-3]|5[2-9]|6[0-3])\.ffn_.*=iq2_s blk\.([4-9]|1[01])\.ffn_.*=iq2_xs ffn_down=iq2_xxs ffn_gate=iq2_xxs ffn_up=iq2_xxs attn_q=iq2_s attn_gate=iq2_s ssm_out=iq2_s output=iq4_xs token_embd=iq4_xs
AD-IQ1_M         | q8_0 | blk\.([0-3]|5[2-9]|6[0-3])\.ffn_.*=iq2_xs blk\.([4-9]|1[01])\.ffn_.*=iq2_xxs ffn_down=iq2_xxs ffn_gate=iq1_m ffn_up=iq1_m attn_q=iq2_s attn_gate=iq2_s ssm_out=iq2_s output=iq4_xs token_embd=iq4_xs
LADDEREOF
    echo "wrote /ladder.txt, $(grep -c '|' /ladder.txt) rungs"
    echo "check the sizes first:   ladder /ladder.txt --dry"
}

# Pull somebody else's build so it can be measured against the same reference.
# Comparisons across different references mean nothing.
# Every publisher names their files the same way, so they go in separate
# directories or they overwrite each other and the measurement belongs to
# whoever downloaded last, with nothing in the log to say who.
get_external() {
    if [ -z "$2" ]; then
        echo "get_external REPO PATTERN"
        echo "  get_external unsloth/Qwen3.8-27B-GGUF '*UD-IQ3_XXS*'"
        return 1
    fi
    local org dest
    org=$(echo "$1" | cut -d/ -f1)
    dest=/gguf/external/$org
    mkdir -p "$dest"
    hf download "$1" --include "$2" --local-dir "$dest"
    ls -la "$dest"/*.gguf 2>/dev/null
    echo
    echo "measured as $org--<file>, so the publisher is in the log name"
}


# ================================================================== results

results() {
    python3 - << 'PYEOF'
import glob, json, os, re, socket

def num(pattern, text):
    m = re.search(pattern, text)
    return float(m.group(1)) if m else None

# One row per file. Quality and speed used to land in separate rows because
# they were keyed differently; they are the same file measured two ways.
rows = {}

def row(name):
    return rows.setdefault(name, {'name': name})

failed = []

for path in glob.glob('/logs/kld-*.log'):
    stem = os.path.basename(path)[4:-4]
    evalset, name = stem.split('--', 1) if '--' in stem else ('neutral', stem)
    text = open(path, errors='ignore').read()
    mean = num(r'Mean\s+KLD:\s+([0-9.]+)', text)
    if mean is None:
        failed.append(name)
        continue
    r = row(name)
    r.setdefault('quality', {})[evalset] = {
        'mean_kld':   mean,
        'q99_kld':    num(r'99\.0%\s+KLD:\s+([0-9.]+)', text),
        'median_kld': num(r'Median\s+KLD:\s+([0-9.]+)', text),
        'top1_pct':   num(r'Same top p:\s+([0-9.]+)', text),
        'rms_dp_pct': num(r'RMS\s+.p\s*:\s+([0-9.]+)', text),
    }

for path in glob.glob('/logs/bench-*.json'):
    name = os.path.basename(path)[6:-5]
    r = row(name)
    try:
        for entry in json.load(open(path)):
            key = 'pp512_tps' if entry.get('n_prompt', 0) > 0 else 'tg128_tps'
            r[key] = entry.get('avg_ts')
            r['gpu'] = entry.get('gpu_info')
            r['build'] = entry.get('build_commit')
    except Exception as e:
        r['bench_error'] = str(e)

# Other publishers all name their files identically, so each one lives under
# /gguf/external/<publisher>/ and is logged as <publisher>--<file>. The path
# has to be rebuilt from that or the row ends up with no size, and a row with
# no size cannot be placed on a size axis.
def find_file(name):
    cands = [os.path.join('/gguf', name + '.gguf'),
             os.path.join('/gguf/experimental', name + '.gguf'),
             os.path.join('/gguf/external', name + '.gguf')]
    if '--' in name:
        org, base = name.split('--', 1)
        cands.append('/gguf/external/%s/%s.gguf' % (org, base))
    cands += glob.glob('/gguf/external/*/%s.gguf' % name)
    for p in cands:
        if os.path.exists(p):
            return p
    return None

for name, r in rows.items():
    p = find_file(name)
    if p:
        r['size_bytes'] = os.path.getsize(p)
        r['size_gb'] = round(os.path.getsize(p) / 1e9, 2)
    if 'UD-' in name or name.startswith('unsloth--'):
        r['publisher'] = 'unsloth'
    elif 'AD-' in name:
        r['publisher'] = 'atomicchat'
    elif p and p.startswith('/gguf/external/'):
        parts = p.split('/')
        r['publisher'] = parts[3] if len(parts) > 4 else 'external'
    else:
        r['publisher'] = 'atomicchat'

out = sorted(rows.values(), key=lambda r: r.get('size_gb') or 0)
for r in out:
    r['measured_on'] = socket.gethostname()

json.dump(out, open('/logs/results.json', 'w'), indent=2)

sets = sorted({s for r in out for s in r.get('quality', {})})
primary = 'neutral' if 'neutral' in sets else (sets[0] if sets else None)

print('%-46s %7s %10s %8s %9s %9s' % ('file', 'GB', 'mean KLD', 'top-1', 'pp512', 'tg128'))
for r in out:
    q = r.get('quality', {}).get(primary, {})
    print('%-46s %7s %10s %8s %9s %9s' % (
        r['name'][-46:],
        r.get('size_gb', ''),
        round(q['mean_kld'], 6) if q.get('mean_kld') is not None else '',
        round(q['top1_pct'], 2) if q.get('top1_pct') is not None else '',
        round(r['pp512_tps'], 1) if r.get('pp512_tps') is not None else '',
        round(r['tg128_tps'], 1) if r.get('tg128_tps') is not None else ''))

if len(sets) > 1:
    print()
    print('eval sets in the file: %s (table shows %s)' % (', '.join(sets), primary))
nosize = [r for r in out if not r.get('size_gb')]
if nosize:
    print()
    print('%d row(s) have no size on this box, so they cannot be plotted yet:' % len(nosize))
    for r in nosize:
        print('   ', r['name'])
    print('the file lives on another box. Run results there too, then merge_results.')

if failed:
    print()
    print('%d run(s) produced no KLD, left out:' % len(failed))
    for n in failed:
        print('   ', n)
print()
print('written to /logs/results.json')
PYEOF
    # Per box, so four boxes measuring different quants do not overwrite one
    # another. merge_results stitches them back together.
    autopush /logs/results.json results-$(hostname).json
}

# Pull every box's results file and stitch them into one.
merge_results() {
    need_preset || return 1
    repo_ok || return 1
    mkdir -p /merged
    hf download $METRICS --repo-type $METRICS_KIND --include "results-*.json" --local-dir /merged
    python3 - << 'MERGEEOF'
import glob, json

rows = {}
for path in sorted(glob.glob('/merged/**/results-*.json', recursive=True)):
    for r in json.load(open(path)):
        cur = rows.setdefault(r['name'], {})
        for k, v in r.items():
            if k == 'quality':
                cur.setdefault('quality', {}).update(v)
            elif v is not None:
                cur[k] = v

out = sorted(rows.values(), key=lambda r: r.get('size_gb') or 0)
json.dump(out, open('/logs/results.json', 'w'), indent=2)

print('%-46s %7s %10s %8s %9s   %s' % ('file', 'GB', 'mean KLD', 'top-1', 'tg128', 'from'))
for r in out:
    q = r.get('quality', {}).get('neutral', {})
    print('%-46s %7s %10s %8s %9s   %s' % (
        r['name'][-46:], r.get('size_gb', ''),
        round(q['mean_kld'], 6) if q.get('mean_kld') is not None else '',
        round(q['top1_pct'], 2) if q.get('top1_pct') is not None else '',
        round(r['tg128_tps'], 1) if r.get('tg128_tps') is not None else '',
        r.get('measured_on', '')))
print()
print('%d files merged into /logs/results.json' % len(out))
MERGEEOF
    hf_put /logs/results.json results.json "$METRICS" "$METRICS_KIND"
}


# ================================================================== upload

# The .kld holds fp16 logits over the whole vocabulary for every scored token.
# People asked for it so they can compare their own quants against the same
# reference, so it gets published. Over 45 GB it goes up as numbered parts.
push_base() {
    need_preset || return 1
    repo_ok || return 1
    if [ ! -f /logs/base-$EVALSET.log ]; then
        echo "no base log yet. Run base first."
        return 1
    fi

    hf_put /logs/base-$EVALSET.log logs/base-$EVALSET.log "$METRICS" "$METRICS_KIND"
    [ -f /logs/env.txt ] && hf_put /logs/env.txt logs/env.txt "$METRICS" "$METRICS_KIND"
    [ -f /logs/llama-commit.txt ] && hf_put /logs/llama-commit.txt logs/llama-commit.txt "$METRICS" "$METRICS_KIND"

    if [ ! -f $BASE ]; then
        echo "no .kld on disk, only the log went up"
        return 0
    fi

    SIZE=$(stat -c %s $BASE)
    echo
    echo "reference blob: $(( SIZE / 1000000000 )) GB"

    # A manifest, not a hash. The hub already verifies transport, and the file
    # is not bit reproducible across different GPU counts and drivers, so a
    # published checksum would mislead rather than help. The one thing worth
    # checking on the far side is that reassembly produced a whole file, and
    # the byte count does that instantly.
    echo "size_bytes $SIZE" > /kld/base-$EVALSET.manifest.txt
    echo "context $CTX" >> /kld/base-$EVALSET.manifest.txt
    echo "eval_set $EVALSET" >> /kld/base-$EVALSET.manifest.txt
    cat /kld/base-$EVALSET.manifest.txt

    if [ "$HASH_BLOB" = "1" ]; then
        echo "HASH_BLOB=1, hashing. Reads the whole file, takes a while."
        sha256sum $BASE | sed "s|/kld/||" > /kld/base-$EVALSET.kld.sha256
        cat /kld/base-$EVALSET.kld.sha256
    fi

    write_kld_readme

    if [ $SIZE -gt 45000000000 ]; then
        echo
        echo "over the 50 GB per-file limit, splitting into 45 GB parts"
        rm -rf /kld/parts
        mkdir -p /kld/parts
        split -b 45G -d --additional-suffix=.part $BASE /kld/parts/base-$EVALSET.kld.
        ls -lh /kld/parts
        cp /kld/base-$EVALSET.manifest.txt /kld/parts/
        [ -f /kld/base-$EVALSET.kld.sha256 ] && cp /kld/base-$EVALSET.kld.sha256 /kld/parts/
        cp /kld/README.md /kld/parts/
        echo
        ask "upload the parts now?" || return 0
        hf_put_dir /kld/parts kld "$METRICS" "$METRICS_KIND"
    else
        ask "upload the blob now?" || return 0
        hf_put $BASE kld/base-$EVALSET.kld "$METRICS" "$METRICS_KIND"
        hf_put /kld/base-$EVALSET.manifest.txt kld/base-$EVALSET.manifest.txt "$METRICS" "$METRICS_KIND"
        hf_put /kld/README.md kld/README.md "$METRICS" "$METRICS_KIND"
    fi
    echo "done"
}

write_kld_readme() {
    cat > /kld/README.md << READMEEOF
# KL divergence reference logits

Produced by \`llama-perplexity --kl-divergence-base\`. The file holds fp16
logits over the full vocabulary for every scored token, so any quantization of
this model can be measured against the same reference we used.

- model    : bf16 weights of $MAIN
- corpus   : $EVALSET held-out set from AtomicChat/calib-corpora
- context  : $CTX
- llama.cpp: see logs/llama-commit.txt
- hardware : see logs/env.txt

## If the file is split

Parts are named \`base-$EVALSET.kld.NN.part\`. Reassemble in order, then check
the hash:

\`\`\`bash
cat base-$EVALSET.kld.*.part > base-$EVALSET.kld
sha256sum -c base-$EVALSET.kld.sha256
\`\`\`

## Using it

\`\`\`bash
llama-perplexity -m your-quant.gguf -f eval_neutral.txt \\
  --kl-divergence-base base-$EVALSET.kld --kl-divergence -c $CTX -ngl 99
\`\`\`

Your corpus and context must match the ones above or the numbers mean nothing.
READMEEOF
    echo "wrote /kld/README.md"
}

# Pull the reference from the metrics repo, reassembling parts if needed.
get_base() {
    need_preset || return 1
    hf download $METRICS --repo-type $METRICS_KIND --include "kld/*" --local-dir /kld

    if [ -f /kld/kld/base-$EVALSET.kld ]; then
        mv /kld/kld/base-$EVALSET.kld $BASE
    elif ls /kld/kld/base-$EVALSET.kld.*.part > /dev/null 2>&1; then
        echo "reassembling parts"
        cat /kld/kld/base-$EVALSET.kld.*.part > $BASE
        rm -f /kld/kld/base-$EVALSET.kld.*.part
        if [ -f /kld/kld/base-$EVALSET.manifest.txt ]; then
            WANT=$(grep size_bytes /kld/kld/base-$EVALSET.manifest.txt | cut -d" " -f2)
            GOT=$(stat -c %s $BASE)
            if [ "$WANT" = "$GOT" ]; then
                echo "size matches the manifest: $GOT bytes"
            else
                echo "SIZE MISMATCH: expected $WANT, got $GOT. A part is missing or truncated."
                return 1
            fi
        fi
    fi
    ls -lh $BASE
}

# Straight box to box, no hub round trip. Much faster on a local-ish link.
#   send_base root@1.2.3.4 41234
send_base() {
    if [ -z "$1" ]; then
        echo "send_base user@host [ssh_port]"
        echo "on vast, get host and port from: vastai ssh-url <id>"
        return 1
    fi
    if [ ! -f $BASE ]; then
        echo "no reference at $BASE"
        return 1
    fi
    which rsync > /dev/null 2>&1 || apt-get install -y -qq rsync
    ls -lh $BASE
    rsync -avP -e "ssh -p ${2:-22} -o StrictHostKeyChecking=no" $BASE $1:$BASE
}

# Every box uploads its own logs as it goes, so merging results across boxes is
# just pulling them all down and re-running the parser.
pull_logs() {
    need_preset || return 1
    repo_ok || return 1
    hf download $METRICS --repo-type $METRICS_KIND --include "logs/*" --local-dir /tmp/pulled
    cp -n /tmp/pulled/logs/* /logs/ 2>/dev/null
    ls /logs | wc -l
    echo "logs in /logs now, running the parser over all of them"
    results
}

get_imatrix() {
    need_preset || return 1
    hf download $METRICS --repo-type $METRICS_KIND --include "imatrix/imatrix.gguf" --local-dir /tmp/im
    mkdir -p /imatrix
    cp /tmp/im/imatrix/imatrix.gguf /imatrix/imatrix.gguf
    ls -lh /imatrix/imatrix.gguf
}


push_logs() {
    need_preset || return 1
    repo_ok || return 1
    du -sh /logs
    if hf_put_dir /logs logs "$METRICS" "$METRICS_KIND"; then
        echo "  $(ls /logs | wc -l) files up -> $METRICS"
    else
        echo "  FAILED"
        return 1
    fi
}

push_results() {
    need_preset || return 1
    repo_ok || return 1
    if [ ! -f /logs/results.json ]; then
        echo "no results.json. Run results first."
        return 1
    fi
    hf_put /logs/results.json results.json "$METRICS" "$METRICS_KIND"
}

push_model() {
    need_preset || return 1
    if [ ! -f "$1" ]; then echo "no file at $1"; return 1; fi
    SIZE=$(stat -c %s $1)
    if [ $SIZE -gt 50000000000 ]; then
        echo "Over the 50 GB per-file limit on Hugging Face. Use push_model_split."
        return 1
    fi
    ls -lh $1
    hf_put "$1" "$(basename $1)" "$MAIN" model
}

# Upload every quant on this box that is not in the repo yet. Idempotent, so
# it is the fix for any upload that failed while the file itself is fine.
push_quants() {
    need_preset || return 1
    token_check > /dev/null || return 1

    local have f name
    have=$(python3 - "$MAIN" << 'HAVEEOF'
import sys
from huggingface_hub import HfApi
try:
    for f in HfApi().list_repo_files(sys.argv[1]):
        if f.endswith(".gguf"):
            print(f)
except Exception:
    pass
HAVEEOF
)
    for f in $(quant_files); do
        name=$(basename "$f")
        if echo "$have" | grep -qx "$name"; then
            echo "already up: $name"
            continue
        fi
        echo "$name  $(du -h "$f" | cut -f1)"
        hf_put "$f" "$name" "$MAIN" model && echo "  up" || echo "  failed"
    done
    echo
    echo "quants live in   $MAIN"
    echo "kld and logs in  $METRICS"
}

# Remove files from the model repo. The cli spells this differently between
# versions, so it goes through the API like every other upload.
del_model() {
    need_preset || return 1
    token_check > /dev/null || return 1
    if [ -z "$1" ]; then
        echo "del_model NAME [NAME ...]      names without the .gguf"
        return 1
    fi
    python3 - "$MAIN" "$@" << 'DELEOF'
import sys
from huggingface_hub import HfApi
api = HfApi()
repo = sys.argv[1]
for name in sys.argv[2:]:
    path = name if name.endswith(".gguf") else name + ".gguf"
    try:
        api.delete_file(path_in_repo=path, repo_id=repo, repo_type="model")
        print("deleted", path)
    except Exception as e:
        print("skip   %s: %s" % (path, str(e).splitlines()[0]))
DELEOF
}

# Every AD file built before edge weighting was measured. They are superseded
# by the same names built with it, and leaving both in the history means two
# different layouts under one name.
del_old_ad() {
    del_model \
        Qwen3.8-27B-AD-Q8_0 Qwen3.8-27B-AD-Q6_K Qwen3.8-27B-AD-Q6_K-Q5_K \
        Qwen3.8-27B-AD-Q5_K Qwen3.8-27B-AD-Q5_K-Q4_K Qwen3.8-27B-AD-Q4_K \
        Qwen3.8-27B-AD-Q4_K-IQ4_XS Qwen3.8-27B-AD-IQ4_XS \
        Qwen3.8-27B-AD-IQ4_XS-IQ3_S Qwen3.8-27B-AD-IQ3_S \
        Qwen3.8-27B-AD-IQ3_S-IQ3_XXS Qwen3.8-27B-AD-IQ3_XXS \
        Qwen3.8-27B-AD-IQ3_XXS-IQ2_S Qwen3.8-27B-AD-IQ2_S \
        Qwen3.8-27B-AD-IQ2_S-IQ2_XS Qwen3.8-27B-AD-IQ2_XS \
        Qwen3.8-27B-AD-IQ2_XS-IQ2_XXS Qwen3.8-27B-AD-IQ2_XXS \
        Qwen3.8-27B-AD-IQ2_XXS-IQ1_M Qwen3.8-27B-AD-IQ2_XXS-IQ1_S
}

push_model_split() {
    need_preset || return 1
    if [ ! -f "$1" ]; then echo "no file at $1"; return 1; fi
    PREFIX=/gguf/split/$(basename $1 .gguf)
    mkdir -p /gguf/split
    $BIN/llama-gguf-split --split --split-max-size 45G $1 $PREFIX
    ls -lh $PREFIX*
    hf_put_dir /gguf/split "$(basename $1 .gguf)" "$MAIN" model
    echo "Uploaded as a folder of shards. Readers point at shard 1."
}

push_card() {
    need_preset || return 1
    if [ ! -f "$1" ]; then echo "no file at $1"; return 1; fi
    hf_put "$1" README.md "$MAIN" model
}


# ================================================================== flows

run_base_box() {
    need_preset || return 1
    get_eval
    get_bf16
    base
    push_base
}

wait_base() {
    need_preset || return 1
    echo "Polling $METRICS for the reference. Ctrl-C to stop."
    while true; do
        get_base 2>/dev/null
        if [ -f $BASE ]; then
            echo "Got it."
            return 0
        fi
        echo "not there yet, $(date +%H:%M:%S), retry in 30s"
        sleep 30
    done
}

run_quant_box() {
    need_preset || return 1
    get_eval
    get_quants
    wait_base
    kld_all
    push_logs
    push_results
}

# ================================================================== MLX
#
# The MLX side of the pipeline. Same rules as everything above: one function
# at a time, in the foreground, nothing hidden in the background.
#
# Read this once, it saves an afternoon:
#
#   A GGUF quant is a file. An MLX quant is a whole checkpoint directory:
#   config.json + sharded safetensors + tokenizer. The loader takes the
#   directory or the repo id, not a file inside it. That is why every rung of
#   the ladder needs its own repository on the hub.
#
#   mlx_vlm.convert is the primary tool here, not mlx_lm.convert. It is the
#   one that keeps the vision tower, and its output is what the desktop app
#   loads directly. Reach for mlx-lm only for a method that mlx-vlm does not
#   have, and run mlx_caps to find out which those are on THIS box rather
#   than trusting anyone's list, this file included.
#
#   The method used to pick the numbers leaves no trace in the output. Plain
#   rounding, awq, gptq, distillation and per layer allocation all write the
#   same affine quantized checkpoint. Nothing downstream needs to know which
#   was used. That is why a checkpoint made by an mlx-lm command can be loaded
#   by mlx-vlm once the directory has the shape mlx-vlm expects, which is what
#   mlx_reattach is for.
#
#   One bit is a special case worth knowing about. mlx-vlm can RUN a one bit
#   affine checkpoint: bits: 1 in the config, packed uint32 weights, scales
#   and biases, group size 32, 64 or 128. Nothing in the stack PRODUCES one.
#   See mlx_1bit_status.

MLXROOT=${MLXROOT:-/mlx}
MLX_SRC=${MLX_SRC:-/src}
MLX_EVAL=${MLX_EVAL:-/eval/neutral.txt}
MLX_CTX=${MLX_CTX:-512}
MLX_STEP=${MLX_STEP:-64}
MLX_REF=${MLX_REF:-}
MLX_GROUP=${MLX_GROUP:-64}
MLX_BACKEND=${MLX_BACKEND:-cuda13}
MLX_SENS=${MLX_SENS:-/mlx/sensitivities.json}


# ------------------------------------------------------------------ setup

# MLX_BACKEND picks the wheel: cuda13 for an Nvidia box on CUDA 13, cuda12 for
# CUDA 12, cpu for a machine with no GPU at all. Plain conversion works on cpu,
# measurement and distillation do not.
mlx_setup() {
    echo "backend  : $MLX_BACKEND"
    echo "installs : mlx, mlx-vlm with the training extra, mlx-lm with it too"
    echo
    echo "the training extra brings the optimizer and autograd pieces. Without"
    echo "it the learned quantization entry points import-error even when they"
    echo "are present in the package."
    echo
    case "$MLX_BACKEND" in
        cuda13) echo "CUDA 13 wheel, needs an Nvidia driver 580 or newer:" ;;
        cuda12) echo "CUDA 12 wheel." ;;
        cpu)    echo "CPU only. Conversion works, everything else crawls." ;;
        *) echo "unknown MLX_BACKEND: $MLX_BACKEND (cuda13, cuda12 or cpu)"
           return 1 ;;
    esac
    nvidia-smi --query-gpu=name,driver_version --format=csv 2>/dev/null
    echo
    ask "install?" || return 1

    pip install --break-system-packages -q -U "mlx[$MLX_BACKEND]" || return 1
    pip install --break-system-packages -q -U "mlx-vlm[train]" || return 1
    pip install --break-system-packages -q -U "mlx-lm[train]" || return 1
    mkdir -p $MLXROOT /logs

    echo
    mlx_check
    echo
    echo "now run mlx_caps. Do not skip it: it is the only thing that says"
    echo "what these versions actually give you."
}

mlx_check() {
    python3 - << 'MLXCHKEOF' | tee /logs/mlx-env.txt
import importlib, sys, time

def ver(name):
    try:
        return importlib.import_module(name).__version__
    except Exception as e:
        return "NOT INSTALLED (%s)" % str(e).splitlines()[0][:50]

print("mlx      :", ver("mlx"))
print("mlx_vlm  :", ver("mlx_vlm"))
print("mlx_lm   :", ver("mlx_lm"))

try:
    import mlx.core as mx
except Exception as e:
    print("mlx does not import at all:", e)
    sys.exit(1)

print("device   :", mx.default_device())

a = mx.random.normal((2048, 2048))
mx.eval(a)
t = time.time()
for _ in range(10):
    b = a @ a
mx.eval(b)
dt = time.time() - t
gflops = 10 * 2 * 2048 ** 3 / dt / 1e9
print("matmul   : %.0f GFLOP/s over ten 2048x2048 runs" % gflops)
if gflops < 200:
    print()
    print("that is processor speed, not GPU speed. Either the cpu wheel got")
    print("installed or the CUDA backend did not pick up the card.")
MLXCHKEOF
    echo
    echo "written to /logs/mlx-env.txt. Every published number should carry"
    echo "these versions. The flags on these tools move between releases."
}

# What the INSTALLED packages actually expose. Written because a blog post from
# two months ago said mlx-vlm has no learned quantization, its README says the
# converter does awq, and neither is evidence about the version on this disk.
# Nothing in this file should be planned around a claim this function has not
# confirmed.
mlx_caps() {
    mkdir -p /logs/mlx-help
    echo "=============== console entry points ==============="
    local c
    for c in mlx_vlm.convert mlx_vlm.generate mlx_vlm.server \
             mlx_lm.convert mlx_lm.dwq mlx_lm.awq mlx_lm.gptq \
             mlx_lm.dynamic_quant mlx_lm.evaluate mlx_lm.perplexity \
             mlx_lm.upload; do
        if command -v $c > /dev/null 2>&1; then
            echo "  [x] $c"
        else
            echo "  [ ] $c"
        fi
    done

    echo
    echo "=============== inside the mlx-vlm package ==============="
    python3 - << 'CAPSEOF'
import os, re, sys

try:
    import mlx_vlm
except Exception as e:
    print("mlx_vlm does not import:", e)
    sys.exit(0)

root = os.path.dirname(mlx_vlm.__file__)
print("package at", root)

wanted = ["awq", "dwq", "gptq", "dynamic_quant", "quant_predicate",
          "QUANT_RECIPES", "skip_multimodal", "skip_vision", "turboquant"]
hits = {w: [] for w in wanted}
for dirpath, _, files in os.walk(root):
    if "tests" in dirpath:
        continue
    for f in files:
        if not f.endswith(".py"):
            continue
        p = os.path.join(dirpath, f)
        try:
            src = open(p, encoding="utf-8", errors="replace").read()
        except Exception:
            continue
        for w in wanted:
            if re.search(r"\b%s\b" % re.escape(w), src, re.IGNORECASE):
                hits[w].append(os.path.relpath(p, root))

for w in wanted:
    n = len(hits[w])
    mark = "[x]" if n else "[ ]"
    where = hits[w][0] if n else ""
    extra = (" and %d more" % (n - 1)) if n > 1 else ""
    print("  %s %-16s %s%s" % (mark, w, where, extra))

print()
print("a name here means the string is in the source, not that a working")
print("command exists. The converter help below is the real test.")
CAPSEOF

    echo
    echo "=============== mlx_vlm.convert options ==============="
    if command -v mlx_vlm.convert > /dev/null 2>&1; then
        mlx_vlm.convert --help > /logs/mlx-help/mlx_vlm.convert.txt 2>&1
        grep -E "^[[:space:]]+-" /logs/mlx-help/mlx_vlm.convert.txt
    else
        echo "not installed"
    fi

    echo
    echo "=============== mlx_lm entry points ==============="
    for c in mlx_lm.convert mlx_lm.dwq mlx_lm.awq mlx_lm.gptq mlx_lm.dynamic_quant; do
        if command -v $c > /dev/null 2>&1; then
            $c --help > /logs/mlx-help/$c.txt 2>&1
            echo "--- $c ---"
            grep -E "^[[:space:]]+--" /logs/mlx-help/$c.txt | head -12
        fi
    done

    echo
    echo "full help pages in /logs/mlx-help/"
    echo
    echo "three things to read out of the above before anything long starts:"
    echo "  1. does mlx_vlm.convert take an awq option, and a dwq or gptq one"
    echo "  2. what the bits and group flags are called here"
    echo "     (--q-bits or --bits, --q-group-size or --group-size)"
    echo "  3. what --skip-vision does in THIS version: leave the tower"
    echo "     unquantized, or drop it from the output entirely"
    echo
    echo "whatever mlx_vlm.convert can do, do THERE. The tower survives and"
    echo "the result loads in the app with nothing reassembled."
}

# One bit is loadable but not producible. This says where that stands on the
# installed versions, because if it ever changes it changes here first.
mlx_1bit_status() {
    echo "=============== loading a one bit checkpoint ==============="
    python3 - << 'ONEBITEOF'
import os, re, sys
try:
    import mlx_vlm
except Exception as e:
    print("  mlx_vlm does not import:", e)
    sys.exit(0)
root = os.path.dirname(mlx_vlm.__file__)
found = []
for dirpath, _, files in os.walk(root):
    for f in files:
        if not f.endswith(".py"):
            continue
        p = os.path.join(dirpath, f)
        src = open(p, encoding="utf-8", errors="replace").read()
        if re.search(r"bits\s*==\s*1\b|one_?bit|1-?bit", src, re.IGNORECASE):
            found.append(os.path.relpath(p, root))
if found:
    print("  supported, the handling lives in:")
    for f in found[:6]:
        print("   ", f)
else:
    print("  no one bit handling found in this version")
ONEBITEOF

    echo
    echo "=============== producing one ==============="
    python3 - << 'ONEBITQEOF'
import mlx.core as mx
w = mx.random.normal((256, 512))
for b in (1, 2, 3, 4, 5, 6, 8):
    try:
        mx.quantize(w, group_size=64, bits=b)
        print("  bits=%d  ok" % b)
    except Exception as e:
        print("  bits=%d  refused: %s" % (b, str(e).splitlines()[0][:60]))
ONEBITQEOF

    echo
    echo "If loading works and producing does not, that gap is the most useful"
    echo "thing on this board. The runtime is already waiting and the contract"
    echo "is written down: packed uint32 weights, scales and biases, group"
    echo "size 32, 64 or 128, and bits: 1 declared in config.json."
}


# ------------------------------------------------------------------ naming

mlx_stem() {
    [ -n "$UPSTREAM" ] && basename "$UPSTREAM"
}

mlx_out() {
    echo "$MLXROOT/$(mlx_stem)-MLX-$1"
}

mlx_repo() {
    echo "AtomicChat/$(mlx_stem)-MLX-$1"
}

mlx_metrics_repo() {
    echo "AtomicChat/$(mlx_stem)-MLX-metrics"
}

mlx_need() {
    if [ -z "$UPSTREAM" ]; then
        echo "no preset loaded. Run use_model NAME REPO first."
        return 1
    fi
    if ! command -v mlx_vlm.convert > /dev/null 2>&1; then
        echo "mlx-vlm is not installed. Run mlx_setup."
        return 1
    fi
}


# ------------------------------------------------------------------ day zero

# mlx_probe SMALL_REPO
#
# Run this before renting anything big. It answers two questions the rest of
# the file depends on: does mlx_vlm.convert keep the vision tower on this
# architecture, and does mlx_lm.convert handle it at all. If mlx-vlm covers
# every method needed, the mlx-lm half of this file never gets used.
mlx_probe() {
    if [ -z "$1" ]; then
        echo "mlx_probe SMALL_REPO_FROM_THE_SAME_FAMILY"
        echo "  mlx_probe Qwen/Qwen3.8-2B"
        echo
        echo "confirm the id exists first:  find_repo Qwen3.8"
        return 1
    fi
    local repo="$1"
    local a=/mlx/probe-vlm
    local b=/mlx/probe-lm
    mkdir -p $MLXROOT /logs

    echo "=============== 1. mlx_vlm.convert, the primary path ==============="
    date
    rm -rf $a
    stdbuf -oL -eL mlx_vlm.convert --hf-path "$repo" --mlx-path $a \
        -q --q-bits 4 --q-group-size 64 2>&1 | tee /logs/mlx-probe-vlm.log
    echo

    echo "=============== 2. mlx_lm.convert, the fallback ==============="
    date
    rm -rf $b
    stdbuf -oL -eL mlx_lm.convert --hf-path "$repo" --mlx-path $b \
        -q --q-bits 4 --q-group-size 64 2>&1 | tee /logs/mlx-probe-lm.log
    echo

    echo "=============== 3. what came out ==============="
    mlx_inspect $a
    echo
    mlx_inspect $b
    echo
    echo "=============== 4. reading the result ==============="
    echo "The vlm build should carry a vision section and vision tensors. If it"
    echo "does, that is the path for everything and the ladder is one tool."
    echo
    echo "The lm build only matters if mlx_caps showed a method that mlx-vlm"
    echo "does not have. If it also kept the vision section and the same tensor"
    echo "prefixes, the two are interchangeable. If it dropped the vision half,"
    echo "that method needs mlx_reattach afterwards. If it refused the model"
    echo "type outright, that method is unavailable here."
}

mlx_inspect() {
    local p="${1:-}"
    if [ -z "$p" ] || [ ! -d "$p" ]; then
        echo "mlx_inspect /path/to/checkpoint"
        return 1
    fi
    echo "--- $p  ($(du -sh $p 2>/dev/null | cut -f1)) ---"
    python3 - "$p" << 'INSPEOF'
import glob, json, os, sys
p = sys.argv[1]

cfg_path = os.path.join(p, "config.json")
if not os.path.exists(cfg_path):
    print("no config.json, this is not a checkpoint directory")
    sys.exit(1)
cfg = json.load(open(cfg_path))

print("model_type   :", cfg.get("model_type"))
print("architectures:", cfg.get("architectures"))
print("vision block :", "YES" if ("vision_config" in cfg) else "no")
print("text block   :", "nested under text_config" if "text_config" in cfg else "flat")

q = cfg.get("quantization")
if q is None:
    print("quantization : none, full precision checkpoint")
else:
    print("quantization : bits=%s group_size=%s mode=%s"
          % (q.get("bits"), q.get("group_size"), q.get("mode")))
    per_layer = {k: v for k, v in q.items() if isinstance(v, dict)}
    if per_layer:
        seen = {}
        for v in per_layer.values():
            key = (v.get("bits"), v.get("group_size"))
            seen[key] = seen.get(key, 0) + 1
        print("per layer    : %d entries" % len(per_layer))
        for (b, g), n in sorted(seen.items(), key=lambda kv: -kv[1]):
            print("   %s tensors at bits=%s group=%s" % (n, b, g))

names = []
try:
    from safetensors import safe_open
    for f in sorted(glob.glob(os.path.join(p, "*.safetensors"))):
        with safe_open(f, framework="np") as h:
            names.extend(list(h.keys()))
except Exception as e:
    print("cannot read tensors:", str(e).splitlines()[0])

if names:
    prefixes = {}
    for n in names:
        head = ".".join(n.split(".")[:2])
        prefixes[head] = prefixes.get(head, 0) + 1
    print("tensors      : %d" % len(names))
    print("prefixes     :")
    for k, v in sorted(prefixes.items(), key=lambda kv: -kv[1])[:8]:
        print("   %-34s %d" % (k, v))
    vis = [n for n in names if "vision" in n or "visual" in n]
    print("vision tensors: %d" % len(vis))
INSPEOF
}


# ------------------------------------------------------------------ weights

mlx_src() {
    mlx_need || return 1
    if [ -d "$MLX_SRC" ] && ls $MLX_SRC/*.safetensors > /dev/null 2>&1; then
        echo "already here: $(du -sh $MLX_SRC | cut -f1) in $MLX_SRC"
        return 0
    fi
    echo "pulling the original weights of $UPSTREAM into $MLX_SRC"
    echo "for a 27B model in bf16 that is around 56 GB"
    ask "download?" || return 1
    mkdir -p $MLX_SRC
    hf download $UPSTREAM --local-dir $MLX_SRC
    du -sh $MLX_SRC
}


# ------------------------------------------------------------------ the ladder

# mlx_quant BITS [GROUP]
# The main rung builder. Goes through mlx_vlm.convert so the vision tower
# survives, which means the result loads in the app with nothing reassembled.
#
# BITS is one of 2 3 4 5 6 8. One is loadable but no tool produces it, see
# mlx_1bit_status.
#
# GROUP is how many neighbouring weights share one scale and one offset. The
# real cost: two 16 bit numbers per group, so group 64 adds 0.5 bits to every
# weight and group 32 adds 1.0. A four bit build at group 64 is 4.5 bits per
# weight on disk. Tighter groups are worth the weight at 2 and 3 bits and are
# not worth it at 6 and 8.
mlx_quant() {
    [ -z "$1" ] && { echo "mlx_quant BITS [GROUP]"; return 1; }
    mlx_need || return 1
    local bits=$1 group=${2:-$MLX_GROUP} label out rc start
    label="${bits}bit"; [ "$group" != "64" ] && label="${bits}bit-g${group}"
    out=$(mlx_out "$label")
    case "$bits" in 2|3|4|5|6|8) ;; *) echo "bits: 2 3 4 5 6 8"; return 1 ;; esac
    echo "target : $out"
    echo "bits   : $bits, group $group, mode affine, method rtn"
    date; start=$(date +%s); rm -rf "$out"
    stdbuf -oL -eL mlx_vlm.convert --hf-path "$UPSTREAM" --mlx-path "$out" \
        -q --q-bits "$bits" --q-group-size "$group" --q-mode affine \
        --quant-method rtn 2>&1 | tee "/logs/mlx-convert-$label.log"
    rc=${PIPESTATUS[0]}
    echo "took $(( $(date +%s) - start )) seconds"
    [ "$rc" != "0" ] && { rm -rf "$out"; return 1; }
    mlx_size "$out"
}

mlx_ver() {
    python3 - << 'VEREOF'
from importlib.metadata import version
for p in ("mlx", "mlx-lm", "mlx-vlm"):
    try:
        print("%-8s %s" % (p, version(p)))
    except Exception as e:
        print("%-8s not installed" % p)
VEREOF
}


mlx_quant_text() {
    echo "mlx_vlm.convert doesn't have --skip-vision flag in this version."
    echo "text only builds go through:  mlx_quant_lm BITS [GROUP]"
    return 1
}

# The fallback text path through mlx-lm. Needed only if the flag above turns
# out to mean something else, or for a method mlx-vlm does not carry.
mlx_quant_lm() {
    if [ -z "$1" ]; then
        echo "mlx_quant_lm BITS [GROUP]      fallback path through mlx-lm"
        return 1
    fi
    mlx_need || return 1
    local bits=$1
    local group=${2:-$MLX_GROUP}
    local out
    out=$(mlx_out "${bits}bit-TextOnly-lm")

    rm -rf "$out"
    stdbuf -oL -eL mlx_lm.convert \
        --hf-path "$UPSTREAM" --mlx-path "$out" \
        -q --q-bits "$bits" --q-group-size "$group" \
        2>&1 | tee "/logs/mlx-convert-${bits}bit-lm.log"
    mlx_size "$out"
}

# The whole uniform ladder with the vision tower, one rung after another.
# Rungs already on disk are skipped, so this is safe to interrupt and resume.
# Group 32 at the bottom where the extra scales pay for themselves, 64 above.
mlx_ladder() {
    mlx_need || return 1
    local spec n=0 total bits group out label
    local specs="8:64 6:64 5:64 4:64 4:32 3:32 2:32"
    total=$(echo $specs | wc -w)
    for spec in $specs; do
        bits=${spec%%:*}
        group=${spec##*:}
        n=$(( n + 1 ))
        label="${bits}bit"
        [ "$group" != "64" ] && label="${bits}bit-g${group}"
        out=$(mlx_out "$label")
        echo
        echo "########## $n of $total: $label ##########"
        if [ -d "$out" ]; then
            echo "already built, skipping. Delete $out to redo it."
            continue
        fi
        mlx_quant "$bits" "$group" || echo "rung $label failed, moving on"
    done
    echo
    echo "on disk now:"
    du -sh $MLXROOT/*/ 2>/dev/null | grep -v probe
}


# ------------------------------------------------------------------ learned

# mlx_awq BITS [SAMPLES]
# Scales and clips the weights before rounding them, using which activation
# channels actually carry signal. Cheaper than distillation.
#
# The mlx-vlm README lists awq on its converter, so this tries there first and
# falls back to mlx-lm. Confirm the real option name with mlx_caps and fix the
# line below if it differs on your version.
mlx_awq() {
    [ -z "$1" ] && { echo "mlx_awq BITS [GROUP] [text|multimodal]"; return 1; }
    mlx_need || return 1
    local bits=$1 group=${2:-$MLX_GROUP} calib=${3:-text} label out start
    label="${bits}bit-AWQ"; [ "$group" != "64" ] && label="${bits}bit-g${group}-AWQ"
    [ "$calib" = "multimodal" ] && label="$label-MM"
    out=$(mlx_out "$label")
    echo "target : $out"
    echo "calib  : $calib"
    date; start=$(date +%s); rm -rf "$out"
    stdbuf -oL -eL mlx_vlm.convert --hf-path "$UPSTREAM" --mlx-path "$out" \
        -q --q-bits "$bits" --q-group-size "$group" --q-mode affine \
        --quant-method awq --calibration "$calib" \
        2>&1 | tee "/logs/mlx-awq-$label.log"
    echo "took $(( $(date +%s) - start )) seconds"
    mlx_size "$out"
}

# mlx_dwq BITS [GROUP] [TEACHER]
# Distillation: gradient descent on the scales and offsets so the quantized
# model's output moves back toward the full model's output.
#
# It repairs damage, so it pays between 2 and 4 bits and does nearly nothing at
# 6 and 8, where there is not enough damage left to repair. The teacher does
# not have to be full precision: an 8 bit build works about as well and halves
# the memory, which is what makes this fit on one card.
mlx_dwq() {
    if [ -z "$1" ]; then
        echo "mlx_dwq BITS [GROUP] [TEACHER_PATH]"
        echo "  mlx_dwq 4"
        echo "  mlx_dwq 2 32"
        echo "  mlx_dwq 3 32 /mlx/<stem>-MLX-8bit      cheaper teacher"
        return 1
    fi
    mlx_need || return 1
    local bits=$1
    local group=${2:-32}
    local teacher=${3:-$UPSTREAM}
    local label="${bits}bit-DWQ"
    [ "$group" != "64" ] && label="${bits}bit-g${group}-DWQ"

    case "$bits" in
        2|3|4) ;;
        *) echo "$bits bits: distillation has almost nothing to work with here."
           ask "run it anyway?" || return 1 ;;
    esac

    local out
    if grep -qi "dwq" /logs/mlx-help/mlx_vlm.convert.txt 2>/dev/null; then
        out=$(mlx_out "$label")
        echo "mlx_vlm.convert advertises dwq, the tower will survive"
        echo "teacher : $teacher"
        echo "target  : $out"
        ask "start?" || return 1
        rm -rf "$out"
        stdbuf -oL -eL mlx_vlm.convert \
            --hf-path "$teacher" --mlx-path "$out" \
            -q --q-bits "$bits" --q-group-size "$group" \
            --quant-method dwq \
            2>&1 | tee "/logs/mlx-dwq-$label.log"
    else
        if ! command -v mlx_lm.dwq > /dev/null 2>&1; then
            echo "no dwq anywhere: not in the mlx_vlm.convert help and no"
            echo "mlx_lm.dwq command. Run mlx_caps and read the output."
            return 1
        fi
        out=$(mlx_out "$label-TextOnly")
        echo "going through mlx-lm. The result is text only and needs"
        echo "mlx_reattach afterwards to see images again."
        echo "teacher : $teacher"
        echo "target  : $out"
        echo
        echo "if this runs out of memory, in this order: point the teacher at"
        echo "the 8 bit build, lower --max-seq-length, then --batch-size 1."
        ask "start?" || return 1
        rm -rf "$out"
        stdbuf -oL -eL mlx_lm.dwq \
            --model "$teacher" --mlx-path "$out" \
            --bits "$bits" --group-size "$group" \
            --num-samples 1024 --batch-size 1 --max-seq-length 512 \
            2>&1 | tee "/logs/mlx-dwq-$label.log"
    fi

    echo
    echo "read the loss curve in the log. Oscillating and not falling means the"
    echo "learning rate is too high. Falling but barely means too low."
    mlx_size "$out"
    autopush "/logs/mlx-dwq-$label.log" "logs/mlx-dwq-$label.log"
}

# Per layer bit allocation. Two different things hide behind this name and
# mlx_caps tells you which one is available:
#   a structural recipe, which lifts the first and last blocks and a few
#   tensor roles by a fixed rule, and
#   a measured allocation, which quantizes one layer at a time, watches how
#   far the output moves, and spends the budget where it moves most.
# The second is the one worth having. The first still beats uniform.
mlx_mixed() {
    if [ -z "$1" ]; then
        echo "mlx_mixed RECIPE"
        echo "  mixed_2_6 mixed_3_4 mixed_3_5 mixed_3_6 mixed_3_8 mixed_4_6 mixed_4_8"
        echo "  read the name as: mixed_2_6 puts the bulk at 2 bits and the"
        echo "  sensitive tensors at 6"
        return 1
    fi
    mlx_need || return 1
    local out; out=$(mlx_out "$1")
    rm -rf "$out"
    stdbuf -oL -eL mlx_vlm.convert --hf-path "$UPSTREAM" --mlx-path "$out" \
        -q --q-mode affine --quant-predicate "$1" \
        2>&1 | tee "/logs/mlx-mixed-$1.log"
    mlx_size "$out"
}

# The sensitivity profile, computed once and reused by every measured rung.
# Same idea as an importance matrix in llama.cpp arrived at from the other
# side: instead of collecting activation statistics it perturbs one layer at a
# time and watches the output move.
mlx_sens() {
    mlx_need || return 1
    if [ -f "$MLX_SENS" ]; then
        echo "already computed: $MLX_SENS"
        ls -lh "$MLX_SENS"
        return 0
    fi
    if ! command -v mlx_lm.dynamic_quant > /dev/null 2>&1; then
        echo "mlx_lm.dynamic_quant is not installed. Run mlx_setup."
        return 1
    fi
    mkdir -p $MLXROOT
    echo "measuring per layer sensitivity of $UPSTREAM"
    echo "this runs the model many times and is the slow step on the MLX side."
    echo "Budget an hour or more on a 27B. It happens once."
    ask "start?" || return 1
    date
    local start
    start=$(date +%s)

    stdbuf -oL -eL mlx_lm.dynamic_quant \
        --model "$UPSTREAM" \
        --mlx-path "$(mlx_out AD-probe)" \
        --target-bpw 4.5 \
        2>&1 | tee /logs/mlx-sensitivity.log

    echo "took $(( $(date +%s) - start )) seconds"
    echo
    echo "the tool writes the profile next to its output. Candidates:"
    find $MLXROOT /root . -maxdepth 3 -name "*sensitiv*" \
        -newer /logs/mlx-sensitivity.log 2>/dev/null | head
    echo "move the right one to $MLX_SENS and the mixed rungs will reuse it."
    autopush /logs/mlx-sensitivity.log logs/mlx-sensitivity.log
}


# ------------------------------------------------------------------ reattach

# mlx_reattach TEXT_CHECKPOINT VISION_CHECKPOINT OUT
#
# Only needed for a method that exists in mlx-lm and not in mlx-vlm. Takes the
# text tensors from the mlx-lm build, which carry the benefit of distillation
# or measured allocation, and writes them next to the untouched tower from an
# mlx-vlm build, under the names the vlm loader expects.
#
# The result is a claim, not a result, until it is loaded and shown a picture.
mlx_reattach() {
    if [ -z "$3" ]; then
        echo "mlx_reattach TEXT_CHECKPOINT VISION_CHECKPOINT OUT"
        echo "  mlx_reattach /mlx/X-4bit-DWQ-TextOnly /mlx/X-4bit /mlx/X-4bit-DWQ"
        echo
        echo "the vision checkpoint is used for its tower and its config only,"
        echo "its text weights are replaced by the ones from the text build."
        return 1
    fi
    python3 - "$1" "$2" "$3" << 'REATTEOF'
import glob, json, os, shutil, sys
from safetensors import safe_open
from safetensors.numpy import save_file

text_p, vis_p, out_p = sys.argv[1:4]

def read_all(path):
    out = {}
    for f in sorted(glob.glob(os.path.join(path, "*.safetensors"))):
        with safe_open(f, framework="np") as h:
            for k in h.keys():
                out[k] = h.get_tensor(k)
    return out

print("reading text build  :", text_p)
tw = read_all(text_p)
print("  %d tensors" % len(tw))
print("reading vision build:", vis_p)
vw = read_all(vis_p)
print("  %d tensors" % len(vw))

if not tw or not vw:
    print("one of them has no tensors, stopping")
    sys.exit(1)

sample = sorted(tw.keys())[len(tw) // 2]
tail = sample.split(".", 1)[1] if "." in sample else sample
cands = [k for k in vw if k.endswith(tail)]
if not cands:
    print()
    print("cannot match tensor names between the two builds.")
    print("  text sample:", sample)
    print("  vlm  sample:", sorted(vw.keys())[len(vw) // 2])
    print("Look at both with mlx_inspect and map them by hand.")
    sys.exit(1)
prefix = cands[0][: len(cands[0]) - len(tail)]
print("language prefix in the vlm build: %r" % prefix)

merged = {}
vision_kept = 0
replaced = 0
missing = []
for k, v in vw.items():
    if k.startswith(prefix):
        src = k[len(prefix):]
        if src in tw:
            merged[k] = tw[src]
            replaced += 1
        else:
            merged[k] = v
            missing.append(src)
    else:
        merged[k] = v
        vision_kept += 1

print("replaced from the text build : %d" % replaced)
print("kept from the vlm build      : %d" % vision_kept)
if missing:
    print("not found in the text build  : %d, kept as they were" % len(missing))
    for m in missing[:10]:
        print("   ", m)

os.makedirs(out_p, exist_ok=True)
for f in glob.glob(os.path.join(vis_p, "*")):
    if f.endswith(".safetensors") or f.endswith(".safetensors.index.json"):
        continue
    if os.path.isfile(f):
        shutil.copy2(f, out_p)

tcfg = json.load(open(os.path.join(text_p, "config.json")))
vcfg = json.load(open(os.path.join(out_p, "config.json")))
if "quantization" in tcfg:
    vcfg["quantization"] = tcfg["quantization"]
    json.dump(vcfg, open(os.path.join(out_p, "config.json"), "w"), indent=2)
    print("copied the quantization block from the text build into the config")

save_file(merged, os.path.join(out_p, "model.safetensors"))
size = os.path.getsize(os.path.join(out_p, "model.safetensors"))
print()
print("wrote %s, %.2f GB" % (out_p, size / 1e9))
print()
print("now load it and show it a picture. Until then this is a claim.")
REATTEOF
}


# ------------------------------------------------------------------ size

mlx_size() {
    local p="${1:-}"
    if [ -z "$p" ] || [ ! -d "$p" ]; then
        echo "mlx_size /path/to/checkpoint"
        return 1
    fi
    python3 - "$p" "$MLX_SRC" << 'SIZEEOF'
import glob, json, os, sys
p, src = sys.argv[1], sys.argv[2]

total = sum(os.path.getsize(f) for f in glob.glob(os.path.join(p, "*.safetensors")))
print("on disk      : %.2f GB  (%.2f GiB)" % (total / 1e9, total / 2 ** 30))

params = None
idx = os.path.join(src, "model.safetensors.index.json")
if os.path.exists(idx):
    meta = json.load(open(idx)).get("metadata", {})
    if "total_size" in meta:
        params = meta["total_size"] // 2

if params:
    print("parameters   : %.2f B  (from the bf16 source index)" % (params / 1e9))
    print("bits/weight  : %.3f" % (total * 8 / params))
else:
    print("parameters   : unknown, no source index at %s" % idx)
    print("               run mlx_src to get the real bits per weight")
SIZEEOF
}


# ------------------------------------------------------------------ measurement

# There is no tool for this upstream. Both libraries ship perplexity and
# downstream task evaluation, neither of which compares a quant against a
# reference model.
#
# In llama.cpp the reference logits go to disk because the two runs are
# separate processes. Here both models are objects in one process, so the
# divergence is accumulated on the fly and nothing large is written.
#
# Every number this prints is defined in the script. Read the definitions
# before putting them in a table next to somebody else's.

mlx_ref() {
    if [ -z "$1" ]; then
        echo "mlx_ref PATH_OR_REPO      current: ${MLX_REF:-not set}"
        echo "  mlx_ref $UPSTREAM"
        echo "  mlx_ref $(mlx_out 8bit)   cheaper, and shifts every number"
        return 1
    fi
    MLX_REF=$1
    echo "reference is now $MLX_REF"
    echo "put it in the card. A number measured against a different reference"
    echo "is not the same number."
}

mlx_kld() {
    local q="${1:-}"
    if [ -z "$q" ]; then
        echo "mlx_kld /path/to/checkpoint"
        ls -d $MLXROOT/*/ 2>/dev/null | sed "s/^/   /"
        return 1
    fi
    if [ -z "$MLX_REF" ]; then
        echo "no reference set. Run mlx_ref first."
        return 1
    fi
    if [ ! -f "$MLX_EVAL" ]; then
        echo "no corpus at $MLX_EVAL. Run get_eval."
        return 1
    fi
    local name
    name=$(basename "$q")
    mkdir -p /logs

    echo "reference : $MLX_REF"
    echo "quant     : $q"
    echo "corpus    : $MLX_EVAL"
    echo "window    : $MLX_CTX tokens, math in blocks of $MLX_STEP"
    echo

    stdbuf -oL -eL python3 - "$MLX_REF" "$q" "$MLX_EVAL" "$MLX_CTX" "$MLX_STEP" \
        "/logs/mlx-kld-$name.json" 2>&1 | tee "/logs/mlx-kld-$name.log" << 'KLDEOF'
import json, math, os, sys, time
import mlx.core as mx

# mlx_lm.load opens a vision checkpoint text-only, which is what is wanted
# here: this measures the language half, the half every quant touches.
from mlx_lm import load

ref_path, qnt_path, corpus, ctx, step, out_json = sys.argv[1:7]
ctx, step = int(ctx), int(step)

print("loading the reference")
ref_model, tok = load(ref_path)
print("loading the quant")
qnt_model, _ = load(qnt_path)

text = open(corpus, encoding="utf-8", errors="replace").read()
ids = tok.encode(text)
n_win = len(ids) // ctx
print("corpus: %d tokens, %d windows of %d" % (len(ids), n_win, ctx))
if n_win < 2:
    print("too few windows to say anything. Longer corpus or shorter window.")
    sys.exit(1)

klds = []
top1_hits = 0
top1_total = 0
dp_abs = []
dp_rel = []
t0 = time.time()

for w in range(n_win):
    chunk = ids[w * ctx:(w + 1) * ctx]
    x = mx.array([chunk])
    lr = ref_model(x)[0]
    lq = qnt_model(x)[0]
    mx.eval(lr, lq)

    # position t predicts token t+1, so the last position has no target
    for s in range(0, ctx - 1, step):
        e = min(s + step, ctx - 1)
        a = lr[s:e].astype(mx.float32)
        b = lq[s:e].astype(mx.float32)
        logp = a - mx.logsumexp(a, axis=-1, keepdims=True)
        logq = b - mx.logsumexp(b, axis=-1, keepdims=True)
        p = mx.exp(logp)
        k = mx.sum(p * (logp - logq), axis=-1)
        hits = mx.sum(mx.argmax(a, axis=-1) == mx.argmax(b, axis=-1))

        tgt = mx.array(chunk[s + 1:e + 1])
        rows = mx.arange(e - s)
        p_true = mx.exp(logp[rows, tgt])
        q_true = mx.exp(logq[rows, tgt])
        d_abs = (q_true - p_true) * 100.0
        d_rel = (q_true - p_true) / mx.maximum(p_true, 1e-12) * 100.0

        mx.eval(k, hits, d_abs, d_rel)
        klds.extend([float(v) for v in k])
        top1_hits += int(hits)
        top1_total += (e - s)
        dp_abs.extend([float(v) for v in d_abs])
        dp_rel.extend([float(v) for v in d_rel])

    del lr, lq

    if (w + 1) % 5 == 0 or w == n_win - 1:
        el = time.time() - t0
        rate = (w + 1) / el
        eta = (n_win - w - 1) / rate
        print("  window %d/%d   %.2f win/s   eta %d min %02d s"
              % (w + 1, n_win, rate, eta // 60, eta % 60), flush=True)

klds.sort()

def q(frac):
    return klds[min(len(klds) - 1, int(len(klds) * frac))]

def rms(v):
    return math.sqrt(sum(x * x for x in v) / len(v))

res = {
    "reference": ref_path,
    "quant": qnt_path,
    "corpus": os.path.basename(corpus),
    "window": ctx,
    "tokens_scored": len(klds),
    "mean_kld": sum(klds) / len(klds),
    "median_kld": q(0.50),
    "p90_kld": q(0.90),
    "p95_kld": q(0.95),
    "p99_kld": q(0.99),
    "max_kld": klds[-1],
    "top1_agree_pct": 100.0 * top1_hits / top1_total,
    "mean_delta_p_points": sum(dp_abs) / len(dp_abs),
    "rms_delta_p_points": rms(dp_abs),
    "mean_delta_p_relative_pct": sum(dp_rel) / len(dp_rel),
    "rms_delta_p_relative_pct": rms(dp_rel),
}
json.dump(res, open(out_json, "w"), indent=2)

print()
print("tokens scored     : %d" % res["tokens_scored"])
print("mean KLD          : %.6f" % res["mean_kld"])
print("median KLD        : %.6f" % res["median_kld"])
print("90 / 95 / 99 pct  : %.6f  %.6f  %.6f" % (res["p90_kld"], res["p95_kld"], res["p99_kld"]))
print("max KLD           : %.6f" % res["max_kld"])
print("same top-1        : %.3f %%" % res["top1_agree_pct"])
print("delta p, points   : mean %.4f   rms %.4f" % (res["mean_delta_p_points"], res["rms_delta_p_points"]))
print("delta p, relative : mean %.4f %%  rms %.4f %%" % (res["mean_delta_p_relative_pct"], res["rms_delta_p_relative_pct"]))
print()
print("definitions used here:")
print("  KLD is the sum over the vocabulary of p*(log p - log q), reference")
print("  first. Same top-1 is how often both models rank the same token")
print("  highest. Delta p in points is (q - p)*100 on the probability of the")
print("  token that actually comes next in the corpus, relative is the same")
print("  divided by p. llama.cpp reports one of those two under the name RMS")
print("  delta p. Check which one before putting these beside GGUF numbers.")
print()
print("written to %s" % out_json)
KLDEOF

    autopush "/logs/mlx-kld-$name.log" "logs/mlx-kld-$name.log"
    autopush "/logs/mlx-kld-$name.json" "logs/mlx-kld-$name.json"
}

mlx_kld_all() {
    if [ -z "$MLX_REF" ]; then
        echo "no reference set. Run mlx_ref first."
        return 1
    fi
    local d name todo="" n=0 total
    for d in $MLXROOT/*/; do
        [ -f "$d/config.json" ] || continue
        name=$(basename "$d")
        case "$name" in probe-*) continue ;; esac
        [ "$d" = "$MLX_REF/" ] && continue
        if [ -f "/logs/mlx-kld-$name.json" ] && [ "$KLD_FORCE" != "1" ]; then
            echo "already measured: $name"
            continue
        fi
        todo="$todo $d"
    done
    if [ -z "$todo" ]; then
        echo "everything here is measured. KLD_FORCE=1 mlx_kld_all to redo."
        mlx_results
        return 0
    fi
    total=$(echo $todo | wc -w)
    for d in $todo; do
        n=$(( n + 1 ))
        echo
        echo "########## $n of $total ##########"
        mlx_kld "${d%/}"
    done
    mlx_results
}

mlx_results() {
    python3 - "$MLXROOT" << 'RESEOF'
import glob, json, os, sys
root = sys.argv[1]

rows = []
for p in sorted(glob.glob("/logs/mlx-kld-*.json")):
    try:
        r = json.load(open(p))
    except Exception:
        continue
    name = os.path.basename(p)[len("mlx-kld-"):-len(".json")]
    d = os.path.join(root, name)
    size = 0
    if os.path.isdir(d):
        size = sum(os.path.getsize(f) for f in glob.glob(os.path.join(d, "*.safetensors")))
    r["name"] = name
    r["size_gb"] = round(size / 1e9, 2) if size else None
    rows.append(r)

rows.sort(key=lambda r: r.get("size_gb") or 0)
json.dump(rows, open("/logs/mlx-results.json", "w"), indent=2)

print("%-44s %8s %11s %10s %11s" % ("build", "GB", "mean KLD", "top-1 %", "rms dp pts"))
for r in rows:
    print("%-44s %8s %11s %10s %11s" % (
        r["name"][-44:], r.get("size_gb", ""),
        round(r["mean_kld"], 6),
        round(r["top1_agree_pct"], 3),
        round(r["rms_delta_p_points"], 3)))
print()
print("%d builds, written to /logs/mlx-results.json" % len(rows))
if rows:
    print("reference for all of these: %s" % rows[0]["reference"])
RESEOF
    autopush /logs/mlx-results.json "mlx-results-$(hostname).json"
}


# ------------------------------------------------------------------ publishing

mlx_push() {
    if [ -z "$2" ]; then
        echo "mlx_push CHECKPOINT LABEL"
        echo "  mlx_push /mlx/Qwen3.8-27B-MLX-4bit 4bit"
        return 1
    fi
    mlx_need || return 1
    token_check > /dev/null || return 1
    if [ ! -d "$1" ]; then
        echo "no checkpoint at $1"
        return 1
    fi
    scan_secrets "$1" || return 1
    local repo
    repo=$(mlx_repo "$2")
    echo "$1  ->  $repo   ($(du -sh $1 | cut -f1))"
    ask "upload?" || return 1
    stdbuf -oL -eL mlx_lm.upload --path "$1" --upload-repo "$repo" \
        2>&1 | tee "/logs/mlx-upload-$2.log"
    echo
    echo "users load it as:  mlx_vlm.generate --model $repo --image pic.jpg"
}

mlx_push_all() {
    mlx_need || return 1
    local d name label
    for d in $MLXROOT/*/; do
        [ -f "$d/config.json" ] || continue
        name=$(basename "$d")
        case "$name" in probe-*) continue ;; esac
        label=${name#"$(mlx_stem)-MLX-"}
        echo
        echo "=== $name ==="
        mlx_push "${d%/}" "$label"
    done
}


# ------------------------------------------------------------------ orientation

mlx_status() {
    echo
    if [ -z "$UPSTREAM" ]; then
        echo "  [ ] preset             ->  use_model NAME REPO"
    else
        echo "  [x] upstream: $UPSTREAM"
        echo "      builds: $MLXROOT/$(mlx_stem)-MLX-*"
        echo "      repos : $(mlx_repo '<label>')"
    fi
    if command -v mlx_vlm.convert > /dev/null 2>&1; then
        echo "  [x] mlx-vlm installed  (the primary tool)"
    else
        echo "  [ ] mlx-vlm            ->  mlx_setup"
    fi
    if [ -f /logs/mlx-help/mlx_vlm.convert.txt ]; then
        echo "  [x] capabilities known ->  mlx_caps to refresh"
    else
        echo "  [ ] capabilities       ->  mlx_caps    (do not skip this)"
    fi
    if [ -d /mlx/probe-vlm ]; then
        echo "  [x] probe done         ->  mlx_inspect /mlx/probe-vlm"
    else
        echo "  [ ] probe not done     ->  mlx_probe <small model, same family>"
    fi
    if [ -d "$MLX_SRC" ] && ls $MLX_SRC/*.safetensors > /dev/null 2>&1; then
        echo "  [x] source weights     ($(du -sh $MLX_SRC 2>/dev/null | cut -f1))"
    else
        echo "  [ ] source weights     ->  mlx_src"
    fi
    local n
    n=$(ls -d $MLXROOT/*/ 2>/dev/null | grep -v probe | wc -l)
    echo "  [$([ $n -gt 0 ] && echo x || echo ' ')] builds on disk: $n   ->  mlx_ladder"
    if [ -n "$MLX_REF" ]; then
        echo "  [x] reference: $MLX_REF"
    else
        echo "  [ ] reference          ->  mlx_ref PATH_OR_REPO"
    fi
    n=$(ls /logs/mlx-kld-*.json 2>/dev/null | wc -l)
    echo "  [$([ $n -gt 0 ] && echo x || echo ' ')] measurements: $n     ->  mlx_kld_all"
    echo
    echo "  full list: mlx_help"
    echo
}

mlx_help() {
cat << 'MLXHELPEOF'

SETUP        mlx_setup | mlx_check | mlx_caps | mlx_status
             mlx_caps is not optional. It reports what the installed versions
             actually give you, which is the only thing worth planning around.

DAY ZERO     mlx_probe SMALL_REPO | mlx_inspect PATH | mlx_1bit_status

WEIGHTS      mlx_src

LADDER       mlx_quant BITS [GROUP]       primary, keeps the vision tower
             mlx_quant_text BITS [GROUP]  same rung without the tower
             mlx_quant_lm BITS [GROUP]    fallback through mlx-lm
             mlx_ladder                   the whole uniform grid
             bits: 2 3 4 5 6 8. One loads but nothing produces it.

LEARNED      mlx_awq BITS [SAMPLES]
             mlx_dwq BITS [GROUP] [TEACHER]
             mlx_sens
             mlx_mixed RECIPE | mlx_mixed BPW LOW HIGH

VISION       mlx_reattach TEXT VISION OUT
             only for a method mlx-vlm does not have

MEASURE      mlx_ref PATH | mlx_kld PATH | mlx_kld_all | mlx_results
             mlx_size PATH

PUBLISH      mlx_push PATH LABEL | mlx_push_all

ORDER THAT WORKS

  mlx_setup ; mlx_check ; mlx_caps
  mlx_1bit_status                    read it, that gap is the opportunity
  mlx_probe Qwen/Qwen3.8-2B          read the verdict before continuing
  mlx_src
  mlx_ladder                         the uniform grid, tower included
  mlx_ref <upstream or the 8bit build>
  mlx_kld_all ; mlx_results          numbers before publishing, not after
  mlx_push_all
  mlx_awq 4 ; mlx_dwq 4 32 ; mlx_dwq 3 32 ; mlx_dwq 2 32
  mlx_sens ; mlx_mixed 2.6 2 4 ; mlx_mixed 3.4 3 5
  mlx_kld_all ; mlx_results          one graph out of the whole grid

MLXHELPEOF
}

# ================================================================== MEASURE
#
# One protocol for both engines, and a Mac that starts from nothing.
#
# Three things this block exists to fix.
#
#   The KLD script written earlier did not follow llama.cpp's conventions.
#   llama.cpp scores only the second half of each chunk, so every scored token
#   has at least half a context behind it. The earlier script scored from
#   position zero, which counts tokens the model predicted almost blind, and
#   those are exactly where a quant looks worst. Every MLX number measured
#   before this block is therefore pessimistic and has to be redone.
#
#   Speed and quality are different measurements and people keep mixing their
#   settings. Quality is llama-perplexity with -c 4096: context is how much
#   text the model sees while predicting. Speed is llama-bench with -p 512
#   -n 128: those are how many tokens to feed and to generate for timing, and
#   -d is where context enters. Both are in the existing bench function
#   already; nothing here changes the numbers, it only adds depth.
#
#   A Mac cannot use the paths this file assumes. Nothing can be created at /
#   on macOS, so FROOT redirects everything under $HOME/foundry there and
#   stays empty on Linux, where /gguf and /logs keep working as before.

FOUNDRY_OS=$(uname -s)
if [ "$FOUNDRY_OS" = "Darwin" ]; then
    FROOT=${FROOT:-$HOME/foundry}
else
    FROOT=${FROOT:-}
fi

GGUF_DIR=${GGUF_DIR:-$FROOT/gguf}
MLX_DIR=${MLX_DIR:-$FROOT/mlx}
LOG_DIR=${LOG_DIR:-$FROOT/logs}
EVAL_DIR=${EVAL_DIR:-$FROOT/eval}

# One protocol, written down once. Change it here or nowhere.
SPEED_PROMPT=${SPEED_PROMPT:-512}
SPEED_GEN=${SPEED_GEN:-128}
SPEED_DEPTHS=${SPEED_DEPTHS:-"0 4096 16384"}
SPEED_REPS=${SPEED_REPS:-5}
SPEED_COOL=${SPEED_COOL:-45}

# Quality. These match the GGUF metrics repo exactly so the two ladders can go
# in one table: context 4096, the neutral held-out set, second half scored.
Q_CTX=${Q_CTX:-4096}
Q_FIRST=${Q_FIRST:-}          # empty means ctx/2, which is what llama.cpp uses
Q_STEP=${Q_STEP:-64}          # how many positions the vocabulary math takes at once


is_mac() { [ "$FOUNDRY_OS" = "Darwin" ]; }


# ------------------------------------------------------------------ mac setup

# A Mac Studio with nothing on it. Run these in order.
mac_setup() {
    if ! is_mac; then
        echo "this is not a Mac. On Linux use setup and build."
        return 1
    fi
    echo "=== command line tools ==="
    if ! xcode-select -p > /dev/null 2>&1; then
        echo "installing, accept the dialog then rerun mac_setup"
        xcode-select --install
        return 1
    fi
    echo "present at $(xcode-select -p)"

    echo
    echo "=== homebrew ==="
    if ! command -v brew > /dev/null 2>&1; then
        echo "not installed. Install it, then rerun:"
        echo '  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'
        return 1
    fi
    brew list cmake > /dev/null 2>&1 || brew install cmake
    brew list git > /dev/null 2>&1 || brew install git
    echo "cmake $(cmake --version | head -1)"

    echo
    echo "=== python packages ==="
    pip3 install --break-system-packages -q -U "huggingface_hub[hf_xet]" \
        mlx mlx-lm mlx-vlm || return 1

    mkdir -p $GGUF_DIR $MLX_DIR $LOG_DIR $EVAL_DIR
    echo
    mac_info
    echo
    echo "next:  mac_build   then   mac_memory"
}

mac_info() {
    is_mac || { echo "not a Mac"; return 1; }
    {
        echo "chip     : $(sysctl -n machdep.cpu.brand_string)"
        echo "cores    : $(sysctl -n hw.ncpu)"
        echo "memory   : $(( $(sysctl -n hw.memsize) / 1000000000 )) GB unified"
        echo "macos    : $(sw_vers -productVersion)"
        python3 - << 'PYEOF'
from importlib.metadata import version
for p in ("mlx", "mlx-lm", "mlx-vlm", "huggingface_hub"):
    try:
        print("%-16s %s" % (p, version(p)))
    except Exception:
        print("%-16s not installed" % p)
PYEOF
    } | tee $LOG_DIR/mac-env.txt
    echo
    echo "written to $LOG_DIR/mac-env.txt, it ships with the results"
}

# macOS caps how much of the unified memory the GPU may wire down. The default
# is around three quarters of what is installed, which on a 64 GB machine is
# roughly 48 GB. A 29.5 GB model plus its context fits, a 51 GB one does not,
# and the failure looks like a crash rather than a message.
mac_memory() {
    is_mac || return 1
    local total limit
    total=$(( $(sysctl -n hw.memsize) / 1048576 ))
    limit=$(sysctl -n iogpu.wired_limit_mb 2>/dev/null || echo 0)
    echo "installed        : $(( total / 1024 )) GB"
    if [ "$limit" = "0" ]; then
        echo "gpu wired limit  : default, about $(( total * 75 / 100 / 1024 )) GB"
    else
        echo "gpu wired limit  : $(( limit / 1024 )) GB (set explicitly)"
    fi
    echo
    echo "largest model that fits comfortably: leave 8 GB for the system and"
    echo "for the attention cache, so about $(( total * 75 / 100 / 1024 - 8 )) GB of weights."
    echo
    echo "to raise it, until the next reboot:"
    echo "  sudo sysctl iogpu.wired_limit_mb=$(( total * 90 / 100 ))"
    echo
    echo "Raising it starves the system. Do it for one measurement, not as a"
    echo "permanent setting, and never past 90 percent."
}

mac_build() {
    is_mac || return 1
    mkdir -p $FROOT
    if [ -d $FROOT/llama.cpp ]; then
        echo "already cloned, pulling"
        git -C $FROOT/llama.cpp pull --ff-only
    else
        git clone https://github.com/ggml-org/llama.cpp $FROOT/llama.cpp || return 1
    fi
    cd $FROOT/llama.cpp
    git rev-parse --short HEAD > $LOG_DIR/llama-commit.txt
    echo "commit $(cat $LOG_DIR/llama-commit.txt)"

    # Metal is on by default on macOS. Naming it makes the build reproducible
    # rather than dependent on what cmake guesses.
    cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON || return 1
    cmake --build build -j "$(sysctl -n hw.ncpu)" --target \
        llama-bench llama-cli llama-perplexity llama-mtmd-cli || return 1

    MACBIN=$FROOT/llama.cpp/build/bin
    ls -la $MACBIN
    echo
    echo "MACBIN=$MACBIN"
    echo "next:  mac_get   to pull the ladders"
}

MACBIN=${MACBIN:-$FROOT/llama.cpp/build/bin}

# Pull our own published ladders onto the Mac. Nothing is built here, this
# machine only measures.
mac_get() {
    local what="${1:-}"
    if [ -z "$what" ]; then
        echo "mac_get gguf | mlx | eval | all"
        return 1
    fi
    mkdir -p $GGUF_DIR $MLX_DIR $EVAL_DIR
    case "$what" in
        gguf|all)
            echo "=== gguf ladder, about 250 GB ==="
            ask "download?" && hf download AtomicChat/Qwen3.8-27B-GGUF \
                --include "*.gguf" --exclude "*BF16*" --exclude "*bf16*" \
                --local-dir $GGUF_DIR
            ;;&
        mlx|all)
            echo "=== mlx ladder, one directory per rung ==="
            local r
            for r in mixed_3_4 4bit mixed_4_6 5bit 6bit 8bit; do
                echo "  $r"
                hf download AtomicChat/Qwen3.8-27B-MLX-$r \
                    --local-dir $MLX_DIR/Qwen3.8-27B-MLX-$r > /dev/null || echo "   failed"
            done
            ;;&
        eval|all)
            hf download AtomicChat/calib-corpora --repo-type dataset \
                --include "eval/neutral/eval_neutral.txt" --local-dir $EVAL_DIR
            find $EVAL_DIR -name "eval_neutral.txt" -exec cp {} $EVAL_DIR/neutral.txt \;
            ls -lh $EVAL_DIR/neutral.txt
            ;;
    esac
    echo
    du -sh $GGUF_DIR $MLX_DIR 2>/dev/null
}


# ------------------------------------------------------------------ speed

# Both engines, one protocol. Nothing here is engine specific except which
# binary runs: prompt length, generation length, cache depth, repetitions and
# cooldown are identical, so the two columns can sit in one table.
#
# Speed is the one number where comparing across engines is legitimate. A user
# on a Mac does not care whose kernels are faster, only how many tokens per
# second appear. Quality is the opposite and needs the reference treatment.

speed_note() {
    echo "protocol : $SPEED_PROMPT prompt tokens, $SPEED_GEN generated,"
    echo "           depths [$SPEED_DEPTHS], $SPEED_REPS repetitions, median"
    echo "           $SPEED_COOL s cooldown between models"
    echo
    echo "prefill is how fast your prompt is read, decode is how fast the answer"
    echo "is written. Decode at depth 0 is the best case nobody experiences,"
    echo "because the attention cache grows with the conversation and reading it"
    echo "back is what slows generation down. The depth 4096 row is the honest one."
}

speed_gguf() {
    local f="${1:-}"
    if [ -z "$f" ] || [ ! -f "$f" ]; then
        echo "speed_gguf /path/to/quant.gguf"
        ls $GGUF_DIR/*.gguf 2>/dev/null | sed "s/^/   /"
        return 1
    fi
    if [ ! -x "$MACBIN/llama-bench" ] && [ ! -x "$BIN/llama-bench" ]; then
        echo "no llama-bench. Run mac_build, or build on Linux."
        return 1
    fi
    local bench=$MACBIN/llama-bench
    [ -x "$bench" ] || bench=$BIN/llama-bench

    local name d depth_flag=""
    name=$(basename "$f" .gguf)
    if "$bench" --help 2>&1 | grep -q -- "-d,"; then
        depth_flag=yes
    else
        echo "this llama-bench has no -d flag, depth 0 only. Update llama.cpp."
    fi
    mkdir -p $LOG_DIR

    echo "### gguf: $name   $(du -h "$f" | cut -f1)"
    if [ -n "$depth_flag" ]; then
        for d in $SPEED_DEPTHS; do
            echo "  depth $d"
            "$bench" -m "$f" -p "$SPEED_PROMPT" -n "$SPEED_GEN" -d "$d" \
                -ngl 99 -r "$SPEED_REPS" -o json \
                > "$LOG_DIR/speed-gguf-$name-d$d.json" 2>"$LOG_DIR/speed-gguf-$name-d$d.err"
        done
    else
        "$bench" -m "$f" -p "$SPEED_PROMPT" -n "$SPEED_GEN" \
            -ngl 99 -r "$SPEED_REPS" -o json \
            > "$LOG_DIR/speed-gguf-$name-d0.json" 2>"$LOG_DIR/speed-gguf-$name-d0.err"
    fi
    echo "  done"
}

speed_mlx() {
    local d="${1:-}"
    if [ -z "$d" ] || [ ! -f "$d/config.json" ]; then
        echo "speed_mlx /path/to/checkpoint"
        ls -d $MLX_DIR/*/ 2>/dev/null | sed "s/^/   /"
        return 1
    fi
    kld_install > /dev/null
    local name
    name=$(basename "${d%/}")
    echo "### mlx: $name   $(du -sh "$d" | cut -f1)"
    stdbuf -oL -eL python3 $FROOT/speed_mlx.py "${d%/}" \
        "$SPEED_PROMPT" "$SPEED_GEN" "$SPEED_REPS" "$SPEED_DEPTHS" \
        "$LOG_DIR/speed-mlx-$name.json"
}

# Everything on this machine, both engines, with cooldowns. The first model is
# repeated at the end: if that repeat differs by more than a few percent the
# machine got hot and the whole session should be thrown away.
speed_all() {
    speed_note
    echo
    mac_info 2>/dev/null | head -4
    echo
    local first="" f d
    for f in $GGUF_DIR/*.gguf; do
        [ -f "$f" ] || continue
        case "$f" in *mmproj*|*BF16*|*bf16*) continue ;; esac
        [ -z "$first" ] && first="$f"
        speed_gguf "$f"
        echo "cooling $SPEED_COOL s"
        sleep $SPEED_COOL
    done
    for d in $MLX_DIR/*/; do
        [ -f "$d/config.json" ] || continue
        case "$d" in *bf16*|*probe*) continue ;; esac
        speed_mlx "${d%/}"
        echo "cooling $SPEED_COOL s"
        sleep $SPEED_COOL
    done
    if [ -n "$first" ]; then
        echo
        echo "### thermal check: repeating the first model ###"
        local n
        n=$(basename "$first" .gguf)
        cp "$LOG_DIR/speed-gguf-$n-d0.json" "$LOG_DIR/thermal-before.json" 2>/dev/null
        speed_gguf "$first"
        cp "$LOG_DIR/speed-gguf-$n-d0.json" "$LOG_DIR/thermal-after.json" 2>/dev/null
    fi
    speed_report
}

speed_report() {
    python3 - "$LOG_DIR" << 'SPEEDREPEOF'
import glob, json, os, sys
log = sys.argv[1]
rows = []

for p in sorted(glob.glob(os.path.join(log, "speed-gguf-*.json"))):
    try:
        data = json.load(open(p))
    except Exception:
        continue
    base = os.path.basename(p)[len("speed-gguf-"):-len(".json")]
    name, _, depth = base.rpartition("-d")
    pp = tg = None
    for e in data:
        if e.get("n_prompt", 0) > 0:
            pp = e.get("avg_ts")
        elif e.get("n_gen", 0) > 0:
            tg = e.get("avg_ts")
    rows.append({"engine": "gguf", "name": name, "depth": int(depth or 0),
                 "prefill": pp, "decode": tg})

for p in sorted(glob.glob(os.path.join(log, "speed-mlx-*.json"))):
    try:
        data = json.load(open(p))
    except Exception:
        continue
    for e in data.get("runs", []):
        rows.append({"engine": "mlx", "name": data["name"], "depth": e["depth"],
                     "prefill": e["prefill_tps"], "decode": e["decode_tps"]})

rows.sort(key=lambda r: (r["depth"], r["engine"], r["name"]))
print()
print("%-6s %-42s %7s %10s %10s" % ("engine", "build", "depth", "prefill", "decode"))
print("-" * 80)
for r in rows:
    print("%-6s %-42s %7d %10s %10s" % (
        r["engine"], r["name"][-42:], r["depth"],
        "%.1f" % r["prefill"] if r["prefill"] else "",
        "%.2f" % r["decode"] if r["decode"] else ""))

json.dump(rows, open(os.path.join(log, "speed.json"), "w"), indent=2)
print()
print("written to %s/speed.json" % log)

a = os.path.join(log, "thermal-before.json")
b = os.path.join(log, "thermal-after.json")
if os.path.exists(a) and os.path.exists(b):
    def tg(p):
        for e in json.load(open(p)):
            if e.get("n_gen", 0) > 0:
                return e.get("avg_ts")
        return None
    x, y = tg(a), tg(b)
    if x and y:
        drift = 100.0 * (y - x) / x
        print()
        print("thermal check: first model %.2f then %.2f tok/s, drift %+.1f %%" % (x, y, drift))
        if abs(drift) > 5:
            print("more than 5 percent. The machine was heat limited, redo the session.")
SPEEDREPEOF
}


# ------------------------------------------------------------------ quality

# Writes the two python files this block needs. Kept as files on disk rather
# than heredocs inside a pipeline, because a heredoc attached to the wrong end
# of a pipe silently feeds the script to tee instead of to python.
kld_install() {
    mkdir -p $FROOT $LOG_DIR
    cat > $FROOT/kld.py << 'KLDPYEOF'
"""Per token KL divergence of a quantized MLX checkpoint against a reference.

    python3 kld.py REF QUANT CORPUS CTX FIRST STEP OUT.json

Follows llama.cpp's llama-perplexity --kl-divergence conventions so the numbers
can sit in one table with GGUF numbers measured the same way:

  the corpus is cut into independent chunks of CTX tokens, no cache carried
  between them

  inside each chunk only positions FIRST and later are scored, and llama.cpp
  uses FIRST = CTX/2 so every scored token has at least half a context behind
  it. Scoring from position zero counts tokens the model predicted almost
  blind, which is where a quant looks worst, and makes every number pessimistic

  the last position of a chunk has no target and is dropped

  KLD is sum over the vocabulary of p*(log p - log q) with p from the
  reference, so it is zero when the two agree

  delta p is the change in the probability the model assigns to the token that
  actually comes next, in percentage points, which is what llama.cpp reports

Perplexity of both models is printed as well. That is the cross check: the
reference perplexity here should match the reference perplexity llama.cpp
reports on the same corpus at the same context. If it does, the two engines
agree at full precision and quant against own-reference numbers are
comparable across them. If it does not, the gap is the correction.
"""

import json
import math
import os
import sys
import time

import mlx.core as mx
from mlx_lm import load


def logits_of(out):
    return out.logits if hasattr(out, "logits") else out


def main():
    ref_path, qnt_path, corpus, ctx, first, step, out_json = sys.argv[1:8]
    ctx, step = int(ctx), int(step)
    first = int(first) if first else ctx // 2
    same = os.path.realpath(ref_path) == os.path.realpath(qnt_path)

    print("reference : %s" % ref_path, flush=True)
    print("quant     : %s" % qnt_path, flush=True)
    print("context   : %d, scoring from position %d" % (ctx, first), flush=True)
    if same:
        print("SELF TEST: reference against itself, expect 0 and 100 %", flush=True)

    ref_model, tok = load(ref_path)
    qnt_model = ref_model if same else load(qnt_path)[0]

    ids = tok.encode(open(corpus, encoding="utf-8", errors="replace").read())
    n_chunk = len(ids) // ctx
    per_chunk = ctx - 1 - first
    print("corpus    : %d tokens, %d chunks of %d, %d scored per chunk"
          % (len(ids), n_chunk, ctx, per_chunk), flush=True)
    if n_chunk < 1 or per_chunk < 1:
        print("not enough text for one chunk at this context")
        sys.exit(1)

    klds = []
    dp = []
    nll_ref = 0.0
    nll_qnt = 0.0
    top1 = 0
    scored = 0
    t0 = time.time()

    for c in range(n_chunk):
        chunk = ids[c * ctx:(c + 1) * ctx]
        x = mx.array([chunk])
        lr = logits_of(ref_model(x))[0]
        lq = lr if same else logits_of(qnt_model(x))[0]
        mx.eval(lr, lq)

        for s in range(first, ctx - 1, step):
            e = min(s + step, ctx - 1)
            a = lr[s:e].astype(mx.float32)
            b = a if same else lq[s:e].astype(mx.float32)
            logp = a - mx.logsumexp(a, axis=-1, keepdims=True)
            logq = logp if same else b - mx.logsumexp(b, axis=-1, keepdims=True)
            p = mx.exp(logp)
            k = mx.sum(p * (logp - logq), axis=-1)

            am_p = mx.argmax(a, axis=-1)
            hits = mx.sum(am_p == (am_p if same else mx.argmax(b, axis=-1)))

            tgt = mx.array(chunk[s + 1:e + 1])
            rows = mx.arange(e - s)
            lp_true = logp[rows, tgt]
            lq_true = logq[rows, tgt]
            d = (mx.exp(lq_true) - mx.exp(lp_true)) * 100.0

            mx.eval(k, hits, lp_true, lq_true, d)
            klds.extend(float(v) for v in k)
            dp.extend(float(v) for v in d)
            nll_ref += -float(mx.sum(lp_true))
            nll_qnt += -float(mx.sum(lq_true))
            top1 += int(hits)
            scored += (e - s)

        del lr
        if not same:
            del lq

        if (c + 1) % 2 == 0 or c == n_chunk - 1:
            el = time.time() - t0
            eta = (n_chunk - c - 1) * el / (c + 1)
            print("  chunk %d/%d   %.1f s each   eta %d min %02d s"
                  % (c + 1, n_chunk, el / (c + 1), eta // 60, eta % 60), flush=True)

    klds.sort()

    def q(f):
        return klds[min(len(klds) - 1, int(len(klds) * f))]

    def rms(v):
        return math.sqrt(sum(x * x for x in v) / len(v))

    res = {
        "reference": ref_path,
        "quant": qnt_path,
        "corpus": os.path.basename(corpus),
        "context": ctx,
        "score_from": first,
        "chunks": n_chunk,
        "tokens_scored": scored,
        "reference_ppl": math.exp(nll_ref / scored),
        "quant_ppl": math.exp(nll_qnt / scored),
        "mean_kld": sum(klds) / len(klds),
        "median_kld": q(0.50),
        "p90_kld": q(0.90),
        "p95_kld": q(0.95),
        "p99_kld": q(0.99),
        "max_kld": klds[-1],
        "top1_agree_pct": 100.0 * top1 / scored,
        "mean_delta_p_points": sum(dp) / len(dp),
        "rms_delta_p_points": rms(dp),
    }
    json.dump(res, open(out_json, "w"), indent=2)

    print()
    print("tokens scored     : %d" % res["tokens_scored"])
    print("reference ppl     : %.4f" % res["reference_ppl"])
    print("quant ppl         : %.4f" % res["quant_ppl"])
    print("mean KLD          : %.6f" % res["mean_kld"])
    print("median KLD        : %.6f" % res["median_kld"])
    print("90 / 95 / 99 pct  : %.6f  %.6f  %.6f"
          % (res["p90_kld"], res["p95_kld"], res["p99_kld"]))
    print("max KLD           : %.6f" % res["max_kld"])
    print("same top-1        : %.3f %%" % res["top1_agree_pct"])
    print("delta p, points   : mean %.4f   rms %.4f"
          % (res["mean_delta_p_points"], res["rms_delta_p_points"]))
    print()
    if same:
        ok = res["mean_kld"] < 1e-9 and res["top1_agree_pct"] > 99.999
        print("SELF TEST %s" % ("PASSED" if ok else "FAILED, the script is wrong"))
    else:
        print("compare reference ppl against what llama.cpp reports for the")
        print("same corpus at context %d. If they agree, the engines agree at" % ctx)
        print("full precision and these numbers belong in the same table as")
        print("the GGUF ones.")
    print("written to %s" % out_json)


if __name__ == "__main__":
    main()
KLDPYEOF

    cat > $FROOT/speed_mlx.py << 'SPEEDPYEOF'
"""MLX half of the speed protocol, timed to llama-bench's definitions.

    python3 speed_mlx.py CHECKPOINT PROMPT GEN REPS "0 4096" OUT.json

prefill: PROMPT tokens in one pass, tokens over seconds.
decode : GEN single token passes reusing the cache, tokens over seconds.
depth  : how much context is already in the cache before timing starts. Filling
         it is setup and is not timed.

A warmup pass runs first, because the first call compiles kernels and would
otherwise be counted as the model being slow.
"""

import json
import sys
import time

import mlx.core as mx
from mlx_lm import load

try:
    from mlx_lm.models.cache import make_prompt_cache
except Exception:
    make_prompt_cache = None


def logits_of(out):
    return out.logits if hasattr(out, "logits") else out


def one_depth(model, depth, prompt, gen, reps):
    pre, dec = [], []
    for _ in range(reps):
        cache = make_prompt_cache(model) if make_prompt_cache else None
        if depth > 0 and cache is not None:
            mx.eval(logits_of(model(mx.array([[1] * depth]), cache=cache)))

        x = mx.array([[1] * prompt])
        t0 = time.perf_counter()
        out = model(x, cache=cache) if cache is not None else model(x)
        mx.eval(logits_of(out))
        pre.append(prompt / (time.perf_counter() - t0))

        if cache is None:
            dec.append(float("nan"))
            continue
        tok = mx.array([[1]])
        t0 = time.perf_counter()
        for _ in range(gen):
            mx.eval(logits_of(model(tok, cache=cache)))
        dec.append(gen / (time.perf_counter() - t0))

    pre.sort()
    dec.sort()
    return pre[len(pre) // 2], dec[len(dec) // 2]


def main():
    path, prompt, gen, reps, depths, out_json = sys.argv[1:7]
    prompt, gen, reps = int(prompt), int(gen), int(reps)
    depths = [int(d) for d in depths.split()]

    name = path.rstrip("/").split("/")[-1]
    model, _ = load(path)
    mx.eval(logits_of(model(mx.array([[1] * 64]))))

    runs = []
    for d in depths:
        print("  depth %d ..." % d, end="", flush=True)
        try:
            pp, tg = one_depth(model, d, prompt, gen, reps)
        except Exception as e:
            print(" failed: %s" % str(e).splitlines()[0][:70])
            continue
        print("  prefill %.1f   decode %.2f tok/s" % (pp, tg))
        runs.append({"depth": d, "prefill_tps": pp, "decode_tps": tg})

    json.dump({"name": name, "path": path, "engine": "mlx",
               "prompt_tokens": prompt, "gen_tokens": gen, "reps": reps,
               "runs": runs}, open(out_json, "w"), indent=2)
    print("  written to %s" % out_json)


if __name__ == "__main__":
    main()
SPEEDPYEOF
    echo "wrote $FROOT/kld.py and $FROOT/speed_mlx.py"
}

# The first thing to run after kld_install. Measures the reference against
# itself: a correct implementation gives exactly zero divergence and exactly
# 100 percent agreement. Anything else means the script is wrong and no number
# it produces is worth reading.
mlx_kld_selftest() {
    local ref="${1:-$MLX_REF}"
    [ -d "$ref" ] || { echo "mlx_kld_selftest /path/to/reference"; return 1; }
    kld_install > /dev/null
    head -c 40000 "$MLX_EVAL" > /tmp/selftest-corpus.txt
    echo "20 chunks on a slice, not the whole corpus"
    echo "this checks tokenization, chunking, the scoring window and target"
    echo "alignment. It does NOT check the divergence formula: with one model"
    echo "the script short circuits. ppl_compare is what validates that."
    python3 $FROOT/kld.py "$ref" "$ref" /tmp/selftest-corpus.txt 512 256 \
        "$Q_STEP" "$LOG_DIR/selftest.json"
}

# mlx_kld2 CHECKPOINT
# The corrected measurement: context 4096, second half scored, llama.cpp's
# conventions. Replaces mlx_kld, whose numbers were pessimistic.
mlx_kld2() {
    local q="${1:-}"
    if [ -z "$q" ] || [ ! -f "$q/config.json" ]; then
        echo "mlx_kld2 /path/to/checkpoint"
        ls -d $MLX_DIR/*/ 2>/dev/null | sed "s/^/   /"
        return 1
    fi
    [ -z "$MLX_REF" ] && { echo "no reference. Run mlx_ref first."; return 1; }
    [ -f "$MLX_EVAL" ] || { echo "no corpus at $MLX_EVAL"; return 1; }
    kld_install > /dev/null
    local name
    name=$(basename "${q%/}")
    mkdir -p $LOG_DIR
    stdbuf -oL -eL python3 $FROOT/kld.py "$MLX_REF" "${q%/}" "$MLX_EVAL" \
        "$Q_CTX" "$Q_FIRST" "$Q_STEP" "$LOG_DIR/kld2-$name.json" \
        2>&1 | tee "$LOG_DIR/kld2-$name.log"
    autopush "$LOG_DIR/kld2-$name.json" "logs/kld2-$name.json"
    autopush "$LOG_DIR/kld2-$name.log" "logs/kld2-$name.log"
}

mlx_kld2_all() {
    [ -z "$MLX_REF" ] && { echo "no reference. Run mlx_ref first."; return 1; }
    local d name n=0 todo=""
    for d in $MLX_DIR/*/; do
        [ -f "$d/config.json" ] || continue
        name=$(basename "${d%/}")
        case "$name" in *probe*|*bf16*) continue ;; esac
        [ -f "$LOG_DIR/kld2-$name.json" ] && [ "$KLD_FORCE" != "1" ] \
            && { echo "already measured: $name"; continue; }
        todo="$todo $d"
    done
    [ -z "$todo" ] && { echo "nothing left"; mlx_results2; return 0; }
    for d in $todo; do
        n=$(( n + 1 ))
        echo; echo "########## $n of $(echo $todo | wc -w) ##########"
        mlx_kld2 "${d%/}"
    done
    mlx_results2
}

mlx_results2() {
    python3 - "$MLX_DIR" "$LOG_DIR" << 'RES2EOF'
import glob, json, os, sys
root, log = sys.argv[1], sys.argv[2]
rows = []
for p in sorted(glob.glob(os.path.join(log, "kld2-*.json"))):
    try:
        r = json.load(open(p))
    except Exception:
        continue
    name = os.path.basename(p)[len("kld2-"):-len(".json")]
    d = os.path.join(root, name)
    size = 0
    if os.path.isdir(d):
        size = sum(os.path.getsize(f) for f in glob.glob(os.path.join(d, "*.safetensors")))
    r["name"] = name
    r["size_gb"] = round(size / 1e9, 2) if size else None
    rows.append(r)
rows.sort(key=lambda r: r.get("size_gb") or 0)
json.dump(rows, open(os.path.join(log, "mlx-results2.json"), "w"), indent=2)
print("%-42s %8s %11s %10s %9s" % ("build", "GB", "mean KLD", "top-1 %", "ppl"))
for r in rows:
    print("%-42s %8s %11s %10s %9s" % (
        r["name"][-42:], r.get("size_gb", ""),
        round(r["mean_kld"], 6), round(r["top1_agree_pct"], 3),
        round(r["quant_ppl"], 4)))
print()
if rows:
    print("reference: %s, ppl %.4f, context %d, scored from %d, %d chunks"
          % (rows[0]["reference"], rows[0]["reference_ppl"], rows[0]["context"],
             rows[0]["score_from"], rows[0]["chunks"]))
RES2EOF
}


# ------------------------------------------------------------------ calibration

# The one experiment that makes GGUF and MLX numbers comparable at all.
#
# Both engines run the SAME original weights at full precision on the SAME
# corpus at the SAME context, and we compare perplexity. If they agree, their
# kernels agree where it matters and each engine's quant-against-own-reference
# numbers belong in one table. If they disagree, the gap is a correction that
# has to be stated rather than ignored.
#
# The GGUF side of this number is already published: 4.5219 plus or minus
# 0.0238 at context 4096 on eval_neutral.
ppl_compare() {
    echo "=============== what we are comparing ==============="
    echo "the same weights, unquantized, in two engines, same corpus, same"
    echo "context. Not two quants: two references."
    echo
    echo "GGUF side, already measured and published:"
    echo "  reference ppl 4.5219 +/- 0.0238   context 4096   87 chunks"
    echo
    echo "MLX side, run it now:"
    echo
    if [ -z "$MLX_REF" ]; then
        echo "  set the bf16 checkpoint first:  mlx_ref /mlx/<stem>-MLX-bf16"
        return 1
    fi
    kld_install > /dev/null
    stdbuf -oL -eL python3 $FROOT/kld.py "$MLX_REF" "$MLX_REF" "$MLX_EVAL" \
        "$Q_CTX" "$Q_FIRST" "$Q_STEP" "$LOG_DIR/ppl-mlx-bf16.json" \
        2>&1 | tee "$LOG_DIR/ppl-mlx-bf16.log"
    echo
    echo "to redo the GGUF side on this box for a like for like check:"
    echo "  $BIN/llama-perplexity -m <bf16>.gguf -f $MLX_EVAL -c $Q_CTX -ngl 99"
    echo
    echo "reading the result: within about one percent means the engines agree"
    echo "and the two ladders can share a table. A larger gap is a correction"
    echo "that goes in the methodology section, not something to hide."
}


help_measure() {
cat << 'HELPMEOF'

MAC, FROM NOTHING
  mac_setup            xcode tools, brew, cmake, python packages
  mac_build            llama.cpp with Metal
  mac_memory           how much of unified memory the GPU may actually use
  mac_get gguf|mlx|eval|all
  mac_info             chip, memory, library versions, written to the log

SPEED, BOTH ENGINES, ONE PROTOCOL
  speed_note           what the settings mean and why
  speed_gguf FILE | speed_mlx DIR | speed_all | speed_report
  SPEED_PROMPT=512 SPEED_GEN=128 SPEED_DEPTHS="0 4096 16384" SPEED_REPS=5

QUALITY, llama.cpp CONVENTIONS
  kld_install          writes kld.py and speed_mlx.py to disk
  mlx_kld_selftest     reference against itself, must give 0 and 100 percent
  mlx_kld2 DIR | mlx_kld2_all | mlx_results2
  Q_CTX=4096 Q_FIRST=(ctx/2) Q_STEP=64

CROSS ENGINE
  ppl_compare          full precision perplexity in both engines, same corpus

ORDER

  on the Mac:
    mac_setup ; mac_build ; mac_memory ; mac_get all
    speed_note ; speed_all

  on the rented box:
    kld_install ; mlx_kld_selftest        <- read PASSED before anything else
    ppl_compare                           <- against the published 4.5219
    mlx_ref /mlx/<stem>-MLX-bf16
    mlx_kld2_all ; mlx_results2

HELPMEOF
}


# ================================================================== ADDENDUM
#
# Append this after the MEASURE block. Three gaps it closes:
#
#   nothing built the MLX reference, it was a raw command typed by hand
#   nothing downloaded our published MLX ladder onto a Linux box
#   nothing downloaded other publishers' MLX builds
#
# It also redefines save_state. A later definition of a shell function wins, so
# no edit to the original is needed: the version below simply replaces it from
# the point this file is sourced. It writes everything the old one wrote plus
# the variables the MLX and measurement work depends on, which otherwise vanish
# in a fresh tmux pane.

save_state() {
    cat > $STATE << STATEEOF
MAIN=$MAIN
RECIPE=$RECIPE
METRICS=$METRICS
METRICS_KIND=$METRICS_KIND
UPSTREAM=$UPSTREAM
EVALSET=$EVALSET
EVAL=$EVAL
BASE=$BASE
CTX=$CTX
IM_TOKENS_EXACT=$IM_TOKENS_EXACT
IM_MODEL=$IM_MODEL
IM_CORPUS=$IM_CORPUS
IM_CTX=$IM_CTX
GPUS=$GPUS
AUTOPUSH=$AUTOPUSH
INCLUDE_EXPERIMENTAL=$INCLUDE_EXPERIMENTAL
MLX_REF=$MLX_REF
MLX_EVAL=$MLX_EVAL
MLX_SRC=$MLX_SRC
MLXROOT=$MLXROOT
MLX_DIR=$MLX_DIR
Q_CTX=$Q_CTX
Q_FIRST=$Q_FIRST
Q_STEP=$Q_STEP
STATEEOF
}

# The build function takes a CUDA architecture number and getting it wrong
# produces a binary that runs on nothing. It is not guessable from the card
# name: H100 and H200 are both 90, RTX 5090 is 120, B200 is 100. Read it off
# the card instead of remembering.
cuda_arch() {
    local cap
    cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1)
    if [ -z "$cap" ]; then
        echo "no nvidia-smi, cannot tell"
        return 1
    fi
    local arch
    arch=$(echo "$cap" | tr -d '.')
    echo "compute capability $cap on $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
    echo "build with:  build $arch"
}

# What the box has to hold before anything starts. Running out of disk halfway
# through a 210 GB download is a wasted hour, and vast bills for the hour.
disk_plan() {
    echo "what this job needs on disk:"
    echo "  /src        original weights                          56 GB"
    echo "  MLX bf16    the reference checkpoint                   51 GB"
    echo "  GGUF bf16   the other engine's reference               51 GB"
    echo "  our MLX     six rungs                                 119 GB"
    echo "  external    fourteen community builds                 210 GB"
    echo "  ------------------------------------------------------------"
    echo "  total                                            about 490 GB"
    echo
    echo "plus the hub cache, which can hold a second copy of anything"
    echo "downloaded without --local-dir. Take 1 TB minimum, 2 TB comfortably."
    echo
    df -h / | tail -1
    echo
    echo "HF_HOME is $HF_HOME. If space runs short mid-job:  rm -rf $HF_HOME/*"
}


# ------------------------------------------------------------------ reference

# The unquantized MLX checkpoint every measurement is taken against. Not a
# build for use: it is 51 GB and exists so that divergence has a fixed zero.
#
# Conversion with no quantization flag is a format change, not arithmetic, so
# this reproduces byte for byte anywhere. That is what makes it publishable as
# a shared point of comparison.
mlx_reference() {
    mlx_need || return 1
    local out
    out=$(mlx_out bf16)
    if [ -d "$out" ] && ls "$out"/*.safetensors > /dev/null 2>&1; then
        echo "already here: $out  $(du -sh $out | cut -f1)"
        mlx_ref "$out"
        return 0
    fi
    if [ ! -d "$MLX_SRC" ] || ! ls $MLX_SRC/*.safetensors > /dev/null 2>&1; then
        echo "no original weights at $MLX_SRC. Run mlx_src first."
        return 1
    fi
    echo "converting $MLX_SRC to an MLX checkpoint with no quantization"
    echo "target: $out, about 51 GB"
    date
    local start
    start=$(date +%s)
    stdbuf -oL -eL mlx_vlm.convert --hf-path "$MLX_SRC" --mlx-path "$out" \
        --dtype bfloat16 2>&1 | tee "$LOG_DIR/mlx-reference.log"
    echo "took $(( $(date +%s) - start )) seconds"
    mlx_inspect "$out"
    echo
    mlx_ref "$out"
    save_state
}


# ------------------------------------------------------------------ our builds

# Pull our own published MLX ladder onto this box, one directory per rung.
OUR_MLX=${OUR_MLX:-"mixed_3_4 4bit mixed_4_6 5bit 6bit 8bit"}

mlx_get_ours() {
    mlx_need || return 1
    mkdir -p $MLX_DIR
    local r repo dest n=0
    for r in $OUR_MLX; do
        n=$(( n + 1 ))
        repo=$(mlx_repo "$r")
        dest=$MLX_DIR/$(mlx_stem)-MLX-$r
        echo
        echo "########## $n: $repo ##########"
        if [ -f "$dest/config.json" ]; then
            echo "already here, $(du -sh $dest | cut -f1)"
            continue
        fi
        hf download "$repo" --local-dir "$dest" || echo "  FAILED"
        du -sh "$dest" 2>/dev/null
    done
    echo
    du -sh $MLX_DIR
}


# ------------------------------------------------------------------ external

# Other publishers' MLX builds, measured here against our reference. Their
# published figures were taken against their own reference and harness, so
# they cannot go in a table with ours. Their files can.
#
# The list is only builds that keep the vision tower and are in the size range
# where the argument is. Text only builds are excluded on purpose: they are
# smaller for free because they drop a 0.92 GB tower, and putting them on a
# size axis next to a build that carries one is a comparison of two different
# things.
#
# Each lands flat as ext--<publisher>--<name>, so the publisher is in every log
# name and the existing measurement loop picks them up without recursing.
EXTERNAL_MLX="
lukaskremla/Qwen3.8-27B-2bit-MLX
mlx-works/Qwen3.8-27B-oQ2e-mtp
lukaskremla/Qwen3.8-27B-3bit-MLX
leonsarmiento/Qwen3.8-27B-3bit-mtp-mlx
maglun/Qwen3.8-27B-MLX-Mixed-3.80bpw
mlx-works/Qwen3.8-27B-oQ3e-mtp
rapid-mlx/Qwen3.8-27B-mixed-3.5bpw-MLX
mlx-community/Qwen3.8-27B-mxfp4
mlx-community/Qwen3.8-27B-4bit
WaveCut/Qwen3.8-27B-MLX-4bit-DWQ
mlx-community/Qwen3.8-27B-oQ4
True2456/Qwen3.8-27B-AWQ-4.85bpw
mlx-community/Qwen3.8-27B-OptiQ-4bit
"

mlx_ext_list() {
    local r n=0
    echo "these get downloaded by mlx_get_external:"
    for r in $EXTERNAL_MLX; do
        n=$(( n + 1 ))
        printf "  %2d  %s\n" $n "$r"
    done
    echo
    echo "about 210 GB total. Edit EXTERNAL_MLX to change the list."
}

mlx_get_external() {
    mkdir -p $MLX_DIR
    local r pub name dest n=0 total
    total=$(echo $EXTERNAL_MLX | wc -w)
    for r in $EXTERNAL_MLX; do
        n=$(( n + 1 ))
        pub=$(echo "$r" | cut -d/ -f1)
        name=$(echo "$r" | cut -d/ -f2)
        dest=$MLX_DIR/ext--$pub--$name
        echo
        echo "########## $n of $total: $r ##########"
        if [ -f "$dest/config.json" ]; then
            echo "already here, $(du -sh $dest | cut -f1)"
            continue
        fi
        hf download "$r" --local-dir "$dest" || { echo "  FAILED"; continue; }
        du -sh "$dest"
    done
    echo
    echo "on disk now:"
    du -sh $MLX_DIR/ext--* 2>/dev/null
    echo
    echo "measure them with the same command as ours:  mlx_kld2_all"
}

# What is downloaded, what is measured, and what is neither.
mlx_audit() {
    local d name have=0 done_=0 miss=""
    for d in $MLX_DIR/*/; do
        [ -f "$d/config.json" ] || continue
        name=$(basename "${d%/}")
        case "$name" in *probe*|*bf16*) continue ;; esac
        have=$(( have + 1 ))
        if [ -f "$LOG_DIR/kld2-$name.json" ]; then
            done_=$(( done_ + 1 ))
        else
            miss="$miss $name"
        fi
    done
    echo "checkpoints on disk : $have"
    echo "measured            : $done_"
    if [ -n "$miss" ]; then
        echo "not measured yet:"
        for name in $miss; do echo "   $name"; done
        echo
        echo "run:  mlx_kld2_all"
    fi
    echo
    echo "reference           : ${MLX_REF:-NOT SET, run mlx_reference}"
    echo "corpus              : $MLX_EVAL"
    echo "context             : $Q_CTX, scoring from ${Q_FIRST:-half of it}"
}

# ================================================================== probe
#
# Everything about a new model that can be known before renting anything.
#
# Reads the config, the tokenizer files and the SAFETENSORS HEADERS. A
# safetensors file starts with eight bytes holding the length of a json header,
# and that header lists every tensor with its dtype and its shape. Two HTTP
# range requests per shard therefore give the complete tensor layout of a 360 GB
# checkpoint for a few megabytes, which is the difference between knowing what
# the ladder will cost and guessing at it.
#
#   probe_arch Qwen/Qwen3.8-Flash-Next
#
# What to read out of the output, in order:
#   1. architectures     -> does the converter know this class at all
#   2. auto_map          -> if present the model ships its own modeling code,
#                           so transformers works today and the NVFP4 and MLX
#                           routes are open even while llama.cpp is not
#   3. dtype             -> bf16 or an already quantized checkpoint
#   4. the tensor table  -> where the parameters actually are, which decides
#                           the ladder and what imatrix can and cannot reach
probe_arch() {
    if [ -z "$1" ]; then
        echo "probe_arch UPSTREAM_REPO"
        echo "  probe_arch Qwen/Qwen3.8-Flash-Next"
        return 1
    fi
    local repo="$1" dir=/probe
    mkdir -p $dir /logs

    echo "=============== 1. small files only, no weights ==============="
    hf download "$repo" --local-dir $dir \
        --include "*.json" --include "*.py" --include "*.jinja" --include "*.md" \
        || return 1
    du -sh $dir
    echo
    ls $dir

    echo
    echo "=============== 2. what this model says it is ==============="
    python3 - "$repo" "$dir" 2>&1 | tee /logs/probe-$(basename $repo).log << 'PROBEEOF'
import json, os, sys, collections

repo, d = sys.argv[1], sys.argv[2]
cfg = json.load(open(os.path.join(d, "config.json")))
inner = cfg.get("text_config") or cfg

arch = cfg.get("architectures") or ["UNKNOWN"]
print("architectures : %s" % ", ".join(arch))
print("model_type    : %s" % cfg.get("model_type"))
print("dtype         : %s" % (cfg.get("torch_dtype") or cfg.get("dtype")))
print("quantization  : %s" % (cfg.get("quantization_config") or "none, full precision"))

auto = cfg.get("auto_map")
py = [f for f in os.listdir(d) if f.endswith(".py")]
print()
if auto or py:
    print("CUSTOM CODE   : yes")
    for k, v in (auto or {}).items():
        print("   %-34s %s" % (k, v))
    for f in sorted(py):
        print("   ships %s" % f)
    print("   transformers runs this today with trust_remote_code=True, so the")
    print("   NVFP4 and MLX routes do not wait on llama.cpp.")
else:
    print("CUSTOM CODE   : no, this needs a transformers release that knows it")

print()
print("--- shape of the network ---")
for k in ("num_hidden_layers", "hidden_size", "intermediate_size",
          "moe_intermediate_size", "num_experts", "num_experts_per_tok",
          "num_attention_heads", "num_key_value_heads", "head_dim",
          "vocab_size", "max_position_embeddings", "full_attention_interval",
          "linear_conv_kernel_dim", "num_nextn_predict_layers"):
    v = inner.get(k, cfg.get(k))
    if v is not None:
        extra = ""
        if k in ("hidden_size", "intermediate_size", "moe_intermediate_size"):
            extra = "   %%256 = %d %s" % (v % 256,
                     "ok" if v % 256 == 0 else "<- BREAKS k/i quants below 4.5 bpw")
        print("  %-26s %s%s" % (k, v, extra))

print()
print("--- anything n-gram, residual or sparse-attention shaped ---")
hits = 0
def walk(node, path=""):
    global hits
    if isinstance(node, dict):
        for k, v in node.items():
            p = "%s.%s" % (path, k) if path else k
            if any(t in k.lower() for t in ("ngram", "n_gram", "residual",
                                            "sparse", "hash", "sub_table",
                                            "subtable", "over_enc", "expand")):
                print("  %-40s %s" % (p, v if not isinstance(v, (dict, list)) else type(v).__name__))
                hits += 1
            walk(v, p)
    elif isinstance(node, list):
        for i, v in enumerate(node[:4]):
            walk(v, "%s[%d]" % (path, i))
walk(cfg)
if hits == 0:
    print("  nothing, the new parts may be named differently. Read config.json.")

print()
print("vision tower  : %s" % ("yes, mmproj needed" if
      ("vision_config" in cfg or os.path.exists(os.path.join(d, "preprocessor_config.json")))
      else "no"))
PROBEEOF

    echo
    echo "=============== 3. does OUR converter know this class ==============="
    local cls
    cls=$(python3 -c "import json;print((json.load(open('/probe/config.json')).get('architectures') or ['?'])[0])")
    echo "looking for $cls in /llama.cpp"
    if [ -d /llama.cpp ]; then
        if grep -rq "$cls" /llama.cpp/conversion/ /llama.cpp/convert_hf_to_gguf.py 2>/dev/null; then
            echo "  FOUND. convert_hf_to_gguf.py can read this model."
        else
            echo "  NOT FOUND. The converter will abort with 'Model $cls is not"
            echo "  supported'. No GGUF is possible until that class is added and"
            echo "  the graph is written in C++."
        fi
    else
        echo "  no /llama.cpp on this box yet, run build first"
    fi

    echo
    echo "=============== 4. tensor layout, from the shard headers ==============="
    python3 - "$repo" << 'HDREOF'
import json, os, re, sys, collections
from huggingface_hub import hf_hub_url, get_hf_file_metadata
import requests

repo = sys.argv[1]
idx = "/probe/model.safetensors.index.json"
if os.path.exists(idx):
    shards = sorted(set(json.load(open(idx))["weight_map"].values()))
else:
    shards = ["model.safetensors"]
print("%d shard(s), reading only their headers" % len(shards))

DT = {"BF16": 2, "F16": 2, "F32": 4, "F8_E4M3": 1, "F8_E5M2": 1,
      "I8": 1, "U8": 1, "I32": 4, "I64": 8, "F64": 8}

tensors = {}
got = 0
for i, s in enumerate(shards):
    url = hf_hub_url(repo, s)
    try:
        r = requests.get(url, headers={"Range": "bytes=0-7"}, timeout=30)
        n = int.from_bytes(r.content[:8], "little")
        r = requests.get(url, headers={"Range": "bytes=8-%d" % (8 + n - 1)}, timeout=60)
        head = json.loads(r.content[:n])
    except Exception as e:
        print("  shard %s unreadable: %s" % (s, str(e).splitlines()[0][:60]))
        continue
    for name, meta in head.items():
        if name == "__metadata__":
            continue
        tensors[name] = meta
    got += 1
    if (i + 1) % 10 == 0 or i == len(shards) - 1:
        print("  %d/%d headers read, %d tensors so far" % (i + 1, len(shards), len(tensors)),
              flush=True)

if not tensors:
    print("could not read any header. Falling back to the index only.")
    sys.exit(0)

def group(name):
    n = re.sub(r"\.\d+\.", ".N.", name)
    n = re.sub(r"^(model|language_model|transformer)\.", "", n)
    n = re.sub(r"^layers\.N\.", "blk.", n)
    return n.rsplit(".weight", 1)[0].rsplit(".bias", 1)[0]

g = collections.defaultdict(lambda: {"n": 0, "params": 0, "bytes": 0,
                                     "dtype": set(), "shape": None})
for name, m in tensors.items():
    p = 1
    for x in m["shape"]:
        p *= x
    e = g[group(name)]
    e["n"] += 1
    e["params"] += p
    e["bytes"] += p * DT.get(m["dtype"], 2)
    e["dtype"].add(m["dtype"])
    if e["shape"] is None:
        e["shape"] = m["shape"]

total_p = sum(e["params"] for e in g.values())
total_b = sum(e["bytes"] for e in g.values())
print()
print("%d tensors, %.1f B parameters, %.0f GB on the wire"
      % (len(tensors), total_p / 1e9, total_b / 1e9))
print()
print("%-46s %5s %11s %8s %8s %s" % ("tensor group", "count", "params", "share", "GB", "dtype"))
for k, e in sorted(g.items(), key=lambda kv: -kv[1]["params"]):
    if e["params"] < total_p * 0.001:
        continue
    print("%-46s %5d %10.2fB %7.1f%% %8.1f %s"
          % (k[:46], e["n"], e["params"] / 1e9, 100 * e["params"] / total_p,
             e["bytes"] / 1e9, ",".join(sorted(e["dtype"]))))

emb = sum(e["params"] for k, e in g.items()
          if any(t in k.lower() for t in ("ngram", "n_gram", "embed", "sub_table", "subtable")))
print()
print("lookup-shaped parameters (embedding and n-gram tables): %.1f B, %.0f%% of the model"
      % (emb / 1e9, 100 * emb / total_p))
print("Those are gather operations, not matmuls, so llama-imatrix collects no")
print("statistics for them and they quantize blind however good the corpus is.")

json.dump({k: {"count": v["n"], "params": v["params"], "bytes": v["bytes"],
               "dtype": sorted(v["dtype"]), "shape": v["shape"]}
           for k, v in g.items()},
          open("/logs/tensor-layout.json", "w"), indent=2)
print()
print("written to /logs/tensor-layout.json")
HDREOF

    echo
    echo "next: if the converter does not know the class, nothing downstream can"
    echo "run. Decide there before renting anything with a GPU in it."
}

# ================================================================== state
# Read back whatever the last pane set, so a fresh tmux tab is not amnesiac.
if [ -f /state.sh ]; then
    . /state.sh
fi
