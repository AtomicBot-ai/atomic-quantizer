#!/usr/bin/env bash
# Importance matrix over the model's calibration build, split into shards (foundry im_shard / im_merge_all).
#
# The matrix is a sum over chunks, so shards over disjoint chunk ranges of the
# same corpus merge into exactly the single-run result. Shards run side by side
# on this box's GPUs (IM_INDEX="0 1" on a 4 GPU box = two pairs) or on other
# boxes; the box that owns index 0 merges once every shard is on the hub.
#
# env: STEM, LLAMA_COMMIT, RECIPE (calib-corpora build), IM_TOTAL, IM_INDEX; optional
#      IM_CTX=512, IM_BATCH=8192, IM_WAIT_MIN=240
# out: $METRICS imatrix/shard-I-of-N.gguf, imatrix/imatrix.gguf, imatrix/imatrix.stats.txt
. "${PIPE:-/opt/pipeline}/lib/node_common.sh"
: "${RECIPE:?RECIPE, the calib-corpora build name, e.g. qwen3.8-27b}"
IM_TOTAL=${IM_TOTAL:-1}
IM_INDEX=${IM_INDEX:-$(seq -s ' ' 0 $(( IM_TOTAL - 1 )))}
IM_CTX=${IM_CTX:-512}
IM_BATCH=${IM_BATCH:-auto}
IM_WAIT_MIN=${IM_WAIT_MIN:-240}
IMD=$WORK/imatrix

if [ "$FORCE" != 1 ] && hf_has "$METRICS" dataset "imatrix/imatrix.gguf"; then
    say "imatrix already in $METRICS"
    done_node '"skipped": true'
fi

ensure_llama
ensure_bf16
mkdir -p "$IMD" "$WORK/corpus"
CORPUS=$WORK/corpus/calib_train.txt
if [ ! -f "$CORPUS" ]; then
    hf_get "$CALIB_REPO" dataset "$CALIB_REV" "builds/$RECIPE/*" "$WORK/corpus/dl" || fail 1 "no build $RECIPE in $CALIB_REPO"
    mv "$WORK/corpus/dl/builds/$RECIPE/calib_train.txt" "$CORPUS"
    cp "$WORK/corpus/dl/builds/$RECIPE/manifest.json" "$WORK/corpus/manifest.json"
fi
# the build log has the exact token count; bytes/4 is only a fallback (foundry im_size)
TOKENS=$("$PY" -c "import json; print(json.load(open('$WORK/corpus/manifest.json'))['calib_train']['tokens'])" 2>/dev/null \
         || echo $(( $(stat -c %s "$CORPUS") / 4 )))
CHUNKS=$(( TOKENS / IM_CTX ))
PER=$(( CHUNKS / IM_TOTAL + 1 ))

NGPU=$(nvidia-smi -L | wc -l)
set -- $IM_INDEX
LOCAL=$#
GPER=$(( NGPU / LOCAL ))
[ "$GPER" -ge 1 ] || fail 1 "$LOCAL shards on $NGPU GPUs"

# weights + logits (batch x vocab x 4 bytes) must fit the GPUs a shard gets (foundry im_shard)
ensure_inventory
VOCAB=$("$PY" -c "
import json
inv = json.load(open('$INVENTORY'))
print(next(t['shape'][1] for t in inv['tensors'] if t['name'] == 'token_embd.weight'))")
MODEL_BYTES=$("$PY" -c "
import json, math
inv = json.load(open('$INVENTORY'))
print(sum(math.prod(t['shape']) * (4 if t['type'] == 'f32' else 2) for t in inv['tensors']))")
PER_GPU_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
HAVE_GB=$(( PER_GPU_MB * GPER / 1000 ))
# largest batch that fits: 8192 is fastest (foundry), 27B on a 5090 pair only takes 4096
for b in ${IM_BATCH/auto/8192 4096 2048 1024}; do
    NEED_GB=$(( MODEL_BYTES / 1000000000 + VOCAB * b * 4 / 1000000000 + 4 ))
    if [ "$NEED_GB" -le "$HAVE_GB" ]; then IM_BATCH=$b; break; fi
done
[ "$NEED_GB" -le "$HAVE_GB" ] \
    || fail 1 "a shard needs ~${NEED_GB} GB even at batch 1024 but gets ${HAVE_GB} GB; run fewer shards per box"

say "$TOKENS tokens, $CHUNKS chunks of $IM_CTX, $IM_TOTAL shards of ~$PER; here: $IM_INDEX on $GPER GPU(s) each"
pids=()
slot=0
for I in $IM_INDEX; do
    OUT=$IMD/shard-$I-of-$IM_TOTAL.gguf
    if [ "$FORCE" != 1 ] && hf_has "$METRICS" dataset "imatrix/shard-$I-of-$IM_TOTAL.gguf"; then
        say "shard $I already on the hub"; slot=$(( slot + 1 )); continue
    fi
    FROM=$(( I * PER ))
    LIMIT="--chunks $PER"
    [ "$I" -eq $(( IM_TOTAL - 1 )) ] && LIMIT=""   # the last shard runs to the end of the corpus
    GPUS=$(seq -s, $(( slot * GPER )) $(( slot * GPER + GPER - 1 )))
    (
        CUDA_VISIBLE_DEVICES=$GPUS stdbuf -oL -eL "$BIN/llama-imatrix" -m "$BF16" -f "$CORPUS" -o "$OUT" \
            -ngl 99 -c "$IM_CTX" -b "$IM_BATCH" -ub "$IM_BATCH" --parse-special --output-frequency 20 \
            --from-chunk "$FROM" $LIMIT > "$LOGS/imatrix-shard-$I-of-$IM_TOTAL.log" 2>&1
        hf_up "$OUT" "imatrix/shard-$I-of-$IM_TOTAL.gguf" "$METRICS" dataset
        push_log "$LOGS/imatrix-shard-$I-of-$IM_TOTAL.log"
    ) &
    pids+=($!)
    say "shard $I: chunks from $FROM ${LIMIT:-to the end}, GPUs $GPUS, pid $!"
    slot=$(( slot + 1 ))
done
for p in "${pids[@]}"; do wait "$p" || fail 1 "a shard failed, see logs/imatrix-shard-*.log"; done

case " $IM_INDEX " in *" 0 "*) ;; *) done_node "\"shards\": \"$IM_INDEX\", \"merged\": false" ;; esac

# index 0 merges once every shard is on the hub
deadline=$(( $(date +%s) + IM_WAIT_MIN * 60 ))
for (( I=0; I<IM_TOTAL; I++ )); do
    until hf_has "$METRICS" dataset "imatrix/shard-$I-of-$IM_TOTAL.gguf"; do
        [ "$(date +%s)" -lt "$deadline" ] || fail 1 "shard $I never arrived"
        say "waiting for shard $I"; sleep 60
    done
done
hf_get "$METRICS" dataset - "imatrix/shard-*-of-$IM_TOTAL.gguf" "$WORK/imdl"
LIST=$(ls "$WORK"/imdl/imatrix/shard-*-of-$IM_TOTAL.gguf | sort -V | paste -sd, -)
# this llama-imatrix wants one comma separated list, and the model even for a pure merge
"$BIN/llama-imatrix" -m "$BF16" --in-file "$LIST" -o "$IMD/imatrix.gguf" > "$LOGS/imatrix-merge.log" 2>&1 \
    || { tail -20 "$LOGS/imatrix-merge.log"; fail 1 "merge failed"; }
"$BIN/llama-imatrix" -m "$BF16" --in-file "$IMD/imatrix.gguf" --show-statistics > "$IMD/imatrix.stats.txt" 2>&1 || true
[ "$(wc -l < "$IMD/imatrix.stats.txt")" -gt 20 ] || { cat "$IMD/imatrix.stats.txt"; fail 3 "--show-statistics printed almost nothing, the merged matrix is suspect"; }
ENTRIES=$(grep -oP 'loaded \K[0-9]+(?= importance matrix entries)' "$IMD/imatrix.stats.txt" | head -1)
hf_up "$IMD/imatrix.gguf" imatrix/imatrix.gguf "$METRICS" dataset
hf_up "$IMD/imatrix.stats.txt" imatrix/imatrix.stats.txt "$METRICS" dataset
push_log "$LOGS/imatrix-merge.log"
done_node "\"shards\": $IM_TOTAL, \"chunks\": $CHUNKS, \"entries\": \"${ENTRIES:-?}\", \"merged\": true"
