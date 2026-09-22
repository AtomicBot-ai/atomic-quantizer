#!/usr/bin/env bash
# HF safetensors -> BF16 GGUF (+ mmproj) -> inventory.json, all published to the metrics repo.
#
# env: MODEL (hub id), STEM, LLAMA_COMMIT; optional MODEL_REV, KEEP_MTP=1, REPO_SUFFIX
# out: $METRICS bf16/$STEM-BF16*.gguf, inventory.json, logs/convert.log, logs/llama-commit.txt
#      $MAIN    mmproj-$STEM-{F16,BF16}.gguf when the model has a vision tower
. "${PIPE:-/opt/pipeline}/lib/node_common.sh"
: "${MODEL:?MODEL, the upstream hub id}"
KEEP_MTP=${KEEP_MTP:-1}

if [ "$FORCE" != 1 ] && hf_has "$METRICS" dataset "inventory.json" && hf_has "$METRICS" dataset "bf16/$STEM-BF16*.gguf"; then
    say "BF16 and inventory already in $METRICS"
    done_node '"skipped": true'
fi

ensure_llama
hf_ensure "$METRICS" dataset
hf_ensure "$MAIN" model

SRC=$WORK/src
say "downloading $MODEL ${MODEL_REV:+@ $MODEL_REV}"
hf_get "$MODEL" model "${MODEL_REV:--}" "*" "$SRC" > "$LOGS/download.log" 2>&1 || fail 1 "download of $MODEL failed"
MODEL_SHA=$(hf_py "$MODEL" "${MODEL_REV:--}" <<'PY'
import sys
from huggingface_hub import HfApi
print(HfApi().model_info(sys.argv[1], revision=None if sys.argv[2] == "-" else sys.argv[2]).sha)
PY
)
echo "$MODEL $MODEL_SHA" > "$LOGS/source.txt"

# foundry make_bf16. The MTP head stays by default: the published Qwen3.8 files
# ship it and advertise --spec-type draft-mtp. KEEP_MTP=0 drops it.
mkdir -p "$WORK/gguf"
NEXTN=""
[ "$KEEP_MTP" = 1 ] || NEXTN="--no-nextn"
say "converting to $BF16 ${NEXTN:-(MTP kept)}"
"$PY" "$LLAMA/convert_hf_to_gguf.py" "$SRC" --outtype bf16 $NEXTN --outfile "$BF16" > "$LOGS/convert.log" 2>&1 \
    || { tail -20 "$LOGS/convert.log"; push_log "$LOGS/convert.log"; fail 2 "convert_hf_to_gguf.py refused $MODEL (architecture not supported at $LLAMA_COMMIT?)"; }

# foundry check_blocks: metadata and tensors must agree or the file will not load
if ! "$PY" - "$BF16" > "$LOGS/check-blocks.txt" <<'PY'
import re, sys
from gguf import GGUFReader
r = GGUFReader(sys.argv[1])
arch = r.fields["general.architecture"].contents()
declared = int(r.fields[f"{arch}.block_count"].contents())
seen = {int(m.group(1)) for t in r.tensors if (m := re.match(r"blk\.(\d+)\.", t.name))}
print(f"architecture {arch}, declared {declared} blocks, present {len(seen)}")
if declared != (max(seen) + 1 if seen else 0):
    print("MISMATCH: the file converts but will not load. An MTP head the weights never shipped? Use KEEP_MTP=0.")
    sys.exit(3)
PY
then
    cat "$LOGS/check-blocks.txt"; push_log "$LOGS/check-blocks.txt"
    fail 3 "block count mismatch"
fi
cat "$LOGS/check-blocks.txt"

"$PY" "$PIPE/lib/gguf_inventory.py" "$BF16" -o "$WORK/inventory.json"

# foundry make_mmproj: one projector serves the whole ladder
MMPROJ=0
if [ -f "$SRC/preprocessor_config.json" ]; then
    for t in f16 bf16; do
        out=$WORK/gguf/mmproj-$STEM-${t^^}.gguf
        "$PY" "$LLAMA/convert_hf_to_gguf.py" "$SRC" --mmproj --outtype $t --outfile "$out" > "$LOGS/mmproj-$t.log" 2>&1 \
            || fail 1 "mmproj $t conversion failed, see mmproj-$t.log"
        hf_up "$out" "$(basename "$out")" "$MAIN" model
        MMPROJ=$(( MMPROJ + 1 ))
    done
fi

# files over the 50 GB hub limit go up in 45 GB shards (foundry push_model_split)
mkdir -p "$WORK/gguf/bf16"
SIZE=$(stat -c %s "$BF16")
if [ "$SIZE" -gt $(( 48 * 1024 ** 3 )) ]; then
    "$BIN/llama-gguf-split" --split --split-max-size 45G "$BF16" "$WORK/gguf/bf16/$STEM-BF16" > "$LOGS/split.log" 2>&1
else
    ln -f "$BF16" "$WORK/gguf/bf16/$STEM-BF16.gguf"
fi
say "uploading BF16 ($(( SIZE / 1000000000 )) GB)"
hf_up_dir "$WORK/gguf/bf16" bf16 "$METRICS" dataset
hf_up "$WORK/inventory.json" inventory.json "$METRICS" dataset
for f in convert.log check-blocks.txt source.txt llama-commit.txt llama-version.txt env-$NODE.txt; do push_log "$LOGS/$f"; done

done_node "\"model_sha\": \"$MODEL_SHA\", \"bf16_bytes\": $SIZE, \"mmproj\": $MMPROJ"
