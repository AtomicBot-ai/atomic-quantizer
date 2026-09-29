#!/usr/bin/env bash
# Quantize rungs of the reviewed ladder, verify each against the ladder, measure KLD, publish.
# One rung at a time: quantize -> verify -> KLD -> upload -> delete (foundry ladder()).
#
# env: STEM, LLAMA_COMMIT, PROFILE (profiles/<name>.yaml); optional RUNGS (labels, default all
#      non-control rungs of ladder.json), KLD=1, KEEP_FILES=0, EVALSET, CTX
# in:  $METRICS bf16/, inventory.json, imatrix/imatrix.gguf, kld/, ladder/ (written by the driver)
# out: $MAIN <STEM>-<LABEL>.gguf (45 GB shards above 48 GB)
#      $METRICS logs/quantize-<LABEL>.log, logs/kld-<EVALSET>--<STEM>-<LABEL>.log, results/rows/<name>.json
. "${PIPE:-/opt/pipeline}/lib/node_common.sh"
: "${PROFILE:?PROFILE, e.g. dense-hybrid}"
KLD=${KLD:-1}
KEEP_FILES=${KEEP_FILES:-0}
PROFILE_FILE=$PIPE/profiles/$PROFILE.yaml
[ -f "$PROFILE_FILE" ] || fail 1 "no profile $PROFILE_FILE"

ensure_llama
ensure_bf16
ensure_inventory
mkdir -p "$WORK/quants" "$WORK/results/rows"

# the reviewed ladder, and proof it is the one this profile and inventory produce
hf_get "$METRICS" dataset - "ladder/*" "$WORK" || fail 1 "no ladder/ in $METRICS: run the ladder stage and review it first"
"$PY" "$PIPE/lib/ladder_gen.py" --inventory "$INVENTORY" --profile "$PROFILE_FILE" --out "$WORK/ladder-check" > "$LOGS/ladder-check.txt" \
    || { cat "$LOGS/ladder-check.txt"; fail 3 "the profile refuses this inventory"; }
"$PY" - "$WORK/ladder/ladder.json" "$WORK/ladder-check/ladder.json" <<'PY' || fail 3 "ladder/ on the hub is not what $PROFILE generates for this inventory"
import json, sys
a, b = (json.load(open(p)) for p in sys.argv[1:3])
sa = {r["label"]: r["rules_sha256"] for r in a["rungs"]}
sb = {r["label"]: r["rules_sha256"] for r in b["rungs"]}
bad = [k for k in sa if sa[k] != sb.get(k)]
if bad:
    print("rules differ for", bad)
    sys.exit(1)
PY

# rung table: label|ftype|file_type|imatrix|flags
RUNG_TABLE=$("$PY" - "$WORK/ladder/ladder.json" "${RUNGS:-}" <<'PY'
import json, sys
lad = json.load(open(sys.argv[1]))
want = sys.argv[2].split()
for r in lad["rungs"]:
    if (want and r["label"] not in want) or (not want and r.get("control")):
        continue
    print("|".join([r["label"], r["ftype"], str(r["file_type"] or ""), "1" if r["imatrix"] else "0", " ".join(r["flags"])]))
PY
)
[ -n "$RUNG_TABLE" ] || fail 1 "no rungs selected (RUNGS='${RUNGS:-}')"

IMATRIX=$WORK/imatrix/imatrix.gguf
# (no grep -q in a pipe: under pipefail the writer's SIGPIPE would turn a match into a miss)
if [[ "$(cut -d'|' -f4 <<< "$RUNG_TABLE")" == *1* ]] && [ ! -f "$IMATRIX" ]; then
    hf_get "$METRICS" dataset - "imatrix/imatrix.gguf" "$WORK" || fail 1 "no imatrix in $METRICS"
fi

BASE=$WORK/kld/base-$EVALSET.kld
KLD_ARGS=()
if [ "$KLD" = 1 ]; then
    ensure_eval
    # the measurement repeats the reference's protocol exactly, whatever this node's env says
    MANIFEST=$WORK/kld/dl/kld/base-$EVALSET.manifest.txt
    hf_get "$METRICS" dataset - "kld/base-$EVALSET.manifest.txt" "$WORK/kld/dl" || fail 1 "no KLD reference in $METRICS"
    CTX=$(awk '$1=="context"{print $2}' "$MANIFEST")
    KLD_CHUNKS=$(awk '$1=="chunks"{print $2}' "$MANIFEST"); KLD_CHUNKS=${KLD_CHUNKS:-0}
    WANT_SHA=$(awk '$1=="eval_sha256"{print $2}' "$MANIFEST")
    [ -z "$WANT_SHA" ] || [ "$(sha256sum "$EVAL" | cut -d' ' -f1)" = "$WANT_SHA" ] \
        || fail 1 "eval corpus differs from the one the reference was measured on"
    # shellcheck disable=SC2207
    KLD_ARGS=(-c "$CTX" $(kld_chunk_args))
    if [ ! -f "$BASE" ]; then
        hf_get "$METRICS" dataset - "kld/base-$EVALSET.kld*" "$WORK/kld/dl"
        if ls "$WORK"/kld/dl/kld/base-$EVALSET.kld.*.part >/dev/null 2>&1; then
            cat "$WORK"/kld/dl/kld/base-$EVALSET.kld.*.part > "$BASE"
        else
            mv "$WORK/kld/dl/kld/base-$EVALSET.kld" "$BASE"
        fi
        rm -rf "$WORK"/kld/dl/kld/base-$EVALSET.kld*
    fi
    WANT=$(awk '$1=="size_bytes"{print $2}' "$MANIFEST")
    [ "$(stat -c %s "$BASE")" = "$WANT" ] || fail 1 "reference is $(stat -c %s "$BASE") bytes, manifest says $WANT"
fi

has_quant() { hf_has "$MAIN" model "$1.gguf" || hf_has "$MAIN" model "$1-00001-of-*.gguf"; }

built=0; skipped=0
while IFS='|' read -r LABEL FTYPE FILE_TYPE USE_IM FLAGS; do
    NAME=$STEM-$LABEL
    QLOG=$LOGS/quantize-$LABEL.log
    KLOG=$LOGS/kld-$EVALSET--$NAME.log
    if [ "$FORCE" != 1 ] && has_quant "$NAME" && { [ "$KLD" != 1 ] || hf_has "$METRICS" dataset "logs/kld-$EVALSET--$NAME.log"; }; then
        say "$LABEL already published and measured"; skipped=$(( skipped + 1 )); continue
    fi
    OUT=$WORK/quants/$NAME.gguf
    args=(--tensor-type-file "$WORK/ladder/$LABEL.types")
    [ "$USE_IM" = 1 ] && args=(--imatrix "$IMATRIX" "${args[@]}")
    [ -n "$FILE_TYPE" ] && args+=(--override-kv "general.file_type=int:$FILE_TYPE")
    # shellcheck disable=SC2206
    [ -n "$FLAGS" ] && args+=($FLAGS)

    say "$LABEL: quantizing ($FTYPE, ${FILE_TYPE:-own} file_type)"
    START=$(date +%s)
    stdbuf -oL -eL "$BIN/llama-quantize" "${args[@]}" "$BF16" "$OUT" "$FTYPE" "$(nproc)" > "$QLOG" 2>&1 \
        || { rm -f "$OUT"; push_log "$QLOG"; fail 1 "$LABEL: llama-quantize failed, see $(basename "$QLOG")"; }
    QSEC=$(( $(date +%s) - START ))
    if ! "$PY" "$PIPE/lib/verify_quant.py" --inventory "$INVENTORY" --profile "$PROFILE_FILE" --label "$LABEL" \
            --log "$QLOG" --commit "$LLAMA_COMMIT" | tee "$LOGS/verify-$LABEL.txt"; then
        rm -f "$OUT"; push_log "$QLOG"; push_log "$LOGS/verify-$LABEL.txt"
        fail 3 "$LABEL: the file is not what the ladder asked for, deleted"
    fi
    SIZE=$(stat -c %s "$OUT")

    if [ "$KLD" = 1 ]; then
        say "$LABEL: KLD"
        stdbuf -oL -eL "$BIN/llama-perplexity" -m "$OUT" -f "$EVAL" --kl-divergence-base "$BASE" --kl-divergence \
            "${KLD_ARGS[@]}" -ngl $NGL > "$KLOG" 2>&1 || { push_log "$KLOG"; fail 1 "$LABEL: KLD run failed"; }
        "$PY" "$PIPE/lib/results.py" row --name "$NAME" --label "$LABEL" --kld "$KLOG" --quant-log "$QLOG" \
            --size-bytes "$SIZE" ${FILE_TYPE:+--file-type "$FILE_TYPE"} --commit "$(cat "$LOGS/llama-commit.txt")" \
            --evalset "$EVALSET" -o "$WORK/results/rows/$NAME.json"
        say "$LABEL: $(grep -oP 'Mean\s+KLD:\s+\K[0-9.]+' "$KLOG") mean KLD, $(( SIZE / 1000000 )) MB, quantize ${QSEC}s"
    fi

    # flat names at the root; above the 50 GB file limit, flat 45 GB shards (foundry push_model_split)
    if [ "$SIZE" -gt $(( 48 * 1024 ** 3 )) ]; then
        mkdir -p "$WORK/quants/split-$LABEL"
        "$BIN/llama-gguf-split" --split --split-max-size 45G "$OUT" "$WORK/quants/split-$LABEL/$NAME" > "$LOGS/split-$LABEL.log" 2>&1
        hf_up_dir "$WORK/quants/split-$LABEL" . "$MAIN" model
        rm -rf "$WORK/quants/split-$LABEL"
    else
        hf_up "$OUT" "$NAME.gguf" "$MAIN" model
    fi
    push_log "$QLOG"; push_log "$LOGS/verify-$LABEL.txt"
    if [ "$KLD" = 1 ]; then
        push_log "$KLOG"
        hf_up "$WORK/results/rows/$NAME.json" "results/rows/$NAME.json" "$METRICS" dataset
    fi
    [ "$KEEP_FILES" = 1 ] || rm -f "$OUT"
    built=$(( built + 1 ))
done <<< "$RUNG_TABLE"

push_log "$LOGS/llama-commit.txt"; push_log "$LOGS/env-$NODE.txt"
done_node "\"built\": $built, \"skipped\": $skipped"
