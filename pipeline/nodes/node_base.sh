#!/usr/bin/env bash
# The KLD reference: BF16 logits over the held-out corpus, measured once per model.
#
# env: STEM, LLAMA_COMMIT; optional EVALSET=neutral, CTX=4096, KLD_CHUNKS=0 (all), SELFCHECK=1
# out: $METRICS kld/base-$EVALSET.kld[.NN.part], kld/base-$EVALSET.manifest.txt, logs/base-$EVALSET.log
#      The manifest carries the protocol (context, chunks, corpus sha) node_quant must repeat.
. "${PIPE:-/opt/pipeline}/lib/node_common.sh"
SELFCHECK=${SELFCHECK:-1}
BASE=$WORK/kld/base-$EVALSET.kld

if [ "$FORCE" != 1 ] && hf_has "$METRICS" dataset "kld/base-$EVALSET.manifest.txt"; then
    say "reference already in $METRICS"
    done_node '"skipped": true'
fi

ensure_llama
ensure_bf16
ensure_eval
mkdir -p "$WORK/kld"
# shellcheck disable=SC2046
set -- $(kld_chunk_args)

# foundry base: all GPUs, the BF16 file does not fit on fewer
say "writing the reference over $(wc -c < "$EVAL") bytes of $EVALSET at ctx $CTX ${1:+($2 chunks)}"
unset CUDA_VISIBLE_DEVICES
stdbuf -oL -eL "$BIN/llama-perplexity" -m "$BF16" -f "$EVAL" --kl-divergence-base "$BASE" -c "$CTX" -ngl $NGL "$@" \
    > "$LOGS/base-$EVALSET.log" 2>&1 || { tail -20 "$LOGS/base-$EVALSET.log"; fail 1 "llama-perplexity failed"; }
push_log "$LOGS/base-$EVALSET.log"; push_log "$LOGS/env-$NODE.txt"
[ -s "$BASE" ] || fail 1 "no reference written"

# the reference measured against itself must be exactly zero, or every later number is noise
if [ "$SELFCHECK" = 1 ]; then
    stdbuf -oL -eL "$BIN/llama-perplexity" -m "$BF16" -f "$EVAL" --kl-divergence-base "$BASE" --kl-divergence \
        -c "$CTX" -ngl $NGL "$@" > "$LOGS/kld-selfcheck.log" 2>&1
    push_log "$LOGS/kld-selfcheck.log"
    SELF=$(grep -oP 'Mean\s+KLD:\s+\K[-0-9.eE+]+' "$LOGS/kld-selfcheck.log" | head -1 || true)
    "$PY" -c "import sys; sys.exit(0 if abs(float('${SELF:-1}')) < 1e-5 else 1)" \
        || fail 3 "self-check KLD is ${SELF:-missing}, the reference does not reproduce itself"
fi

# foundry push_base: a byte count to verify reassembly, not a hash (the blob is not bit reproducible)
SIZE=$(stat -c %s "$BASE")
mkdir -p "$WORK/kld/up"
{
    echo "size_bytes $SIZE"
    echo "context $CTX"
    echo "chunks $KLD_CHUNKS"
    echo "eval_set $EVALSET"
    echo "eval_sha256 $(sha256sum "$EVAL" | cut -d' ' -f1)"
    echo "calib_rev $CALIB_REV"
    echo "llama_commit $(cat "$LOGS/llama-commit.txt")"
} > "$WORK/kld/up/base-$EVALSET.manifest.txt"
if [ "$SIZE" -gt 45000000000 ]; then
    split -b 45G -d --additional-suffix=.part "$BASE" "$WORK/kld/up/base-$EVALSET.kld."
else
    ln -f "$BASE" "$WORK/kld/up/base-$EVALSET.kld"
fi
say "uploading the reference ($(( SIZE / 1000000 )) MB)"
hf_up_dir "$WORK/kld/up" kld "$METRICS" dataset

PPL=$(grep -oP 'Final estimate: PPL = \K[0-9.]+' "$LOGS/base-$EVALSET.log" | head -1 || true)
done_node "\"base_bytes\": $SIZE, \"ppl\": \"${PPL:-?}\", \"selfcheck_kld\": \"${SELF:-skipped}\""
