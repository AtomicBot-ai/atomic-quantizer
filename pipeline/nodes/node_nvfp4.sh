#!/usr/bin/env bash
# BF16 safetensors -> NVFP4 safetensors with llm-compressor, calibrated on the model's own build.
# Port of atomic-forge src/node_nvfp4.sh.tpl (worked on five gemma-4 models in July 2026),
# with the calibration corpus taken from calib-corpora builds/<RECIPE> like the GGUF imatrix.
#
# env: MODEL, TARGET_NVFP4, RECIPE, STEM; optional CALIB_REV, NVFP4_SAMPLES=256, NVFP4_SEQ=2048
# out: $TARGET_NVFP4 (private), compressed-tensors checkpoint for vLLM
set -euo pipefail
[ -f /root/.hf_env ] && . /root/.hf_env
: "${HF_TOKEN:?no HF_TOKEN in /root/.hf_env}"
: "${MODEL:?}" "${TARGET_NVFP4:?}" "${RECIPE:?}"
export HF_TOKEN HF_HUB_DISABLE_PROGRESS_BARS=1
CALIB_REPO=${CALIB_REPO:-AtomicChat/calib-corpora}
CALIB_REV=${CALIB_REV:-814d662f6c94d207fc8f38545a1b4abea11484b5}
NVFP4_SAMPLES=${NVFP4_SAMPLES:-256}
NVFP4_SEQ=${NVFP4_SEQ:-2048}
FORCE=${FORCE:-0}
WORK=/root/work/$(basename "$MODEL")-nvfp4
OUT=$WORK/nvfp4
mkdir -p "$WORK"
T0=$(date +%s)
say() { echo "[$(date -u +%H:%M:%S) +$(( $(date +%s) - T0 ))s] $*"; }
SENT=""
trap 'rc=$?; if [ -z "$SENT" ] && [ $rc -ne 0 ]; then echo "NODE_FAIL node_nvfp4 unexpected exit $rc near line $LINENO"; fi' EXIT

VENV=/opt/venv-nvfp4
if [ ! -x $VENV/bin/python ]; then
    say "installing the stack (bare image)"
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq >/dev/null && apt-get install -y -qq python3-venv python3-pip git curl >/dev/null
    python3 -m venv $VENV
    $VENV/bin/pip install -q -U pip "huggingface_hub>=1.0" datasets
    # torchvision: multimodal processors need it; llmcompressor last so its transformers window wins
    $VENV/bin/pip install -q vllm torchvision
    $VENV/bin/pip install -q -U llmcompressor
fi
PY=$VENV/bin/python
$PY -c "import transformers, llmcompressor; print('transformers', transformers.__version__, 'llmcompressor', llmcompressor.__version__)"

if [ "$FORCE" != 1 ] && $PY -c "
from huggingface_hub import HfApi; import sys
sys.exit(0 if HfApi().file_exists('$TARGET_NVFP4', 'config.json') else 1)" 2>/dev/null; then
    SENT=1; echo "NODE_DONE node_nvfp4 {\"skipped\": true}"; exit 0
fi

say "downloading $MODEL and calib build $RECIPE"
$PY - "$MODEL" "$WORK/hf" "$CALIB_REPO" "$CALIB_REV" "$RECIPE" "$WORK/corpus" <<'PY'
import sys
from huggingface_hub import snapshot_download
model, dest, crepo, crev, recipe, cdest = sys.argv[1:7]
snapshot_download(model, local_dir=dest)
snapshot_download(crepo, repo_type="dataset", revision=crev, allow_patterns=[f"builds/{recipe}/calib_train.txt"], local_dir=cdest)
PY
CORPUS=$WORK/corpus/builds/$RECIPE/calib_train.txt

# NVFP4 quantizes activations too, so llm-compressor wants calibration and traces the model
# layer by layer. Hybrid attention broke that tracing on gemma-4; the ladder of fallbacks is
# from the forge script: sequential -> basic pipeline -> NVFP4A16 (weights only, no data).
run_calib() {  # $1 scheme, $2 pipeline ("" = auto), $3 with-data | data-free
    SCHEME="$1" PIPE="$2" MODE="$3" WORK="$WORK" OUT="$OUT" CORPUS="$CORPUS" N="$NVFP4_SAMPLES" SEQ="$NVFP4_SEQ" $PY - <<'PY'
import os
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

recipe = QuantizationModifier(
    targets="Linear", scheme=os.environ["SCHEME"],
    ignore=["lm_head", "re:.*embed_tokens", "re:.*gate$", "re:.*norm.*", "re:.*router.*", "re:.*visual.*",
            "re:.*vision_tower.*", "re:.*embed_vision.*", "re:.*audio_tower.*", "re:.*embed_audio.*",
            "re:.*multi_modal_projector.*", "re:.*mtp.*"],
)
kw = dict(model=os.path.join(os.environ["WORK"], "hf"), recipe=recipe, output_dir=os.environ["OUT"])
if os.environ["PIPE"]:
    kw["pipeline"] = os.environ["PIPE"]
if os.environ["MODE"] == "with-data":
    from datasets import Dataset
    # calib_train.txt separates documents with a blank line (build manifest: document_separator)
    docs = [d.strip() for d in open(os.environ["CORPUS"]).read().split("\n\n") if len(d.strip()) > 200]
    docs = docs[:: max(1, len(docs) // int(os.environ["N"]))][: int(os.environ["N"])]
    kw.update(dataset=Dataset.from_dict({"text": docs}), max_seq_length=int(os.environ["SEQ"]),
              num_calibration_samples=len(docs))
oneshot(**kw)
PY
}

MODE=""
if run_calib NVFP4 "" with-data > "$WORK/calib.log" 2>&1; then MODE="NVFP4 sequential"
elif rm -rf "$OUT" && run_calib NVFP4 basic with-data >> "$WORK/calib.log" 2>&1; then MODE="NVFP4 basic"
elif rm -rf "$OUT" && run_calib NVFP4A16 "" data-free >> "$WORK/calib.log" 2>&1; then MODE="NVFP4A16 data-free"
else tail -30 "$WORK/calib.log"; SENT=1; echo "NODE_FAIL node_nvfp4 all three calibration paths failed"; exit 1
fi
say "calibrated: $MODE"

# native NVFP4 kernels are Blackwell only; elsewhere the smoke test may fail without the file being wrong
SMOKE=ok
OUT_DIR=$OUT $PY - <<'PY' > "$WORK/smoke.log" 2>&1 || SMOKE="failed (not fatal off Blackwell)"
import os
from vllm import LLM, SamplingParams
llm = LLM(model=os.environ["OUT_DIR"], max_model_len=4096)
print(llm.generate(["Write a Python retry wrapper."], SamplingParams(max_tokens=64))[0].outputs[0].text[:200])
PY
say "vLLM smoke: $SMOKE"

$PY - "$TARGET_NVFP4" "$OUT" "$WORK" "$MODE" <<'PY'
import os, sys
from huggingface_hub import HfApi
target, out, work, mode = sys.argv[1:5]
api = HfApi()
api.create_repo(target, private=True, exist_ok=True)
api.upload_folder(folder_path=out, repo_id=target, commit_message=f"pipeline: {mode}")
for f in ("calib.log", "smoke.log"):
    api.upload_file(path_or_fileobj=os.path.join(work, f), path_in_repo=f"logs/{f}", repo_id=target)
PY
SENT=1
echo "NODE_DONE node_nvfp4 {\"target\": \"$TARGET_NVFP4\", \"mode\": \"$MODE\", \"smoke\": \"$SMOKE\", \"seconds\": $(( $(date +%s) - T0 ))}"
