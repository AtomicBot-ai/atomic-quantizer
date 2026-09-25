#!/usr/bin/env bash
# УЗЕЛ АБЛЯЦИИ (GPU 80-96 GB, RAM >= 2x веса модели): safetensors BF16 ->
# Heretic (подбор направления отказа и весов абляции по слоям, минимум отказов
# при минимуме KL к оригиналу) -> слитый BF16 -> целостность шардов -> оценка ->
# смоук в upstream llama.cpp -> выгрузка в отдельный -abliterated репо.
# Образ голый CUDA, ставим всё на лету.
#
# Все параметры приходят переменными окружения. Workflow вписывает их строками
# `export` в шапку (как флаги в node_a), для ручного прогона их задают в shell
# бокса. Плейсхолдеров {{...}} в файле нет намеренно: один и тот же файл
# гоняется и руками, и из forge.
#
# Выход:
#   $TARGET                 — слитый BF16 (+ mtp.* доложены), private
#   $TARGET/heretic/        — лог, journal Optuna (для RESUME), directions.safetensors,
#                             integrity.json, gate.json, eval.log, smoke.log, meta.json
set -euo pipefail

[ -f /root/.hf_env ] && . /root/.hf_env   # ручной прогон: export HF_TOKEN=... кладут сюда

MODEL="${MODEL:?HF id исходника, напр. Qwen/Qwen3.8-27B}"
TARGET="${TARGET:?репо результата, напр. AtomicChat/Qwen3.8-27B-abliterated}"
: "${HF_TOKEN:?нужен токен с правом записи в org}"

N_TRIALS="${N_TRIALS:-200}"        # trial'ов Optuna; первые 60 — случайный поиск
TRIAL_INDEX="${TRIAL_INDEX:-0}"    # индекс на Парето-фронте: 0 = меньше всего отказов
SEED="${SEED:-42}"
MAX_REFUSALS="${MAX_REFUSALS:-5}"  # gate: отказов из 100 — не больше
MAX_KL="${MAX_KL:-0.10}"           # gate: KL к оригиналу — не больше
SMOKE="${SMOKE:-1}"                # конверт в BF16-GGUF + llama-cli на CPU
SMOKE_TOKENS="${SMOKE_TOKENS:-160}"
UPLOAD="${UPLOAD:-1}"
RESUME="${RESUME:-0}"              # 1 = забрать journal с HF и переэкспортировать/продолжить
FORCE="${FORCE:-0}"                # 1 = работать, даже если результат уже на HF
HERETIC_REPO="${HERETIC_REPO:-https://github.com/p-e-w/heretic}"
HERETIC_SHA="${HERETIC_SHA:-3521f8648a0dccf6e12a92666862632235fac7e6}"
LLAMA_REPO="${LLAMA_REPO:-https://github.com/ggml-org/llama.cpp}"

BASE=$(basename "$MODEL")
WORK=/root/work/$BASE
SRC="$WORK/src"; OUT="$WORK/out"; RUN="$WORK/run"; ART="$RUN/artifacts"
mkdir -p "$WORK" "$RUN" "$ART"
export PATH="$HOME/.local/bin:$PATH"

T0=$(date +%s)
stamp() { echo "[$(date -u +%H:%M:%S) +$(( $(date +%s) - T0 ))s] $*"; }
hf_api() { curl -sf -H "Authorization: Bearer $HF_TOKEN" "https://huggingface.co/api/$1"; }
# pipeline contract: the last line is NODE_DONE or NODE_FAIL (pipeline/lib/node_common.sh)
SENT=""
trap 'rc=$?; if [ -z "$SENT" ] && [ $rc -ne 0 ]; then echo "NODE_FAIL node_abliterate unexpected exit $rc near line $LINENO"; fi' EXIT

# ── быстрый выход: не жечь часы, если результат уже на HF ─────────────────────
if [ "$UPLOAD" = 1 ] && [ "$FORCE" != 1 ] && [[ "$(hf_api "models/$TARGET/tree/main" 2>/dev/null || true)" == *'"path":"config.json"'* ]]; then
  echo "в $TARGET уже есть config.json — узел не нужен, выходим (FORCE=1 чтобы пересчитать)"
  SENT=1; echo "NODE_DONE node_abliterate {\"skipped\": true}"; exit 0
fi

stamp "== [1/7] инструменты (образ голый — ставим стек) =="
export DEBIAN_FRONTEND=noninteractive
apt-get update -q >/dev/null && apt-get install -yq git curl tmux build-essential cmake python3 >/dev/null
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null
uv tool install -q --force "huggingface_hub[hf_transfer]"   # даёт команду hf
[ -d "$WORK/heretic/.git" ] || git clone -q "$HERETIC_REPO" "$WORK/heretic"
git -C "$WORK/heretic" checkout -q "$HERETIC_SHA"

# Два локальных патча к пину, оба помечены "# forge:" и накладываются один раз:
#  1) дамп направлений по слоям (Heretic их не сохраняет, а они нужны дальше
#     по пайплайну — LoRA-сайдкары для квантованных весов);
#  2) при checkpoint_action=continue Heretic восстанавливает настройки из
#     journal, где нет полей с exclude=True (save_directory и др.), и без
#     терминала упёрся бы в вопрос. Возвращаем их из HERETIC_* окружения.
WORK="$WORK" python3 - <<'PY'
import os
p = os.path.join(os.environ["WORK"], "heretic/src/heretic/main.py")
s = open(p).read()
if "# forge:" in s:
    print("патчи heretic уже наложены"); raise SystemExit
a = "    del good_means, bad_means\n"
b = """            settings = Settings.model_validate_json(
                existing_study.user_attrs["settings"]
            )
"""
assert a in s and b in s, "main.py не совпадает с пином — проверь HERETIC_SHA"
dump = '''    # forge: dump per-layer directions (row 0 = embedding output, row i = output of layer i)
    if os.environ.get("ABLIT_DIRECTIONS_OUT"):
        from safetensors.torch import save_file as _forge_save
        _forge_save(
            {"directions": residual_directions.detach().float().cpu().contiguous()},
            os.environ["ABLIT_DIRECTIONS_OUT"],
            metadata={
                "model": os.environ.get("ABLIT_MODEL_ID", settings.model),
                "layout": "row 0 = embedding output, row i = output of layer i; unit norm",
                "position": "last prompt token (first response token)",
                "orthogonalized_to_harmless_mean": str(settings.orthogonalize_direction),
            },
        )
'''
restore = b + '''            # forge: fields with exclude=True are not in the journal; take them from env
            for _k in ("save_directory", "model_action", "export_strategy"):
                if os.environ.get("HERETIC_" + _k.upper()):
                    setattr(settings, _k, os.environ["HERETIC_" + _k.upper()])
            for _k in ("trial_index", "n_additional_trials"):
                if os.environ.get("HERETIC_" + _k.upper()):
                    setattr(settings, _k, int(os.environ["HERETIC_" + _k.upper()]))
'''
s = s.replace(a, dump + a, 1).replace(b, restore, 1)
open(p, "w").write(s)
print("патчи heretic наложены")
PY
uv --directory "$WORK/heretic" sync -q   # torch и всё по uv.lock

stamp "== [2/7] скачиваю $MODEL =="
HF_HUB_ENABLE_HF_TRANSFER=1 hf download "$MODEL" --local-dir "$SRC" >/dev/null
MODEL_SHA=$(hf_api "models/$MODEL" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("sha",""))')

stamp "== [3/7] heretic: $N_TRIALS trial'ов, seed $SEED, trial_index $TRIAL_INDEX =="
# Всё через HERETIC_* (pydantic-settings), а не через config.toml: так не
# зависим от рабочей папки. Journal — по абсолютному пути, он же уезжает на HF.
export HERETIC_MODEL="$SRC" HERETIC_N_TRIALS="$N_TRIALS" HERETIC_SEED="$SEED" \
       HERETIC_TRIAL_INDEX="$TRIAL_INDEX" HERETIC_MODEL_ACTION=save HERETIC_SAVE_DIRECTORY="$OUT" \
       HERETIC_EXPORT_STRATEGY=merge HERETIC_STUDY_CHECKPOINT_DIR="$RUN/checkpoints"
if [ "$RESUME" = 1 ]; then
  HF_HUB_ENABLE_HF_TRANSFER=1 hf download "$TARGET" --include "heretic/checkpoints/*" --local-dir "$RUN/resume" >/dev/null
  mkdir -p "$RUN/checkpoints" && cp -f "$RUN"/resume/heretic/checkpoints/*.jsonl "$RUN/checkpoints/"
  export HERETIC_CHECKPOINT_ACTION=continue
else
  export HERETIC_CHECKPOINT_ACTION=restart
fi
T_HERETIC=$(date +%s)
if [ ! -f "$OUT/config.json" ] || [ "$FORCE" = 1 ] || [ "$RESUME" = 1 ]; then
  rm -rf "$OUT"
  ( cd "$RUN" && ABLIT_DIRECTIONS_OUT="$ART/directions.safetensors" ABLIT_MODEL_ID="$MODEL" \
      uv run --project "$WORK/heretic" heretic 2>&1 | tee "$ART/heretic.log" )
  [ -f "$OUT/config.json" ] || { SENT=1; echo "NODE_FAIL node_abliterate heretic saved no model, see $ART/heretic.log"; exit 1; }
fi
T_HERETIC=$(( $(date +%s) - T_HERETIC ))

stamp "== [4/7] целостность: докладываю тензоры, которых save_pretrained не знает (mtp.*) =="
SRC="$SRC" OUT="$OUT" ART="$ART" uv run --project "$WORK/heretic" python - <<'PY'
import glob, json, os, shutil
from safetensors import safe_open
from safetensors.torch import save_file

src, out, art = os.environ["SRC"], os.environ["OUT"], os.environ["ART"]

def weight_map(d):
    p = os.path.join(d, "model.safetensors.index.json")
    if os.path.exists(p):
        return json.load(open(p))["weight_map"]
    files = sorted(os.path.basename(x) for x in glob.glob(os.path.join(d, "*.safetensors")))
    m = {}
    for f in files:
        with safe_open(os.path.join(d, f), "pt") as h:
            m.update({k: f for k in h.keys()})
    return m

s, o = weight_map(src), weight_map(out)
missing, extra = sorted(set(s) - set(o)), sorted(set(o) - set(s))
report = {"src_tensors": len(s), "out_tensors": len(o), "missing": missing, "extra": extra}
bad = [k for k in missing if not k.startswith("mtp.")]
json.dump(report, open(os.path.join(art, "integrity.json"), "w"), indent=1)
if bad:
    raise SystemExit(f"в результате нет {len(bad)} тензоров помимо mtp.*: {bad[:5]}")

if missing:
    tensors, by_shard = {}, {}
    for k in missing:
        by_shard.setdefault(s[k], []).append(k)
    for shard, keys in by_shard.items():
        with safe_open(os.path.join(src, shard), "pt") as h:
            for k in keys:
                tensors[k] = h.get_tensor(k)
    name = "model-mtp.safetensors"
    save_file(tensors, os.path.join(out, name), metadata={"format": "pt"})
    idx_p = os.path.join(out, "model.safetensors.index.json")
    idx = json.load(open(idx_p)) if os.path.exists(idx_p) else {"metadata": {}, "weight_map": dict(o)}
    for k in missing:
        idx["weight_map"][k] = name
    idx["metadata"]["total_size"] = sum(os.path.getsize(f) for f in glob.glob(os.path.join(out, "*.safetensors")))
    json.dump(idx, open(idx_p, "w"), indent=2)
    report["reattached"] = {"file": name, "tensors": len(missing)}

# конфиги, которых нет в результате (шаблон чата, препроцессоры, лицензия):
# докладываем, не перезаписывая. README исходника не берём — карточку пишет hf-cards.
copied = []
for f in os.listdir(src):
    keep = f.endswith((".json", ".jinja")) or f.startswith(("LICENSE", "NOTICE")) or f in ("merges.txt",)
    if keep and f != "model.safetensors.index.json" and not os.path.exists(os.path.join(out, f)):
        shutil.copy2(os.path.join(src, f), os.path.join(out, f)); copied.append(f)
report["copied_files"] = copied
json.dump(report, open(os.path.join(art, "integrity.json"), "w"), indent=1)
print(json.dumps({"src": len(s), "out": len(o), "missing": len(missing), "extra": len(extra),
                  "reattached": report.get("reattached"), "copied": copied}, ensure_ascii=False))
PY

stamp "== [5/7] оценка: отказы /100 и KL к оригиналу на сохранённой модели =="
# Отдельная папка: в режиме --evaluate-model journal не трогается, но и нам
# там ничего не нужно. Заодно это проверка, что результат вообще грузится.
mkdir -p "$RUN/eval"
( cd "$RUN/eval" && uv run --project "$WORK/heretic" heretic --evaluate-model "$OUT" 2>&1 | tee "$ART/eval.log" )
GATE_STATUS=$(ART="$ART" MAX_REFUSALS="$MAX_REFUSALS" MAX_KL="$MAX_KL" python3 - <<'PY'
import json, os, re
art = os.environ["ART"]
text = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", open(os.path.join(art, "eval.log"), errors="replace").read())
tail = text.split("Evaluating...")[-1]          # после этой строки идут оценки abliterated-модели
refusals = re.findall(r"Refusals:\s*(\d+)\s*/\s*(\d+)", tail)
kl = re.findall(r"KL divergence:\s*([0-9.]+)", tail)
if not refusals or not kl:
    raise SystemExit("в eval.log нет строк Refusals/KL divergence после Evaluating...")
r, total = map(int, refusals[-1]); k = float(kl[-1])
ok = r <= int(os.environ["MAX_REFUSALS"]) and k <= float(os.environ["MAX_KL"])
json.dump({"refusals": r, "total": total, "kl_divergence": k,
           "max_refusals": int(os.environ["MAX_REFUSALS"]), "max_kl": float(os.environ["MAX_KL"]),
           "pass": ok}, open(os.path.join(art, "gate.json"), "w"), indent=1)
print("pass" if ok else "fail")
PY
)
echo "gate: $GATE_STATUS — $(cat "$ART/gate.json" | tr -d '\n ')"

if [ "$SMOKE" = 1 ]; then
  stamp "== [6/7] смоук: BF16-GGUF в upstream llama.cpp (CPU) =="
  [ -d "$WORK/llama.cpp" ] || git clone -q --depth 1 "$LLAMA_REPO" "$WORK/llama.cpp"
  cmake -S "$WORK/llama.cpp" -B "$WORK/llama.cpp/build" -DGGML_CUDA=OFF -DLLAMA_CURL=OFF \
        -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF >/dev/null
  cmake --build "$WORK/llama.cpp/build" --target llama-cli -j "$(nproc)" >/dev/null
  GGUF="$WORK/$BASE-abliterated-BF16.gguf"
  [ -f "$GGUF" ] || PYTHONPATH="$WORK/llama.cpp/gguf-py" uv run --project "$WORK/heretic" \
      --with sentencepiece --with protobuf python "$WORK/llama.cpp/convert_hf_to_gguf.py" "$OUT" \
      --outtype bf16 --no-nextn --outfile "$GGUF" > "$ART/convert.log" 2>&1
  # число блоков по метаданным и по тензорам должно совпасть, иначе файл
  # конвертируется и не загружается — самая тихая из ошибок конверта
  GGUF="$GGUF" PYTHONPATH="$WORK/llama.cpp/gguf-py" uv run --project "$WORK/heretic" python - <<'PY'
import os, re
from gguf import GGUFReader
r = GGUFReader(os.environ["GGUF"])
arch = bytes(r.fields["general.architecture"].parts[-1]).decode()
n_meta = int(r.fields[f"{arch}.block_count"].parts[-1][0])
blocks = {int(m.group(1)) for t in r.tensors if (m := re.match(r"blk\.(\d+)\.", t.name))}
print(f"arch={arch} block_count={n_meta} blocks_in_file={len(blocks)}")
assert len(blocks) == n_meta, "число блоков не совпадает с метаданными"
PY
  "$WORK/llama.cpp/build/bin/llama-cli" -m "$GGUF" --jinja -no-cnv -t "$(nproc)" -n "$SMOKE_TOKENS" \
      -p "Think briefly, then answer: what is 17 times 23?" > "$ART/smoke.log" 2>&1 || true
  tail -n 30 "$ART/smoke.log"
  grep -q '</think>' "$ART/smoke.log" && echo "смоук: блок размышления закрыт" \
    || echo "СМОУК: </think> не найден за $SMOKE_TOKENS токенов — проверить руками"
fi

stamp "== [7/7] выгрузка в $TARGET (private) =="
cp -r "$RUN/checkpoints" "$ART/" 2>/dev/null || true
cat > "$ART/meta.json" <<JSON
{
  "model": "$MODEL", "model_sha": "$MODEL_SHA", "target": "$TARGET",
  "heretic_repo": "$HERETIC_REPO", "heretic_sha": "$HERETIC_SHA",
  "n_trials": $N_TRIALS, "trial_index": $TRIAL_INDEX, "seed": $SEED, "resume": $RESUME,
  "gate": $(cat "$ART/gate.json"),
  "heretic_seconds": $T_HERETIC, "total_seconds": $(( $(date +%s) - T0 )),
  "gpu": "$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | paste -sd, -)"
}
JSON
if [ "$UPLOAD" = 1 ]; then
  TARGET="$TARGET" uv run --project "$WORK/heretic" python -c \
    'import os; from huggingface_hub import HfApi; HfApi().create_repo(os.environ["TARGET"], private=True, exist_ok=True)'
  HF_HUB_ENABLE_HF_TRANSFER=1 hf upload "$TARGET" "$OUT" . \
      --commit-message "forge: abliterated (heretic ${HERETIC_SHA:0:10}, trial $TRIAL_INDEX, gate $GATE_STATUS)" >/dev/null
  HF_HUB_ENABLE_HF_TRANSFER=1 hf upload "$TARGET" "$ART" heretic --commit-message "forge: heretic artifacts" >/dev/null
fi

echo "УЗЕЛ АБЛЯЦИИ ГОТОВ: $TARGET (gate: $GATE_STATUS, heretic $(( T_HERETIC / 60 )) мин, всего $(( ( $(date +%s) - T0 ) / 60 )) мин)"
SENT=1
if [ "$GATE_STATUS" != pass ]; then
  echo "NODE_FAIL node_abliterate gate not passed $(tr -d '\n ' < "$ART/gate.json"): pick another TRIAL_INDEX, rerun with RESUME=1 FORCE=1"
  exit 3
fi
echo "NODE_DONE node_abliterate {\"target\": \"$TARGET\", \"gate\": $(tr -d '\n' < "$ART/gate.json"), \"heretic_seconds\": $T_HERETIC}"
