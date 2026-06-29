#!/usr/bin/env bash
# Run multi-VLM evaluation on adversarial outputs (MCA / MCA2-style folders).
#
# Usage (from repo root):
#   bash eval/run_multi_vlm_eval.sh
# Override any variable inline:
#   VARIANT=MCA RUN_HASH='...' bash eval/run_multi_vlm_eval.sh
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ -d "$REPO_ROOT/FOA_attack" ]]; then
  # shellcheck source=/dev/null
  source "$REPO_ROOT/FOA_attack/bin/activate"
fi

: "${API_KEYS_FILE:=$REPO_ROOT/api_keys.yaml}"

# Experiment folder under foa_attack_output6_0_99_use_<VARIANT>/img/<RUN_HASH>/
: "${VARIANT:=MCA2}"
: "${RUN_HASH:=73fa66c7fe9308d2e8d0e8945325f347_target_noise_masking}"
: "${OUTPUT_ROOT:=foa_attack_output6_0_99_use_${VARIANT}}"

: "${KEYWORDS_JSON:=$REPO_ROOT/resources/images/target_images/keywords.json}"
: "${TARGET_DIR:=$REPO_ROOT/resources/images/target_images_copy}"

SAMPLE_ROOT="$REPO_ROOT/${OUTPUT_ROOT}/img/${RUN_HASH}"
ADV_DIR="${ADV_DIR_OVERRIDE:-$SAMPLE_ROOT/nips17}"
: "${OUT_DIR:=$REPO_ROOT/eval/multi_vlm_results_${VARIANT}_$(echo "$RUN_HASH" | cut -c1-8)}"

MODELS=(qwen claude4.0 gpt4o gpt5)

echo "[repo]       $REPO_ROOT"
echo "[sample]     $SAMPLE_ROOT"
echo "[adv_dir]    $ADV_DIR"
echo "[out_dir]    $OUT_DIR"

if [[ ! -d "$SAMPLE_ROOT" ]]; then
  echo "ERROR: sample_root missing: $SAMPLE_ROOT" >&2
  exit 1
fi
if [[ ! -d "$ADV_DIR" ]]; then
  echo "ERROR: adv_dir missing: $ADV_DIR" >&2
  exit 1
fi

exec python3 "$REPO_ROOT/eval/multi_vlm_eval.py" \
  --api_keys_file "$API_KEYS_FILE" \
  --sample_root "$SAMPLE_ROOT" \
  --adv_dir "$ADV_DIR" \
  --target_dir "$TARGET_DIR" \
  --keywords_json "$KEYWORDS_JSON" \
  --models "${MODELS[@]}" \
  --out_dir "$OUT_DIR" \
  "$@"
