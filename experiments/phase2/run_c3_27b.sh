#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${HMO_PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
PROTOCOL="${HMO_C3_PROTOCOL:-${ROOT}/refine-logs/c3_27b_protocol.json}"
MODEL_ID="Qwen/Qwen3.5-27B"
MODEL_REVISION="fc05daec18b0a78c049392ed2e771dde82bdf654"
MODEL_PATH="${HMO_C3_MODEL_PATH:-${HMO_MODEL_ROOT:-/data/hmo/models}/Qwen3.5-27B}"
RESULTS_ROOT="${HMO_C3_RESULTS_ROOT:-${HMO_RESULTS_ROOT:-/data/hmo/results}/c3_27b}"
PROBE_ROOT="${HMO_C3_PROBE_ROOT:-${RESULTS_ROOT}/probe_cache}"
ARCHIVE="${HMO_LONGBENCH_ARCHIVE:-/data/hmo/datasets/LongBench/data.zip}"
PYTHON="${HMO_PYTHON:-python}"
TARGET="${1:-validate}"

cd "${ROOT}"

validate_protocol() {
  "${PYTHON}" -m experiments.phase2.e3_v2.c3_protocol \
    --project-root "${ROOT}" \
    --protocol "${PROTOCOL}"
}

require_clean_commit() {
  git diff --quiet
  git diff --cached --quiet
  test -z "$(git status --short --untracked-files=normal)"
  echo "[c3] commit: $(git rev-parse HEAD)"
}

require_single_gpu() {
  if [[ -z "${CUDA_VISIBLE_DEVICES:-}" || "${CUDA_VISIBLE_DEVICES}" == *,* ]]; then
    echo "[fatal] Export CUDA_VISIBLE_DEVICES to exactly one physical GPU." >&2
    exit 2
  fi
  "${PYTHON}" -c 'import torch; assert torch.cuda.is_available() and torch.cuda.device_count() == 1; print(torch.cuda.get_device_name(0))'
}

require_model() {
  if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
    echo "[fatal] Missing pinned 27B model at ${MODEL_PATH}" >&2
    exit 2
  fi
}

run_synthetic() {
  local stage_set="$1"
  local run_dir="${RESULTS_ROOT}/synthetic_${stage_set}"
  mkdir -p "${run_dir}" "${PROBE_ROOT}"
  "${PYTHON}" experiments/phase2/e3_v2/run_pareto.py \
    --model-path "${MODEL_PATH}" \
    --model-id "${MODEL_ID}" \
    --model-revision "${MODEL_REVISION}" \
    --protocol "${PROTOCOL}" \
    --stage-set "${stage_set}" \
    --run-dir "${run_dir}" \
    --probe-cache-dir "${PROBE_ROOT}" \
    --resume
}

run_native() {
  local run_dir="${RESULTS_ROOT}/native_core"
  if [[ ! -f "${ARCHIVE}" ]]; then
    echo "[fatal] Missing pinned LongBench archive at ${ARCHIVE}" >&2
    exit 2
  fi
  mkdir -p "${run_dir}" "${PROBE_ROOT}"
  "${PYTHON}" experiments/phase2/e3_v2/run_native_tasks.py \
    --model-path "${MODEL_PATH}" \
    --model-id "${MODEL_ID}" \
    --model-revision "${MODEL_REVISION}" \
    --archive "${ARCHIVE}" \
    --protocol "${PROTOCOL}" \
    --stage-set formal \
    --run-dir "${run_dir}" \
    --probe-cache-dir "${PROBE_ROOT}" \
    --resume
}

case "${TARGET}" in
  validate)
    validate_protocol
    ;;
  preflight)
    validate_protocol
    require_clean_commit
    require_single_gpu
    require_model
    run_synthetic preflight
    ;;
  core-synthetic)
    validate_protocol
    require_clean_commit
    require_single_gpu
    require_model
    run_synthetic core
    ;;
  core-native)
    validate_protocol
    require_clean_commit
    require_single_gpu
    require_model
    run_native
    ;;
  status)
    find "${RESULTS_ROOT}" -maxdepth 2 -type f \
      \( -name '*.jsonl' -o -name '*summary.json' \) -print 2>/dev/null || true
    ;;
  *)
    echo "Usage: $0 {validate|preflight|core-synthetic|core-native|status}" >&2
    exit 2
    ;;
esac
