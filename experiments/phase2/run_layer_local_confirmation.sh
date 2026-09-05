#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${HMO_PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
PYTHON="${HMO_PYTHON:-/home/pz/miniconda3/envs/hmo_research_v6/bin/python}"
MODEL_PATH="${HMO_9B_MODEL_PATH:-/mnt/nvme0/hmo/models/Qwen3.5-9B}"
ARCHIVE="${HMO_LONGBENCH_ARCHIVE:-/mnt/nvme0/hmo/datasets/LongBench/5e628be450b7e67fb7ae6e201bd6d8f7056f7672/data.zip}"
PROTOCOL="${HMO_LAYER_LOCAL_PROTOCOL:-${ROOT}/refine-logs/layer_local_confirmation_protocol.json}"
RESULTS_ROOT="${HMO_LAYER_LOCAL_RESULTS_ROOT:-/mnt/nvme0/hmo/runs/layer_local_confirmation_9b_v1}"
PROBE_ROOT="${HMO_LAYER_LOCAL_PROBE_ROOT:-${RESULTS_ROOT}/probe_cache}"
TARGET="${1:-validate}"

cd "${ROOT}"

validate() {
  "${PYTHON}" -c 'from pathlib import Path; from experiments.phase2.e3_v2.run_layer_local_confirmation import load_confirmation_protocol; payload, digest = load_confirmation_protocol(Path("'"${PROTOCOL}"'")); print({"status": "valid", "protocol_sha256": digest, "case_count": payload["case_count"]})'
}

require_runtime() {
  git diff --quiet
  git diff --cached --quiet
  test -z "$(git status --short --untracked-files=normal)"
  if [[ -z "${CUDA_VISIBLE_DEVICES:-}" || "${CUDA_VISIBLE_DEVICES}" == *,* ]]; then
    echo "[fatal] Export exactly one CUDA_VISIBLE_DEVICES value." >&2
    exit 2
  fi
  "${PYTHON}" -c 'import torch; assert torch.cuda.is_available() and torch.cuda.device_count() == 1; print(torch.cuda.get_device_name(0))'
  test -f "${MODEL_PATH}/config.json"
  test -f "${ARCHIVE}"
}

run_package() {
  local run_dir="$1"
  shift
  mkdir -p "${run_dir}" "${PROBE_ROOT}"
  "${PYTHON}" -m experiments.phase2.e3_v2.run_layer_local_confirmation \
    --model-path "${MODEL_PATH}" \
    --model-id Qwen/Qwen3.5-9B \
    --model-revision c202236235762e1c871ad0ccb60c8ee5ba337b9a \
    --archive "${ARCHIVE}" \
    --protocol "${PROTOCOL}" \
    --run-dir "${run_dir}" \
    --probe-cache-dir "${PROBE_ROOT}" \
    --resume \
    "$@"
}

case "${TARGET}" in
  validate)
    validate
    ;;
  smoke)
    validate
    require_runtime
    run_package "${RESULTS_ROOT}/smoke" --limit 1
    ;;
  formal)
    validate
    require_runtime
    run_package "${RESULTS_ROOT}/formal"
    ;;
  status)
    find "${RESULTS_ROOT}" -maxdepth 3 -type f \
      \( -name '*.jsonl' -o -name '*summary.json' -o -name 'run_manifest.json' \) \
      -print 2>/dev/null || true
    ;;
  *)
    echo "Usage: $0 {validate|smoke|formal|status}" >&2
    exit 2
    ;;
esac
