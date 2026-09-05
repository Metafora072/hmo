#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${HMO_PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
PYTHON="${HMO_PYTHON:-/home/pz/miniconda3/envs/hmo_research_v6/bin/python}"
PROTOCOL="${HMO_MECH9B_PROTOCOL:-${ROOT}/refine-logs/chunkkv_mechanism_transfer_9b_protocol.json}"
MODEL_PATH="${HMO_9B_MODEL_PATH:-/mnt/nvme0/hmo/models/Qwen3.5-9B}"
RESULTS_ROOT="${HMO_MECH9B_RESULTS_ROOT:-/mnt/nvme0/hmo/runs/chunkkv_mechanism_9b}"
PROBE_ROOT="${HMO_MECH9B_PROBE_ROOT:-${RESULTS_ROOT}/probe_cache}"
TARGET="${1:-validate}"

cd "${ROOT}"

validate() {
  "${PYTHON}" -c 'from pathlib import Path; from experiments.phase2.e3_v2.run_pareto import load_pareto_protocol, resolve_pareto_stage_set; payload, digest = load_pareto_protocol(Path("'"${PROTOCOL}"'")); stages, systems, equal_bytes, budgets = resolve_pareto_stage_set(payload, "formal"); print({"status": "valid", "protocol_sha256": digest, "stages": stages, "systems": systems, "equal_byte_systems": equal_bytes, "budgets": budgets, "execution": payload["execution"]})'
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
}

run_formal() {
  mkdir -p "${RESULTS_ROOT}" "${PROBE_ROOT}"
  "${PYTHON}" -m experiments.phase2.e3_v2.run_pareto \
    --model-path "${MODEL_PATH}" \
    --model-id Qwen/Qwen3.5-9B \
    --model-revision c202236235762e1c871ad0ccb60c8ee5ba337b9a \
    --protocol "${PROTOCOL}" \
    --stage-set formal \
    --run-dir "${RESULTS_ROOT}/formal" \
    --probe-cache-dir "${PROBE_ROOT}" \
    --resume
}

case "${TARGET}" in
  validate)
    validate
    ;;
  formal)
    validate
    require_runtime
    run_formal
    ;;
  status)
    find "${RESULTS_ROOT}" -maxdepth 3 -type f \
      \( -name '*.jsonl' -o -name '*summary.json' -o -name 'run_manifest.json' \) \
      -print 2>/dev/null || true
    ;;
  *)
    echo "Usage: $0 {validate|formal|status}" >&2
    exit 2
    ;;
esac
