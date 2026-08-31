#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${HMO_PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
cd "$ROOT"

export HMO_PROJECT_ROOT="$ROOT"
export HMO_DATA_ROOT="${HMO_DATA_ROOT:-$(cd "${ROOT}/.." && pwd)}"
export HMO_MODEL_ROOT="${HMO_MODEL_ROOT:-${HMO_DATA_ROOT}/model}"
export HMO_RESULTS_ROOT="${HMO_RESULTS_ROOT:-${ROOT}/experiments/results}"
export HMO_CONDA_ENV="${HMO_CONDA_ENV:-hmo_research}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:128,garbage_collection_threshold:0.8}"

resolve_conda_sh() {
  if [[ -n "${HMO_CONDA_SH:-}" && -f "${HMO_CONDA_SH}" ]]; then
    printf '%s\n' "${HMO_CONDA_SH}"
    return 0
  fi

  if [[ -n "${CONDA_EXE:-}" ]]; then
    local conda_exe_real conda_base_guess
    conda_exe_real="$(readlink -f "${CONDA_EXE}" 2>/dev/null || printf '%s' "${CONDA_EXE}")"
    conda_base_guess="$(cd "$(dirname "${conda_exe_real}")/.." 2>/dev/null && pwd || true)"
    if [[ -n "${conda_base_guess}" && -f "${conda_base_guess}/etc/profile.d/conda.sh" ]]; then
      printf '%s\n' "${conda_base_guess}/etc/profile.d/conda.sh"
      return 0
    fi
  fi

  if command -v conda >/dev/null 2>&1; then
    local conda_base
    conda_base="$(conda info --base 2>/dev/null || true)"
    if [[ -n "${conda_base}" && -f "${conda_base}/etc/profile.d/conda.sh" ]]; then
      printf '%s\n' "${conda_base}/etc/profile.d/conda.sh"
      return 0
    fi
  fi

  local candidates=(
    "/data/miniconda/etc/profile.d/conda.sh"
    "${HOME}/anaconda3/etc/profile.d/conda.sh"
    "${HOME}/miniconda3/etc/profile.d/conda.sh"
    "/opt/conda/etc/profile.d/conda.sh"
    "/usr/local/anaconda3/etc/profile.d/conda.sh"
  )
  local candidate
  for candidate in "${candidates[@]}"; do
    if [[ -f "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  return 1
}

CONDA_SH="$(resolve_conda_sh || true)"
if [[ -z "${CONDA_SH}" ]]; then
  echo "[fatal] Could not locate conda.sh. Export HMO_CONDA_SH=/path/to/conda.sh and retry."
  exit 1
fi
# Conda activation hooks reference optional variables that are incompatible
# with nounset. Keep strict mode for the experiment script itself.
set +u
source "${CONDA_SH}"
conda activate "${HMO_CONDA_ENV}"
set -u

TARGET="${1:-}"
shift || true

if [[ -z "${TARGET}" ]]; then
  echo "Usage: bash experiments/phase2/run_single_a100.sh <target> [extra args]"
  echo "Targets: preflight e1_timing e1_formal e2 e3 e3_analyze e4 e5_smoke e5 e6 all_qwen"
  exit 1
fi

run_py() {
  local script="$1"
  shift
  python "$script" "$@"
}

show_model_dir() {
  local path="$1"
  echo "[preflight] ${path}"
  if [[ -d "${path}" ]]; then
    ls -lah "${path}" | sed -n '1,20p'
  else
    echo "  missing"
  fi
}

case "${TARGET}" in
  preflight)
    echo "[preflight] project root: ${HMO_PROJECT_ROOT}"
    echo "[preflight] data root:    ${HMO_DATA_ROOT}"
    echo "[preflight] model root:   ${HMO_MODEL_ROOT}"
    echo "[preflight] results root: ${HMO_RESULTS_ROOT}"
    echo "[preflight] conda env:    ${HMO_CONDA_ENV}"
    echo "[preflight] conda.sh:     ${CONDA_SH}"
    echo
    echo "[preflight] python version"
    python --version
    echo
    echo "[preflight] gpu"
    nvidia-smi
    echo
    show_model_dir "${HMO_MODEL_ROOT}/Qwen3.5-27B"
    echo
    show_model_dir "${HMO_MODEL_ROOT}/Qwen3.5-27B-GPTQ-Int4"
    echo
    show_model_dir "${HMO_MODEL_ROOT}/Kimi-Linear-48B-A3B-Instruct-GPTQ-Int4"
    echo
    mkdir -p "${HMO_RESULTS_ROOT}"
    echo "[preflight] py_compile"
    python -m py_compile \
      experiments/phase2/e1_main/run.py \
      experiments/phase2/e2_ablation/run.py \
      experiments/phase2/e3_mechanism/run.py \
      experiments/phase2/e3_mechanism/analyze.py \
      experiments/phase2/e4_sensitivity/run.py \
      experiments/phase2/e5_kimi/run.py \
      experiments/phase2/e6_overhead/run.py \
      experiments/phase2/runner.py \
      experiments/utils/model_loader.py \
      experiments/utils/dataset_utils.py \
      experiments/utils/eval_harness.py \
      experiments/utils/hmo_controller.py
    ;;

  e1_timing)
    run_py experiments/phase2/e1_main/run.py \
      --timing-test \
      --gpu_id 0 \
      "$@"
    ;;

  e1_formal)
    run_py experiments/phase2/e1_main/run.py \
      --resume \
      --gpu_id 0 \
      --run-name a100_formal \
      "$@"
    ;;

  e2)
    run_py experiments/phase2/e2_ablation/run.py \
      --resume \
      --gpu_id 0 \
      "$@"
    ;;

  e3)
    run_py experiments/phase2/e3_mechanism/run.py \
      --resume \
      --gpu_id 0 \
      "$@"
    ;;

  e3_analyze)
    run_py experiments/phase2/e3_mechanism/analyze.py "$@"
    ;;

  e4)
    run_py experiments/phase2/e4_sensitivity/run.py \
      --resume \
      --gpu_id 0 \
      "$@"
    ;;

  e5_smoke)
    run_py experiments/phase2/e5_kimi/run.py \
      --resume \
      --gpu_id 0 \
      --n_samples 2 \
      --methods full_kv \
      "$@"
    ;;

  e5)
    run_py experiments/phase2/e5_kimi/run.py \
      --resume \
      --gpu_id 0 \
      "$@"
    ;;

  e6)
    run_py experiments/phase2/e6_overhead/run.py \
      --resume \
      --gpu_id 0 \
      "$@"
    ;;

  all_qwen)
    bash experiments/phase2/run_single_a100.sh e1_timing
    bash experiments/phase2/run_single_a100.sh e1_formal
    bash experiments/phase2/run_single_a100.sh e2
    bash experiments/phase2/run_single_a100.sh e3
    bash experiments/phase2/run_single_a100.sh e3_analyze
    bash experiments/phase2/run_single_a100.sh e4
    bash experiments/phase2/run_single_a100.sh e6
    ;;

  *)
    echo "Unknown target: ${TARGET}"
    exit 1
    ;;
esac
