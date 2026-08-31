#!/usr/bin/env bash
# Portable HMO environment template. Override any value before sourcing.

HMO_PROJECT_ROOT_DEFAULT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export HMO_PROJECT_ROOT="${HMO_PROJECT_ROOT:-$HMO_PROJECT_ROOT_DEFAULT}"
export HMO_DATA_ROOT="${HMO_DATA_ROOT:-$HMO_PROJECT_ROOT}"
export HMO_MODEL_ROOT="${HMO_MODEL_ROOT:-$HMO_PROJECT_ROOT/models}"
export HMO_RESULTS_ROOT="${HMO_RESULTS_ROOT:-$HMO_PROJECT_ROOT/experiments/results}"
export HMO_CONDA_SH="${HMO_CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
export HMO_CONDA_ENV="${HMO_CONDA_ENV:-hmo_research}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

if [[ -f "$HMO_CONDA_SH" ]]; then
  source "$HMO_CONDA_SH"
  conda activate "$HMO_CONDA_ENV"
fi

cd "$HMO_PROJECT_ROOT"
