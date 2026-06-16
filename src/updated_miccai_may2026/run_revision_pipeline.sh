#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-smoke}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(cd "${SCRIPT_DIR}/.." && pwd)"

module load gcc/14.2.0 python3/3.11.8

export MPLCONFIGDIR="${WORKSPACE}/logs/matplotlib_cache"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
mkdir -p "${MPLCONFIGDIR}" "${WORKSPACE}/logs"

if [[ "${MODE}" == "smoke" ]]; then
  python3 "${SCRIPT_DIR}/revision_analysis_pipeline.py" --mode smoke --steps all
elif [[ "${MODE}" == "full" ]]; then
  python3 "${SCRIPT_DIR}/revision_analysis_pipeline.py" --mode full --steps all --workers 120 --rf-workers 96
else
  python3 "${SCRIPT_DIR}/revision_analysis_pipeline.py" "$@"
fi
