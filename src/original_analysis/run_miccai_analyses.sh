#!/bin/bash
# ==============================================================================
# MICCAI 2026 — Run All Analyses in Parallel
# TACC Vista: 144 cores ARM Neoverse-V2, 243 GB RAM
# ==============================================================================

set -e

BASE_DIR="/home1/11021/harshtirhekar/WORK/GaitAnalysis/Parkinson-Disease-using-LRRK2"
SCRIPTS_DIR="${BASE_DIR}/scripts/miccai"
LOG_DIR="${BASE_DIR}/logs/miccai"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p "${LOG_DIR}"

echo "============================================================"
echo "MICCAI 2026 — Full Analysis Pipeline"
echo "Started: $(date)"
echo "Infrastructure: TACC Vista, 144 cores ARM Neoverse-V2"
echo "============================================================"

# Activate virtual environment if it exists
if [ -f "${BASE_DIR}/venv/bin/activate" ]; then
    source "${BASE_DIR}/venv/bin/activate"
    echo "Virtual environment activated"
fi

# Check Python dependencies
echo ""
echo "Checking dependencies..."
python3 -c "import sklearn, numpy, pandas, scipy; print('Core dependencies: OK')" || {
    echo "ERROR: Missing core dependencies"
    exit 1
}
python3 -c "import hdbscan; print('HDBSCAN: OK')" || {
    echo "WARNING: hdbscan not installed. Installing..."
    pip install hdbscan
}

echo ""
echo "============================================================"
echo "Running all 5 analyses in parallel..."
echo "============================================================"

# Run all 5 scripts in parallel (background processes)
# Each script uses internal ProcessPoolExecutor with 120 workers

echo "[1/5] Bayesian GMM Mega-Sweep (2,912 configs)..."
python3 "${SCRIPTS_DIR}/bayesian_gmm_megasweep.py" \
    > "${LOG_DIR}/run_bgmm_${TIMESTAMP}.stdout" 2>&1 &
PID_BGMM=$!

echo "[2/5] Phenotype Hierarchy Collapse (70→k*)..."
python3 "${SCRIPTS_DIR}/collapse_phenotypes.py" \
    > "${LOG_DIR}/run_collapse_${TIMESTAMP}.stdout" 2>&1 &
PID_COLLAPSE=$!

echo "[3/5] Noise Filtering (RMT + SVD)..."
python3 "${SCRIPTS_DIR}/noise_filtering_rmt.py" \
    > "${LOG_DIR}/run_noise_${TIMESTAMP}.stdout" 2>&1 &
PID_NOISE=$!

echo "[4/5] Causal Inference (SEM + Granger)..."
python3 "${SCRIPTS_DIR}/causal_inference_sem.py" \
    > "${LOG_DIR}/run_causal_${TIMESTAMP}.stdout" 2>&1 &
PID_CAUSAL=$!

echo "[5/5] Multi-DB Integration (ComBat + SNR)..."
python3 "${SCRIPTS_DIR}/multi_db_integration.py" \
    > "${LOG_DIR}/run_multidb_${TIMESTAMP}.stdout" 2>&1 &
PID_MULTIDB=$!

echo ""
echo "All analyses launched. PIDs:"
echo "  BGMM:      $PID_BGMM"
echo "  Collapse:   $PID_COLLAPSE"
echo "  Noise:      $PID_NOISE"
echo "  Causal:     $PID_CAUSAL"
echo "  Multi-DB:   $PID_MULTIDB"
echo ""
echo "Waiting for completion..."

# Wait for all
FAILED=0

wait $PID_MULTIDB && echo "[5/5] Multi-DB Integration: DONE ✓" || { echo "[5/5] Multi-DB Integration: FAILED ✗"; FAILED=$((FAILED+1)); }
wait $PID_NOISE && echo "[3/5] Noise Filtering: DONE ✓" || { echo "[3/5] Noise Filtering: FAILED ✗"; FAILED=$((FAILED+1)); }
wait $PID_CAUSAL && echo "[4/5] Causal Inference: DONE ✓" || { echo "[4/5] Causal Inference: FAILED ✗"; FAILED=$((FAILED+1)); }
wait $PID_COLLAPSE && echo "[2/5] Phenotype Collapse: DONE ✓" || { echo "[2/5] Phenotype Collapse: FAILED ✗"; FAILED=$((FAILED+1)); }
wait $PID_BGMM && echo "[1/5] BGMM Mega-Sweep: DONE ✓" || { echo "[1/5] BGMM Mega-Sweep: FAILED ✗"; FAILED=$((FAILED+1)); }

echo ""
echo "============================================================"
echo "PIPELINE COMPLETE"
echo "Finished: $(date)"
echo "Failed: ${FAILED}/5"
echo ""
echo "Results in: ${BASE_DIR}/analysis_results/miccai/"
echo "Logs in: ${LOG_DIR}/"
echo "============================================================"

# List results
echo ""
echo "Generated files:"
ls -lh "${BASE_DIR}/analysis_results/miccai/" 2>/dev/null || echo "  (no results yet)"
