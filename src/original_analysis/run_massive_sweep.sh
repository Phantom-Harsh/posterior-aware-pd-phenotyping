#!/bin/bash
#===============================================================================
# MASSIVE HYPERPARAMETER SWEEP - 50+ Configurations
# Runs ALL algorithm sweeps in TRUE PARALLEL on 144 cores
# Each algorithm has its own script, own log, own results
#===============================================================================

cd /home1/11021/harshtirhekar/WORK/GaitAnalysis/Parkinson-Disease-using-LRRK2

# Setup
module load gcc/14.2.0 python3/3.11.8
source venv/bin/activate

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "================================================================================"
echo "MASSIVE HYPERPARAMETER SWEEP - 50+ CONFIGURATIONS"
echo "Started: $(date)"
echo "Using 144 ARM Neoverse-V2 cores"
echo "================================================================================"

mkdir -p logs/sweep analysis_results/sweep

echo ""
echo "Launching 5 algorithm sweeps in PARALLEL..."
echo ""

# Launch each algorithm sweep as independent background process
# Each uses its own cores via internal parallelization

python scripts/sweep/sweep_hdbscan.py 2>&1 | tee logs/sweep/hdbscan_sweep_${TIMESTAMP}.log &
PID1=$!
echo "[1] HDBSCAN (12 configs): PID $PID1"

python scripts/sweep/sweep_kmeans.py 2>&1 | tee logs/sweep/kmeans_sweep_${TIMESTAMP}.log &
PID2=$!
echo "[2] K-Means (12 configs): PID $PID2"

python scripts/sweep/sweep_gmm.py 2>&1 | tee logs/sweep/gmm_sweep_${TIMESTAMP}.log &
PID3=$!
echo "[3] GMM (12 configs): PID $PID3"

python scripts/sweep/sweep_hierarchical.py 2>&1 | tee logs/sweep/hierarchical_sweep_${TIMESTAMP}.log &
PID4=$!
echo "[4] Hierarchical (15 configs): PID $PID4"

python scripts/sweep/sweep_umap_hdbscan.py 2>&1 | tee logs/sweep/umap_hdbscan_sweep_${TIMESTAMP}.log &
PID5=$!
echo "[5] UMAP+HDBSCAN (12 configs): PID $PID5"

echo ""
echo "Total: 63 configurations across 5 algorithms"
echo "All running in parallel using 144 cores"
echo ""
echo "Waiting for all processes to complete..."
wait $PID1 $PID2 $PID3 $PID4 $PID5

echo ""
echo "================================================================================"
echo "ALL SWEEPS COMPLETE"
echo "Finished: $(date)"
echo "================================================================================"
echo ""
echo "Results saved in: analysis_results/sweep/"
ls -la analysis_results/sweep/*.json 2>/dev/null | tail -10
echo ""
echo "Logs saved in: logs/sweep/"
ls -la logs/sweep/*.log 2>/dev/null | tail -10
