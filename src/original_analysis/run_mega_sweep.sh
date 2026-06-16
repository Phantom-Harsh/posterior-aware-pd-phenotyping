#!/bin/bash
#===============================================================================
# MEGA HYPERPARAMETER SWEEP - 370+ CONFIGURATIONS
# Runs ON 144 ARM Neoverse-V2 cores
# 
# Total: HDBSCAN (100) + K-Means (50) + GMM (60) + Hierarchical (60) + UMAP+HDBSCAN (100)
#      = 370 configurations
#===============================================================================

cd /home1/11021/harshtirhekar/WORK/GaitAnalysis/Parkinson-Disease-using-LRRK2

# Setup
module load gcc/14.2.0 python3/3.11.8
source venv/bin/activate

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "================================================================================"
echo "MEGA HYPERPARAMETER SWEEP - 370+ CONFIGURATIONS"
echo "Using 144 ARM Neoverse-V2 cores"
echo "Started: $(date)"
echo "================================================================================"

mkdir -p logs/mega_sweep analysis_results/mega_sweep

echo ""
echo "Launching 5 MEGA sweeps in PARALLEL..."
echo ""
echo "  [1] HDBSCAN: 100 configurations"
echo "  [2] K-Means: 50 configurations"
echo "  [3] GMM: 60 configurations"
echo "  [4] Hierarchical: 60 configurations"
echo "  [5] UMAP+HDBSCAN: 100 configurations"
echo "  ─────────────────────────────────"
echo "  TOTAL: 370 configurations"
echo ""

# Launch all in parallel
python scripts/mega_sweep/mega_hdbscan.py 2>&1 | tee logs/mega_sweep/hdbscan_mega_${TIMESTAMP}.log &
PID1=$!
echo "Started HDBSCAN: PID $PID1"

python scripts/mega_sweep/mega_kmeans.py 2>&1 | tee logs/mega_sweep/kmeans_mega_${TIMESTAMP}.log &
PID2=$!
echo "Started K-Means: PID $PID2"

python scripts/mega_sweep/mega_gmm.py 2>&1 | tee logs/mega_sweep/gmm_mega_${TIMESTAMP}.log &
PID3=$!
echo "Started GMM: PID $PID3"

python scripts/mega_sweep/mega_hierarchical.py 2>&1 | tee logs/mega_sweep/hierarchical_mega_${TIMESTAMP}.log &
PID4=$!
echo "Started Hierarchical: PID $PID4"

python scripts/mega_sweep/mega_umap_hdbscan.py 2>&1 | tee logs/mega_sweep/umap_hdbscan_mega_${TIMESTAMP}.log &
PID5=$!
echo "Started UMAP+HDBSCAN: PID $PID5"

echo ""
echo "All 5 mega sweeps running in parallel..."
echo "Waiting for completion..."
wait $PID1 $PID2 $PID3 $PID4 $PID5

echo ""
echo "================================================================================"
echo "ALL MEGA SWEEPS COMPLETE"
echo "Finished: $(date)"
echo "================================================================================"
echo ""
echo "Results:"
ls -la analysis_results/mega_sweep/*.json 2>/dev/null
echo ""
echo "Logs:"
ls -la logs/mega_sweep/*.log 2>/dev/null | tail -10
