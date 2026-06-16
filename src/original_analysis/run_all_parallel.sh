#!/bin/bash
# Run all clustering algorithms in parallel - uses all 72 cores
# Each script runs independently

cd /home1/11021/harshtirhekar/WORK/GaitAnalysis/Parkinson-Disease-using-LRRK2

# Load modules
module load gcc/14.2.0 python3/3.11.8
source venv/bin/activate

echo "=========================================="
echo "PARALLEL CLUSTERING ANALYSIS LAUNCHER"
echo "Started: $(date)"
echo "=========================================="

# Create logs directory
mkdir -p logs

# Run all scripts in parallel (& runs in background)
echo "Launching all algorithms in parallel..."

python scripts/run_kmeans_gmm.py 2>&1 | tee logs/kmeans_gmm_$(date +%Y%m%d_%H%M%S).log &
PID1=$!

python scripts/run_hdbscan.py 2>&1 | tee logs/hdbscan_$(date +%Y%m%d_%H%M%S).log &
PID2=$!

python scripts/run_hierarchical.py 2>&1 | tee logs/hierarchical_$(date +%Y%m%d_%H%M%S).log &
PID3=$!

python scripts/run_umap_hdbscan.py 2>&1 | tee logs/umap_hdbscan_$(date +%Y%m%d_%H%M%S).log &
PID4=$!

echo "Running processes:"
echo "  K-Means/GMM: PID $PID1"
echo "  HDBSCAN: PID $PID2"
echo "  Hierarchical: PID $PID3"
echo "  UMAP+HDBSCAN: PID $PID4"

# Wait for all to complete
echo ""
echo "Waiting for all processes to complete..."
wait $PID1 $PID2 $PID3 $PID4

echo ""
echo "=========================================="
echo "ALL ANALYSES COMPLETE"
echo "Finished: $(date)"
echo "=========================================="
echo ""
echo "Results in: analysis_results/"
ls -la analysis_results/*.json 2>/dev/null | tail -10
echo ""
echo "Logs in: logs/"
ls -la logs/*.log 2>/dev/null | tail -10
