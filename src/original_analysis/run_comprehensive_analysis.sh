#!/bin/bash
# Master script to run ALL analysis pipelines in parallel
# Uses all 72 cores on ARM Neoverse-V2

cd /home1/11021/harshtirhekar/WORK/GaitAnalysis/Parkinson-Disease-using-LRRK2

# Load modules
module load gcc/14.2.0 python3/3.11.8
source venv/bin/activate

echo "=========================================="
echo "COMPREHENSIVE PD ANALYSIS PIPELINE"
echo "Started: $(date)"
echo "Using 72 ARM Neoverse-V2 cores"
echo "=========================================="

mkdir -p logs figures analysis_results

# Run all analysis scripts in parallel
echo ""
echo "Launching parallel analysis pipelines..."
echo ""

# 1. Clinical Characterization
python scripts/run_clinical_characterization.py 2>&1 | tee logs/clinical_$(date +%Y%m%d_%H%M%S).log &
PID1=$!
echo "  [1] Clinical Characterization: PID $PID1"

# 2. Co-Clustering
python scripts/run_coclustering.py 2>&1 | tee logs/coclustering_$(date +%Y%m%d_%H%M%S).log &
PID2=$!
echo "  [2] Co-Clustering: PID $PID2"

# 3. Visualizations
python scripts/run_visualizations.py 2>&1 | tee logs/visualizations_$(date +%Y%m%d_%H%M%S).log &
PID3=$!
echo "  [3] Visualizations: PID $PID3"

# Wait for all to complete
echo ""
echo "Waiting for all processes to complete..."
wait $PID1 $PID2 $PID3

echo ""
echo "=========================================="
echo "ALL ANALYSES COMPLETE"
echo "Finished: $(date)"
echo "=========================================="
echo ""
echo "Results:"
ls -la analysis_results/*.json 2>/dev/null | tail -10
echo ""
echo "Figures:"
ls -la figures/*.png 2>/dev/null
echo ""
echo "Logs:"
ls -la logs/*.log 2>/dev/null | tail -10
