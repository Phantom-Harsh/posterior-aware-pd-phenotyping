#!/bin/bash
#===============================================================================
# MEGA CO-CLUSTERING SWEEP - 550+ CONFIGURATIONS
# Combines: Biclustering (200) + Coclustering (200) + MultiView (150)
#===============================================================================

cd /home1/11021/harshtirhekar/WORK/GaitAnalysis/Parkinson-Disease-using-LRRK2

# Setup
module load gcc/14.2.0 python3/3.11.8
source venv/bin/activate

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "================================================================================"
echo "MEGA CO-CLUSTERING SWEEP - 550+ CONFIGURATIONS"
echo "Using 144 ARM Neoverse-V2 cores"
echo "Started: $(date)"
echo "================================================================================"

mkdir -p logs/coclustering analysis_results/coclustering

echo ""
echo "Launching 3 CO-CLUSTERING sweeps in PARALLEL..."
echo ""
echo "  [1] SPECTRAL BICLUSTERING: 200 configurations (patient × feature)"
echo "  [2] SPECTRAL CO-CLUSTERING: 200 configurations"
echo "  [3] MULTI-VIEW CLUSTERING: 150 configurations (motor + cognitive)"
echo "  ─────────────────────────────────────────────────────────────────"
echo "  TOTAL: 550 configurations"
echo ""

# Launch all in parallel
python scripts/coclustering/mega_biclustering.py 2>&1 | tee logs/coclustering/biclustering_${TIMESTAMP}.log &
PID1=$!
echo "Started Biclustering: PID $PID1"

python scripts/coclustering/mega_coclustering.py 2>&1 | tee logs/coclustering/coclustering_${TIMESTAMP}.log &
PID2=$!
echo "Started Co-clustering: PID $PID2"

python scripts/coclustering/mega_multiview.py 2>&1 | tee logs/coclustering/multiview_${TIMESTAMP}.log &
PID3=$!
echo "Started Multi-View: PID $PID3"

echo ""
echo "All 3 sweeps running in parallel..."
echo "Waiting for completion..."
wait $PID1 $PID2 $PID3

echo ""
echo "================================================================================"
echo "ALL CO-CLUSTERING SWEEPS COMPLETE"
echo "Finished: $(date)"
echo "================================================================================"
echo ""
echo "Results:"
ls -la analysis_results/coclustering/*.json 2>/dev/null
echo ""
echo "Logs:"
ls -la logs/coclustering/*.log 2>/dev/null | tail -10
