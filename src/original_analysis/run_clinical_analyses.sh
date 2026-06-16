#!/bin/bash
#===============================================================================
# MASTER SCRIPT: Run All Clinical Analyses
# Infrastructure: 144-core ARM Neoverse-V2, 243GB RAM
# Purpose: Early Intervention + Personalized Medicine for Parkinson's Disease
#===============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}===============================================================================${NC}"
echo -e "${BLUE}  CLINICAL ANALYSIS SUITE - EARLY INTERVENTION + PERSONALIZED MEDICINE${NC}"
echo -e "${BLUE}  Infrastructure: 144-core ARM Neoverse-V2, 243GB RAM${NC}"
echo -e "${BLUE}===============================================================================${NC}"

# Change to project directory
cd /home1/11021/harshtirhekar/WORK/GaitAnalysis/Parkinson-Disease-using-LRRK2

# Load modules
echo -e "\n${YELLOW}Loading modules...${NC}"
module load gcc/14.2.0 python3/3.11.8 2>/dev/null || echo "Modules may already be loaded"

# Check if venv exists, create if not
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv venv
fi

# Activate virtual environment
echo -e "${YELLOW}Activating virtual environment...${NC}"
source venv/bin/activate

# Install/upgrade dependencies
echo -e "${YELLOW}Checking dependencies...${NC}"
pip install --upgrade pip -q
pip install pandas numpy scikit-learn -q 2>/dev/null || echo "Dependencies may already be installed"

# Set parallel processing environment
export OMP_NUM_THREADS=144
export NUMEXPR_MAX_THREADS=144
export MKL_NUM_THREADS=144

# Create output directories
echo -e "${YELLOW}Creating output directories...${NC}"
mkdir -p analysis_results/progression
mkdir -p analysis_results/treatment
mkdir -p analysis_results/prodromal

# Timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="logs/clinical_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

echo -e "\n${GREEN}===============================================================================${NC}"
echo -e "${GREEN}  RUNNING 3 CLINICAL ANALYSES IN PARALLEL${NC}"
echo -e "${GREEN}===============================================================================${NC}"
echo ""

#===============================================================================
# Run all 3 analyses in parallel
#===============================================================================

echo -e "${BLUE}[1/3] Starting Progression Trajectory Analysis...${NC}"
python scripts/clinical/progression_trajectories.py > "$LOG_DIR/progression.log" 2>&1 &
PID1=$!

echo -e "${BLUE}[2/3] Starting Treatment-Phenotype Matrix Analysis...${NC}"
python scripts/clinical/treatment_phenotype_matrix.py > "$LOG_DIR/treatment.log" 2>&1 &
PID2=$!

echo -e "${BLUE}[3/3] Starting Prodromal → Phenotype Prediction...${NC}"
python scripts/clinical/prodromal_phenotype_prediction.py > "$LOG_DIR/prodromal.log" 2>&1 &
PID3=$!

echo -e "\n${YELLOW}All 3 analyses running in parallel...${NC}"
echo "PIDs: Progression=$PID1, Treatment=$PID2, Prodromal=$PID3"
echo ""

#===============================================================================
# Wait for all to complete
#===============================================================================

echo -e "${YELLOW}Waiting for analyses to complete...${NC}"

wait $PID1
STATUS1=$?
if [ $STATUS1 -eq 0 ]; then
    echo -e "${GREEN}✓ Progression Trajectory Analysis completed successfully${NC}"
else
    echo -e "${RED}✗ Progression Trajectory Analysis failed (exit code: $STATUS1)${NC}"
fi

wait $PID2
STATUS2=$?
if [ $STATUS2 -eq 0 ]; then
    echo -e "${GREEN}✓ Treatment-Phenotype Matrix completed successfully${NC}"
else
    echo -e "${RED}✗ Treatment-Phenotype Matrix failed (exit code: $STATUS2)${NC}"
fi

wait $PID3
STATUS3=$?
if [ $STATUS3 -eq 0 ]; then
    echo -e "${GREEN}✓ Prodromal Prediction completed successfully${NC}"
else
    echo -e "${RED}✗ Prodromal Prediction failed (exit code: $STATUS3)${NC}"
fi

#===============================================================================
# Summary
#===============================================================================

echo ""
echo -e "${BLUE}===============================================================================${NC}"
echo -e "${BLUE}  ANALYSIS COMPLETE${NC}"
echo -e "${BLUE}===============================================================================${NC}"
echo ""
echo -e "Logs saved to: ${LOG_DIR}/"
echo ""
echo -e "${GREEN}Results:${NC}"
echo "  - Progression: analysis_results/progression/"
echo "  - Treatment:   analysis_results/treatment/"
echo "  - Prodromal:   analysis_results/prodromal/"
echo ""

# Show latest results
echo -e "${YELLOW}Latest result files:${NC}"
ls -la analysis_results/progression/*.json 2>/dev/null | tail -1 || echo "  (no progression results yet)"
ls -la analysis_results/treatment/*.json 2>/dev/null | tail -1 || echo "  (no treatment results yet)"
ls -la analysis_results/prodromal/*.json 2>/dev/null | tail -1 || echo "  (no prodromal results yet)"

echo ""
echo -e "${BLUE}===============================================================================${NC}"
echo -e "${BLUE}  CLINICAL IMPACT SUMMARY${NC}"
echo -e "${BLUE}===============================================================================${NC}"
echo ""
echo -e "${GREEN}Analysis 1: Progression Trajectories${NC}"
echo "  Purpose: Identify RAPID PROGRESSORS for early intervention"
echo "  Output:  Patients clustered by progression speed"
echo ""
echo -e "${GREEN}Analysis 2: Treatment-Phenotype Matrix${NC}"
echo "  Purpose: Match phenotype to optimal treatment"
echo "  Output:  Treatment response rates by phenotype"
echo ""
echo -e "${GREEN}Analysis 3: Prodromal → Phenotype${NC}"
echo "  Purpose: Predict future phenotype from prodromal markers"
echo "  Output:  Early intervention before symptoms emerge"
echo ""
echo -e "${BLUE}===============================================================================${NC}"

# Return overall exit status
if [ $STATUS1 -eq 0 ] && [ $STATUS2 -eq 0 ] && [ $STATUS3 -eq 0 ]; then
    echo -e "${GREEN}All analyses completed successfully!${NC}"
    exit 0
else
    echo -e "${RED}Some analyses failed - check logs for details${NC}"
    exit 1
fi
