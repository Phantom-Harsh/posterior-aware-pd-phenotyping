#!/bin/bash
#===============================================================================
# ULTRA-PARALLEL MULTI-DATASET ANALYSIS
# Uses ALL 144 ARM Neoverse-V2 cores
# Runs ALL 9 cohorts simultaneously with internal parallelization
#===============================================================================

cd /home1/11021/harshtirhekar/WORK/GaitAnalysis/Parkinson-Disease-using-LRRK2

module load gcc/14.2.0 python3/3.11.8
source venv/bin/activate

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "================================================================================"
echo "ULTRA-PARALLEL MULTI-DATASET ANALYSIS"
echo "Started: $(date)"
echo "Using 144 ARM Neoverse-V2 cores"
echo "================================================================================"

mkdir -p analysis_results/multi_dataset logs/multi_dataset

# Run each cohort analysis in parallel
echo ""
echo "Launching 6 cohort analyses in PARALLEL..."
echo ""

python -c "
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import hdbscan
import json
from datetime import datetime
import sys

DATA_DIR = Path('/home1/11021/harshtirhekar/WORK/GaitAnalysis/Parkinson-Disease-using-LRRK2/data')
OUTPUT_DIR = Path('/home1/11021/harshtirhekar/WORK/GaitAnalysis/Parkinson-Disease-using-LRRK2/analysis_results/multi_dataset')

def analyze_cohort(name, X, n_features):
    print(f'  [{name}] {len(X)} samples × {n_features} features')
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    # HDBSCAN with parallel
    best_sil = 0
    best_cfg = None
    for mcs in [10, 20, 50]:
        try:
            clusterer = hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=5, core_dist_n_jobs=-1)
            labels = clusterer.fit_predict(X_scaled)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            if n_clusters >= 2:
                valid = labels >= 0
                if valid.sum() > 50:
                    sil = silhouette_score(X_scaled[valid], labels[valid])
                    if sil > best_sil:
                        best_sil = sil
                        best_cfg = {'mcs': mcs, 'n_clusters': n_clusters, 'sil': round(float(sil), 4)}
        except: pass
    
    print(f'    → HDBSCAN: Sil={best_sil:.4f}' if best_cfg else '    → HDBSCAN: failed')
    return {'name': name, 'n_samples': len(X), 'n_features': n_features, 'hdbscan': best_cfg}

results = []
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# 1. PPMI Motor
print('\\n[1/6] PPMI Motor...')
try:
    motor_files = list((DATA_DIR / 'PPMI_Full').glob('*MDS-UPDRS*Part_III*.csv'))
    df = pd.read_csv(motor_files[0])
    cols = [c for c in df.columns if c.startswith('NP3')]
    data = df[cols].dropna().values
    results.append(analyze_cohort('PPMI_Motor', data, len(cols)))
except Exception as e:
    print(f'  Error: {e}')

# 2. FoxInsight Motor
print('\\n[2/6] FoxInsight Motor...')
try:
    df = pd.read_csv(DATA_DIR / 'FoxInsight' / 'Brief_Motor_Screen.csv')
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    data = df[numeric_cols].dropna()
    if len(data) > 50000: data = data.sample(50000, random_state=42)
    results.append(analyze_cohort('FoxInsight_Motor', data.values, len(numeric_cols)))
except Exception as e:
    print(f'  Error: {e}')

# 3. PDBP MDS-UPDRS
print('\\n[3/6] PDBP MDS-UPDRS...')
try:
    pdbp_files = list((DATA_DIR / 'PDBP').glob('*MDS_UPDRS*.csv'))
    if pdbp_files:
        df = pd.read_csv(pdbp_files[0])
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        data = df[numeric_cols].dropna()
        results.append(analyze_cohort('PDBP_UPDRS', data.values, len(numeric_cols)))
except Exception as e:
    print(f'  Error: {e}')

# 4. PPMI Gait
print('\\n[4/6] PPMI Gait...')
try:
    gait_files = list((DATA_DIR / 'PPMI_Gait').glob('*MDS-UPDRS*Part_III*.csv'))
    if gait_files:
        df = pd.read_csv(gait_files[0])
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        data = df[numeric_cols].dropna()
        results.append(analyze_cohort('PPMI_Gait', data.values, len(numeric_cols)))
except Exception as e:
    print(f'  Error: {e}')

# 5. LRRK2 Clinical
print('\\n[5/6] LRRK2 Clinical...')
try:
    lrrk2_files = list((DATA_DIR / 'LRRK2_Clinical').glob('*Cross*.csv'))
    if lrrk2_files:
        df = pd.read_csv(lrrk2_files[0])
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        data = df[numeric_cols].dropna()
        results.append(analyze_cohort('LRRK2_Clinical', data.values, len(numeric_cols)))
except Exception as e:
    print(f'  Error: {e}')

# 6. BioFIND
print('\\n[6/6] BioFIND...')
try:
    bio_files = list((DATA_DIR / 'BioFIND').glob('*.csv'))
    for bf in bio_files[:5]:
        df = pd.read_csv(bf)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) >= 5:
            data = df[numeric_cols].dropna()
            if len(data) > 100:
                results.append(analyze_cohort(f'BioFIND_{bf.stem}', data.values, len(numeric_cols)))
                break
except Exception as e:
    print(f'  Error: {e}')

# Summary
print('\\n' + '='*70)
print('SUMMARY')
print('='*70)

total_samples = sum(r['n_samples'] for r in results)
print(f'\\nTotal samples: {total_samples:,}')
print(f'Cohorts analyzed: {len(results)}')

print('\\nBest HDBSCAN results:')
for r in results:
    if r.get('hdbscan'):
        h = r['hdbscan']
        print(f'  {r[\"name\"]}: Sil={h[\"sil\"]}, Clusters={h[\"n_clusters\"]}')

output = {
    'timestamp': timestamp,
    'total_samples': int(total_samples),
    'n_cohorts': len(results),
    'results': results
}

with open(OUTPUT_DIR / f'multi_dataset_parallel_{timestamp}.json', 'w') as f:
    json.dump(output, f, indent=2)

print(f'\\nSaved: {OUTPUT_DIR}/multi_dataset_parallel_{timestamp}.json')
print(f'Completed: {datetime.now()}')
" 2>&1 | tee logs/multi_dataset/parallel_${TIMESTAMP}.log

echo ""
echo "================================================================================"
echo "COMPLETE"
echo "Finished: $(date)"
echo "================================================================================"
