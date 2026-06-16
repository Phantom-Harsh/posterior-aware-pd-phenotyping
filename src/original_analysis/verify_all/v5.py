#!/usr/bin/env python3
"""
INVESTIGATE DISCREPANCY: Why does reproduction give different results?
"""

import pandas as pd
import numpy as np
from sklearn.mixture import BayesianGaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from pathlib import Path

BASE_DIR = Path("/home1/11021/harshtirhekar/WORK/GaitAnalysis/Parkinson-Disease-using-LRRK2")

print("="*100)
print("INVESTIGATING WHY ORIGINAL HAD SILHOUETTE=0.535 BUT REPRODUCTION GIVES 0.207")
print("="*100)

# Load data
updrs = pd.read_csv(BASE_DIR / "data/PPMI_Gait/MDS-UPDRS_Part_III_06Jan2025.csv")
baseline = updrs[updrs['EVENT_ID'] == 'BL'].copy()
motor_items = [col for col in baseline.columns if col.startswith('NP3') and col != 'NP3TOT']
complete = baseline[['PATNO'] + motor_items].dropna()

print(f"\nOriginal dataset: {len(complete)} patients")

# ============================================================================
# TEST 1: Run with ALL 4,166 patients (including outliers)
# ============================================================================
print("\n" + "="*100)
print("TEST 1: Clustering WITH outliers (n=4,166)")
print("="*100)

X_with_outliers = complete[motor_items].values
scaler1 = StandardScaler()
X_scaled_outliers = scaler1.fit_transform(X_with_outliers)

# Try 4 components (what original claimed)
bgm_4 = BayesianGaussianMixture(
    n_components=4,
    covariance_type='full',
    max_iter=200,
    random_state=42,
    weight_concentration_prior_type='dirichlet_process'
)
bgm_4.fit(X_scaled_outliers)
labels_4 = bgm_4.predict(X_scaled_outliers)
sil_4 = silhouette_score(X_scaled_outliers, labels_4)

print(f"4 clusters: Silhouette = {sil_4:.3f}")
print(f"Cluster sizes: {np.bincount(labels_4)}")

# Try 5 components
bgm_5 = BayesianGaussianMixture(
    n_components=5,
    covariance_type='full',
    max_iter=200,
    random_state=42,
    weight_concentration_prior_type='dirichlet_process'
)
bgm_5.fit(X_scaled_outliers)
labels_5 = bgm_5.predict(X_scaled_outliers)
sil_5 = silhouette_score(X_scaled_outliers, labels_5)

print(f"5 clusters: Silhouette = {sil_5:.3f}")
print(f"Cluster sizes: {np.bincount(labels_5)}")

# ============================================================================
# TEST 2: Check if NP3TOT column exists and matches
# ============================================================================
print("\n" + "="*100)
print("TEST 2: Checking if NP3TOT (pre-calculated total) was used")
print("="*100)

if 'NP3TOT' in complete.columns:
    nptot_available = complete['NP3TOT'].notna().sum()
    print(f"NP3TOT column exists: {nptot_available} non-null values")
    
    # Check if NP3TOT matches sum
    calculated_total = complete[motor_items].sum(axis=1)
    nptot_values = complete['NP3TOT']
    
    matches = (calculated_total == nptot_values).sum()
    print(f"NP3TOT matches calculated sum: {matches}/{len(complete)} cases")
    
    # If NP3TOT was used instead of summing
    if nptot_values.max() <= 132:
        print(f"✅ NP3TOT has valid range (max={nptot_values.max()})")
    else:
        print(f"❌ NP3TOT also has outliers (max={nptot_values.max()})")

# ============================================================================
# TEST 3: Different random seeds
# ============================================================================
print("\n" + "="*100)
print("TEST 3: Testing different random seeds")
print("="*100)

for seed in [42, 0, 123]:
    bgm_test = BayesianGaussianMixture(
        n_components=4,
        covariance_type='full',
        max_iter=200,
        random_state=seed,
        weight_concentration_prior_type='dirichlet_process'
    )
    bgm_test.fit(X_scaled_outliers)
    labels_test = bgm_test.predict(X_scaled_outliers)
    sil_test = silhouette_score(X_scaled_outliers, labels_test)
    
    print(f"Seed={seed}: Silhouette = {sil_test:.3f}")

# ============================================================================
# CONCLUSION
# ============================================================================
print("\n" + "="*100)
print("POSSIBLE EXPLANATIONS:")
print("="*100)

print("""
1. Original analysis may have used NP3TOT column instead of summing 33 items
2. Original analysis may have had implicit data cleaning we don't know about
3. Original analysis may have used different random seed
4. The 0.535 score in logs might be from a different run/dataset

RECOMMENDATION:
  Check your ORIGINAL comprehensive_dopaminergic.py code - does it:
  - Use NP3TOT column?
  - Have any data cleaning steps?
  - Use a different random seed?
""")

print("="*100)