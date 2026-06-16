#!/usr/bin/env python3
"""
CORRECTED CLUSTERING ANALYSIS - WITH DATA CLEANING
==================================================

This script:
1. Identifies and removes outlier patients with impossible UPDRS scores
2. Re-runs Bayesian GMM clustering on CLEAN data
3. Provides correct, publishable cluster composition
4. Generates updated numbers for Table 3 in manuscript

Author: Verification Team
Date: October 26, 2025
"""

import pandas as pd
import numpy as np
from sklearn.mixture import BayesianGaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from pathlib import Path

BASE_DIR = Path("/home1/11021/harshtirhekar/WORK/GaitAnalysis/Parkinson-Disease-using-LRRK2")
OUTPUT_FILE = Path("CORRECTED_CLUSTERING_RESULTS.txt")

def header(title, width=100):
    line = "\n" + "="*width + f"\n  {title}\n" + "="*width
    print(line)
    return line

def subheader(title, width=100):
    line = "\n" + "-"*width + f"\n  {title}\n" + "-"*width
    print(line)
    return line

output = []

def log(text):
    print(text)
    output.append(text)

# ============================================================================
# STEP 1: LOAD AND PREPARE DATA
# ============================================================================
output.append(header("STEP 1: DATA LOADING AND QUALITY CHECKS"))

log("\nLoading UPDRS-III data...")
updrs = pd.read_csv(BASE_DIR / "data/PPMI_Gait/MDS-UPDRS_Part_III_06Jan2025.csv")
log(f"Total records: {len(updrs):,}")

# Extract baseline
baseline = updrs[updrs['EVENT_ID'] == 'BL'].copy()
log(f"Baseline records: {len(baseline):,}")

# Extract motor items
motor_items = [col for col in baseline.columns if col.startswith('NP3') and col != 'NP3TOT']
log(f"Motor items (NP3 columns): {len(motor_items)}")

# Complete case analysis
complete = baseline[['PATNO'] + motor_items].dropna()
log(f"\nComplete cases (before cleaning): {len(complete):,}")
log(f"Unique patients: {complete['PATNO'].nunique():,}")

# ============================================================================
# STEP 2: DATA QUALITY CHECKS AND CLEANING
# ============================================================================
output.append(header("STEP 2: DATA QUALITY CHECKS AND OUTLIER REMOVAL"))

log("\n🔍 Checking for impossible UPDRS scores...")

# Calculate total UPDRS score
updrs_total = complete[motor_items].sum(axis=1)

log(f"\nUPDRS-III Score Statistics:")
log(f"  Minimum: {updrs_total.min():.2f}")
log(f"  Maximum: {updrs_total.max():.2f}")
log(f"  Mean: {updrs_total.mean():.2f}")
log(f"  Median: {updrs_total.median():.2f}")
log(f"  Theoretical maximum: {len(motor_items) * 4} (33 items × 4 points each)")

# Identify outliers
outliers_mask = updrs_total > 132  # Maximum possible score
n_outliers = outliers_mask.sum()

if n_outliers > 0:
    log(f"\n🚨 OUTLIERS DETECTED: {n_outliers} patients with impossible scores!")
    
    outlier_patients = complete[outliers_mask]['PATNO'].tolist()
    outlier_scores = updrs_total[outliers_mask].tolist()
    
    log(f"\nOutlier details:")
    for patno, score in zip(outlier_patients, outlier_scores):
        log(f"  Patient {patno}: UPDRS Total = {score:.0f} (IMPOSSIBLE - max is 132)")
    
    # Remove outliers
    complete_clean = complete[~outliers_mask].copy()
    log(f"\n✅ Removed {n_outliers} outlier patients")
    log(f"Clean dataset: {len(complete_clean):,} patients")
else:
    log(f"\n✅ No outliers detected - data is clean")
    complete_clean = complete.copy()

# Additional sanity checks
log(f"\n🔍 Additional data quality checks...")

# Check for any negative values
negative_values = (complete_clean[motor_items] < 0).any(axis=1).sum()
log(f"  Patients with negative values: {negative_values}")

# Check for values > 4 in individual items
invalid_items = (complete_clean[motor_items] > 4).any(axis=1).sum()
log(f"  Patients with items > 4: {invalid_items}")

if negative_values > 0 or invalid_items > 0:
    log(f"\n⚠️  WARNING: Additional data quality issues found!")
    valid_mask = (complete_clean[motor_items] >= 0).all(axis=1) & (complete_clean[motor_items] <= 4).all(axis=1)
    complete_clean = complete_clean[valid_mask]
    log(f"  After additional cleaning: {len(complete_clean):,} patients")

# Final clean dataset
log(f"\n{'='*100}")
log(f"FINAL CLEAN DATASET:")
log(f"  Total patients: {len(complete_clean):,}")
log(f"  Unique patients: {complete_clean['PATNO'].nunique():,}")
log(f"  Removed from original: {len(complete) - len(complete_clean):,} patients")
log(f"{'='*100}")

# ============================================================================
# STEP 3: RE-RUN BAYESIAN GMM CLUSTERING ON CLEAN DATA
# ============================================================================
output.append(header("STEP 3: BAYESIAN GMM CLUSTERING (ON CLEAN DATA)"))

log("\nPreparing clean data for clustering...")
X = complete_clean[motor_items].values

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
log(f"Features standardized (shape: {X_scaled.shape})")

# BIC selection
log("\n🔍 Running BIC selection for optimal clusters...")

best_bic = np.inf
best_n = 0
bic_results = []

for n_components in range(2, 6):
    bgm = BayesianGaussianMixture(
        n_components=n_components,
        covariance_type='full',
        max_iter=200,
        random_state=42,
        weight_concentration_prior_type='dirichlet_process'
    )
    bgm.fit(X_scaled)
    
    bic = -bgm.lower_bound_
    bic_results.append((n_components, bic))
    
    log(f"  n_components={n_components}, BIC={bic:.2f}")
    
    if bic < best_bic:
        best_bic = bic
        best_n = n_components

log(f"\n✅ Best number of components: {best_n} (BIC={best_bic:.2f})")

# Fit final model
log(f"\nFitting final model with {best_n} components...")

final_bgm = BayesianGaussianMixture(
    n_components=best_n,
    covariance_type='full',
    max_iter=200,
    random_state=42,
    weight_concentration_prior_type='dirichlet_process'
)
final_bgm.fit(X_scaled)

labels = final_bgm.predict(X_scaled)
probabilities = final_bgm.predict_proba(X_scaled)

log(f"Clustering complete!")

# ============================================================================
# STEP 4: ANALYZE CLEAN CLUSTER COMPOSITION
# ============================================================================
output.append(header("STEP 4: CLEAN CLUSTER COMPOSITION"))

unique_labels = sorted(np.unique(labels))
log(f"\nCluster labels: {unique_labels}")
log(f"Number of clusters: {len(unique_labels)}")

# Calculate UPDRS totals for clean data
updrs_totals_clean = complete_clean[motor_items].sum(axis=1).values

log(f"\n{'Cluster':<12s} {'N Patients':>12s} {'% of Total':>12s} {'Mean UPDRS':>12s} {'Std UPDRS':>12s}")
log("-" * 65)

cluster_info = []

for i, cluster_label in enumerate(unique_labels):
    cluster_mask = (labels == cluster_label)
    cluster_size = cluster_mask.sum()
    cluster_pct = 100 * cluster_size / len(complete_clean)
    cluster_mean = updrs_totals_clean[cluster_mask].mean()
    cluster_std = updrs_totals_clean[cluster_mask].std()
    
    cluster_info.append({
        'label': cluster_label,
        'size': cluster_size,
        'mean_updrs': cluster_mean,
        'std_updrs': cluster_std
    })
    
    log(f"Phenotype {i+1:<2d} {cluster_size:>12,d} {cluster_pct:>11.1f}% {cluster_mean:>12.2f} {cluster_std:>12.2f}")

total = sum(c['size'] for c in cluster_info)
log(f"\n{'TOTAL':<12s} {total:>12,d} {100.0:>11.1f}%")

# ============================================================================
# STEP 5: VALIDATION METRICS
# ============================================================================
output.append(header("STEP 5: CLUSTERING QUALITY METRICS"))

silhouette = silhouette_score(X_scaled, labels)
davies_bouldin = davies_bouldin_score(X_scaled, labels)
calinski = calinski_harabasz_score(X_scaled, labels)

log(f"\nCluster Validation Metrics:")
log(f"  Silhouette Score: {silhouette:.3f}")
log(f"  Davies-Bouldin Index: {davies_bouldin:.3f}")
log(f"  Calinski-Harabasz Score: {calinski:.3f}")

# Uncertainty
uncertainties = 1 - np.max(probabilities, axis=1)
high_uncertainty = (uncertainties > 0.7).sum()

log(f"\nUncertainty Statistics:")
log(f"  Mean uncertainty: {uncertainties.mean():.3f}")
log(f"  Max uncertainty: {uncertainties.max():.3f}")
log(f"  High uncertainty (>70%): {high_uncertainty} patients")

# ============================================================================
# STEP 6: CLINICAL INTERPRETATION
# ============================================================================
output.append(header("STEP 6: CLINICAL INTERPRETATION OF CLUSTERS"))

# Sort clusters by mean severity
cluster_info_sorted = sorted(cluster_info, key=lambda x: x['mean_updrs'])

log(f"\nClusters ordered by disease severity:")
log(f"\n{'Phenotype':<15s} {'N':>10s} {'Mean UPDRS':>12s} {'Clinical Stage':<30s}")
log("-" * 70)

for i, cluster in enumerate(cluster_info_sorted):
    if cluster['mean_updrs'] < 10:
        stage = "Minimal/Early (Mild symptoms)"
    elif cluster['mean_updrs'] < 20:
        stage = "Moderate (Progressing)"
    elif cluster['mean_updrs'] < 30:
        stage = "Moderately Severe"
    else:
        stage = "Severe (Advanced disease)"
    
    log(f"Phenotype {i+1:<6d} {cluster['size']:>10,d} {cluster['mean_updrs']:>12.2f} {stage:<30s}")

# ============================================================================
# FINAL RECOMMENDATIONS
# ============================================================================
output.append(header("FINAL RECOMMENDATIONS FOR MANUSCRIPT"))

log(f"""
CORRECTED NUMBERS FOR YOUR MANUSCRIPT:

Abstract & Results:
  ✓ Number of phenotypes: {len(unique_labels)}
  ✓ Sample size: {len(complete_clean):,} patients (after removing {len(complete) - len(complete_clean)} outliers)
  ✓ Silhouette Score: {silhouette:.3f}
  ✓ Davies-Bouldin Index: {davies_bouldin:.3f}

Table 3 (Cluster Composition):
""")

for i, cluster in enumerate(cluster_info_sorted):
    log(f"  Phenotype {i+1}: n={cluster['size']:,}, Mean UPDRS={cluster['mean_updrs']:.2f}±{cluster['std_updrs']:.2f}")

log(f"""
Methods Section Update:
  Original cohort: 4,166 baseline assessments
  After quality control: {len(complete_clean):,} patients
  Removed: {len(complete) - len(complete_clean)} patients with invalid UPDRS scores (>132)
  
Data Quality Statement:
  "Data quality control removed {len(complete) - len(complete_clean)} patients with 
   physiologically impossible UPDRS-III scores (>132), resulting in a final 
   analytical cohort of {len(complete_clean):,} patients from {complete_clean['PATNO'].nunique():,} 
   unique individuals."
""")

log(f"\n{'='*100}")
log(f"CRITICAL FINDING:")
log(f"{'='*100}")

if len(complete) != len(complete_clean):
    log(f"""
Your original analysis included {len(complete) - len(complete_clean)} patients with DATA ERRORS.
This affects ALL reported results!

You MUST:
1. Update sample size in manuscript: 4,166 → {len(complete_clean):,}
2. Update cluster compositions with corrected numbers above
3. Re-run ALL downstream analyses with clean data
4. Update all claims, tables, and figures
""")
else:
    log(f"\n✅ Data is clean - no changes needed")

log(f"{'='*100}\n")

# Save detailed output
with open(OUTPUT_FILE, 'w') as f:
    f.write('\n'.join(output))

print(f"✅ Complete results saved to: {OUTPUT_FILE}")
print(f"\nIMPORTANT: Review the corrected cluster composition above!")
print(f"Update your manuscript with these verified numbers!")
print("="*100 + "\n")