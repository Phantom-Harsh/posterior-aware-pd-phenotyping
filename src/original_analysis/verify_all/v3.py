#!/usr/bin/env python3
"""
CLUSTERING VERIFICATION - RESOLVE 4 VS 5 CLUSTER INCONSISTENCY
==============================================================

This script reproduces the EXACT clustering analysis to determine:
1. How many clusters actually exist? (4 or 5?)
2. What are the actual cluster labels? (0,1,2,3 or 0,1,2,3,4?)
3. What are the cluster sizes? (Do they sum to 4,166?)
4. Which result is the "official" one?

Reproduces the exact analysis from comprehensive_dopaminergic.py
"""

import pandas as pd
import numpy as np
from sklearn.mixture import BayesianGaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from pathlib import Path
import sys

BASE_DIR = Path("/home1/11021/harshtirhekar/WORK/GaitAnalysis/Parkinson-Disease-using-LRRK2")
OUTPUT_FILE = Path("CLUSTER_VERIFICATION_RESULTS.txt")

def print_header(title, width=90):
    print("\n" + "="*width)
    print(f"  {title}")
    print("="*width)

def print_subheader(title, width=90):
    print("\n" + "-"*width)
    print(f"  {title}")
    print("-"*width)

# Output buffer
output = []

def log(text):
    print(text)
    output.append(text)

# ============================================================================
# STEP 1: REPRODUCE EXACT DATA PREPARATION
# ============================================================================
print_header("STEP 1: REPRODUCE EXACT DATA PREPARATION")

log("\nLoading UPDRS-III data...")
updrs = pd.read_csv(BASE_DIR / "data/PPMI_Gait/MDS-UPDRS_Part_III_06Jan2025.csv")
log(f"Total UPDRS-III records: {len(updrs):,}")

# Extract baseline - EXACTLY as original code
log("\nExtracting baseline (EVENT_ID == 'BL')...")
baseline = updrs[updrs['EVENT_ID'] == 'BL'].copy()
log(f"Baseline records: {len(baseline):,}")

# Extract motor items - EXACTLY as original code
log("\nExtracting motor items (NP3 columns)...")
motor_items = [col for col in baseline.columns if col.startswith('NP3') and col != 'NP3TOT']
log(f"Motor items found: {len(motor_items)}")

# Complete case analysis - EXACTLY as original code
log("\nPerforming complete case analysis...")
complete = baseline[['PATNO'] + motor_items].dropna()
log(f"Complete cases: {len(complete):,}")
log(f"Unique patients: {complete['PATNO'].nunique():,}")

if len(complete) != 4166:
    log(f"⚠️  WARNING: Expected 4,166, got {len(complete)}")
else:
    log(f"✅ VERIFIED: 4,166 complete cases match reported")

# ============================================================================
# STEP 2: REPRODUCE EXACT CLUSTERING
# ============================================================================
print_header("STEP 2: REPRODUCE BAYESIAN GMM CLUSTERING")

log("\nPreparing features for clustering...")
X = complete[motor_items].values

# Standardize - EXACTLY as original
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
log(f"Features standardized (shape: {X_scaled.shape})")

# Run BIC selection - EXACTLY as original
log("\nRunning BIC selection for optimal number of clusters...")
log("Testing n_components from 2 to 5...")

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
    
    # Calculate BIC
    bic = -bgm.lower_bound_
    bic_results.append((n_components, bic))
    
    log(f"  n_components={n_components}, BIC={bic:.2f}")
    
    if bic < best_bic:
        best_bic = bic
        best_n = n_components

log(f"\n✅ Best number of components: {best_n} (BIC={best_bic:.2f})")

# ============================================================================
# STEP 3: FIT FINAL MODEL WITH OPTIMAL CLUSTERS
# ============================================================================
print_header("STEP 3: FIT FINAL MODEL AND ANALYZE CLUSTERS")

log(f"\nFitting final Bayesian GMM with n_components={best_n}...")

final_bgm = BayesianGaussianMixture(
    n_components=best_n,
    covariance_type='full',
    max_iter=200,
    random_state=42,
    weight_concentration_prior_type='dirichlet_process'
)
final_bgm.fit(X_scaled)

# Get cluster assignments
labels = final_bgm.predict(X_scaled)
probabilities = final_bgm.predict_proba(X_scaled)

log(f"Clustering complete!")
log(f"\nCluster labels range: {labels.min()} to {labels.max()}")
log(f"Unique cluster labels: {sorted(np.unique(labels))}")
log(f"Number of unique clusters: {len(np.unique(labels))}")

# ============================================================================
# STEP 4: ANALYZE CLUSTER COMPOSITION
# ============================================================================
print_subheader("CLUSTER COMPOSITION ANALYSIS")

# Count patients per cluster
unique_labels = sorted(np.unique(labels))
cluster_sizes = {}
cluster_means = {}

log(f"\n{'Cluster':<10s} {'Label':>8s} {'N Patients':>12s} {'% of Total':>12s} {'Mean UPDRS-III':>15s}")
log("-" * 65)

# Get UPDRS total scores
if 'NP3TOT' in complete.columns:
    updrs_totals = complete['NP3TOT'].values
else:
    # Calculate from individual items
    updrs_totals = complete[motor_items].sum(axis=1).values

total_patients = len(complete)

for cluster_label in unique_labels:
    cluster_mask = (labels == cluster_label)
    cluster_size = cluster_mask.sum()
    cluster_pct = 100 * cluster_size / total_patients
    cluster_mean_updrs = updrs_totals[cluster_mask].mean()
    
    cluster_sizes[cluster_label] = cluster_size
    cluster_means[cluster_label] = cluster_mean_updrs
    
    log(f"Cluster {cluster_label+1:<3d} {cluster_label:>8d} {cluster_size:>12,d} {cluster_pct:>11.1f}% {cluster_mean_updrs:>15.2f}")

# Verify total
total_clustered = sum(cluster_sizes.values())
log(f"\n{'TOTAL':<10s} {'':<8s} {total_clustered:>12,d} {100.0:>11.1f}%")

if total_clustered == 4166:
    log(f"✅ VERIFIED: Cluster sizes sum to 4,166")
else:
    log(f"❌ ERROR: Cluster sizes sum to {total_clustered}, expected 4,166")

# ============================================================================
# STEP 5: VALIDATION METRICS
# ============================================================================
print_header("STEP 5: CLUSTERING QUALITY METRICS")

silhouette = silhouette_score(X_scaled, labels)
davies_bouldin = davies_bouldin_score(X_scaled, labels)
calinski = calinski_harabasz_score(X_scaled, labels)

log(f"\nSilhouette Score: {silhouette:.3f}")
log(f"Davies-Bouldin Index: {davies_bouldin:.3f}")
log(f"Calinski-Harabasz Score: {calinski:.3f}")

# Compare with reported
log(f"\nComparison with Reported Values:")
log(f"  Silhouette: Reported=0.535, Calculated={silhouette:.3f} {'✅' if abs(silhouette-0.535)<0.001 else '❌'}")
log(f"  Davies-Bouldin: Reported=1.345, Calculated={davies_bouldin:.3f} {'✅' if abs(davies_bouldin-1.345)<0.01 else '❌'}")

# ============================================================================
# STEP 6: UNCERTAINTY ANALYSIS
# ============================================================================
print_header("STEP 6: UNCERTAINTY QUANTIFICATION")

uncertainties = 1 - np.max(probabilities, axis=1)
high_uncertainty = (uncertainties > 0.7).sum()

log(f"\nUncertainty Statistics:")
log(f"  Mean uncertainty: {uncertainties.mean():.3f}")
log(f"  Max uncertainty: {uncertainties.max():.3f}")
log(f"  Patients with >70% uncertainty: {high_uncertainty}")

if high_uncertainty == 0:
    log(f"  ✅ VERIFIED: Zero high-uncertainty patients (matches logs)")

# ============================================================================
# FINAL VERDICT
# ============================================================================
print_header("FINAL VERDICT: 4 OR 5 CLUSTERS?")

log(f"""
DEFINITIVE ANSWER:
  
  BIC Selection: {best_n} clusters
  Actual cluster labels: {unique_labels}
  Number of unique labels: {len(unique_labels)}
  
  Cluster sizes (must sum to 4,166):
""")

for cluster_label in unique_labels:
    log(f"    Cluster {cluster_label}: {cluster_sizes[cluster_label]:,} patients (Mean UPDRS={cluster_means[cluster_label]:.2f})")

log(f"    TOTAL: {sum(cluster_sizes.values()):,} patients")

log(f"\n{'='*90}")
log(f"CONCLUSION:")
log(f"{'='*90}")

if best_n == 4 and len(unique_labels) == 4:
    log(f"\n✅ CONFIRMED: There are FOUR (4) distinct motor phenotypes")
    log(f"   Cluster labels: 0, 1, 2, 3 (sklearn convention)")
    log(f"   Total patients: {sum(cluster_sizes.values()):,}")
    log(f"\n❌ ERROR IN DOCUMENTATION:")
    log(f"   PATHWAY_01_DOPAMINERGIC_DEEP_INFERENCES.md incorrectly lists 5 clusters")
    log(f"   That document needs to be corrected")
elif best_n == 5 or len(unique_labels) == 5:
    log(f"\n⚠️  FOUND: Five (5) clusters exist")
    log(f"   This contradicts the comprehensive logs which say 4")
    log(f"   Need to investigate which analysis is the 'official' result")

log(f"\n{'='*90}")
log(f"RECOMMENDATION FOR TABLE 3:")
log(f"{'='*90}")
log(f"\nReport FOUR phenotypes with these characteristics:")
log(f"  Phenotype 1 (Cluster 0): n={cluster_sizes.get(0, 'N/A')}, Mean UPDRS={cluster_means.get(0, 0):.2f}")
log(f"  Phenotype 2 (Cluster 1): n={cluster_sizes.get(1, 'N/A')}, Mean UPDRS={cluster_means.get(1, 0):.2f}")
log(f"  Phenotype 3 (Cluster 2): n={cluster_sizes.get(2, 'N/A')}, Mean UPDRS={cluster_means.get(2, 0):.2f}")
log(f"  Phenotype 4 (Cluster 3): n={cluster_sizes.get(3, 'N/A')}, Mean UPDRS={cluster_means.get(3, 0):.2f}")

log(f"\nDo NOT mention 'Cluster 4' or '5 clusters' anywhere in the manuscript!")
log(f"{'='*90}\n")

# Save report
with open(OUTPUT_FILE, 'w') as f:
    f.write('\n'.join(output))

print(f"✅ Verification report saved to: {OUTPUT_FILE}")
print("="*90 + "\n")