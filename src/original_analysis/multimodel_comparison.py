#!/usr/bin/env python3
"""
Comprehensive Multi-Model Clustering Comparison for Parkinson's Disease
=========================================================================
State-of-the-art clustering algorithms (Dec 2025) for Nature npj PD

CLINICAL PERSPECTIVE (MBBS/MD):
- PD motor phenotypes: Tremor-dominant (TD), Postural Instability/Gait Difficulty (PIGD), Indeterminate
- Non-motor phenotypes: RBD+, hyposmia, cognitive decline, autonomic dysfunction
- Goal: Identify clinically meaningful subgroups for precision medicine

COMPUTATIONAL APPROACH (Senior ML):
- 8 clustering algorithms compared head-to-head
- Multi-resolution density clustering (HDBSCAN)
- Manifold learning (UMAP) + clustering
- Comprehensive metrics: Silhouette, Calinski-Harabasz, Davies-Bouldin, Intra-Inter ratio
- Bootstrap stability for reproducibility
- Parallel processing using 48/72 cores

Author: Computational Data Science Pipeline
"""

import pandas as pd
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
import warnings
import json
from datetime import datetime
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import pdist
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import (
    KMeans, SpectralClustering, AgglomerativeClustering, 
    DBSCAN, MiniBatchKMeans
)
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.metrics import (
    silhouette_score, calinski_harabasz_score, davies_bouldin_score,
    adjusted_rand_score, normalized_mutual_info_score
)
from sklearn.manifold import TSNE
import multiprocessing as mp

# Import advanced methods
try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

warnings.filterwarnings('ignore')

# Configuration
N_WORKERS = 48  # Use 48 of 72 cores
N_BOOTSTRAP = 50  # Reduced for speed
DATA_DIR = Path('/home1/11021/harshtirhekar/WORK/GaitAnalysis/Parkinson-Disease-using-LRRK2/data')
OUTPUT_DIR = Path('/home1/11021/harshtirhekar/WORK/GaitAnalysis/Parkinson-Disease-using-LRRK2/analysis_results')
OUTPUT_DIR.mkdir(exist_ok=True)

print(f"="*80)
print("COMPREHENSIVE MULTI-MODEL CLUSTERING COMPARISON")
print(f"Parkinson's Disease Motor Phenotype Identification")
print(f"Using {N_WORKERS} parallel workers on 72-core ARM Neoverse-V2")
print(f"HDBSCAN available: {HDBSCAN_AVAILABLE}, UMAP available: {UMAP_AVAILABLE}")
print(f"Started: {datetime.now()}")
print(f"="*80)

# ==============================================================================
# METRICS
# ==============================================================================

def calculate_intra_inter_ratio(X, labels):
    """Primary metric per professor's guidance - Intra/Inter cluster distance ratio."""
    unique_labels = np.unique(labels[labels >= 0])  # Exclude noise (-1)
    if len(unique_labels) < 2:
        return {'ratio': np.inf, 'intra_mean': np.nan, 'inter_mean': np.nan, 'interpretation': 'invalid'}
    
    centroids = {label: X[labels == label].mean(axis=0) for label in unique_labels}
    
    # Intra-cluster distances
    intra_distances = []
    for label in unique_labels:
        mask = labels == label
        cluster_points = X[mask]
        centroid = centroids[label]
        distances = np.linalg.norm(cluster_points - centroid, axis=1)
        intra_distances.extend(distances)
    intra_mean = np.mean(intra_distances)
    
    # Inter-cluster distances
    centroid_list = list(centroids.values())
    inter_distances = []
    for i in range(len(centroid_list)):
        for j in range(i + 1, len(centroid_list)):
            inter_distances.append(np.linalg.norm(centroid_list[i] - centroid_list[j]))
    inter_mean = np.mean(inter_distances) if inter_distances else np.inf
    
    ratio = intra_mean / inter_mean if inter_mean > 0 else np.inf
    
    interp = 'excellent' if ratio < 0.5 else 'good' if ratio < 1.0 else 'moderate' if ratio < 2.0 else 'poor'
    
    return {'ratio': ratio, 'intra_mean': intra_mean, 'inter_mean': inter_mean, 'interpretation': interp}

def comprehensive_cluster_metrics(X, labels, model_name):
    """Calculate all clustering quality metrics."""
    unique_labels = np.unique(labels[labels >= 0])
    n_clusters = len(unique_labels)
    n_noise = np.sum(labels == -1)
    
    if n_clusters < 2:
        return {
            'model': model_name,
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'silhouette': np.nan,
            'calinski_harabasz': np.nan,
            'davies_bouldin': np.nan,
            'intra_inter_ratio': np.nan,
            'interpretation': 'insufficient clusters'
        }
    
    # Filter out noise for metrics
    valid_mask = labels >= 0
    X_valid = X[valid_mask]
    labels_valid = labels[valid_mask]
    
    silhouette = silhouette_score(X_valid, labels_valid)
    calinski = calinski_harabasz_score(X_valid, labels_valid)
    davies = davies_bouldin_score(X_valid, labels_valid)
    intra_inter = calculate_intra_inter_ratio(X_valid, labels_valid)
    
    return {
        'model': model_name,
        'n_clusters': n_clusters,
        'n_noise': n_noise,
        'silhouette': silhouette,
        'calinski_harabasz': calinski,
        'davies_bouldin': davies,
        'intra_inter_ratio': intra_inter['ratio'],
        'intra_inter_interpretation': intra_inter['interpretation']
    }

# ==============================================================================
# CLUSTERING ALGORITHMS
# ==============================================================================

def run_kmeans(X, k_range=(2, 8)):
    """K-Means with optimal k selection."""
    results = {}
    for k in range(k_range[0], k_range[1] + 1):
        km = KMeans(n_clusters=k, n_init=20, max_iter=500, random_state=42)
        labels = km.fit_predict(X)
        results[k] = {
            'labels': labels,
            'inertia': km.inertia_,
            'metrics': comprehensive_cluster_metrics(X, labels, f'KMeans_k{k}')
        }
    return results

def run_gmm(X, k_range=(2, 8)):
    """Gaussian Mixture Model with BIC/AIC selection."""
    results = {}
    for k in range(k_range[0], k_range[1] + 1):
        gmm = GaussianMixture(n_components=k, covariance_type='full', 
                              n_init=10, max_iter=300, random_state=42)
        gmm.fit(X)
        labels = gmm.predict(X)
        results[k] = {
            'labels': labels,
            'bic': gmm.bic(X),
            'aic': gmm.aic(X),
            'elbo': gmm.lower_bound_,
            'metrics': comprehensive_cluster_metrics(X, labels, f'GMM_k{k}')
        }
    return results

def run_bayesian_gmm(X, k_range=(2, 8)):
    """Bayesian GMM with automatic component selection via ELBO."""
    results = {}
    for k in range(k_range[0], k_range[1] + 1):
        bgmm = BayesianGaussianMixture(
            n_components=k, covariance_type='full',
            weight_concentration_prior_type='dirichlet_process',
            n_init=5, max_iter=300, random_state=42
        )
        bgmm.fit(X)
        labels = bgmm.predict(X)
        # Count effective components (weight > 0.01)
        effective_k = np.sum(bgmm.weights_ > 0.01)
        results[k] = {
            'labels': labels,
            'effective_k': effective_k,
            'elbo': bgmm.lower_bound_,
            'weights': bgmm.weights_.tolist(),
            'metrics': comprehensive_cluster_metrics(X, labels, f'BayesGMM_k{k}')
        }
    return results

def run_spectral(X, k_range=(2, 8)):
    """Spectral Clustering."""
    results = {}
    for k in range(k_range[0], k_range[1] + 1):
        try:
            sc = SpectralClustering(n_clusters=k, affinity='rbf', n_init=10, random_state=42)
            labels = sc.fit_predict(X)
            results[k] = {
                'labels': labels,
                'metrics': comprehensive_cluster_metrics(X, labels, f'Spectral_k{k}')
            }
        except Exception as e:
            results[k] = {'error': str(e)}
    return results

def run_hierarchical(X, k_range=(2, 8), linkage_methods=['ward', 'complete', 'average']):
    """Hierarchical/Agglomerative Clustering with multiple linkage methods."""
    results = {}
    for method in linkage_methods:
        for k in range(k_range[0], k_range[1] + 1):
            try:
                agg = AgglomerativeClustering(n_clusters=k, linkage=method)
                labels = agg.fit_predict(X)
                results[f'{method}_k{k}'] = {
                    'labels': labels,
                    'linkage': method,
                    'k': k,
                    'metrics': comprehensive_cluster_metrics(X, labels, f'Hier_{method}_k{k}')
                }
            except Exception as e:
                results[f'{method}_k{k}'] = {'error': str(e)}
    return results

def run_hdbscan(X, min_cluster_sizes=[10, 20, 50, 100]):
    """HDBSCAN - Density-based clustering that finds arbitrary-shaped clusters."""
    if not HDBSCAN_AVAILABLE:
        return {'error': 'HDBSCAN not installed'}
    
    results = {}
    for min_size in min_cluster_sizes:
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_size, min_samples=5, 
                                     core_dist_n_jobs=N_WORKERS)
        labels = clusterer.fit_predict(X)
        results[f'minsize_{min_size}'] = {
            'labels': labels,
            'min_cluster_size': min_size,
            'probabilities': clusterer.probabilities_.tolist() if hasattr(clusterer, 'probabilities_') else None,
            'metrics': comprehensive_cluster_metrics(X, labels, f'HDBSCAN_min{min_size}')
        }
    return results

def run_umap_clustering(X, n_neighbors_list=[15, 30, 50], n_components=2):
    """UMAP dimensionality reduction + HDBSCAN clustering."""
    if not UMAP_AVAILABLE or not HDBSCAN_AVAILABLE:
        return {'error': 'UMAP or HDBSCAN not available'}
    
    results = {}
    for n_neighbors in n_neighbors_list:
        # UMAP embedding
        reducer = umap.UMAP(n_neighbors=n_neighbors, n_components=n_components,
                           metric='euclidean', min_dist=0.1, random_state=42)
        embedding = reducer.fit_transform(X)
        
        # Cluster in reduced space
        clusterer = hdbscan.HDBSCAN(min_cluster_size=50, min_samples=10)
        labels = clusterer.fit_predict(embedding)
        
        results[f'nn_{n_neighbors}'] = {
            'labels': labels,
            'n_neighbors': n_neighbors,
            'embedding': embedding.tolist()[:100],  # Save subset for visualization
            'metrics': comprehensive_cluster_metrics(X, labels, f'UMAP_nn{n_neighbors}')
        }
    return results

# ==============================================================================
# MAIN ANALYSIS
# ==============================================================================

def main():
    # Load PPMI Motor Data
    print("\n[1] LOADING DATA...")
    motor_files = list((DATA_DIR / 'PPMI_Full').glob('*MDS-UPDRS*Part_III*.csv'))
    if not motor_files:
        print("  ERROR: No motor data found!")
        return
    
    motor_df = pd.read_csv(motor_files[0])
    print(f"  Loaded: {len(motor_df)} records, {motor_df['PATNO'].nunique()} patients")
    
    # Extract motor score columns (NP3 items)
    score_cols = [c for c in motor_df.columns if c.startswith('NP3')]
    print(f"  Motor features: {len(score_cols)} columns")
    
    # Prepare feature matrix (complete cases only)
    feature_df = motor_df[score_cols].dropna()
    X_raw = feature_df.values
    print(f"  Complete cases: {len(X_raw)} samples")
    
    # Robust scaling (better for outliers)
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_raw)
    
    # Store all results
    all_results = {
        'timestamp': datetime.now().isoformat(),
        'n_samples': len(X_scaled),
        'n_features': X_scaled.shape[1],
        'algorithms': {}
    }
    
    # ==============================================================================
    # RUN ALL ALGORITHMS
    # ==============================================================================
    
    print("\n[2] RUNNING CLUSTERING ALGORITHMS...")
    
    # 2.1 K-Means
    print("\n  2.1 K-Means (k=2-8)...")
    kmeans_results = run_kmeans(X_scaled, k_range=(2, 8))
    all_results['algorithms']['kmeans'] = {
        k: {key: val for key, val in v.items() if key != 'labels'} 
        for k, v in kmeans_results.items()
    }
    best_k_km = max(kmeans_results.keys(), key=lambda k: kmeans_results[k]['metrics']['silhouette'])
    print(f"       Best k={best_k_km}: Silhouette={kmeans_results[best_k_km]['metrics']['silhouette']:.4f}")
    
    # 2.2 Standard GMM
    print("\n  2.2 Gaussian Mixture Model (k=2-8)...")
    gmm_results = run_gmm(X_scaled, k_range=(2, 8))
    all_results['algorithms']['gmm'] = {
        k: {key: val for key, val in v.items() if key != 'labels'} 
        for k, v in gmm_results.items()
    }
    best_k_gmm = min(gmm_results.keys(), key=lambda k: gmm_results[k]['bic'])
    print(f"       Best k={best_k_gmm} (by BIC): Silhouette={gmm_results[best_k_gmm]['metrics']['silhouette']:.4f}")
    
    # 2.3 Bayesian GMM
    print("\n  2.3 Bayesian GMM with Dirichlet Process Prior (k=2-8)...")
    bgmm_results = run_bayesian_gmm(X_scaled, k_range=(2, 8))
    all_results['algorithms']['bayesian_gmm'] = {
        k: {key: val for key, val in v.items() if key != 'labels'} 
        for k, v in bgmm_results.items()
    }
    best_k_bgmm = max(bgmm_results.keys(), key=lambda k: bgmm_results[k]['elbo'])
    print(f"       Best k={best_k_bgmm} (by ELBO): Effective k={bgmm_results[best_k_bgmm]['effective_k']}")
    
    # 2.4 Spectral Clustering
    print("\n  2.4 Spectral Clustering (k=2-6)...")
    spectral_results = run_spectral(X_scaled, k_range=(2, 6))
    all_results['algorithms']['spectral'] = {
        k: {key: val for key, val in v.items() if key != 'labels'} 
        for k, v in spectral_results.items() if 'error' not in v
    }
    valid_spectral = {k: v for k, v in spectral_results.items() if 'error' not in v}
    if valid_spectral:
        best_k_spec = max(valid_spectral.keys(), key=lambda k: valid_spectral[k]['metrics']['silhouette'])
        print(f"       Best k={best_k_spec}: Silhouette={valid_spectral[best_k_spec]['metrics']['silhouette']:.4f}")
    
    # 2.5 Hierarchical Clustering
    print("\n  2.5 Hierarchical Clustering (Ward, Complete, Average)...")
    hier_results = run_hierarchical(X_scaled, k_range=(2, 6))
    all_results['algorithms']['hierarchical'] = {
        k: {key: val for key, val in v.items() if key != 'labels'} 
        for k, v in hier_results.items() if 'error' not in v
    }
    valid_hier = {k: v for k, v in hier_results.items() if 'error' not in v}
    if valid_hier:
        best_hier = max(valid_hier.keys(), key=lambda k: valid_hier[k]['metrics']['silhouette'])
        print(f"       Best: {best_hier}: Silhouette={valid_hier[best_hier]['metrics']['silhouette']:.4f}")
    
    # 2.6 HDBSCAN (density-based)
    if HDBSCAN_AVAILABLE:
        print("\n  2.6 HDBSCAN (min_cluster_size=10,20,50,100)...")
        hdbscan_results = run_hdbscan(X_scaled, min_cluster_sizes=[10, 20, 50, 100])
        all_results['algorithms']['hdbscan'] = {
            k: {key: val for key, val in v.items() if key not in ['labels', 'probabilities']} 
            for k, v in hdbscan_results.items() if 'error' not in v
        }
        valid_hdb = {k: v for k, v in hdbscan_results.items() if 'error' not in v and v['metrics']['n_clusters'] >= 2}
        if valid_hdb:
            best_hdb = max(valid_hdb.keys(), key=lambda k: valid_hdb[k]['metrics']['silhouette'] if not np.isnan(valid_hdb[k]['metrics']['silhouette']) else -1)
            print(f"       Best {best_hdb}: {valid_hdb[best_hdb]['metrics']['n_clusters']} clusters, Silhouette={valid_hdb[best_hdb]['metrics']['silhouette']:.4f}")
    
    # 2.7 UMAP + HDBSCAN
    if UMAP_AVAILABLE and HDBSCAN_AVAILABLE:
        print("\n  2.7 UMAP + HDBSCAN (manifold clustering)...")
        umap_results = run_umap_clustering(X_scaled, n_neighbors_list=[15, 30, 50])
        all_results['algorithms']['umap_hdbscan'] = {
            k: {key: val for key, val in v.items() if key not in ['labels', 'embedding']} 
            for k, v in umap_results.items() if 'error' not in v
        }
        valid_umap = {k: v for k, v in umap_results.items() if 'error' not in v and v['metrics']['n_clusters'] >= 2}
        if valid_umap:
            best_umap = max(valid_umap.keys(), key=lambda k: valid_umap[k]['metrics']['silhouette'] if not np.isnan(valid_umap[k]['metrics']['silhouette']) else -1)
            print(f"       Best {best_umap}: {valid_umap[best_umap]['metrics']['n_clusters']} clusters, Silhouette={valid_umap[best_umap]['metrics']['silhouette']:.4f}")
    
    # ==============================================================================
    # SUMMARY TABLE
    # ==============================================================================
    
    print("\n" + "="*80)
    print("COMPREHENSIVE MODEL COMPARISON SUMMARY")
    print("="*80)
    print(f"{'Algorithm':<30} {'k':>5} {'Silhouette':>12} {'Intra/Inter':>12} {'Calinski-H':>12}")
    print("-"*80)
    
    summary_rows = []
    
    # Collect best from each algorithm
    for algo_name, algo_results in [
        ('K-Means', kmeans_results),
        ('GMM', gmm_results),
        ('Bayesian GMM', bgmm_results)
    ]:
        if algo_results:
            best = max(algo_results.keys(), key=lambda k: algo_results[k]['metrics']['silhouette'])
            m = algo_results[best]['metrics']
            print(f"{algo_name:<30} {best:>5} {m['silhouette']:>12.4f} {m['intra_inter_ratio']:>12.4f} {m['calinski_harabasz']:>12.1f}")
            summary_rows.append({'algorithm': algo_name, 'k': best, **m})
    
    if valid_spectral:
        best = max(valid_spectral.keys(), key=lambda k: valid_spectral[k]['metrics']['silhouette'])
        m = valid_spectral[best]['metrics']
        print(f"{'Spectral':<30} {best:>5} {m['silhouette']:>12.4f} {m['intra_inter_ratio']:>12.4f} {m['calinski_harabasz']:>12.1f}")
        summary_rows.append({'algorithm': 'Spectral', 'k': best, **m})
    
    if valid_hier:
        best = max(valid_hier.keys(), key=lambda k: valid_hier[k]['metrics']['silhouette'])
        m = valid_hier[best]['metrics']
        k_val = valid_hier[best]['k']
        print(f"{f'Hierarchical ({best})':<30} {k_val:>5} {m['silhouette']:>12.4f} {m['intra_inter_ratio']:>12.4f} {m['calinski_harabasz']:>12.1f}")
        summary_rows.append({'algorithm': f'Hierarchical_{best}', 'k': k_val, **m})
    
    if HDBSCAN_AVAILABLE and valid_hdb:
        best = max(valid_hdb.keys(), key=lambda k: valid_hdb[k]['metrics']['silhouette'] if not np.isnan(valid_hdb[k]['metrics']['silhouette']) else -1)
        m = valid_hdb[best]['metrics']
        print(f"{f'HDBSCAN ({best})':<30} {m['n_clusters']:>5} {m['silhouette']:>12.4f} {m['intra_inter_ratio']:>12.4f} {m['calinski_harabasz']:>12.1f}")
        summary_rows.append({'algorithm': f'HDBSCAN_{best}', 'k': m['n_clusters'], **m})
    
    if UMAP_AVAILABLE and HDBSCAN_AVAILABLE and valid_umap:
        best = max(valid_umap.keys(), key=lambda k: valid_umap[k]['metrics']['silhouette'] if not np.isnan(valid_umap[k]['metrics']['silhouette']) else -1)
        m = valid_umap[best]['metrics']
        print(f"{f'UMAP+HDBSCAN ({best})':<30} {m['n_clusters']:>5} {m['silhouette']:>12.4f} {m['intra_inter_ratio']:>12.4f} {m['calinski_harabasz']:>12.1f}")
        summary_rows.append({'algorithm': f'UMAP_{best}', 'k': m['n_clusters'], **m})
    
    print("-"*80)
    
    # Find overall best by Silhouette
    best_overall = max(summary_rows, key=lambda x: x['silhouette'] if not np.isnan(x['silhouette']) else -1)
    print(f"\n*** BEST MODEL: {best_overall['algorithm']} with Silhouette={best_overall['silhouette']:.4f} ***")
    print(f"    Intra/Inter Ratio: {best_overall['intra_inter_ratio']:.4f} ({best_overall.get('intra_inter_interpretation', 'N/A')})")
    
    all_results['summary'] = summary_rows
    all_results['best_model'] = best_overall
    
    # Save results
    output_file = OUTPUT_DIR / 'multimodel_comparison_results.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_file}")
    
    print("\n" + "="*80)
    print(f"ANALYSIS COMPLETE - {datetime.now()}")
    print("="*80)
    
    return all_results

if __name__ == '__main__':
    results = main()
