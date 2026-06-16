#!/usr/bin/env python3
"""
1000+ Configuration Mega Sweep with MDS-UPDRS Part III Motor Data
For npj Parkinson's Disease Paper - Cross-Cohort Replication Analysis

Uses 100+ cores for parallel execution of HDBSCAN configurations
Plus Bayesian GMM comparison for methodological rigor
"""

import json
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import warnings
warnings.filterwarnings('ignore')

# Configuration
N_CORES = min(mp.cpu_count(), 120)
RESULTS_DIR = Path("analysis_results")
DATA_DIR = Path("data")
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

print(f"=" * 80)
print(f"1000+ CONFIG MEGA SWEEP WITH REAL MDS-UPDRS PART III DATA")
print(f"Timestamp: {TIMESTAMP}")
print(f"Available cores: {mp.cpu_count()}, Using: {N_CORES}")
print(f"=" * 80)

# ============================================================================
# SECTION 1: LOAD AND PREPARE MOTOR DATA
# ============================================================================

def load_motor_data():
    """Load MDS-UPDRS Part III motor data."""
    print("\n[DATA LOADING] Searching for MDS-UPDRS Part III files...")
    
    # Priority order for data files
    data_files = [
        DATA_DIR / "PPMI_Full" / "MDS_UPDRS_Part_III-Archived_20Dec2025.csv",
        DATA_DIR / "BioFIND" / "MDS_UPDRS_Part_III__Post_Dose__18Dec2025.csv"
    ]
    
    combined_data = []
    
    for df_path in data_files:
        if df_path.exists():
            print(f"  Loading: {df_path.name}")
            try:
                df = pd.read_csv(df_path, low_memory=False)
                print(f"    Shape: {df.shape}")
                
                # Find motor score columns (NP3xxx pattern)
                np3_cols = [c for c in df.columns if c.startswith('NP3')]
                if np3_cols:
                    print(f"    Motor columns: {len(np3_cols)}")
                    motor_df = df[np3_cols].dropna()
                    print(f"    Complete cases: {len(motor_df)}")
                    combined_data.append(motor_df)
            except Exception as e:
                print(f"    Error: {e}")
    
    if combined_data:
        # Combine all data with matching columns
        all_cols = set(combined_data[0].columns)
        for df in combined_data[1:]:
            all_cols = all_cols.intersection(set(df.columns))
        
        common_cols = list(all_cols)
        combined = pd.concat([df[common_cols] for df in combined_data], ignore_index=True)
        print(f"\n[DATA] Combined dataset: {combined.shape[0]} samples × {combined.shape[1]} features")
        return combined.values, common_cols
    
    return None, None


# ============================================================================
# SECTION 2: HDBSCAN CONFIGURATION SWEEP
# ============================================================================

def generate_configs():
    """Generate 1000+ HDBSCAN configurations."""
    configs = []
    config_id = 0
    
    # Expanded parameter grid
    min_cluster_sizes = [5, 8, 10, 12, 15, 18, 20, 25, 30, 35, 40, 50, 60, 75, 100]
    min_samples_list = [2, 3, 5, 7, 10, 12, 15, 20, 25, 30]
    methods = ['eom', 'leaf']
    metrics = ['euclidean', 'manhattan']
    epsilons = [0.0, 0.1, 0.2, 0.5]
    
    for mcs in min_cluster_sizes:
        for ms in min_samples_list:
            for method in methods:
                for metric in metrics:
                    for eps in epsilons:
                        if ms <= mcs:  # Valid combination only
                            configs.append({
                                'config_id': config_id,
                                'min_cluster_size': mcs,
                                'min_samples': ms,
                                'method': method,
                                'metric': metric,
                                'epsilon': eps
                            })
                            config_id += 1
    
    return configs


def run_hdbscan_worker(args):
    """Worker function for parallel HDBSCAN execution."""
    config, X = args
    
    try:
        import hdbscan
        from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
        
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=config['min_cluster_size'],
            min_samples=config['min_samples'],
            cluster_selection_method=config['method'],
            metric=config['metric'],
            cluster_selection_epsilon=config['epsilon'],
            core_dist_n_jobs=1  # Single core per worker
        )
        labels = clusterer.fit_predict(X)
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        noise_pct = (labels == -1).sum() / len(labels) * 100
        
        # Metrics (only for valid clustering)
        if n_clusters >= 2:
            mask = labels != -1
            if mask.sum() >= n_clusters * 2:
                sil = silhouette_score(X[mask], labels[mask])
                ch = calinski_harabasz_score(X[mask], labels[mask])
                db = davies_bouldin_score(X[mask], labels[mask])
            else:
                sil, ch, db = -1, -1, -1
        else:
            sil, ch, db = -1, -1, -1
        
        return {
            **config,
            'n_clusters': n_clusters,
            'noise_pct': round(noise_pct, 2),
            'silhouette': round(sil, 4) if sil > 0 else sil,
            'calinski_harabasz': round(ch, 2) if ch > 0 else ch,
            'davies_bouldin': round(db, 4) if db > 0 else db,
            'status': 'success'
        }
    except Exception as e:
        return {**config, 'status': 'failed', 'error': str(e)[:100]}


def run_mega_sweep(X):
    """Run 1000+ configuration mega sweep."""
    configs = generate_configs()
    n_configs = len(configs)
    
    print(f"\n[MEGA SWEEP] Running {n_configs} configurations using {N_CORES} cores")
    print(f"  Data shape: {X.shape}")
    
    # Prepare parallel arguments
    args_list = [(config, X) for config in configs]
    
    results = []
    completed = 0
    
    with ProcessPoolExecutor(max_workers=N_CORES) as executor:
        futures = {executor.submit(run_hdbscan_worker, args): args[0]['config_id'] 
                   for args in args_list}
        
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            completed += 1
            
            if completed % 200 == 0 or completed == n_configs:
                print(f"  Progress: {completed}/{n_configs} ({100*completed/n_configs:.1f}%)")
    
    # Analyze results
    successful = [r for r in results if r.get('status') == 'success' and r.get('silhouette', -1) > 0]
    successful.sort(key=lambda x: x['silhouette'], reverse=True)
    
    print(f"\n[RESULTS SUMMARY]")
    print(f"  Total configs: {n_configs}")
    print(f"  Successful with valid silhouette: {len(successful)}")
    
    if successful:
        top = successful[0]
        print(f"\n  🏆 BEST CONFIGURATION:")
        print(f"    Silhouette: {top['silhouette']}")
        print(f"    n_clusters: {top['n_clusters']}")
        print(f"    min_cluster_size: {top['min_cluster_size']}")
        print(f"    min_samples: {top['min_samples']}")
        print(f"    method: {top['method']}")
        print(f"    metric: {top['metric']}")
        print(f"    epsilon: {top['epsilon']}")
        print(f"    noise_pct: {top['noise_pct']}%")
    
    return {
        'timestamp': TIMESTAMP,
        'n_samples': len(X),
        'n_features': X.shape[1],
        'n_configs': n_configs,
        'n_successful': len(successful),
        'n_cores_used': N_CORES,
        'top_10': successful[:10],
        'all_results': results
    }


# ============================================================================
# SECTION 3: BAYESIAN GMM COMPARISON
# ============================================================================

def run_bayesian_gmm(X, n_components_range=range(3, 20)):
    """Run Bayesian GMM for comparison with HDBSCAN."""
    print(f"\n[BAYESIAN GMM] Running comparison with {len(n_components_range)} component configs...")
    
    from sklearn.mixture import BayesianGaussianMixture
    from sklearn.metrics import silhouette_score
    
    results = []
    
    for n_comp in n_components_range:
        try:
            bgm = BayesianGaussianMixture(
                n_components=n_comp,
                covariance_type='full',
                max_iter=300,
                n_init=3,
                random_state=42
            )
            labels = bgm.fit_predict(X)
            n_actual = len(set(labels))
            
            sil = silhouette_score(X, labels) if n_actual >= 2 else -1
            
            results.append({
                'n_components': n_comp,
                'n_actual_clusters': n_actual,
                'silhouette': round(sil, 4),
                'converged': bgm.converged_
            })
            
            print(f"  n_components={n_comp}: actual={n_actual}, sil={sil:.4f}")
        except Exception as e:
            results.append({'n_components': n_comp, 'error': str(e)})
    
    # Find best
    valid_results = [r for r in results if r.get('silhouette', -1) > 0]
    best = max(valid_results, key=lambda x: x['silhouette']) if valid_results else None
    
    if best:
        print(f"\n  🏆 BEST BAYESIAN GMM:")
        print(f"    n_components: {best['n_components']}")
        print(f"    Silhouette: {best['silhouette']}")
    
    return {
        'method': 'BayesianGMM',
        'results': results,
        'best': best
    }


# ============================================================================
# SECTION 4: CROSS-COHORT VERIFICATION
# ============================================================================

def verify_multi_cohort():
    """Verify multi-cohort statistics from JSON."""
    print("\n[MULTI-COHORT VERIFICATION]")
    
    mc_file = RESULTS_DIR / "multi_dataset" / "maximum_parallel_20251228_084258.json"
    
    if not mc_file.exists():
        print("  ⚠ Multi-cohort file not found")
        return None
    
    with open(mc_file) as f:
        data = json.load(f)
    
    total_samples = data.get('total_samples', 0)
    n_cohorts = data.get('n_cohorts', 0)
    n_successful = data.get('n_successful', 0)
    
    print(f"  ✓ Total samples: {total_samples:,}")
    print(f"  ✓ Number of cohorts: {n_cohorts}")
    print(f"  ✓ Successful cohorts: {n_successful}")
    
    # Cohort details
    cohort_stats = []
    for cohort in data.get('results', []):
        name = cohort.get('name')
        n_samples = cohort.get('n_samples', 0)
        best = cohort.get('hdbscan_best', {})
        if best:
            sil = best.get('sil', 'N/A')
            n_clusters = best.get('n_clusters', 'N/A')
        else:
            sil = 'failed'
            n_clusters = 'N/A'
        
        cohort_stats.append({
            'name': name,
            'n_samples': n_samples,
            'silhouette': sil,
            'n_clusters': n_clusters
        })
        print(f"    • {name}: {n_samples:,} samples, sil={sil}")
    
    return {
        'total_samples': total_samples,
        'n_cohorts': n_cohorts,
        'n_successful': n_successful,
        'cohort_details': cohort_stats
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    all_results = {
        'timestamp': TIMESTAMP,
        'infrastructure': {
            'cores_available': mp.cpu_count(),
            'cores_used': N_CORES
        }
    }
    
    # Verify multi-cohort data first
    all_results['multi_cohort'] = verify_multi_cohort()
    
    # Load motor data
    X, feature_names = load_motor_data()
    
    if X is not None and len(X) >= 100:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        all_results['data'] = {
            'n_samples': len(X),
            'n_features': len(feature_names),
            'feature_names': feature_names
        }
        
        # Run mega sweep
        all_results['mega_sweep'] = run_mega_sweep(X_scaled)
        
        # Run Bayesian GMM comparison
        all_results['bayesian_gmm'] = run_bayesian_gmm(X_scaled)
        
        # Comparison summary
        hdb_best = all_results['mega_sweep']['top_10'][0] if all_results['mega_sweep']['top_10'] else None
        gmm_best = all_results['bayesian_gmm']['best']
        
        print("\n" + "=" * 80)
        print("HDBSCAN vs BAYESIAN GMM COMPARISON")
        print("=" * 80)
        if hdb_best and gmm_best:
            print(f"  HDBSCAN Best Silhouette:  {hdb_best['silhouette']} ({hdb_best['n_clusters']} clusters)")
            print(f"  Bayesian GMM Silhouette:  {gmm_best['silhouette']} ({gmm_best['n_actual_clusters']} clusters)")
            improvement = ((hdb_best['silhouette'] - gmm_best['silhouette']) / gmm_best['silhouette'] * 100) if gmm_best['silhouette'] > 0 else 0
            print(f"  HDBSCAN improvement: {improvement:+.1f}%")
    else:
        print("\n⚠ Insufficient data for mega sweep")
    
    # Save results
    output_file = RESULTS_DIR / f"mega_sweep_1000_configs_{TIMESTAMP}.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n✓ Results saved to: {output_file}")
    
    return all_results


if __name__ == "__main__":
    results = main()
