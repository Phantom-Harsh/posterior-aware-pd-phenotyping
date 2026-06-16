#!/usr/bin/env python3
"""
Comprehensive Data Verification & 1000+ Configuration HPC Sweep
For npj Parkinson's Disease Paper Resubmission

This script:
1. Reverifies all statistics from source JSON files
2. Runs cross-cohort phenotype replication analysis
3. Executes 1000+ HDBSCAN configuration sweep using 100+ cores
4. Compares HDBSCAN vs Bayesian GMM (for methodological rigor)
5. Generates clinical interpretation summaries

Author: Research Team
Date: December 2025
Infrastructure: TACC Grace (144 cores, Neoverse-V2)
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
N_CORES = min(mp.cpu_count(), 120)  # Use up to 120 cores, leave some for system
RESULTS_DIR = Path("analysis_results")
DATA_DIR = Path("data")
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

print(f"=" * 80)
print(f"COMPREHENSIVE VERIFICATION & 1000+ CONFIG MEGA SWEEP")
print(f"Timestamp: {TIMESTAMP}")
print(f"Available cores: {mp.cpu_count()}, Using: {N_CORES}")
print(f"=" * 80)

# ============================================================================
# SECTION 1: VERIFY ALL EXISTING STATISTICS
# ============================================================================

def verify_progression_data():
    """Verify longitudinal progression trajectory statistics."""
    print("\n[1/5] VERIFYING PROGRESSION TRAJECTORY DATA...")
    
    prog_file = RESULTS_DIR / "progression" / "REAL_progression_trajectories_20251230_055929.json"
    
    if not prog_file.exists():
        return {"status": "ERROR", "message": f"File not found: {prog_file}"}
    
    with open(prog_file) as f:
        data = json.load(f)
    
    # Extract and verify statistics
    results = {
        "file": str(prog_file),
        "n_patients_analyzed": data.get("n_patients_analyzed"),
        "n_total_records": data.get("n_total_records"),
        "best_k": data.get("best_k"),
        "clustering_silhouette_k3": None,
        "clusters": {}
    }
    
    # Get silhouette for k=3
    for cr in data.get("clustering_results", []):
        if cr.get("k") == 3:
            results["clustering_silhouette_k3"] = cr.get("silhouette")
    
    # Cluster characterization
    for cluster_name, cluster_data in data.get("cluster_characterization", {}).items():
        results["clusters"][cluster_name] = {
            "n_patients": cluster_data.get("n_patients"),
            "pct_of_cohort": cluster_data.get("pct_of_cohort"),
            "baseline_mean": cluster_data.get("baseline_score", {}).get("mean"),
            "final_mean": cluster_data.get("final_score", {}).get("mean"),
            "progression_rate": cluster_data.get("progression_rate", {}).get("mean"),
            "trajectory_type": cluster_data.get("trajectory_type")
        }
    
    # Summary stats
    summary = data.get("summary_statistics", {})
    results["mean_progression_rate"] = summary.get("mean_progression_rate")
    results["mean_visits_per_patient"] = summary.get("mean_visits")
    
    # Compute derived statistics
    total_patients = sum(c.get("n_patients", 0) for c in results["clusters"].values())
    results["computed_total_patients"] = total_patients
    results["verification_match"] = total_patients == results["n_patients_analyzed"]
    
    # Identify rapid progressors
    rapid_cluster = None
    for cname, cdata in results["clusters"].items():
        if cdata.get("trajectory_type") == "RAPID_PROGRESSOR":
            rapid_cluster = cname
            results["rapid_progressor_n"] = cdata["n_patients"]
            results["rapid_progressor_pct"] = cdata["pct_of_cohort"]
            results["rapid_progressor_rate"] = cdata["progression_rate"]
    
    print(f"  ✓ Patients analyzed: {results['n_patients_analyzed']}")
    print(f"  ✓ Total records: {results['n_total_records']}")
    print(f"  ✓ Mean visits/patient: {results['mean_visits_per_patient']:.2f}")
    print(f"  ✓ Rapid progressors: {results.get('rapid_progressor_n', 'N/A')} ({results.get('rapid_progressor_pct', 'N/A')}%)")
    print(f"  ✓ Rapid progression rate: {results.get('rapid_progressor_rate', 'N/A')} UPDRS pts/visit")
    print(f"  ✓ Verification match: {results['verification_match']}")
    
    return results


def verify_hdbscan_clustering():
    """Verify HDBSCAN mega sweep results."""
    print("\n[2/5] VERIFYING HDBSCAN CLUSTERING RESULTS...")
    
    hdbscan_file = RESULTS_DIR / "mega_sweep" / "hdbscan_mega_20251228_080721.json"
    
    if not hdbscan_file.exists():
        return {"status": "ERROR", "message": f"File not found: {hdbscan_file}"}
    
    with open(hdbscan_file) as f:
        data = json.load(f)
    
    results = {
        "file": str(hdbscan_file),
        "algorithm": data.get("algorithm"),
        "n_configs_tested": data.get("n_configs"),
        "n_successful": data.get("n_successful"),
        "best_config": None
    }
    
    # Find best configuration by silhouette score
    best_sil = -1
    for config in data.get("results", []):
        sil = config.get("silhouette", 0)
        if sil > best_sil:
            best_sil = sil
            results["best_config"] = {
                "config_id": config.get("config_id"),
                "min_cluster_size": config.get("min_cluster_size"),
                "min_samples": config.get("min_samples"),
                "method": config.get("method"),
                "n_clusters": config.get("n_clusters"),
                "silhouette": config.get("silhouette"),
                "intra_inter_ratio": config.get("intra_inter_ratio"),
                "noise_pct": config.get("noise_pct")
            }
    
    results["best_silhouette"] = best_sil
    
    print(f"  ✓ Configurations tested: {results['n_configs_tested']}")
    print(f"  ✓ Successful: {results['n_successful']}")
    print(f"  ✓ Best silhouette: {results['best_silhouette']:.4f}")
    print(f"  ✓ Best config n_clusters: {results['best_config']['n_clusters']}")
    print(f"  ✓ Best config params: min_cluster_size={results['best_config']['min_cluster_size']}, min_samples={results['best_config']['min_samples']}")
    
    return results


def verify_multi_cohort_data():
    """Verify multi-cohort parallel analysis results."""
    print("\n[3/5] VERIFYING MULTI-COHORT DATA...")
    
    mc_file = RESULTS_DIR / "multi_dataset" / "maximum_parallel_20251228_084258.json"
    
    if not mc_file.exists():
        return {"status": "ERROR", "message": f"File not found: {mc_file}"}
    
    with open(mc_file) as f:
        data = json.load(f)
    
    results = {
        "file": str(mc_file),
        "cohorts": [],
        "total_samples": 0
    }
    
    for cohort in data.get("cohort_results", []):
        cohort_info = {
            "name": cohort.get("name"),
            "n_samples": cohort.get("n_samples"),
            "n_features": cohort.get("n_features"),
            "status": cohort.get("status", "success" if cohort.get("hdbscan_best") else "failed")
        }
        if cohort.get("hdbscan_best"):
            cohort_info["best_silhouette"] = cohort["hdbscan_best"].get("sil")
            cohort_info["n_clusters"] = cohort["hdbscan_best"].get("n_clusters")
            cohort_info["noise_pct"] = cohort["hdbscan_best"].get("noise_pct")
        
        results["cohorts"].append(cohort_info)
        results["total_samples"] += cohort.get("n_samples", 0)
    
    results["n_cohorts"] = len(results["cohorts"])
    
    print(f"  ✓ Number of cohorts: {results['n_cohorts']}")
    print(f"  ✓ Total samples: {results['total_samples']:,}")
    for c in results["cohorts"]:
        status = "✓" if c["status"] == "success" else "✗"
        print(f"    {status} {c['name']}: {c['n_samples']:,} samples")
    
    return results


def verify_prodromal_prediction():
    """Verify prodromal prediction results."""
    print("\n[4/5] VERIFYING PRODROMAL PREDICTION DATA...")
    
    prod_file = RESULTS_DIR / "prodromal" / "REAL_prodromal_prediction_20251230_060003.json"
    
    if not prod_file.exists():
        return {"status": "ERROR", "message": f"File not found: {prod_file}"}
    
    with open(prod_file) as f:
        data = json.load(f)
    
    model_results = data.get("model_results", {})
    
    results = {
        "file": str(prod_file),
        "n_patients_rbd": data.get("n_patients_rbd"),
        "n_patients_upsit": data.get("n_patients_upsit"),
        "n_patients_updrs": data.get("n_patients_updrs"),
        "n_complete_samples": model_results.get("n_samples"),
        "features_used": model_results.get("features_used"),
        "classes": model_results.get("classes"),
        "model_accuracies": {}
    }
    
    # Extract model accuracies
    for model_name, model_data in model_results.get("model_results", {}).items():
        results["model_accuracies"][model_name] = {
            "mean_accuracy": model_data.get("mean_accuracy"),
            "std_accuracy": model_data.get("std_accuracy")
        }
    
    # Find best model
    best_acc = 0
    best_model = None
    for model, acc_data in results["model_accuracies"].items():
        if acc_data["mean_accuracy"] > best_acc:
            best_acc = acc_data["mean_accuracy"]
            best_model = model
    
    results["best_model"] = best_model
    results["best_accuracy"] = best_acc
    
    print(f"  ✓ RBD patients: {results['n_patients_rbd']}")
    print(f"  ✓ UPSIT patients: {results['n_patients_upsit']}")
    print(f"  ✓ UPDRS patients: {results['n_patients_updrs']}")
    print(f"  ✓ Complete samples: {results['n_complete_samples']}")
    print(f"  ✓ Best model: {best_model} (accuracy: {best_acc:.3f})")
    
    return results


def verify_clinical_characterization():
    """Verify clinical characterization of severe phenotype."""
    print("\n[5/5] VERIFYING CLINICAL CHARACTERIZATION...")
    
    char_file = RESULTS_DIR / "clinical_characterization_20251228_065406.json"
    
    if not char_file.exists():
        # Try alternative location
        char_file = RESULTS_DIR / "cache" / "clinical_characterization_20251228_065406.json"
    
    if not char_file.exists():
        return {"status": "WARNING", "message": "Clinical characterization file not found"}
    
    with open(char_file) as f:
        data = json.load(f)
    
    results = {
        "file": str(char_file),
        "data": data
    }
    
    print(f"  ✓ Clinical characterization loaded successfully")
    
    return results


# ============================================================================
# SECTION 2: 1000+ CONFIGURATION MEGA SWEEP
# ============================================================================

def generate_1000_configs():
    """Generate 1000+ HDBSCAN configurations for mega sweep."""
    configs = []
    config_id = 0
    
    # Hyperparameter ranges (expanded for comprehensive sweep)
    min_cluster_sizes = [5, 8, 10, 12, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200]
    min_samples_list = [2, 3, 5, 7, 10, 15, 20, 25, 30, 40, 50]
    methods = ['eom', 'leaf']
    metrics = ['euclidean', 'manhattan', 'cosine']
    cluster_selection_epsilons = [0.0, 0.1, 0.2, 0.3, 0.5]
    
    for mcs in min_cluster_sizes:
        for ms in min_samples_list:
            for method in methods:
                for metric in metrics:
                    for eps in cluster_selection_epsilons:
                        # Skip invalid combinations
                        if ms > mcs:
                            continue
                        
                        configs.append({
                            'config_id': config_id,
                            'min_cluster_size': mcs,
                            'min_samples': ms,
                            'method': method,
                            'metric': metric,
                            'cluster_selection_epsilon': eps
                        })
                        config_id += 1
    
    return configs


def run_single_hdbscan_config(args):
    """Run a single HDBSCAN configuration (for parallel execution)."""
    config, data_matrix = args
    
    try:
        import hdbscan
        from sklearn.metrics import silhouette_score
        from sklearn.preprocessing import StandardScaler
        
        # Scale data
        scaler = StandardScaler()
        X = scaler.fit_transform(data_matrix)
        
        # Run HDBSCAN
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=config['min_cluster_size'],
            min_samples=config['min_samples'],
            cluster_selection_method=config['method'],
            metric=config['metric'],
            cluster_selection_epsilon=config.get('cluster_selection_epsilon', 0.0)
        )
        labels = clusterer.fit_predict(X)
        
        # Calculate metrics
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        noise_pct = (labels == -1).sum() / len(labels) * 100
        
        # Silhouette score (only if we have valid clusters)
        if n_clusters >= 2 and noise_pct < 95:
            # Exclude noise points for silhouette
            mask = labels != -1
            if mask.sum() >= n_clusters * 2:
                sil = silhouette_score(X[mask], labels[mask])
            else:
                sil = -1
        else:
            sil = -1
        
        # Intra/inter cluster distance ratio
        intra_inter = calculate_intra_inter_ratio(X, labels) if n_clusters >= 2 else -1
        
        return {
            'config_id': config['config_id'],
            'min_cluster_size': config['min_cluster_size'],
            'min_samples': config['min_samples'],
            'method': config['method'],
            'metric': config['metric'],
            'cluster_selection_epsilon': config.get('cluster_selection_epsilon', 0.0),
            'n_clusters': n_clusters,
            'noise_pct': noise_pct,
            'silhouette': sil,
            'intra_inter_ratio': intra_inter,
            'status': 'success'
        }
    except Exception as e:
        return {
            'config_id': config['config_id'],
            'status': 'failed',
            'error': str(e)
        }


def calculate_intra_inter_ratio(X, labels):
    """Calculate intra/inter cluster distance ratio."""
    try:
        from scipy.spatial.distance import cdist
        
        unique_labels = set(labels) - {-1}
        if len(unique_labels) < 2:
            return -1
        
        # Calculate cluster centroids
        centroids = []
        for label in unique_labels:
            mask = labels == label
            centroids.append(X[mask].mean(axis=0))
        centroids = np.array(centroids)
        
        # Mean intra-cluster distance
        intra_dists = []
        for label in unique_labels:
            mask = labels == label
            if mask.sum() > 1:
                cluster_data = X[mask]
                centroid = cluster_data.mean(axis=0)
                dists = np.linalg.norm(cluster_data - centroid, axis=1)
                intra_dists.append(dists.mean())
        
        # Mean inter-cluster distance
        inter_dists = cdist(centroids, centroids)
        np.fill_diagonal(inter_dists, np.inf)
        
        mean_intra = np.mean(intra_dists) if intra_dists else 0
        mean_inter = inter_dists[inter_dists < np.inf].mean() if len(centroids) > 1 else 1
        
        return mean_intra / mean_inter if mean_inter > 0 else -1
    except:
        return -1


def run_mega_sweep_parallel(data_matrix, n_cores=N_CORES):
    """Run 1000+ configuration sweep using parallel processing."""
    configs = generate_1000_configs()
    n_configs = len(configs)
    
    print(f"\n[MEGA SWEEP] Running {n_configs} configurations using {n_cores} cores...")
    
    # Prepare arguments for parallel execution
    args_list = [(config, data_matrix) for config in configs]
    
    results = []
    completed = 0
    
    with ProcessPoolExecutor(max_workers=n_cores) as executor:
        futures = {executor.submit(run_single_hdbscan_config, args): args[0]['config_id'] 
                   for args in args_list}
        
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            completed += 1
            
            if completed % 100 == 0:
                print(f"  Progress: {completed}/{n_configs} ({100*completed/n_configs:.1f}%)")
    
    # Sort by silhouette score
    successful = [r for r in results if r.get('status') == 'success' and r.get('silhouette', -1) > 0]
    successful.sort(key=lambda x: x['silhouette'], reverse=True)
    
    print(f"\n[MEGA SWEEP] Complete!")
    print(f"  Total configs: {n_configs}")
    print(f"  Successful: {len(successful)}")
    if successful:
        print(f"  Best silhouette: {successful[0]['silhouette']:.4f}")
        print(f"  Best config: {successful[0]}")
    
    return {
        "timestamp": TIMESTAMP,
        "n_configs": n_configs,
        "n_successful": len(successful),
        "n_cores_used": n_cores,
        "results": results,
        "top_10_configs": successful[:10] if successful else []
    }


# ============================================================================
# SECTION 3: CROSS-COHORT REPLICATION ANALYSIS
# ============================================================================

def run_cross_cohort_replication():
    """Run cross-cohort phenotype replication analysis."""
    print("\n[CROSS-COHORT REPLICATION] Checking phenotype consistency across cohorts...")
    
    # Load multi-cohort results
    mc_file = RESULTS_DIR / "multi_dataset" / "maximum_parallel_20251228_084258.json"
    
    if not mc_file.exists():
        return {"status": "ERROR", "message": "Multi-cohort file not found"}
    
    with open(mc_file) as f:
        mc_data = json.load(f)
    
    # Compare clustering results across successful cohorts
    cohort_results = []
    for cohort in mc_data.get("cohort_results", []):
        if cohort.get("hdbscan_best"):
            cohort_results.append({
                "name": cohort.get("name"),
                "n_samples": cohort.get("n_samples"),
                "silhouette": cohort["hdbscan_best"].get("sil"),
                "n_clusters": cohort["hdbscan_best"].get("n_clusters"),
                "noise_pct": cohort["hdbscan_best"].get("noise_pct")
            })
    
    # Calculate cross-cohort statistics
    if cohort_results:
        silhouettes = [c["silhouette"] for c in cohort_results if c["silhouette"]]
        n_clusters_list = [c["n_clusters"] for c in cohort_results if c["n_clusters"]]
        
        replication_stats = {
            "n_cohorts_analyzed": len(cohort_results),
            "mean_silhouette": np.mean(silhouettes) if silhouettes else None,
            "std_silhouette": np.std(silhouettes) if silhouettes else None,
            "min_silhouette": min(silhouettes) if silhouettes else None,
            "max_silhouette": max(silhouettes) if silhouettes else None,
            "mean_n_clusters": np.mean(n_clusters_list) if n_clusters_list else None,
            "std_n_clusters": np.std(n_clusters_list) if n_clusters_list else None,
            "cohort_details": cohort_results
        }
        
        print(f"  ✓ Cohorts with successful clustering: {len(cohort_results)}")
        print(f"  ✓ Mean silhouette across cohorts: {replication_stats['mean_silhouette']:.4f}")
        print(f"  ✓ Silhouette range: [{replication_stats['min_silhouette']:.4f}, {replication_stats['max_silhouette']:.4f}]")
        
        return replication_stats
    
    return {"status": "ERROR", "message": "No successful cohort results found"}


# ============================================================================
# SECTION 4: BAYESIAN GMM COMPARISON
# ============================================================================

def run_bayesian_gmm_comparison(data_matrix, n_components_range=range(3, 15)):
    """Run Bayesian GMM for methodological comparison with HDBSCAN."""
    print("\n[BAYESIAN GMM] Running comparison analysis...")
    
    try:
        from sklearn.mixture import BayesianGaussianMixture
        from sklearn.metrics import silhouette_score
        from sklearn.preprocessing import StandardScaler
        
        scaler = StandardScaler()
        X = scaler.fit_transform(data_matrix)
        
        results = []
        best_bic = np.inf
        best_result = None
        
        for n_comp in n_components_range:
            bgm = BayesianGaussianMixture(
                n_components=n_comp,
                covariance_type='full',
                max_iter=500,
                n_init=5,
                random_state=42
            )
            labels = bgm.fit_predict(X)
            
            n_actual = len(set(labels))
            sil = silhouette_score(X, labels) if n_actual >= 2 else -1
            bic = bgm.bic(X) if hasattr(bgm, 'bic') else np.inf
            
            result = {
                "n_components": n_comp,
                "n_actual_clusters": n_actual,
                "silhouette": sil,
                "converged": bgm.converged_
            }
            results.append(result)
            
            if sil > 0 and (best_result is None or sil > best_result["silhouette"]):
                best_result = result
        
        print(f"  ✓ Tested {len(n_components_range)} component configurations")
        if best_result:
            print(f"  ✓ Best Bayesian GMM: {best_result['n_components']} components, Silhouette={best_result['silhouette']:.4f}")
        
        return {
            "method": "Bayesian GMM",
            "results": results,
            "best_result": best_result
        }
    except Exception as e:
        return {"status": "ERROR", "error": str(e)}


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    
    all_results = {
        "timestamp": TIMESTAMP,
        "verification_results": {},
        "mega_sweep_results": None,
        "cross_cohort_replication": None,
        "bayesian_gmm_comparison": None
    }
    
    # Section 1: Verify all existing statistics
    print("\n" + "=" * 80)
    print("SECTION 1: VERIFYING ALL EXISTING STATISTICS")
    print("=" * 80)
    
    all_results["verification_results"]["progression"] = verify_progression_data()
    all_results["verification_results"]["hdbscan"] = verify_hdbscan_clustering()
    all_results["verification_results"]["multi_cohort"] = verify_multi_cohort_data()
    all_results["verification_results"]["prodromal"] = verify_prodromal_prediction()
    all_results["verification_results"]["clinical"] = verify_clinical_characterization()
    
    # Section 2: Cross-cohort replication
    print("\n" + "=" * 80)
    print("SECTION 2: CROSS-COHORT REPLICATION ANALYSIS")
    print("=" * 80)
    
    all_results["cross_cohort_replication"] = run_cross_cohort_replication()
    
    # Section 3: Load sample data for mega sweep
    print("\n" + "=" * 80)
    print("SECTION 3: PREPARING DATA FOR MEGA SWEEP")
    print("=" * 80)
    
    # Try to load PPMI motor data for the mega sweep
    ppmi_motor_files = list(DATA_DIR.glob("**/MDS_UPDRS*.csv")) + list(DATA_DIR.glob("**/UPDRS*.csv"))
    
    if ppmi_motor_files:
        print(f"  Found {len(ppmi_motor_files)} UPDRS data files")
        try:
            # Load the first available file
            df = pd.read_csv(ppmi_motor_files[0])
            print(f"  Loaded: {ppmi_motor_files[0].name} ({len(df)} rows)")
            
            # Select numeric columns only
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            motor_cols = [c for c in numeric_cols if any(x in c.upper() for x in ['NP3', 'UPDRS', 'SCORE', 'MOTOR'])]
            
            if motor_cols:
                data_matrix = df[motor_cols].dropna().values
                print(f"  Using {len(motor_cols)} motor features, {len(data_matrix)} samples")
                
                # Run mega sweep if we have enough data
                if len(data_matrix) >= 100:
                    print("\n" + "=" * 80)
                    print("SECTION 4: 1000+ CONFIGURATION MEGA SWEEP")
                    print("=" * 80)
                    
                    all_results["mega_sweep_results"] = run_mega_sweep_parallel(data_matrix[:5000])  # Cap at 5000 for speed
                    
                    # Run Bayesian GMM comparison
                    print("\n" + "=" * 80)
                    print("SECTION 5: BAYESIAN GMM COMPARISON")
                    print("=" * 80)
                    
                    all_results["bayesian_gmm_comparison"] = run_bayesian_gmm_comparison(data_matrix[:5000])
                else:
                    print("  ⚠ Not enough samples for mega sweep")
            else:
                print("  ⚠ No suitable motor columns found")
        except Exception as e:
            print(f"  ⚠ Error loading data: {e}")
    else:
        print("  ⚠ No UPDRS data files found - skipping mega sweep")
    
    # Save results
    output_file = RESULTS_DIR / f"comprehensive_verification_{TIMESTAMP}.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Results saved to: {output_file}")
    
    # Print verification summary
    prog = all_results["verification_results"].get("progression", {})
    hdb = all_results["verification_results"].get("hdbscan", {})
    mc = all_results["verification_results"].get("multi_cohort", {})
    
    print("\n📊 KEY VERIFIED STATISTICS:")
    print(f"  • Progression patients: {prog.get('n_patients_analyzed', 'N/A')}")
    print(f"  • Rapid progressors: {prog.get('rapid_progressor_pct', 'N/A')}%")
    print(f"  • HDBSCAN best silhouette: {hdb.get('best_silhouette', 'N/A')}")
    print(f"  • HDBSCAN phenotypes: {hdb.get('best_config', {}).get('n_clusters', 'N/A')}")
    print(f"  • Total multi-cohort samples: {mc.get('total_samples', 'N/A'):,}")
    
    if all_results.get("mega_sweep_results"):
        ms = all_results["mega_sweep_results"]
        print(f"\n🚀 MEGA SWEEP RESULTS:")
        print(f"  • Configurations tested: {ms.get('n_configs', 'N/A')}")
        print(f"  • Successful: {ms.get('n_successful', 'N/A')}")
        if ms.get("top_10_configs"):
            best = ms["top_10_configs"][0]
            print(f"  • Best new silhouette: {best.get('silhouette', 'N/A'):.4f}")
            print(f"  • Best config n_clusters: {best.get('n_clusters', 'N/A')}")
    
    return all_results


if __name__ == "__main__":
    results = main()
