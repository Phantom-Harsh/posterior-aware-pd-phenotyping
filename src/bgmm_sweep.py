#!/usr/bin/env python3
"""
bgmm_sweep.py — 2,912-configuration parallel BGMM mega-sweep.
MICCAI 2026: Tirhekar, Yadav & Bajaj

Sweeps: K(13) × covariance(4) × prior(2) × alpha_0(7) × mean_precision(4) = 2,912 configs
Selects optimal model via composite rank (Silhouette ↑, DB ↓, CH ↑).
Validates with B=1,000 patient-level bootstrap resamples.
"""

import argparse
import itertools
import logging
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.mixture import BayesianGaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from tqdm import tqdm

from utils import setup_logger, save_json

warnings.filterwarnings("ignore")

# ── Config space ──────────────────────────────────────────────────────────────
K_VALUES       = [3, 5, 8, 10, 12, 15, 20, 25, 30, 40, 50, 70, 100]
COV_TYPES      = ["full", "tied", "diag", "spherical"]
PRIOR_TYPES    = ["dirichlet_process", "dirichlet_distribution"]
ALPHA0_VALUES  = [1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0]
MEAN_PREC      = [1e-6, 1e-4, 1e-2, 1.0]

DOMAIN_SCALED  = ["Tremor_scaled", "Bradykinesia_scaled",
                   "Rigidity_scaled", "Axial_scaled", "Bulbar_scaled"]
K_EFF_THRESHOLD = 0.01   # weight > 1%


def _build_all_configs(n_configs: int) -> list[dict]:
    all_cfgs = list(itertools.product(K_VALUES, COV_TYPES, PRIOR_TYPES,
                                      ALPHA0_VALUES, MEAN_PREC))
    if n_configs < len(all_cfgs):
        rng = np.random.default_rng(42)
        idxs = rng.choice(len(all_cfgs), size=n_configs, replace=False)
        all_cfgs = [all_cfgs[i] for i in sorted(idxs)]
    return [
        dict(K=K, cov=cov, prior=prior, alpha0=a0, mean_prec=mp)
        for K, cov, prior, a0, mp in all_cfgs
    ]


def _fit_one(cfg: dict, X: np.ndarray, seed: int) -> dict | None:
    try:
        model = BayesianGaussianMixture(
            n_components=cfg["K"],
            covariance_type=cfg["cov"],
            weight_concentration_prior_type=cfg["prior"],
            weight_concentration_prior=cfg["alpha0"],
            mean_precision_prior=cfg["mean_prec"],
            max_iter=300,
            n_init=1,
            random_state=seed,
        )
        model.fit(X)
        labels = model.predict(X)
        n_unique = len(np.unique(labels))
        if n_unique < 2:
            return None

        k_eff = int((model.weights_ > K_EFF_THRESHOLD).sum())
        sil  = float(silhouette_score(X, labels, sample_size=min(5000, len(X))))
        db   = float(davies_bouldin_score(X, labels))
        ch   = float(calinski_harabasz_score(X, labels))

        return {**cfg, "k_eff": k_eff, "silhouette": sil,
                "davies_bouldin": db, "calinski_harabasz": ch}
    except Exception:
        return None


def _bootstrap_k_eff(X: np.ndarray, best_cfg: dict, patient_ids: np.ndarray,
                      B: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    unique_patients = np.unique(patient_ids)
    k_effs = []
    for _ in range(B):
        sampled = rng.choice(unique_patients, size=len(unique_patients), replace=True)
        mask = np.isin(patient_ids, sampled)
        X_boot = X[mask]
        try:
            m = BayesianGaussianMixture(
                n_components=best_cfg["K"],
                covariance_type=best_cfg["cov"],
                weight_concentration_prior_type=best_cfg["prior"],
                weight_concentration_prior=best_cfg["alpha0"],
                mean_precision_prior=best_cfg["mean_prec"],
                max_iter=200, n_init=1, random_state=int(rng.integers(1e6)),
            )
            m.fit(X_boot)
            k_effs.append(int((m.weights_ > K_EFF_THRESHOLD).sum()))
        except Exception:
            pass
    return float(np.mean(k_effs)), float(np.std(k_effs))


def run(input_path: str, n_configs: int, n_workers: int,
        output_dir: str, seed: int, demo: bool = False):
    log = setup_logger("bgmm_sweep", output_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load data
    df = pd.read_parquet(input_path)
    feature_cols = [c for c in DOMAIN_SCALED if c in df.columns]
    if not feature_cols:
        # Fallback: use raw domain cols
        feature_cols = ["Tremor", "Bradykinesia", "Rigidity", "Axial", "Bulbar"]
        feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols].dropna().values
    patient_ids = df.loc[df[feature_cols].notna().all(axis=1), "PATNO"].values
    log.info(f"Feature matrix: {X.shape} | {len(np.unique(patient_ids)):,} patients")

    # Build configs
    configs = _build_all_configs(n_configs)
    log.info(f"Sweeping {len(configs):,} configurations with {n_workers} workers …")

    # Parallel sweep
    raw_results = Parallel(n_jobs=n_workers, backend="loky", verbose=0)(
        delayed(_fit_one)(cfg, X, seed) for cfg in tqdm(configs, desc="Sweep")
    )
    results = [r for r in raw_results if r is not None]
    log.info(f"  Valid results: {len(results):,} / {len(configs):,}")

    if not results:
        raise RuntimeError("All configurations failed — check input data.")

    # Composite ranking: silhouette ↑, davies_bouldin ↓, calinski_harabasz ↑
    df_res = pd.DataFrame(results)
    df_res["sil_rank"]  = df_res["silhouette"].rank(ascending=False)
    df_res["db_rank"]   = df_res["davies_bouldin"].rank(ascending=True)
    df_res["ch_rank"]   = df_res["calinski_harabasz"].rank(ascending=False)
    df_res["composite"] = df_res["sil_rank"] + df_res["db_rank"] + df_res["ch_rank"]
    df_res = df_res.sort_values("composite")
    df_res.to_csv(out / "sweep_results.csv", index=False)

    best_row = df_res.iloc[0].to_dict()
    best_cfg = {k: best_row[k] for k in ["K", "cov", "prior", "alpha0", "mean_prec"]}
    log.info(f"Best config: {best_cfg}  |  k_eff={int(best_row['k_eff'])}  "
             f"Sil={best_row['silhouette']:.4f}")

    # Fit best model on full data
    best_model = BayesianGaussianMixture(
        n_components=int(best_cfg["K"]),
        covariance_type=best_cfg["cov"],
        weight_concentration_prior_type=best_cfg["prior"],
        weight_concentration_prior=best_cfg["alpha0"],
        mean_precision_prior=best_cfg["mean_prec"],
        max_iter=500, n_init=3, random_state=seed,
    )
    best_model.fit(X)

    model_path = out / "best_bgmm.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({"model": best_model, "feature_cols": feature_cols,
                     "config": best_cfg}, f)
    log.info(f"Saved best model → {model_path}")

    # Bootstrap k_eff validation
    B = 50 if demo else 1000
    log.info(f"Bootstrap validation: B={B} patient-level resamples …")
    k_eff_mean, k_eff_std = _bootstrap_k_eff(X, best_cfg, patient_ids, B, seed)
    log.info(f"  k_eff* = {k_eff_mean:.1f} ± {k_eff_std:.2f}")

    summary = {
        "n_configs_attempted": len(configs),
        "n_configs_valid": len(results),
        "best_config": best_cfg,
        "best_metrics": {
            "k_eff": int(best_row["k_eff"]),
            "silhouette": float(best_row["silhouette"]),
            "davies_bouldin": float(best_row["davies_bouldin"]),
            "calinski_harabasz": float(best_row["calinski_harabasz"]),
        },
        "bootstrap": {"B": B, "k_eff_mean": k_eff_mean, "k_eff_std": k_eff_std},
    }
    save_json(summary, str(out / "sweep_summary.json"))
    log.info("BGMM sweep complete.")


def main():
    p = argparse.ArgumentParser(description="BGMM Mega-Sweep (MICCAI 2026)")
    p.add_argument("--input",      required=True)
    p.add_argument("--n_configs",  type=int, default=2912)
    p.add_argument("--n_workers",  type=int, default=120)
    p.add_argument("--output_dir", default="results")
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--demo",       action="store_true")
    args = p.parse_args()
    run(args.input, args.n_configs, args.n_workers,
        args.output_dir, args.seed, args.demo)


if __name__ == "__main__":
    main()
