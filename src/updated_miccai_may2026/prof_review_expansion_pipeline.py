#!/usr/bin/env python3
"""
Professor-review expansion experiments for the Nature-track PD motor-state paper.

This script adds the experiments requested after the May 26 manuscript review,
without modifying the original MICCAI submission or overwriting the original
May 2026 revision outputs. All outputs are prefixed with ``prof_review_``.

Implemented:
- subgroup posterior/refit-stability calibration for full and PD-only cohorts
- subgroup fused imaging-to-posterior CV with RF, XGBoost, and LightGBM
- imaging feature-group ablations, permutation importance, and SHAP summaries
- stronger longitudinal baselines and time-gap transition models
- leave-one-domain/PC1/non-target severity residualization sensitivity
- bootstrap confidence intervals for headline metrics and BioFIND endpoints
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

WORKSPACE = Path(__file__).resolve().parents[1]
PROJECT = WORKSPACE.parent
RESULTS = WORKSPACE / "results"
FIGURES = WORKSPACE / "figures"
LOGS = WORKSPACE / "logs"
MPLCONFIG = LOGS / "matplotlib_cache"
MPLCONFIG.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG))

sys.path.insert(0, str(WORKSPACE / "scripts"))
import revision_analysis_pipeline as rp  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from joblib import Parallel, delayed  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402
from scipy import stats  # noqa: E402

from sklearn.base import clone  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    adjusted_rand_score,
    balanced_accuracy_score,
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    normalized_mutual_info_score,
    r2_score,
    roc_auc_score,
)
from sklearn.mixture import BayesianGaussianMixture  # noqa: E402
from sklearn.model_selection import GroupKFold, KFold  # noqa: E402
from sklearn.multioutput import MultiOutputRegressor  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import RobustScaler, StandardScaler  # noqa: E402

from xgboost import XGBClassifier, XGBRegressor  # noqa: E402
from lightgbm import LGBMClassifier, LGBMRegressor  # noqa: E402
from lifelines import CoxPHFitter  # noqa: E402
from hmmlearn.hmm import CategoricalHMM  # noqa: E402
import shap  # noqa: E402
import statsmodels.api as sm  # noqa: E402
import statsmodels.formula.api as smf  # noqa: E402
from statsmodels.genmod.cov_struct import Exchangeable  # noqa: E402


TARGET_COLS = [f"pi_{i}" for i in range(5)]
COHORT_ORDER = ["Full aligned table", "PRIMDIAG == 1", "PD cohort", "PD cohort + enrolled"]


@dataclass
class ExpansionConfig:
    mode: str
    workers: int
    rf_workers: int
    seed: int
    bootstrap_reps: int
    ci_bootstrap_reps: int
    cv_folds: int
    n_estimators: int
    backend: str = "loky"


def setup_logging() -> Path:
    for p in [RESULTS, FIGURES, LOGS]:
        p.mkdir(parents=True, exist_ok=True)
    log_path = LOGS / f"prof_review_expansion_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )
    return log_path


def write_json(path: Path, payload: object) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def normalize(arr: np.ndarray) -> np.ndarray:
    arr = np.clip(arr, 1e-8, None)
    return arr / arr.sum(axis=1, keepdims=True)


def jsd_rows(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    p = normalize(p)
    q = normalize(q)
    m = 0.5 * (p + q)
    return 0.5 * np.sum(p * np.log2((p + 1e-8) / (m + 1e-8)), axis=1) + 0.5 * np.sum(
        q * np.log2((q + 1e-8) / (m + 1e-8)), axis=1
    )


def kl_rows(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    p = normalize(p)
    q = normalize(q)
    return np.sum(p * np.log((p + 1e-8) / (q + 1e-8)), axis=1)


def soft_ece(y: np.ndarray, pred: np.ndarray, n_bins: int = 10) -> float:
    pred = normalize(pred)
    y = normalize(y)
    pred_class = pred.argmax(axis=1)
    confidence = pred.max(axis=1)
    soft_acc = y[np.arange(len(y)), pred_class]
    ece = 0.0
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confidence >= lo) & (confidence < hi if hi < 1 else confidence <= hi)
        if mask.any():
            ece += float(mask.mean()) * abs(float(confidence[mask].mean()) - float(soft_acc[mask].mean()))
    return float(ece)


def binary_ece(y: np.ndarray, prob: np.ndarray, n_bins: int = 10) -> float:
    y = np.asarray(y, dtype=float)
    prob = np.asarray(prob, dtype=float)
    ece = 0.0
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (prob >= lo) & (prob < hi if hi < 1 else prob <= hi)
        if mask.any():
            ece += float(mask.mean()) * abs(float(prob[mask].mean()) - float(y[mask].mean()))
    return float(ece)


def per_state_metrics(y: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    y = normalize(y)
    pred = normalize(pred)
    hard = y.argmax(axis=1)
    out: Dict[str, float] = {}
    for k in range(y.shape[1]):
        if len(np.unique(hard == k)) == 2:
            out[f"state_{k}_auc"] = float(roc_auc_score((hard == k).astype(int), pred[:, k]))
        else:
            out[f"state_{k}_auc"] = float("nan")
        out[f"state_{k}_ece"] = float(_one_state_ece(y[:, k], pred[:, k]))
    return out


def _one_state_ece(y_soft: np.ndarray, p: np.ndarray) -> float:
    ece = 0.0
    bins = np.linspace(0, 1, 11)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if mask.any():
            ece += float(mask.mean()) * abs(float(p[mask].mean()) - float(y_soft[mask].mean()))
    return ece


def calibration_slope_intercept(y: np.ndarray, prob: np.ndarray) -> Tuple[float, float]:
    eps = 1e-6
    prob = np.asarray(prob, dtype=float)
    y = np.asarray(y, dtype=int)
    prob = np.clip(prob, eps, 1 - eps)
    logit = np.log(prob / (1 - prob)).reshape(-1, 1)
    try:
        model = LogisticRegression(solver="lbfgs")
        model.fit(logit, y)
        return float(model.coef_[0, 0]), float(model.intercept_[0])
    except Exception:
        return float("nan"), float("nan")


def cohort_patient_sets(df: pd.DataFrame) -> Dict[str, set[int]]:
    full = set(df["PATNO"].astype(int).unique())
    sets: Dict[str, set[int]] = {"Full aligned table": full}
    ppmi = rp.PPMI

    status_files = sorted(ppmi.glob("Participant_Status_*.csv"))
    if status_files:
        status = read_csv(status_files[0])
        status["PATNO"] = pd.to_numeric(status["PATNO"], errors="coerce")
        status = status.dropna(subset=["PATNO"]).copy()
        status["PATNO"] = status["PATNO"].astype(int)
        cohort_col = next((c for c in status.columns if c.upper() == "COHORT_DEFINITION"), None)
        enroll_col = next((c for c in status.columns if "ENROLL" in c.upper() and "STATUS" in c.upper()), None)
        if cohort_col:
            cohort_text = status[cohort_col].astype(str).str.strip().str.lower()
            pd_mask = cohort_text.isin({"parkinson's disease", "parkinson disease"})
            sets["PD cohort"] = set(status.loc[pd_mask, "PATNO"].astype(int)) & full
        if cohort_col and enroll_col:
            enrolled = status[enroll_col].astype(str).str.contains("enrolled", case=False, na=False)
            cohort_text = status[cohort_col].astype(str).str.strip().str.lower()
            pd_mask = cohort_text.isin({"parkinson's disease", "parkinson disease"})
            sets["PD cohort + enrolled"] = set(status.loc[pd_mask & enrolled, "PATNO"].astype(int)) & full

    diag_files = sorted(ppmi.glob("Primary_Clinical_Diagnosis_*.csv"))
    if diag_files:
        diag = read_csv(diag_files[0])
        if {"PATNO", "PRIMDIAG"}.issubset(diag.columns):
            diag["PATNO"] = pd.to_numeric(diag["PATNO"], errors="coerce")
            diag["PRIMDIAG"] = pd.to_numeric(diag["PRIMDIAG"], errors="coerce")
            diag = diag.dropna(subset=["PATNO"]).copy()
            diag["PATNO"] = diag["PATNO"].astype(int)
            sets["PRIMDIAG == 1"] = set(diag.loc[diag["PRIMDIAG"].eq(1), "PATNO"].astype(int)) & full
    return sets


def _bootstrap_consistency(seed: int, X: np.ndarray, patient_ids: np.ndarray, original: np.ndarray) -> np.ndarray:
    rng = np.random.RandomState(seed)
    unique_patients = np.unique(patient_ids)
    sampled = rng.choice(unique_patients, size=len(unique_patients), replace=True)
    indices = np.concatenate([np.flatnonzero(patient_ids == p) for p in sampled])
    model = rp.make_bgmm(seed, max_iter=300)
    model.fit(X[indices])
    pred = model.predict(X)
    confusion = np.zeros((5, 5), dtype=int)
    for o, p in zip(original, pred):
        if 0 <= int(o) < 5 and 0 <= int(p) < 5:
            confusion[int(o), int(p)] += 1
    row_ind, col_ind = linear_sum_assignment(-confusion)
    mapping = {int(c): int(r) for r, c in zip(row_ind, col_ind)}
    mapped = np.array([mapping.get(int(p), -1) for p in pred])
    return (mapped == original).astype(float)


def run_subgroup_calibration(
    df: pd.DataFrame,
    score_cols: List[str],
    cohort_sets: Dict[str, set[int]],
    cfg: ExpansionConfig,
) -> pd.DataFrame:
    logging.info("Running subgroup refit-stability calibration reps=%d", cfg.bootstrap_reps)
    summaries = []
    all_bins = []
    all_visit = []
    for c_idx, cohort in enumerate(COHORT_ORDER):
        pats = cohort_sets.get(cohort, set())
        sub = df[df["PATNO"].isin(pats)].copy()
        if sub.empty:
            continue
        X = RobustScaler().fit_transform(sub[score_cols].to_numpy(float))
        patient_ids = sub["PATNO"].to_numpy(int)
        original = sub["cluster_label"].to_numpy(int)
        n_jobs = max(1, min(cfg.workers, cfg.bootstrap_reps))
        seeds = [cfg.seed + 10_000 * (c_idx + 1) + r for r in range(cfg.bootstrap_reps)]
        parts = Parallel(n_jobs=n_jobs, backend=cfg.backend)(
            delayed(_bootstrap_consistency)(s, X, patient_ids, original) for s in seeds
        )
        consistency = np.vstack(parts).mean(axis=0)
        conf = sub["max_posterior"].to_numpy(float)
        bins = []
        ece = 0.0
        for lo, hi in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:]):
            mask = (conf >= lo) & (conf < hi if hi < 1 else conf <= hi)
            if not mask.any():
                continue
            mean_conf = float(conf[mask].mean())
            mean_cons = float(consistency[mask].mean())
            gap = abs(mean_conf - mean_cons)
            ece += float(mask.mean()) * gap
            bins.append(
                {
                    "cohort": cohort,
                    "bin_low": lo,
                    "bin_high": hi,
                    "n": int(mask.sum()),
                    "mean_posterior_confidence": mean_conf,
                    "bootstrap_label_consistency": mean_cons,
                    "abs_gap": gap,
                }
            )
        all_bins.extend(bins)
        summaries.append(
            {
                "cohort": cohort,
                "n_visits": int(len(sub)),
                "n_patients": int(sub["PATNO"].nunique()),
                "bootstrap_reps": int(cfg.bootstrap_reps),
                "ece_stability": float(ece),
                "brier_like_score": float(np.mean((conf - consistency) ** 2)),
                "mean_bootstrap_consistency": float(consistency.mean()),
                "mean_posterior_confidence": float(conf.mean()),
                "mean_entropy": float(sub["posterior_entropy"].mean()),
                "mean_gap": float(sub["posterior_gap"].mean()),
            }
        )
        visit = sub[["PATNO", "EVENT_ID", "cluster_label", "max_posterior", "posterior_entropy", "posterior_gap"]].copy()
        visit["cohort"] = cohort
        visit["bootstrap_label_consistency"] = consistency
        all_visit.append(visit)
        logging.info("Calibration %s ECE_stab=%.4f n=%d", cohort, ece, len(sub))

    summary_df = pd.DataFrame(summaries)
    bins_df = pd.DataFrame(all_bins)
    visit_df = pd.concat(all_visit, ignore_index=True)
    summary_df.to_csv(RESULTS / "prof_review_subgroup_calibration_summary.csv", index=False)
    bins_df.to_csv(RESULTS / "prof_review_subgroup_calibration_bins.csv", index=False)
    visit_df.to_csv(RESULTS / "prof_review_subgroup_calibration_visit_consistency.csv", index=False)
    write_json(RESULTS / "prof_review_subgroup_calibration_summary.json", summaries)

    fig, ax = plt.subplots(figsize=(6.4, 5.0), dpi=180)
    ax.plot([0, 1], [0, 1], "--", color="black", linewidth=1)
    for cohort, group in bins_df.groupby("cohort"):
        ax.plot(group["mean_posterior_confidence"], group["bootstrap_label_consistency"], marker="o", label=cohort)
    ax.set_xlabel("Nominal posterior confidence")
    ax.set_ylabel("Patient-blocked refit consistency")
    ax.set_title("Subgroup Refit-Stability Calibration")
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "prof_review_subgroup_calibration.png", bbox_inches="tight")
    plt.close(fig)
    return summary_df


def residualize_features(table: pd.DataFrame, feature_cols: List[str], covars: List[str]) -> pd.DataFrame:
    X_cov = table[covars].replace([np.inf, -np.inf], np.nan)
    X_cov = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(X_cov), columns=covars, index=table.index)
    out = pd.DataFrame(index=table.index)
    for col in feature_cols:
        y = pd.to_numeric(table[col], errors="coerce").to_numpy(float)
        ok = ~np.isnan(y)
        pred = np.full(len(table), np.nan)
        if ok.sum() > len(covars) + 3:
            model = LinearRegression().fit(X_cov.loc[ok], y[ok])
            pred = model.predict(X_cov)
        out[col + "_resid"] = y - pred
    return out


def model_factory(model_name: str, seed: int, n_estimators: int, n_jobs: int, task: str):
    if task == "regression":
        if model_name == "RF":
            return RandomForestRegressor(
                n_estimators=n_estimators, min_samples_leaf=3, random_state=seed, n_jobs=n_jobs
            )
        if model_name == "XGBoost":
            return XGBRegressor(
                n_estimators=n_estimators,
                max_depth=4,
                learning_rate=0.035,
                subsample=0.9,
                colsample_bytree=0.9,
                objective="reg:squarederror",
                tree_method="hist",
                random_state=seed,
                n_jobs=n_jobs,
            )
        if model_name == "LightGBM":
            return LGBMRegressor(
                n_estimators=n_estimators,
                learning_rate=0.035,
                num_leaves=31,
                min_child_samples=10,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=seed,
                n_jobs=n_jobs,
                verbosity=-1,
            )
    else:
        if model_name == "RF":
            return RandomForestClassifier(
                n_estimators=n_estimators,
                min_samples_leaf=4,
                class_weight="balanced_subsample",
                random_state=seed,
                n_jobs=n_jobs,
            )
        if model_name == "XGBoost":
            return XGBClassifier(
                n_estimators=n_estimators,
                max_depth=4,
                learning_rate=0.035,
                subsample=0.9,
                colsample_bytree=0.9,
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                random_state=seed,
                n_jobs=n_jobs,
            )
        if model_name == "LightGBM":
            return LGBMClassifier(
                n_estimators=n_estimators,
                learning_rate=0.035,
                num_leaves=31,
                min_child_samples=10,
                subsample=0.9,
                colsample_bytree=0.9,
                class_weight="balanced",
                random_state=seed,
                n_jobs=n_jobs,
                verbosity=-1,
            )
        if model_name == "Logistic":
            return LogisticRegression(max_iter=2000, class_weight="balanced")
    raise ValueError(model_name)


def soft_cv(
    table: pd.DataFrame,
    feature_cols: List[str],
    target_cols: List[str],
    model_name: str,
    cfg: ExpansionConfig,
) -> Tuple[dict, pd.DataFrame]:
    data = table.dropna(subset=target_cols).copy()
    X = data[feature_cols].replace([np.inf, -np.inf], np.nan)
    y = normalize(data[target_cols].to_numpy(float))
    folds = min(cfg.cv_folds, len(data))
    splits = list(KFold(n_splits=folds, shuffle=True, random_state=cfg.seed).split(X))
    fold_jobs = max(1, min(cfg.workers, len(splits)))
    model_jobs = max(1, min(cfg.rf_workers, max(1, cfg.workers // fold_jobs)))

    def one_fold(train_idx, test_idx, fold_id):
        model = model_factory(model_name, cfg.seed + fold_id, cfg.n_estimators, model_jobs, "regression")
        if model_name in {"XGBoost", "LightGBM"}:
            model = MultiOutputRegressor(model, n_jobs=1)
        pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", RobustScaler()),
                ("model", model),
            ]
        )
        pipe.fit(X.iloc[train_idx], y[train_idx])
        pred = normalize(pipe.predict(X.iloc[test_idx]))
        return test_idx, pred

    parts = Parallel(n_jobs=fold_jobs, backend=cfg.backend)(
        delayed(one_fold)(tr, te, i) for i, (tr, te) in enumerate(splits)
    )
    pred = np.zeros_like(y)
    for idx, p in parts:
        pred[idx] = p
    hard = y.argmax(axis=1)
    try:
        macro_auc = float(roc_auc_score(hard, pred, average="macro", multi_class="ovr"))
    except Exception:
        macro_auc = float("nan")
    summary = {
        "n_samples": int(len(data)),
        "n_features": int(len(feature_cols)),
        "cv_folds": int(folds),
        "model": model_name,
        "macro_auc_hard_state": macro_auc,
        "mean_kl": float(kl_rows(y, pred).mean()),
        "mean_jsd": float(jsd_rows(y, pred).mean()),
        "brier_soft": float(np.mean(np.sum((y - pred) ** 2, axis=1))),
        "ece_soft": soft_ece(y, pred),
        **per_state_metrics(y, pred),
    }
    pred_df = data[["PATNO"]].copy()
    for k in range(y.shape[1]):
        pred_df[f"true_pi_{k}"] = y[:, k]
        pred_df[f"pred_pi_{k}"] = pred[:, k]
    pred_df["hard_state"] = hard
    pred_df["pred_state"] = pred.argmax(axis=1)
    return summary, pred_df


def imaging_to_posterior_expansion(
    df: pd.DataFrame,
    posterior_cols: List[str],
    cohort_sets: Dict[str, set[int]],
    cfg: ExpansionConfig,
) -> pd.DataFrame:
    logging.info("Running subgroup imaging-to-posterior models and ablations")
    targets = rp.patient_level_targets(df, posterior_cols)
    target_cols = [f"pi_{c}" for c in posterior_cols]
    dat, dat_cols = rp.load_datscan_features()
    mri, mri_cols = rp.load_mri_features()
    fusion = targets.merge(dat, on="PATNO", how="inner").merge(mri, on="PATNO", how="inner")
    for col in ["severity", "age_years", "disease_duration_years"]:
        fusion[col] = pd.to_numeric(fusion[col], errors="coerce")

    mean_sbr = [c for c in ["caudate_mean", "putamen_mean", "total_sbr", "ant_putamen_mean", "CP_ratio_mean"] if c in fusion]
    asym = [c for c in dat_cols if c.startswith("AI_")]
    covars = [c for c in ["severity", "age_years", "disease_duration_years"] if c in fusion]
    resid = residualize_features(fusion, dat_cols + mri_cols, covars)
    fusion_resid = pd.concat([fusion[["PATNO"] + target_cols], resid], axis=1)
    resid_cols = list(resid.columns)

    experiments = []
    for cohort in COHORT_ORDER:
        experiments.append((cohort, "fusion_all", fusion[fusion["PATNO"].isin(cohort_sets[cohort])], dat_cols + mri_cols, ["RF", "XGBoost", "LightGBM"]))
    full = fusion[fusion["PATNO"].isin(cohort_sets["Full aligned table"])]
    experiments.extend(
        [
            ("Full aligned table", "DaTSCAN_only", full, dat_cols, ["RF"]),
            ("Full aligned table", "MRI_only", full, mri_cols, ["RF"]),
            ("Full aligned table", "DaTSCAN_mean_SBR", full, mean_sbr, ["RF"]),
            ("Full aligned table", "DaTSCAN_asymmetry", full, asym, ["RF"]),
            ("Full aligned table", "severity_demographics_imaging", full, covars + dat_cols + mri_cols, ["RF"]),
            (
                "Full aligned table",
                "imaging_residualized_for_severity",
                fusion_resid[fusion_resid["PATNO"].isin(cohort_sets["Full aligned table"])],
                resid_cols,
                ["RF"],
            ),
        ]
    )
    summaries = []
    pred_frames = []
    for cohort, feature_set, table, features, models in experiments:
        if table.empty or not features:
            continue
        for model in models:
            logging.info("Imaging CV cohort=%s feature_set=%s model=%s n=%d p=%d", cohort, feature_set, model, len(table), len(features))
            summary, pred = soft_cv(table, features, target_cols, model, cfg)
            summary["cohort"] = cohort
            summary["feature_set"] = feature_set
            summaries.append(summary)
            pred["cohort"] = cohort
            pred["feature_set"] = feature_set
            pred["model"] = model
            pred_frames.append(pred)

    summary_df = pd.DataFrame(summaries)
    pred_df = pd.concat(pred_frames, ignore_index=True)
    summary_df.to_csv(RESULTS / "prof_review_imaging_to_posterior_summary.csv", index=False)
    pred_df.to_csv(RESULTS / "prof_review_imaging_to_posterior_predictions.csv", index=False)

    run_permutation_and_shap(full, dat_cols + mri_cols, target_cols, cfg)

    fig, ax = plt.subplots(figsize=(8.4, 5.0), dpi=180)
    plot = summary_df[summary_df["feature_set"].eq("fusion_all")].copy()
    plot["label"] = plot["cohort"] + "\n" + plot["model"]
    ax.bar(np.arange(len(plot)), plot["mean_jsd"], color="#2b6cb0")
    ax.set_xticks(np.arange(len(plot)), plot["label"], rotation=55, ha="right", fontsize=7)
    ax.set_ylabel("Mean JSD")
    ax.set_title("Subgroup Fused Imaging-to-Posterior Prediction")
    fig.tight_layout()
    fig.savefig(FIGURES / "prof_review_subgroup_imaging_jsd.png", bbox_inches="tight")
    plt.close(fig)
    return summary_df


def run_permutation_and_shap(full: pd.DataFrame, feature_cols: List[str], target_cols: List[str], cfg: ExpansionConfig) -> None:
    logging.info("Running permutation importance and SHAP for full fusion RF")
    data = full.dropna(subset=target_cols).copy()
    X = data[feature_cols].replace([np.inf, -np.inf], np.nan)
    y = normalize(data[target_cols].to_numpy(float))
    folds = list(KFold(n_splits=min(cfg.cv_folds, len(data)), shuffle=True, random_state=cfg.seed).split(X))
    rng = np.random.RandomState(cfg.seed)
    rows = []
    for fold_id, (train_idx, test_idx) in enumerate(folds):
        pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", RobustScaler()),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=cfg.n_estimators,
                        min_samples_leaf=3,
                        random_state=cfg.seed + fold_id,
                        n_jobs=max(1, min(cfg.rf_workers, cfg.workers)),
                    ),
                ),
            ]
        )
        pipe.fit(X.iloc[train_idx], y[train_idx])
        base_pred = normalize(pipe.predict(X.iloc[test_idx]))
        base_jsd = float(jsd_rows(y[test_idx], base_pred).mean())
        for feature in feature_cols:
            X_perm = X.iloc[test_idx].copy()
            vals = X_perm[feature].to_numpy(copy=True)
            rng.shuffle(vals)
            X_perm[feature] = vals
            perm_pred = normalize(pipe.predict(X_perm))
            rows.append(
                {
                    "fold": fold_id,
                    "feature": feature,
                    "baseline_jsd": base_jsd,
                    "permuted_jsd": float(jsd_rows(y[test_idx], perm_pred).mean()),
                    "delta_jsd": float(jsd_rows(y[test_idx], perm_pred).mean() - base_jsd),
                }
            )
    perm = pd.DataFrame(rows)
    perm.to_csv(RESULTS / "prof_review_imaging_permutation_importance.csv", index=False)

    agg = perm.groupby("feature", as_index=False)["delta_jsd"].mean().sort_values("delta_jsd", ascending=False)
    fig, ax = plt.subplots(figsize=(7.4, 5.2), dpi=180)
    top = agg.head(20).sort_values("delta_jsd")
    ax.barh(top["feature"], top["delta_jsd"], color="#2f855a")
    ax.set_xlabel("Mean increase in JSD after permutation")
    ax.set_title("Held-Out Permutation Importance")
    fig.tight_layout()
    fig.savefig(FIGURES / "prof_review_imaging_permutation_importance.png", bbox_inches="tight")
    plt.close(fig)

    # SHAP on a modest sample to keep runtime bounded.
    shap_rows = []
    try:
        sample_n = min(500, len(data))
        sample = data.sample(n=sample_n, random_state=cfg.seed)
        X_sample = sample[feature_cols].replace([np.inf, -np.inf], np.nan)
        imputer = SimpleImputer(strategy="median")
        scaler = RobustScaler()
        X_all = scaler.fit_transform(imputer.fit_transform(X))
        X_s = scaler.transform(imputer.transform(X_sample))
        model = RandomForestRegressor(
            n_estimators=cfg.n_estimators,
            min_samples_leaf=3,
            random_state=cfg.seed,
            n_jobs=max(1, min(cfg.rf_workers, cfg.workers)),
        )
        model.fit(X_all, y)
        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(X_s)
        if isinstance(values, list):
            arr = np.stack(values, axis=-1)
        else:
            arr = np.asarray(values)
        # Accept either (outputs, samples, features) or (samples, features, outputs).
        if arr.shape[0] == y.shape[1] and arr.ndim == 3:
            mean_abs = np.mean(np.abs(arr), axis=(0, 1))
        elif arr.ndim == 3:
            mean_abs = np.mean(np.abs(arr), axis=(0, 2))
        else:
            mean_abs = np.mean(np.abs(arr), axis=0)
        for f, val in zip(feature_cols, mean_abs):
            shap_rows.append({"feature": f, "mean_abs_shap": float(val), "n_sample": int(sample_n)})
    except Exception as exc:
        logging.exception("SHAP failed; recording error")
        shap_rows.append({"feature": "__ERROR__", "mean_abs_shap": np.nan, "n_sample": 0, "error": str(exc)})
    shap_df = pd.DataFrame(shap_rows)
    shap_df.to_csv(RESULTS / "prof_review_imaging_shap_summary.csv", index=False)


def severity_sensitivity(df: pd.DataFrame, cfg: ExpansionConfig) -> pd.DataFrame:
    logging.info("Running severity residualization sensitivity")
    domain_cols = [f"z_{d}" for d in rp.MOTOR_DOMAINS]
    base_covars, _ = rp.build_design_matrix(df)
    results = []

    pc1 = PCA(n_components=1, random_state=cfg.seed).fit_transform(df[domain_cols].to_numpy(float)).ravel()
    variants = ["total_motor", "pc1_motor_burden", "leave_one_domain_total", "non_target_domain_burden", "clinical_only"]
    for variant in variants:
        residuals = pd.DataFrame(index=df.index)
        r2s = {}
        for col in domain_cols:
            cov = base_covars.copy()
            if "severity" in cov.columns:
                cov = cov.drop(columns=["severity"])
            if variant == "total_motor":
                cov["severity_axis"] = pd.to_numeric(df["severity"], errors="coerce")
            elif variant == "pc1_motor_burden":
                cov["severity_axis"] = pc1
            elif variant == "leave_one_domain_total":
                # z-domain burden excluding the target domain.
                others = [c for c in domain_cols if c != col]
                cov["severity_axis"] = df[others].mean(axis=1)
            elif variant == "non_target_domain_burden":
                # Raw profile burden excluding the target domain after robust scaling.
                others = [c for c in domain_cols if c != col]
                cov["severity_axis"] = df[others].sum(axis=1)
            elif variant == "clinical_only":
                pass
            cov = cov.replace([np.inf, -np.inf], np.nan)
            cov = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(cov), columns=cov.columns, index=df.index)
            y = pd.to_numeric(df[col], errors="coerce").to_numpy(float)
            model = LinearRegression().fit(cov, y)
            pred = model.predict(cov)
            residuals[col.replace("z_", "resid_")] = y - pred
            ss_res = float(np.sum((y - pred) ** 2))
            ss_tot = float(np.sum((y - np.mean(y)) ** 2))
            r2s[col.replace("z_", "")] = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        X = RobustScaler().fit_transform(residuals)
        bgmm = rp.make_bgmm(cfg.seed)
        labels = bgmm.fit_predict(X)
        results.append(
            {
                "severity_axis": variant,
                "active_components": int(np.sum(bgmm.weights_ > 0.01)),
                "raw_vs_residual_ari": float(adjusted_rand_score(df["cluster_label"], labels)),
                "raw_vs_residual_nmi": float(normalized_mutual_info_score(df["cluster_label"], labels)),
                **{f"r2_{k}": float(v) for k, v in r2s.items()},
            }
        )
    out = pd.DataFrame(results)
    out.to_csv(RESULTS / "prof_review_severity_sensitivity.csv", index=False)
    return out


def evaluate_binary_cv(
    table: pd.DataFrame,
    features: List[str],
    model_name: str,
    cfg: ExpansionConfig,
) -> Tuple[dict, np.ndarray]:
    X = table[features].replace([np.inf, -np.inf], np.nan)
    y = table["state_transition"].to_numpy(int)
    groups = table["PATNO"].to_numpy(int)
    splits = list(GroupKFold(n_splits=min(cfg.cv_folds, len(np.unique(groups)))).split(X, y, groups))
    fold_jobs = max(1, min(cfg.workers, len(splits)))
    model_jobs = max(1, min(cfg.rf_workers, max(1, cfg.workers // fold_jobs)))

    def one_fold(train_idx, test_idx, fold_id):
        pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", RobustScaler()),
                ("model", model_factory(model_name, cfg.seed + fold_id, cfg.n_estimators, model_jobs, "classification")),
            ]
        )
        pipe.fit(X.iloc[train_idx], y[train_idx])
        if hasattr(pipe.named_steps["model"], "predict_proba"):
            prob = pipe.predict_proba(X.iloc[test_idx])[:, 1]
        else:
            prob = pipe.predict(X.iloc[test_idx])
        pred = (prob >= 0.5).astype(int)
        return test_idx, prob, pred

    parts = Parallel(n_jobs=fold_jobs, backend=cfg.backend)(
        delayed(one_fold)(tr, te, i) for i, (tr, te) in enumerate(splits)
    )
    prob = np.zeros(len(table), dtype=float)
    pred = np.zeros(len(table), dtype=int)
    for idx, p, cls in parts:
        prob[idx] = p
        pred[idx] = cls
    slope, intercept = calibration_slope_intercept(y, prob)
    return (
        {
            "task": "state_transition",
            "model": model_name,
            "auc": float(roc_auc_score(y, prob)) if len(np.unique(y)) == 2 else np.nan,
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "brier": float(brier_score_loss(y, prob)),
            "ece": binary_ece(y, prob),
            "calibration_slope": slope,
            "calibration_intercept": intercept,
        },
        prob,
    )


def longitudinal_expansion(df: pd.DataFrame, posterior_cols: List[str], cfg: ExpansionConfig) -> pd.DataFrame:
    logging.info("Running stronger longitudinal baselines")
    table = rp.build_longitudinal_prediction_rows(df, posterior_cols)
    table["time_gap_months"] = table["time_gap_days"] / 30.4375
    table["log_time_gap"] = np.log1p(table["time_gap_days"].clip(lower=0))

    feature_sets = {
        "current_state_only": [],
        "severity_covariates_only": ["severity", "age_years", "disease_duration_years", "LEDD_mean", "time_gap_days"],
        "motor_domains_only": [f"z_{d}" for d in rp.MOTOR_DOMAINS],
        "posterior_vector_only": [f"pi_{c}" for c in posterior_cols],
        "entropy_gap_only": ["posterior_entropy", "posterior_gap", "max_posterior"],
        "all_features": [
            "current_axial",
            "severity",
            "profile_tremor_axial",
            "posterior_entropy",
            "posterior_gap",
            "max_posterior",
            "time_gap_days",
            "log_time_gap",
            "age_years",
            "disease_duration_years",
            "LEDD_mean",
        ]
        + [f"z_{d}" for d in rp.MOTOR_DOMAINS]
        + [f"pi_{c}" for c in posterior_cols],
    }
    state_dummies = pd.get_dummies(table["current_state"].astype("category"), prefix="state")
    rows = []
    pred_cols = table[["PATNO", "EVENT_ID", "next_EVENT_ID", "state_transition", "next_axial"]].copy()
    for set_name, features in feature_sets.items():
        X_base = table[features].copy() if features else pd.DataFrame(index=table.index)
        X = pd.concat([X_base, state_dummies], axis=1) if set_name in {"current_state_only", "all_features"} else X_base
        feature_cols = list(X.columns)
        if not feature_cols:
            continue
        work = pd.concat([table[["PATNO", "state_transition"]], X], axis=1)
        for model_name in ["Logistic", "RF", "XGBoost", "LightGBM"]:
            try:
                summary, prob = evaluate_binary_cv(work, feature_cols, model_name, cfg)
                summary["feature_set"] = set_name
                rows.append(summary)
                pred_cols[f"transition_prob_{set_name}_{model_name}"] = prob
            except Exception as exc:
                logging.exception("Transition model failed feature_set=%s model=%s", set_name, model_name)
                rows.append({"feature_set": set_name, "model": model_name, "error": str(exc)})

    # Future axial regression baselines.
    X = pd.concat([table[feature_sets["all_features"]], state_dummies], axis=1).replace([np.inf, -np.inf], np.nan)
    y = table["next_axial"].to_numpy(float)
    groups = table["PATNO"].to_numpy(int)
    splits = list(GroupKFold(n_splits=min(cfg.cv_folds, len(np.unique(groups)))).split(X, y, groups))
    reg_rows = []
    for model_name in ["Ridge", "RF", "XGBoost", "LightGBM"]:
        preds = np.zeros(len(table), dtype=float)
        for fold_id, (tr, te) in enumerate(splits):
            if model_name == "Ridge":
                model = Ridge(alpha=1.0)
            else:
                model = model_factory(model_name, cfg.seed + fold_id, cfg.n_estimators, max(1, cfg.rf_workers), "regression")
            pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", RobustScaler()), ("model", model)])
            pipe.fit(X.iloc[tr], y[tr])
            preds[te] = pipe.predict(X.iloc[te])
        reg_rows.append(
            {
                "task": "next_axial_regression",
                "feature_set": "all_features",
                "model": model_name,
                "r2": float(r2_score(y, preds)),
                "rmse": float(math.sqrt(mean_squared_error(y, preds))),
                "mae": float(mean_absolute_error(y, preds)),
            }
        )
        pred_cols[f"next_axial_pred_{model_name}"] = preds

    # Time-gap-stratified transition matrices.
    bins = pd.qcut(table["time_gap_months"].rank(method="first"), q=4, labels=["Q1_short", "Q2", "Q3", "Q4_long"])
    tm_rows = []
    for gap_bin in bins.unique():
        sub = table[bins == gap_bin]
        for a in range(5):
            denom = int((sub["current_state"] == a).sum())
            for b in range(5):
                count = int(((sub["current_state"] == a) & (sub["next_state"] == b)).sum())
                tm_rows.append(
                    {
                        "gap_bin": str(gap_bin),
                        "current_state": a,
                        "next_state": b,
                        "count": count,
                        "probability": count / denom if denom else np.nan,
                        "origin_count": denom,
                    }
                )
    pd.DataFrame(tm_rows).to_csv(RESULTS / "prof_review_time_gap_transition_matrix.csv", index=False)

    # HMM over observed state sequences as a generative trajectory baseline.
    try:
        lengths = []
        seq = []
        for _, g in table.sort_values(["PATNO", "visit_date"]).groupby("PATNO"):
            states = g["current_state"].astype(int).to_numpy()
            if len(states) >= 2:
                seq.extend(states.tolist())
                lengths.append(len(states))
        hmm = CategoricalHMM(n_components=5, n_iter=200, random_state=cfg.seed)
        hmm.fit(np.array(seq).reshape(-1, 1), lengths)
        pd.DataFrame(hmm.transmat_).to_csv(RESULTS / "prof_review_categorical_hmm_transition_matrix.csv", index=False)
    except Exception as exc:
        logging.exception("Categorical HMM failed")
        write_json(RESULTS / "prof_review_categorical_hmm_error.json", {"error": str(exc)})

    # Cox transition hazard model.
    try:
        cox_cols = ["time_gap_months", "state_transition", "severity", "posterior_entropy", "posterior_gap", "age_years", "disease_duration_years"]
        cox = table[cox_cols].replace([np.inf, -np.inf], np.nan).dropna()
        cph = CoxPHFitter(penalizer=0.01)
        cph.fit(cox, duration_col="time_gap_months", event_col="state_transition")
        cph.summary.to_csv(RESULTS / "prof_review_transition_cox_model.csv")
        write_json(RESULTS / "prof_review_transition_cox_concordance.json", {"concordance_index": float(cph.concordance_index_)})
    except Exception as exc:
        logging.exception("Cox transition model failed")
        write_json(RESULTS / "prof_review_transition_cox_error.json", {"error": str(exc)})

    # Population-average longitudinal clinical baselines with patient clustering.
    # These are not the most flexible predictors, but they make the clinical
    # comparison explicit and guard against treating repeated visits as iid.
    try:
        gee_cols = [
            "PATNO",
            "state_transition",
            "next_axial",
            "current_state",
            "current_axial",
            "severity",
            "profile_tremor_axial",
            "posterior_entropy",
            "posterior_gap",
            "max_posterior",
            "time_gap_months",
            "age_years",
            "disease_duration_years",
            "LEDD_mean",
        ]
        gee = table[gee_cols].replace([np.inf, -np.inf], np.nan).dropna().copy()
        gee["current_state"] = gee["current_state"].astype(int)
        formula = (
            "state_transition ~ current_axial + severity + profile_tremor_axial + "
            "posterior_entropy + posterior_gap + max_posterior + time_gap_months + "
            "age_years + disease_duration_years + LEDD_mean + C(current_state)"
        )
        gee_bin = smf.gee(
            formula=formula,
            groups="PATNO",
            data=gee,
            cov_struct=Exchangeable(),
            family=sm.families.Binomial(),
        ).fit(maxiter=100)
        prob = np.clip(gee_bin.predict(gee), 1e-6, 1 - 1e-6)
        cls = (prob >= 0.5).astype(int)
        slope, intercept = calibration_slope_intercept(gee["state_transition"].to_numpy(int), prob)
        rows.append(
            {
                "task": "state_transition",
                "feature_set": "clustered_gee_all_features",
                "model": "GEE_Binomial_Exchangeable",
                "auc": float(roc_auc_score(gee["state_transition"], prob)) if gee["state_transition"].nunique() == 2 else np.nan,
                "balanced_accuracy": float(balanced_accuracy_score(gee["state_transition"], cls)),
                "brier": float(brier_score_loss(gee["state_transition"], prob)),
                "ece": binary_ece(gee["state_transition"].to_numpy(int), prob),
                "calibration_slope": slope,
                "calibration_intercept": intercept,
                "n_rows": int(len(gee)),
                "n_patients": int(gee["PATNO"].nunique()),
            }
        )
        gee_bin.summary2().tables[1].to_csv(RESULTS / "prof_review_transition_gee_binomial_coefficients.csv")

        gaussian_formula = formula.replace("state_transition", "next_axial")
        gee_gauss = smf.gee(
            formula=gaussian_formula,
            groups="PATNO",
            data=gee,
            cov_struct=Exchangeable(),
            family=sm.families.Gaussian(),
        ).fit(maxiter=100)
        pred_axial = gee_gauss.predict(gee)
        reg_rows.append(
            {
                "task": "next_axial_regression",
                "feature_set": "clustered_gee_all_features",
                "model": "GEE_Gaussian_Exchangeable",
                "r2": float(r2_score(gee["next_axial"], pred_axial)),
                "rmse": float(math.sqrt(mean_squared_error(gee["next_axial"], pred_axial))),
                "mae": float(mean_absolute_error(gee["next_axial"], pred_axial)),
                "n_rows": int(len(gee)),
                "n_patients": int(gee["PATNO"].nunique()),
            }
        )
        gee_gauss.summary2().tables[1].to_csv(RESULTS / "prof_review_next_axial_gee_gaussian_coefficients.csv")
    except Exception as exc:
        logging.exception("GEE longitudinal baselines failed")
        write_json(RESULTS / "prof_review_gee_error.json", {"error": str(exc)})

    out = pd.DataFrame(rows + reg_rows)
    out.to_csv(RESULTS / "prof_review_longitudinal_baselines.csv", index=False)
    pred_cols.to_csv(RESULTS / "prof_review_longitudinal_predictions.csv", index=False)
    return out


def eta_squared(groups: List[np.ndarray]) -> float:
    vals = np.concatenate(groups)
    if len(vals) == 0:
        return np.nan
    grand = np.mean(vals)
    ss_between = sum(len(g) * (np.mean(g) - grand) ** 2 for g in groups if len(g))
    ss_total = np.sum((vals - grand) ** 2)
    return float(ss_between / ss_total) if ss_total > 0 else np.nan


def delta_r2(frame: pd.DataFrame, endpoint: str) -> float:
    covars = ["severity", "age_years", "disease_duration_years"]
    use = frame[["assigned_state", endpoint] + [c for c in covars if c in frame.columns]].copy()
    use = use.dropna(subset=[endpoint, "assigned_state"])
    if len(use) < 20:
        return np.nan
    X_base = use[[c for c in covars if c in use.columns]].replace([np.inf, -np.inf], np.nan)
    if X_base.shape[1] == 0:
        X_base = pd.DataFrame({"intercept": np.ones(len(use))}, index=use.index)
    X_base = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(X_base), columns=X_base.columns)
    state = pd.get_dummies(use["assigned_state"].astype(int), prefix="state", drop_first=True).reset_index(drop=True)
    y = use[endpoint].to_numpy(float)
    base = LinearRegression().fit(X_base, y).score(X_base, y)
    full = pd.concat([X_base, state], axis=1)
    return float(LinearRegression().fit(full, y).score(full, y) - base)


def biofind_and_headline_cis(cfg: ExpansionConfig) -> pd.DataFrame:
    logging.info("Running bootstrap CIs for headline metrics and BioFIND endpoints")
    rng = np.random.RandomState(cfg.seed)
    rows = []

    state = pd.read_csv(RESULTS / "patient_state_summary.csv")
    long = pd.read_csv(RESULTS / "longitudinal_prediction_rows.csv")
    calib = pd.read_csv(RESULTS / "prof_review_subgroup_calibration_visit_consistency.csv")
    imaging = pd.read_csv(RESULTS / "prof_review_imaging_to_posterior_predictions.csv")
    bio = pd.read_csv(RESULTS / "biofind_patient_endpoint_table.csv")

    def ci(values: List[float]) -> Tuple[float, float, float]:
        arr = np.array([v for v in values if not pd.isna(v)], dtype=float)
        if len(arr) == 0:
            return np.nan, np.nan, np.nan
        return float(np.mean(arr)), float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))

    def group_indices(frame: pd.DataFrame, id_col: str = "PATNO") -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
        ids = frame[id_col].to_numpy(int)
        unique_ids = np.unique(ids)
        return unique_ids, {int(p): np.flatnonzero(ids == p) for p in unique_ids}

    def sampled_indices(unique_ids: np.ndarray, index_map: Dict[int, np.ndarray]) -> np.ndarray:
        sample = rng.choice(unique_ids, size=len(unique_ids), replace=True)
        return np.concatenate([index_map[int(p)] for p in sample])

    state_modal = state["modal_family_fraction"].to_numpy(float)
    state_transition = state["ever_transitioned_state"].to_numpy(float)
    n_state = len(state)
    vals_modal, vals_transition = [], []
    for _ in range(cfg.ci_bootstrap_reps):
        idx = rng.randint(0, n_state, size=n_state)
        vals_modal.append(float(np.mean(state_modal[idx])))
        vals_transition.append(float(np.mean(state_transition[idx])))
    for metric, vals in [("modal_family_fraction", vals_modal), ("ever_transitioned_fraction", vals_transition)]:
        mean, lo, hi = ci(vals)
        rows.append({"domain": "patient_state", "metric": metric, "mean": mean, "ci_low": lo, "ci_high": hi})

    for cohort in COHORT_ORDER:
        sub = calib[calib["cohort"].eq(cohort)]
        if sub.empty:
            continue
        work = sub[["PATNO", "max_posterior", "bootstrap_label_consistency"]].copy()
        work["sqerr"] = (work["max_posterior"] - work["bootstrap_label_consistency"]) ** 2
        ag = work.groupby("PATNO", sort=False).agg(sse=("sqerr", "sum"), n=("sqerr", "size")).reset_index()
        sse = ag["sse"].to_numpy(float)
        n = ag["n"].to_numpy(float)
        n_pat = len(ag)
        vals = []
        for _ in range(cfg.ci_bootstrap_reps):
            idx = rng.randint(0, n_pat, size=n_pat)
            vals.append(float(np.sum(sse[idx]) / np.sum(n[idx])))
        mean, lo, hi = ci(vals)
        rows.append({"domain": "calibration", "cohort": cohort, "metric": "brier_like_score", "mean": mean, "ci_low": lo, "ci_high": hi})

    # Imaging CIs for key fusion rows.
    key = imaging[(imaging["feature_set"].eq("fusion_all")) & (imaging["model"].isin(["RF", "XGBoost", "LightGBM"]))]
    for (cohort, model), sub in key.groupby(["cohort", "model"]):
        y = sub[[f"true_pi_{k}" for k in range(5)]].to_numpy(float)
        p = sub[[f"pred_pi_{k}" for k in range(5)]].to_numpy(float)
        per_row_jsd = jsd_rows(y, p)
        work = pd.DataFrame({"PATNO": sub["PATNO"].to_numpy(int), "jsd": per_row_jsd})
        ag = work.groupby("PATNO", sort=False).agg(ss=("jsd", "sum"), n=("jsd", "size")).reset_index()
        ss = ag["ss"].to_numpy(float)
        n = ag["n"].to_numpy(float)
        n_pat = len(ag)
        vals = []
        for _ in range(cfg.ci_bootstrap_reps):
            idx = rng.randint(0, n_pat, size=n_pat)
            vals.append(float(np.sum(ss[idx]) / np.sum(n[idx])))
        mean, lo, hi = ci(vals)
        rows.append({"domain": "imaging", "cohort": cohort, "model": model, "metric": "mean_jsd", "mean": mean, "ci_low": lo, "ci_high": hi})

    # Longitudinal AUC CI for strongest rows in new predictions.
    pred = pd.read_csv(RESULTS / "prof_review_longitudinal_predictions.csv")
    y = pred["state_transition"].to_numpy(int)
    unique, idx_map = group_indices(pred)
    for col in [c for c in pred.columns if c.startswith("transition_prob_all_features")]:
        prob = pred[col].to_numpy(float)
        vals = []
        for _ in range(cfg.ci_bootstrap_reps):
            idx = sampled_indices(unique, idx_map)
            if len(np.unique(y[idx])) == 2:
                vals.append(float(roc_auc_score(y[idx], prob[idx])))
        mean, lo, hi = ci(vals)
        rows.append({"domain": "longitudinal", "metric": col + "_auc", "mean": mean, "ci_low": lo, "ci_high": hi})

    # BioFIND endpoint CIs.
    if "assigned_state" in bio.columns:
        endpoints = [
            c
            for c in bio.columns
            if c not in {"PATNO", "assigned_state", "max_posterior", "posterior_entropy"}
            and not c.startswith("pi_")
            and pd.api.types.is_numeric_dtype(bio[c])
        ]
        unique, idx_map = group_indices(bio)
        for endpoint in endpoints:
            eta_vals, delta_vals = [], []
            endpoint_values = bio[endpoint].to_numpy(float)
            state_values = bio["assigned_state"].to_numpy(int)
            for _ in range(cfg.ci_bootstrap_reps):
                idx = sampled_indices(unique, idx_map)
                groups = [endpoint_values[idx[state_values[idx] == k]] for k in range(5)]
                groups = [g[~np.isnan(g)] for g in groups]
                groups = [g for g in groups if len(g)]
                if len(groups) >= 2:
                    eta_vals.append(eta_squared(groups))
                    delta_vals.append(delta_r2(bio.iloc[idx], endpoint))
            mean, lo, hi = ci(eta_vals)
            rows.append({"domain": "biofind", "endpoint": endpoint, "metric": "eta_squared", "mean": mean, "ci_low": lo, "ci_high": hi})
            mean, lo, hi = ci(delta_vals)
            rows.append({"domain": "biofind", "endpoint": endpoint, "metric": "delta_r2", "mean": mean, "ci_low": lo, "ci_high": hi})

    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "prof_review_bootstrap_confidence_intervals.csv", index=False)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "full"], default="full")
    parser.add_argument("--workers", type=int, default=120)
    parser.add_argument("--rf-workers", type=int, default=96)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-reps", type=int, default=None)
    parser.add_argument("--ci-bootstrap-reps", type=int, default=None)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--n-estimators", type=int, default=None)
    args = parser.parse_args()

    cfg = ExpansionConfig(
        mode=args.mode,
        workers=args.workers,
        rf_workers=args.rf_workers,
        seed=args.seed,
        bootstrap_reps=args.bootstrap_reps if args.bootstrap_reps is not None else (32 if args.mode == "smoke" else 250),
        ci_bootstrap_reps=args.ci_bootstrap_reps if args.ci_bootstrap_reps is not None else (100 if args.mode == "smoke" else 1000),
        cv_folds=args.cv_folds,
        n_estimators=args.n_estimators if args.n_estimators is not None else (80 if args.mode == "smoke" else 500),
    )
    log_path = setup_logging()
    start = time.time()
    logging.info("Starting professor-review expansion: %s", cfg)
    df, score_cols, posterior_cols = rp.load_motor_with_assignments()
    df = rp.add_patient_covariates(df)
    cohort_sets = cohort_patient_sets(df)
    logging.info("Cohorts: %s", {k: len(v) for k, v in cohort_sets.items()})

    calib = run_subgroup_calibration(df, score_cols, cohort_sets, cfg)
    sev = severity_sensitivity(df, cfg)
    long = longitudinal_expansion(df, posterior_cols, cfg)
    img = imaging_to_posterior_expansion(df, posterior_cols, cohort_sets, cfg)
    cis = biofind_and_headline_cis(cfg)

    run_summary = {
        "mode": cfg.mode,
        "workers": cfg.workers,
        "rf_workers": cfg.rf_workers,
        "bootstrap_reps": cfg.bootstrap_reps,
        "ci_bootstrap_reps": cfg.ci_bootstrap_reps,
        "cv_folds": cfg.cv_folds,
        "n_estimators": cfg.n_estimators,
        "n_visits": int(len(df)),
        "n_patients": int(df["PATNO"].nunique()),
        "outputs": {
            "calibration_rows": int(len(calib)),
            "severity_sensitivity_rows": int(len(sev)),
            "longitudinal_rows": int(len(long)),
            "imaging_rows": int(len(img)),
            "ci_rows": int(len(cis)),
        },
        "wall_time_seconds": float(time.time() - start),
        "log_path": str(log_path),
    }
    write_json(RESULTS / "prof_review_expansion_run_summary.json", run_summary)
    logging.info("Finished professor-review expansion: %s", run_summary)


if __name__ == "__main__":
    main()
