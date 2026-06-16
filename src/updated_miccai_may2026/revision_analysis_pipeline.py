#!/usr/bin/env python3
"""
Advanced MICCAI revision analysis pipeline.

This script is intentionally scoped to the Updated_MICCAI_May2026 workspace.
It reads source data and prior verified MICCAI outputs, then writes all new
revision outputs to:

  Updated_MICCAI_May2026/results
  Updated_MICCAI_May2026/figures
  Updated_MICCAI_May2026/logs

The implemented analyses directly address the review/meta-review concerns:
patient-level modeling, severity deconfounding, posterior calibration,
imaging-to-posterior soft labels, effect-size-first imaging validation, and
baseline seed stability. The May 2026 extension adds cohort-count auditing,
BioFIND endpoint validation, longitudinal future-state prediction, and
multimodal latent-variable ablations.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from joblib import Parallel, delayed
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import jensenshannon
from scipy import stats

from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    adjusted_rand_score,
    balanced_accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    normalized_mutual_info_score,
    r2_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
from sklearn.model_selection import GroupKFold, KFold
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler

try:
    import statsmodels.api as sm
except Exception:  # pragma: no cover - script should still run partial analyses
    sm = None


WORKSPACE = Path(__file__).resolve().parents[1]
PROJECT = WORKSPACE.parent
DATA = PROJECT / "data"
PPMI = DATA / "PPMI_Full"
BIOFIND = DATA / "BioFIND"
PRIOR_RESULTS = PROJECT / "analysis_results" / "miccai"
RESULTS = WORKSPACE / "results"
FIGURES = WORKSPACE / "figures"
LOGS = WORKSPACE / "logs"
CONFIG_PATH = WORKSPACE / "config" / "revision_config.json"

MOTOR_DOMAINS = {
    "Tremor": [
        "NP3PTRMR",
        "NP3PTRML",
        "NP3KTRMR",
        "NP3KTRML",
        "NP3RTARU",
        "NP3RTALU",
        "NP3RTARL",
        "NP3RTALL",
        "NP3RTALJ",
        "NP3RTCON",
    ],
    "Rigidity": ["NP3RIGN", "NP3RIGRU", "NP3RIGLU", "NP3RIGRL", "NP3RIGLL"],
    "Bradykinesia": [
        "NP3FTAPR",
        "NP3FTAPL",
        "NP3HMOVR",
        "NP3HMOVL",
        "NP3PRSPR",
        "NP3PRSPL",
        "NP3TTAPR",
        "NP3TTAPL",
        "NP3LGAGR",
        "NP3LGAGL",
    ],
    "Axial": ["NP3RISNG", "NP3GAIT", "NP3FRZGT", "NP3PSTBL"],
    "Bulbar": ["NP3SPCH", "NP3FACXP", "NP3POSTR"],
}

PHENO_LABELS = {
    0: "M0: Mod-Trem",
    1: "M1: Sev-Trem",
    2: "M2: Mild-Ax",
    3: "M3: Sev-Ax",
    4: "M4: Mod-Mix",
}

PHENO_COLORS = {
    0: "#2b6cb0",
    1: "#b83232",
    2: "#2f855a",
    3: "#805ad5",
    4: "#dd6b20",
}

FAMILY_MAP = {0: "Tremor", 1: "Tremor", 2: "Mild", 3: "Axial", 4: "Axial"}


@dataclass
class RunConfig:
    mode: str
    steps: List[str]
    seed: int
    workers: int
    rf_workers: int
    parallel_backend: str
    bootstrap_reps: int
    cv_folds: int
    n_estimators: int
    baseline_seeds: int


def setup_logging() -> None:
    for path in [RESULTS, FIGURES, LOGS]:
        path.mkdir(parents=True, exist_ok=True)
    log_path = LOGS / f"revision_pipeline_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )


@contextmanager
def timed_step(name: str, cfg: RunConfig, **metadata):
    meta = " ".join(f"{k}={v}" for k, v in metadata.items())
    logging.info(
        "STEP_START %s workers=%d rf_workers=%d backend=%s %s",
        name,
        cfg.workers,
        cfg.rf_workers,
        cfg.parallel_backend,
        meta,
    )
    start = time.time()
    try:
        yield
    finally:
        logging.info("STEP_END %s wall_time_seconds=%.2f", name, time.time() - start)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)


def find_one(root: Path, pattern: str) -> Path:
    files = sorted(root.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No file matching {pattern} in {root}")
    return files[0]


def best_bgmm_params() -> dict:
    sweep_path = PRIOR_RESULTS / "bgmm_megasweep_20260219_101430.json"
    defaults = {
        "n_components": 5,
        "covariance_type": "tied",
        "weight_concentration_prior_type": "dirichlet_distribution",
        "weight_concentration_prior": 100.0,
        "mean_precision_prior": 1.0,
        "max_iter": 500,
    }
    if not sweep_path.exists():
        return defaults
    try:
        cfg = load_json(sweep_path).get("best_config", {})
    except Exception:
        return defaults
    for key in list(defaults):
        if key in cfg:
            defaults[key] = cfg[key]
    defaults["n_components"] = int(defaults["n_components"])
    defaults["max_iter"] = int(defaults.get("max_iter", 500))
    return defaults


def make_bgmm(seed: int, max_iter: Optional[int] = None) -> BayesianGaussianMixture:
    params = best_bgmm_params()
    if max_iter is not None:
        params["max_iter"] = max_iter
    return BayesianGaussianMixture(
        n_components=params["n_components"],
        covariance_type=params["covariance_type"],
        weight_concentration_prior_type=params["weight_concentration_prior_type"],
        weight_concentration_prior=params["weight_concentration_prior"],
        mean_precision_prior=params["mean_precision_prior"],
        max_iter=params["max_iter"],
        n_init=3,
        random_state=seed,
    )


def parse_date(series: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(series, errors="coerce", format="mixed")
    except TypeError:
        return pd.to_datetime(series, errors="coerce")


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def numeric_frame(frame: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=frame.index)
    for col in cols:
        out[col] = pd.to_numeric(frame[col], errors="coerce") if col in frame.columns else np.nan
    return out


def apply_np3_aliases(frame: pd.DataFrame) -> pd.DataFrame:
    """Handle known BioFIND/PPMI typo variants without changing source files."""
    out = frame.copy()
    rename = {}
    for col in out.columns:
        if col.startswith("PN3"):
            fixed = "NP3" + col[3:]
            if fixed not in out.columns:
                rename[col] = fixed
    if rename:
        out = out.rename(columns=rename)
    return out


def ensure_np3_total(frame: pd.DataFrame, score_cols: Sequence[str]) -> pd.DataFrame:
    out = apply_np3_aliases(frame)
    if "NP3TOT" not in out.columns and "NP3TOT" in score_cols:
        item_cols = [c for c in score_cols if c != "NP3TOT" and c in out.columns]
        out["NP3TOT"] = numeric_frame(out, item_cols).sum(axis=1, min_count=max(1, len(item_cols) // 2))
    return out


def phenotype_distribution(labels: Sequence[int]) -> np.ndarray:
    labels = np.asarray(labels)
    counts = np.array([(labels == i).sum() for i in range(5)], dtype=float)
    if counts.sum() == 0:
        return np.ones(5) / 5
    return counts / counts.sum()


def benjamini_hochberg(p_values: Sequence[float]) -> List[float]:
    p = np.asarray([np.nan if v is None else v for v in p_values], dtype=float)
    out = np.full(len(p), np.nan)
    valid = np.flatnonzero(~np.isnan(p))
    if len(valid) == 0:
        return out.tolist()
    order = valid[np.argsort(p[valid])]
    ranked = p[order] * len(valid) / np.arange(1, len(valid) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out[order] = np.clip(ranked, 0, 1)
    return out.tolist()


def entropy_rows(arr: np.ndarray) -> np.ndarray:
    eps = 1e-12
    return -np.sum(arr * np.log(arr + eps), axis=1)


def normalize_simplex(arr: np.ndarray) -> np.ndarray:
    arr = np.clip(arr, 1e-8, None)
    return arr / arr.sum(axis=1, keepdims=True)


def js_divergence_rows(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    p = normalize_simplex(p)
    q = normalize_simplex(q)
    return np.array([jensenshannon(pi, qi, base=2.0) ** 2 for pi, qi in zip(p, q)])


def kl_divergence_rows(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    p = normalize_simplex(p)
    q = normalize_simplex(q)
    return np.sum(p * np.log((p + 1e-8) / (q + 1e-8)), axis=1)


def load_motor_with_assignments() -> Tuple[pd.DataFrame, List[str], List[str]]:
    motor_path = find_one(PPMI, "MDS-UPDRS_Part_III_*.csv")
    assign_path = PRIOR_RESULTS / "bgmm_posterior_assignments_20260219_101430.csv"
    logging.info("Loading motor data: %s", motor_path)
    logging.info("Loading prior posterior assignments: %s", assign_path)

    motor = read_csv(motor_path)
    score_cols = [c for c in motor.columns if c.startswith("NP3")]

    # This reproduces the row filter/order used by the original BGMM script.
    feature_df = motor[["PATNO", "EVENT_ID", "INFODT", "PDSTATE", "PDMEDYN", "NHY"] + score_cols].dropna(
        subset=score_cols
    )
    feature_df = feature_df.reset_index(drop=True)

    assign = read_csv(assign_path).reset_index(drop=True)
    posterior_cols = [c for c in assign.columns if c.isdigit()]
    if len(assign) != len(feature_df):
        raise ValueError(
            f"Assignment rows ({len(assign)}) do not match filtered motor rows ({len(feature_df)}). "
            "The old assignment file cannot be safely aligned by row order."
        )

    df = pd.concat(
        [
            feature_df,
            assign[["cluster_label", "max_posterior", "flag"] + posterior_cols],
        ],
        axis=1,
    )
    df["PATNO"] = df["PATNO"].astype(int)
    df["visit_date"] = parse_date(df["INFODT"])
    df["visit_order"] = df.groupby("PATNO").cumcount()
    df["severity"] = pd.to_numeric(df.get("NP3TOT", df[score_cols].sum(axis=1)), errors="coerce")

    post = df[posterior_cols].to_numpy(float)
    order = np.sort(post, axis=1)
    df["posterior_gap"] = order[:, -1] - order[:, -2]
    df["posterior_entropy"] = entropy_rows(normalize_simplex(post))
    for domain, items in MOTOR_DOMAINS.items():
        valid = [c for c in items if c in df.columns]
        df[f"domain_{domain}"] = df[valid].mean(axis=1)

    domain_cols = [f"domain_{d}" for d in MOTOR_DOMAINS]
    domain_z = RobustScaler().fit_transform(df[domain_cols])
    for idx, domain in enumerate(MOTOR_DOMAINS):
        df[f"z_{domain}"] = domain_z[:, idx]
    df["profile_tremor_axial"] = df["z_Tremor"] - df["z_Axial"]

    logging.info("Aligned %d visits from %d patients", len(df), df["PATNO"].nunique())
    return df, score_cols, posterior_cols


def add_patient_covariates(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    demo_files = sorted(PPMI.glob("Demographics_*.csv"))
    if demo_files:
        demo = read_csv(demo_files[0])
        keep = [c for c in ["PATNO", "BIRTHDT", "SEX", "GENDER"] if c in demo.columns]
        demo = demo[keep].drop_duplicates("PATNO")
        out = out.merge(demo, on="PATNO", how="left")

    diag_files = sorted(PPMI.glob("PD_Diagnosis_History_*.csv"))
    if diag_files:
        diag = read_csv(diag_files[0])
        keep = [c for c in ["PATNO", "SXDT", "PDDXDT", "DOMSIDE"] if c in diag.columns]
        diag = diag[keep].drop_duplicates("PATNO")
        out = out.merge(diag, on="PATNO", how="left")

    med_files = sorted(PPMI.glob("LEDD_Concomitant_Medication_Log_*.csv"))
    if med_files:
        med = read_csv(med_files[0])
        led_cols = [c for c in med.columns if "LEDD" in c.upper()]
        if led_cols:
            med_summary = med[["PATNO"] + led_cols].copy()
            for c in led_cols:
                med_summary[c] = pd.to_numeric(med_summary[c], errors="coerce")
            med_summary = med_summary.groupby("PATNO")[led_cols].mean().reset_index()
            med_summary["LEDD_mean"] = med_summary[led_cols].mean(axis=1)
            out = out.merge(med_summary[["PATNO", "LEDD_mean"]], on="PATNO", how="left")

    visit_date = out["visit_date"]
    if "BIRTHDT" in out.columns:
        birth = parse_date(out["BIRTHDT"])
        out["age_years"] = (visit_date - birth).dt.days / 365.25
    else:
        out["age_years"] = np.nan

    dx_col = "PDDXDT" if "PDDXDT" in out.columns else "SXDT" if "SXDT" in out.columns else None
    if dx_col:
        dx_date = parse_date(out[dx_col])
        out["disease_duration_years"] = (visit_date - dx_date).dt.days / 365.25
    else:
        out["disease_duration_years"] = np.nan

    sex_col = "SEX" if "SEX" in out.columns else "GENDER" if "GENDER" in out.columns else None
    out["sex_cov"] = out[sex_col].astype(str) if sex_col else "UNK"
    out["pdstate_cov"] = out.get("PDSTATE", pd.Series("UNK", index=out.index)).astype(str)
    out["pdmed_cov"] = out.get("PDMEDYN", pd.Series("UNK", index=out.index)).astype(str)
    return out


def cohort_inclusion_audit(df: pd.DataFrame) -> None:
    logging.info("Running cohort inclusion audit")
    base_keys = ["PATNO"] + (["EVENT_ID"] if "EVENT_ID" in df.columns else [])
    rows = [
        {
            "source_file": "aligned_motor_assignment_table",
            "criterion": "all_complete_np3_rows_aligned_to_original_bgmm_assignments",
            "source_rows": int(len(df)),
            "source_patients": int(df["PATNO"].nunique()),
            "aligned_visits": int(len(df)),
            "aligned_patients": int(df["PATNO"].nunique()),
            "event_aligned_visits": int(len(df)),
            "event_aligned_patients": int(df["PATNO"].nunique()),
            "notes": "This is the exact visit table used for all May 2026 analyses.",
        }
    ]

    def append_filter(source_file: Path, criterion: str, source: pd.DataFrame, mask: pd.Series, notes: str = "") -> None:
        filtered = source.loc[mask.fillna(False)].copy()
        filtered["PATNO"] = pd.to_numeric(filtered["PATNO"], errors="coerce")
        filtered = filtered.dropna(subset=["PATNO"])
        filtered["PATNO"] = filtered["PATNO"].astype(int)
        aligned = df[df["PATNO"].isin(set(filtered["PATNO"]))]
        if "EVENT_ID" in filtered.columns and "EVENT_ID" in df.columns:
            event_keys = filtered[["PATNO", "EVENT_ID"]].drop_duplicates()
            event_aligned = df[base_keys].merge(event_keys, on=["PATNO", "EVENT_ID"], how="inner")
        else:
            event_aligned = aligned
        rows.append(
            {
                "source_file": source_file.name,
                "criterion": criterion,
                "source_rows": int(len(filtered)),
                "source_patients": int(filtered["PATNO"].nunique()),
                "aligned_visits": int(len(aligned)),
                "aligned_patients": int(aligned["PATNO"].nunique()),
                "event_aligned_visits": int(len(event_aligned)),
                "event_aligned_patients": int(event_aligned["PATNO"].nunique()),
                "notes": notes,
            }
        )

    status_files = sorted(PPMI.glob("Participant_Status_*.csv"))
    if status_files:
        status = read_csv(status_files[0])
        status["PATNO"] = pd.to_numeric(status["PATNO"], errors="coerce")
        cohort_col = next((c for c in status.columns if c.upper() == "COHORT_DEFINITION"), None)
        enroll_col = next((c for c in status.columns if "ENROLL" in c.upper() and "STATUS" in c.upper()), None)
        if cohort_col:
            cohort_text = status[cohort_col].astype(str).str.strip().str.lower()
            pd_mask = cohort_text.eq("parkinson's disease") | cohort_text.eq("parkinson disease")
            append_filter(
                status_files[0],
                "COHORT_DEFINITION == Parkinson's Disease",
                status,
                pd_mask,
                "Patient-level filter from PPMI participant status.",
            )
        if enroll_col:
            enrolled_mask = status[enroll_col].astype(str).str.contains("enrolled", case=False, na=False)
            append_filter(
                status_files[0],
                f"{enroll_col} contains Enrolled",
                status,
                enrolled_mask,
                "Enrollment status filter; includes non-PD enrolled cohorts.",
            )
        if cohort_col and enroll_col:
            append_filter(
                status_files[0],
                "Parkinson's Disease cohort and enrolled",
                status,
                pd_mask & enrolled_mask,
                "Intersection of PPMI status cohort and enrollment status.",
            )

    diagnostic_specs = [
        ("Primary_Clinical_Diagnosis_*.csv", "PRIMDIAG", 1, "current_primary_clinical_diagnosis_PD"),
        ("Clinical_Diagnosis_*.csv", "NEWDIAG", 1, "current_clinical_diagnosis_PD"),
        ("Primary_Diagnosis-Archived_*.csv", "PRIMDIAG", 1, "archived_primary_diagnosis_PD"),
    ]
    for pattern, col, value, label in diagnostic_specs:
        for path in sorted(PPMI.glob(pattern))[:1]:
            table = read_csv(path)
            if "PATNO" not in table.columns or col not in table.columns:
                continue
            mask = pd.to_numeric(table[col], errors="coerce").eq(value)
            append_filter(
                path,
                f"{col} == {value} ({label})",
                table,
                mask,
                "Both patient-level and EVENT_ID-aligned counts are reported when EVENT_ID exists.",
            )

    audit = pd.DataFrame(rows)
    audit.to_csv(RESULTS / "cohort_inclusion_audit.csv", index=False)
    write_json(
        RESULTS / "cohort_inclusion_audit.json",
        {
            "n_audit_rows": int(len(audit)),
            "aligned_motor_visits": int(len(df)),
            "aligned_motor_patients": int(df["PATNO"].nunique()),
            "audit": audit.to_dict(orient="records"),
        },
    )


def patient_state_analysis(df: pd.DataFrame, posterior_cols: List[str]) -> None:
    logging.info("Running patient-level posterior state analysis")
    rows = []
    transition = np.zeros((5, 5), dtype=int)
    dwell_lengths: List[int] = []

    for patno, group in df.sort_values(["PATNO", "visit_date", "EVENT_ID"]).groupby("PATNO"):
        states = group["cluster_label"].to_numpy(int)
        families = np.array([FAMILY_MAP[int(s)] for s in states])
        modal_state = int(pd.Series(states).mode().iloc[0])
        modal_family = pd.Series(families).mode().iloc[0]
        modal_fraction = float(np.mean(states == modal_state))
        family_fraction = float(np.mean(families == modal_family))
        post = group[posterior_cols].to_numpy(float)
        mean_post = normalize_simplex(post.mean(axis=0, keepdims=True))[0]

        if len(states) > 1:
            for a, b in zip(states[:-1], states[1:]):
                transition[a, b] += 1
            run_len = 1
            for a, b in zip(states[:-1], states[1:]):
                if a == b:
                    run_len += 1
                else:
                    dwell_lengths.append(run_len)
                    run_len = 1
            dwell_lengths.append(run_len)

        row = {
            "PATNO": int(patno),
            "n_visits": int(len(group)),
            "modal_state": modal_state,
            "modal_label": PHENO_LABELS[modal_state],
            "modal_state_fraction": modal_fraction,
            "modal_family": modal_family,
            "modal_family_fraction": family_fraction,
            "first_state": int(states[0]),
            "last_state": int(states[-1]),
            "ever_transitioned_state": bool(len(np.unique(states)) > 1),
            "posterior_entropy_patient": float(entropy_rows(mean_post.reshape(1, -1))[0]),
            "mean_max_posterior": float(group["max_posterior"].mean()),
            "mean_posterior_gap": float(group["posterior_gap"].mean()),
        }
        for idx, col in enumerate(posterior_cols):
            row[f"patient_pi_{col}"] = float(mean_post[idx])
        rows.append(row)

    summary = pd.DataFrame(rows)
    summary.to_csv(RESULTS / "patient_state_summary.csv", index=False)

    transition_df = pd.DataFrame(
        transition,
        index=[PHENO_LABELS[i] for i in range(5)],
        columns=[PHENO_LABELS[i] for i in range(5)],
    )
    transition_df.to_csv(RESULTS / "state_transition_counts.csv")
    transition_prob = transition / np.maximum(transition.sum(axis=1, keepdims=True), 1)
    pd.DataFrame(
        transition_prob,
        index=transition_df.index,
        columns=transition_df.columns,
    ).to_csv(RESULTS / "state_transition_matrix.csv")

    fig, ax = plt.subplots(figsize=(6.5, 5.2), dpi=180)
    im = ax.imshow(transition_prob, cmap="viridis", vmin=0, vmax=max(0.01, transition_prob.max()))
    ax.set_xticks(range(5), [PHENO_LABELS[i].split(": ")[1] for i in range(5)], rotation=35, ha="right")
    ax.set_yticks(range(5), [PHENO_LABELS[i].split(": ")[1] for i in range(5)])
    ax.set_title("Patient-Level Visit-to-Visit State Transition Matrix")
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{transition_prob[i, j]:.2f}", ha="center", va="center", color="white", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(FIGURES / "state_transition_heatmap.png", bbox_inches="tight")
    plt.close(fig)

    payload = {
        "n_patients": int(summary.shape[0]),
        "n_patients_ge_5_visits": int((summary["n_visits"] >= 5).sum()),
        "mean_modal_state_fraction": float(summary["modal_state_fraction"].mean()),
        "mean_modal_family_fraction": float(summary["modal_family_fraction"].mean()),
        "pct_ever_transitioned_state": float(summary["ever_transitioned_state"].mean()),
        "mean_dwell_visits": float(np.mean(dwell_lengths)) if dwell_lengths else None,
    }
    write_json(RESULTS / "patient_state_metrics.json", payload)


def build_longitudinal_prediction_rows(df: pd.DataFrame, posterior_cols: List[str]) -> pd.DataFrame:
    rows = []
    sort_cols = ["PATNO", "visit_date", "EVENT_ID"]
    for patno, group in df.sort_values(sort_cols).groupby("PATNO"):
        if len(group) < 2:
            continue
        group = group.reset_index(drop=True)
        for idx in range(len(group) - 1):
            cur = group.iloc[idx]
            nxt = group.iloc[idx + 1]
            row = {
                "PATNO": int(patno),
                "EVENT_ID": cur.get("EVENT_ID"),
                "next_EVENT_ID": nxt.get("EVENT_ID"),
                "visit_date": cur.get("visit_date"),
                "next_visit_date": nxt.get("visit_date"),
                "time_gap_days": float((nxt.get("visit_date") - cur.get("visit_date")).days)
                if pd.notna(cur.get("visit_date")) and pd.notna(nxt.get("visit_date"))
                else np.nan,
                "current_state": int(cur["cluster_label"]),
                "next_state": int(nxt["cluster_label"]),
                "state_transition": int(cur["cluster_label"] != nxt["cluster_label"]),
                "current_axial": float(cur["z_Axial"]),
                "next_axial": float(nxt["z_Axial"]),
                "delta_axial": float(nxt["z_Axial"] - cur["z_Axial"]),
                "axial_worsening": int(nxt["z_Axial"] > cur["z_Axial"]),
                "severity": float(cur["severity"]),
                "next_severity": float(nxt["severity"]),
                "delta_severity": float(nxt["severity"] - cur["severity"]),
                "posterior_entropy": float(cur["posterior_entropy"]),
                "posterior_gap": float(cur["posterior_gap"]),
                "max_posterior": float(cur["max_posterior"]),
                "profile_tremor_axial": float(cur["profile_tremor_axial"]),
                "age_years": float(pd.to_numeric(cur.get("age_years"), errors="coerce")),
                "disease_duration_years": float(pd.to_numeric(cur.get("disease_duration_years"), errors="coerce")),
                "LEDD_mean": float(pd.to_numeric(cur.get("LEDD_mean"), errors="coerce")),
            }
            for domain in MOTOR_DOMAINS:
                row[f"z_{domain}"] = float(cur[f"z_{domain}"])
            for col in posterior_cols:
                row[f"pi_{col}"] = float(cur[col])
            rows.append(row)
    return pd.DataFrame(rows)


def _future_regression_fold(
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    X: pd.DataFrame,
    y: np.ndarray,
    seed: int,
    n_estimators: int,
    rf_n_jobs: int,
) -> Tuple[np.ndarray, np.ndarray]:
    pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=n_estimators,
                    min_samples_leaf=4,
                    random_state=seed,
                    n_jobs=rf_n_jobs,
                ),
            ),
        ]
    )
    pipe.fit(X.iloc[train_idx], y[train_idx])
    return test_idx, pipe.predict(X.iloc[test_idx])


def _future_classification_fold(
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    X: pd.DataFrame,
    y: np.ndarray,
    seed: int,
    n_estimators: int,
    rf_n_jobs: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    min_samples_leaf=4,
                    class_weight="balanced_subsample",
                    random_state=seed,
                    n_jobs=rf_n_jobs,
                ),
            ),
        ]
    )
    pipe.fit(X.iloc[train_idx], y[train_idx])
    proba = pipe.predict_proba(X.iloc[test_idx])
    classes = pipe.named_steps["model"].classes_
    pos = np.zeros(len(test_idx), dtype=float)
    if 1 in classes:
        pos = proba[:, list(classes).index(1)]
    return test_idx, pipe.predict(X.iloc[test_idx]), pos


def longitudinal_predictive_model(
    df: pd.DataFrame,
    posterior_cols: List[str],
    cfg: RunConfig,
) -> None:
    logging.info("Running longitudinal future-state predictive model")
    table = build_longitudinal_prediction_rows(df, posterior_cols)
    table.to_csv(RESULTS / "longitudinal_prediction_rows.csv", index=False)
    if table.empty or table["PATNO"].nunique() < 5:
        write_json(
            RESULTS / "longitudinal_predictive_model_summary.json",
            {"error": "Insufficient longitudinal rows for patient-held-out prediction."},
        )
        return

    numeric_features = [
        "current_axial",
        "severity",
        "profile_tremor_axial",
        "posterior_entropy",
        "posterior_gap",
        "max_posterior",
        "time_gap_days",
        "age_years",
        "disease_duration_years",
        "LEDD_mean",
    ] + [f"z_{d}" for d in MOTOR_DOMAINS] + [f"pi_{c}" for c in posterior_cols]
    feature_table = table[numeric_features].replace([np.inf, -np.inf], np.nan)
    state_dummies = pd.get_dummies(table["current_state"].astype("category"), prefix="state")
    X = pd.concat([feature_table, state_dummies], axis=1)
    feature_cols = list(X.columns)

    groups = table["PATNO"].to_numpy(int)
    n_splits = min(cfg.cv_folds, len(np.unique(groups)))
    if n_splits < 2:
        write_json(
            RESULTS / "longitudinal_predictive_model_summary.json",
            {"error": "Fewer than two patient groups available for GroupKFold."},
        )
        return

    splits = list(GroupKFold(n_splits=n_splits).split(X, table["next_axial"].to_numpy(float), groups))
    fold_jobs = max(1, min(cfg.workers, len(splits)))
    rf_n_jobs = max(1, min(cfg.rf_workers, max(1, cfg.workers // fold_jobs)))
    logging.info(
        "Longitudinal CV: rows=%d patients=%d folds=%d fold_jobs=%d rf_n_jobs_per_fold=%d",
        len(table),
        table["PATNO"].nunique(),
        len(splits),
        fold_jobs,
        rf_n_jobs,
    )

    y_next_axial = table["next_axial"].to_numpy(float)
    regression_preds = np.zeros(len(table), dtype=float)
    reg_results = Parallel(n_jobs=fold_jobs, backend=cfg.parallel_backend)(
        delayed(_future_regression_fold)(
            train_idx,
            test_idx,
            X,
            y_next_axial,
            cfg.seed,
            cfg.n_estimators,
            rf_n_jobs,
        )
        for train_idx, test_idx in splits
    )
    for test_idx, pred in reg_results:
        regression_preds[test_idx] = pred

    baseline_next_axial = table["current_axial"].to_numpy(float)
    y_transition = table["state_transition"].to_numpy(int)
    cls_pred = np.zeros(len(table), dtype=int)
    cls_proba = np.zeros(len(table), dtype=float)
    if len(np.unique(y_transition)) > 1:
        cls_results = Parallel(n_jobs=fold_jobs, backend=cfg.parallel_backend)(
            delayed(_future_classification_fold)(
                train_idx,
                test_idx,
                X,
                y_transition,
                cfg.seed,
                cfg.n_estimators,
                rf_n_jobs,
            )
            for train_idx, test_idx in splits
        )
        for test_idx, pred, proba in cls_results:
            cls_pred[test_idx] = pred
            cls_proba[test_idx] = proba

    table["cv_pred_next_axial"] = regression_preds
    table["cv_pred_state_transition"] = cls_pred
    table["cv_proba_state_transition"] = cls_proba
    table.to_csv(RESULTS / "longitudinal_prediction_rows.csv", index=False)

    final_reg = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=cfg.n_estimators,
                    min_samples_leaf=4,
                    random_state=cfg.seed,
                    n_jobs=max(1, min(cfg.rf_workers, cfg.workers)),
                ),
            ),
        ]
    )
    final_reg.fit(X, y_next_axial)
    reg_imp = final_reg.named_steps["model"].feature_importances_

    final_cls_imp = np.zeros(len(feature_cols), dtype=float)
    if len(np.unique(y_transition)) > 1:
        final_cls = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", RobustScaler()),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=cfg.n_estimators,
                        min_samples_leaf=4,
                        class_weight="balanced_subsample",
                        random_state=cfg.seed,
                        n_jobs=max(1, min(cfg.rf_workers, cfg.workers)),
                    ),
                ),
            ]
        )
        final_cls.fit(X, y_transition)
        final_cls_imp = final_cls.named_steps["model"].feature_importances_

    importance = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance_next_axial": reg_imp,
            "importance_state_transition": final_cls_imp,
        }
    ).sort_values("importance_next_axial", ascending=False)
    importance.to_csv(RESULTS / "longitudinal_feature_importance.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.2, 5.0), dpi=180)
    top = importance.head(15).sort_values("importance_next_axial")
    ax.barh(top["feature"], top["importance_next_axial"], color="#805ad5")
    ax.set_xlabel("Random Forest Importance")
    ax.set_title("Future Axial Burden Prediction")
    fig.tight_layout()
    fig.savefig(FIGURES / "longitudinal_predictive_importance.png", bbox_inches="tight")
    plt.close(fig)

    def maybe_auc(y_true: np.ndarray, score: np.ndarray) -> Optional[float]:
        if len(np.unique(y_true)) < 2:
            return None
        try:
            return float(roc_auc_score(y_true, score))
        except Exception:
            return None

    summary = {
        "n_prediction_rows": int(len(table)),
        "n_patients": int(table["PATNO"].nunique()),
        "cv_folds": int(n_splits),
        "next_axial_mae_model": float(mean_absolute_error(y_next_axial, regression_preds)),
        "next_axial_rmse_model": float(math.sqrt(mean_squared_error(y_next_axial, regression_preds))),
        "next_axial_r2_model": float(r2_score(y_next_axial, regression_preds)),
        "next_axial_mae_autoregressive_baseline": float(mean_absolute_error(y_next_axial, baseline_next_axial)),
        "next_axial_rmse_autoregressive_baseline": float(math.sqrt(mean_squared_error(y_next_axial, baseline_next_axial))),
        "next_axial_r2_autoregressive_baseline": float(r2_score(y_next_axial, baseline_next_axial)),
        "state_transition_rate": float(y_transition.mean()),
        "state_transition_balanced_accuracy": float(balanced_accuracy_score(y_transition, cls_pred))
        if len(np.unique(y_transition)) > 1
        else None,
        "state_transition_auc_model": maybe_auc(y_transition, cls_proba),
        "state_transition_auc_entropy_only": maybe_auc(y_transition, table["posterior_entropy"].to_numpy(float)),
        "state_transition_auc_inverse_gap_only": maybe_auc(
            y_transition, -table["posterior_gap"].to_numpy(float)
        ),
        "top_next_axial_features": importance.head(10)["feature"].tolist(),
        "top_state_transition_features": importance.sort_values(
            "importance_state_transition", ascending=False
        )
        .head(10)["feature"]
        .tolist(),
    }
    write_json(RESULTS / "longitudinal_predictive_model_summary.json", summary)


def build_design_matrix(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    use = pd.DataFrame(index=df.index)
    for c in ["severity", "age_years", "disease_duration_years", "LEDD_mean"]:
        if c in df.columns:
            use[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ["sex_cov", "pdstate_cov", "pdmed_cov"]:
        if c in df.columns:
            use[c] = df[c].astype(str)
    use = pd.get_dummies(use, columns=[c for c in use.columns if use[c].dtype == object], dummy_na=True)
    use = use.replace([np.inf, -np.inf], np.nan)
    use = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(use), columns=use.columns, index=df.index)
    return use, list(use.columns)


def severity_deconfounding(df: pd.DataFrame, seed: int) -> None:
    logging.info("Running severity deconfounding")
    domain_cols = [f"z_{d}" for d in MOTOR_DOMAINS]
    covariates, covariate_cols = build_design_matrix(df)
    residuals = pd.DataFrame(index=df.index)
    r2_by_domain = {}

    for col in domain_cols:
        y = pd.to_numeric(df[col], errors="coerce").to_numpy(float)
        model = LinearRegression()
        model.fit(covariates, y)
        pred = model.predict(covariates)
        residuals[col.replace("z_", "resid_")] = y - pred
        ss_res = float(np.sum((y - pred) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2_by_domain[col] = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    X_resid = RobustScaler().fit_transform(residuals)
    bgmm = make_bgmm(seed)
    resid_labels = bgmm.fit_predict(X_resid)
    ari = adjusted_rand_score(df["cluster_label"], resid_labels)
    nmi = normalized_mutual_info_score(df["cluster_label"], resid_labels)

    output = df[["PATNO", "EVENT_ID", "cluster_label", "severity", "profile_tremor_axial"]].copy()
    output["residual_cluster"] = resid_labels
    for c in residuals.columns:
        output[c] = residuals[c]
    output.to_csv(RESULTS / "severity_residualized_visit_profiles.csv", index=False)

    # Matched M1 severe-tremor versus M3 severe-axial visits.
    matched = severity_matched_pairs(df)
    matched.to_csv(RESULTS / "severity_matched_pairs_M1_M3.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.2, 5.2), dpi=180)
    plot_df = df.sample(n=min(len(df), 9000), random_state=seed) if len(df) > 9000 else df
    for label in sorted(plot_df["cluster_label"].unique()):
        sub = plot_df[plot_df["cluster_label"] == label]
        ax.scatter(
            sub["severity"],
            sub["profile_tremor_axial"],
            s=8,
            alpha=0.35,
            color=PHENO_COLORS.get(int(label), "gray"),
            label=PHENO_LABELS.get(int(label), str(label)),
        )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Global Motor Severity (MDS-UPDRS-III Total)")
    ax.set_ylabel("Tremor minus Axial Profile Contrast")
    ax.set_title("Severity/Profile Disentanglement")
    ax.legend(fontsize=7, ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "severity_profile_disentanglement.png", bbox_inches="tight")
    plt.close(fig)

    payload = {
        "n_visits": int(len(df)),
        "residual_bgmm_effective_k": int(np.sum(bgmm.weights_ > 0.01)),
        "raw_vs_residual_ari": float(ari),
        "raw_vs_residual_nmi": float(nmi),
        "domain_r2_explained_by_severity_covariates": r2_by_domain,
        "covariates": covariate_cols,
        "n_matched_M1_M3_pairs": int(len(matched)),
    }
    write_json(RESULTS / "severity_deconfounding_summary.json", payload)


def severity_matched_pairs(df: pd.DataFrame) -> pd.DataFrame:
    patient_rows = (
        df.sort_values(["PATNO", "visit_date"])
        .groupby("PATNO")
        .agg(
            cluster_label=("cluster_label", lambda x: int(pd.Series(x).mode().iloc[0])),
            severity=("severity", "mean"),
            age_years=("age_years", "mean"),
            disease_duration_years=("disease_duration_years", "mean"),
            profile_tremor_axial=("profile_tremor_axial", "mean"),
            sex_cov=("sex_cov", lambda x: str(pd.Series(x).mode().iloc[0])),
        )
        .reset_index()
    )
    a = patient_rows[patient_rows["cluster_label"] == 1].copy()
    b = patient_rows[patient_rows["cluster_label"] == 3].copy()
    if a.empty or b.empty:
        return pd.DataFrame()

    covars = ["severity", "age_years", "disease_duration_years"]
    for c in covars:
        a[c] = pd.to_numeric(a[c], errors="coerce")
        b[c] = pd.to_numeric(b[c], errors="coerce")
    a = a.dropna(subset=["severity"])
    b = b.dropna(subset=["severity"])
    for frame in [a, b]:
        for c in covars:
            frame[c] = frame[c].fillna(patient_rows[c].median())

    scaler = StandardScaler()
    combined = pd.concat([a[covars], b[covars]], axis=0)
    scaler.fit(combined)
    a_x = scaler.transform(a[covars])
    b_x = scaler.transform(b[covars])

    if len(a) <= len(b):
        source, target, sx, tx = a, b, a_x, b_x
    else:
        source, target, sx, tx = b, a, b_x, a_x

    nn = NearestNeighbors(n_neighbors=1).fit(tx)
    dist, idx = nn.kneighbors(sx)
    rows = []
    used = set()
    for src_i, tgt_i in enumerate(idx[:, 0]):
        if int(tgt_i) in used:
            continue
        used.add(int(tgt_i))
        left = source.iloc[src_i]
        right = target.iloc[int(tgt_i)]
        rows.append(
            {
                "PATNO_source": int(left["PATNO"]),
                "source_state": int(left["cluster_label"]),
                "PATNO_target": int(right["PATNO"]),
                "target_state": int(right["cluster_label"]),
                "match_distance": float(dist[src_i, 0]),
                "severity_diff": float(left["severity"] - right["severity"]),
                "profile_contrast_diff": float(left["profile_tremor_axial"] - right["profile_tremor_axial"]),
            }
        )
    return pd.DataFrame(rows)


def _bootstrap_consistency_one(
    seed: int,
    X: np.ndarray,
    patient_ids: np.ndarray,
    original_labels: np.ndarray,
    unique_patients: np.ndarray,
) -> np.ndarray:
    rng = np.random.RandomState(seed)
    sampled_patients = rng.choice(unique_patients, size=len(unique_patients), replace=True)
    boot_indices = np.concatenate([np.flatnonzero(patient_ids == pat) for pat in sampled_patients])
    model = make_bgmm(seed, max_iter=300)
    model.fit(X[boot_indices])
    pred = model.predict(X)
    confusion = np.zeros((5, 5), dtype=int)
    for o, p in zip(original_labels, pred):
        if 0 <= int(o) < 5 and 0 <= int(p) < 5:
            confusion[int(o), int(p)] += 1
    row_ind, col_ind = linear_sum_assignment(-confusion)
    mapping = {int(c): int(r) for r, c in zip(row_ind, col_ind)}
    mapped = np.array([mapping.get(int(p), -1) for p in pred])
    return (mapped == original_labels).astype(float)


def posterior_calibration(
    df: pd.DataFrame,
    score_cols: List[str],
    reps: int,
    workers: int,
    seed: int,
    backend: str,
) -> None:
    n_jobs = max(1, min(workers, reps))
    logging.info("Running posterior bootstrap calibration with %d reps using n_jobs=%d", reps, n_jobs)
    X = RobustScaler().fit_transform(df[score_cols].to_numpy(float))
    patient_ids = df["PATNO"].to_numpy(int)
    original = df["cluster_label"].to_numpy(int)
    unique_patients = np.unique(patient_ids)

    seeds = [seed + 1000 + i for i in range(reps)]
    consistencies = Parallel(n_jobs=n_jobs, backend=backend)(
        delayed(_bootstrap_consistency_one)(s, X, patient_ids, original, unique_patients) for s in seeds
    )
    consistency = np.vstack(consistencies).mean(axis=0)
    conf = df["max_posterior"].to_numpy(float)
    bins = np.linspace(0.0, 1.0, 11)
    rows = []
    ece = 0.0
    n = len(df)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (conf >= lo) & (conf < hi if hi < 1 else conf <= hi)
        if not mask.any():
            continue
        mean_conf = float(conf[mask].mean())
        mean_consistency = float(consistency[mask].mean())
        weight = float(mask.mean())
        ece += weight * abs(mean_conf - mean_consistency)
        rows.append(
            {
                "bin_low": lo,
                "bin_high": hi,
                "n": int(mask.sum()),
                "mean_posterior_confidence": mean_conf,
                "bootstrap_label_consistency": mean_consistency,
                "abs_gap": abs(mean_conf - mean_consistency),
            }
        )
    calib = pd.DataFrame(rows)
    calib.to_csv(RESULTS / "posterior_calibration_bins.csv", index=False)
    brier_like = float(np.mean((conf - consistency) ** 2))

    fig, ax = plt.subplots(figsize=(5.8, 4.8), dpi=180)
    ax.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1)
    if not calib.empty:
        ax.plot(
            calib["mean_posterior_confidence"],
            calib["bootstrap_label_consistency"],
            marker="o",
            color="#2b6cb0",
        )
    ax.set_xlabel("Mean BGMM Posterior Confidence")
    ax.set_ylabel("Bootstrap Label Consistency")
    ax.set_title("Posterior Calibration by Patient-Blocked Bootstrap")
    fig.tight_layout()
    fig.savefig(FIGURES / "posterior_calibration_curve.png", bbox_inches="tight")
    plt.close(fig)

    write_json(
        RESULTS / "posterior_calibration_summary.json",
        {
            "n_visits": n,
            "bootstrap_reps": reps,
            "ece": float(ece),
            "brier_like_score": brier_like,
            "mean_bootstrap_consistency": float(consistency.mean()),
            "mean_posterior_confidence": float(conf.mean()),
            "mean_entropy": float(df["posterior_entropy"].mean()),
            "mean_gap": float(df["posterior_gap"].mean()),
        },
    )


def load_datscan_features() -> Tuple[pd.DataFrame, List[str]]:
    dat_path = find_one(PPMI, "DaTScan_SBR_Analysis_*.csv")
    dat = read_csv(dat_path)
    dat = dat[dat.get("DATSCAN_ANALYZED", "Yes") == "Yes"].copy()
    numeric = [
        "DATSCAN_CAUDATE_R",
        "DATSCAN_CAUDATE_L",
        "DATSCAN_PUTAMEN_R",
        "DATSCAN_PUTAMEN_L",
        "DATSCAN_PUTAMEN_R_ANT",
        "DATSCAN_PUTAMEN_L_ANT",
    ]
    for c in numeric:
        if c in dat.columns:
            dat[c] = pd.to_numeric(dat[c], errors="coerce")
    dat = dat.dropna(subset=["DATSCAN_CAUDATE_R", "DATSCAN_CAUDATE_L", "DATSCAN_PUTAMEN_R", "DATSCAN_PUTAMEN_L"])
    dat = dat.sort_values(["PATNO", "DATSCAN_DATE"]).groupby("PATNO").first().reset_index()

    eps = 1e-8
    dat["caudate_mean"] = (dat["DATSCAN_CAUDATE_R"] + dat["DATSCAN_CAUDATE_L"]) / 2
    dat["putamen_mean"] = (dat["DATSCAN_PUTAMEN_R"] + dat["DATSCAN_PUTAMEN_L"]) / 2
    dat["total_sbr"] = (dat["caudate_mean"] + dat["putamen_mean"]) / 2
    dat["AI_caudate"] = (dat["DATSCAN_CAUDATE_R"] - dat["DATSCAN_CAUDATE_L"]) / (
        dat["DATSCAN_CAUDATE_R"] + dat["DATSCAN_CAUDATE_L"] + eps
    )
    dat["AI_putamen"] = (dat["DATSCAN_PUTAMEN_R"] - dat["DATSCAN_PUTAMEN_L"]) / (
        dat["DATSCAN_PUTAMEN_R"] + dat["DATSCAN_PUTAMEN_L"] + eps
    )
    dat["AI_abs_caudate"] = dat["AI_caudate"].abs()
    dat["AI_abs_putamen"] = dat["AI_putamen"].abs()
    dat["CP_ratio_mean"] = (
        dat["DATSCAN_CAUDATE_R"] / (dat["DATSCAN_PUTAMEN_R"] + eps)
        + dat["DATSCAN_CAUDATE_L"] / (dat["DATSCAN_PUTAMEN_L"] + eps)
    ) / 2
    if "DATSCAN_PUTAMEN_R_ANT" in dat.columns and "DATSCAN_PUTAMEN_L_ANT" in dat.columns:
        dat["ant_putamen_mean"] = (dat["DATSCAN_PUTAMEN_R_ANT"] + dat["DATSCAN_PUTAMEN_L_ANT"]) / 2
        dat["AI_ant_putamen"] = (dat["DATSCAN_PUTAMEN_R_ANT"] - dat["DATSCAN_PUTAMEN_L_ANT"]) / (
            dat["DATSCAN_PUTAMEN_R_ANT"] + dat["DATSCAN_PUTAMEN_L_ANT"] + eps
        )
        dat["AI_abs_ant_putamen"] = dat["AI_ant_putamen"].abs()

    cols = [
        "caudate_mean",
        "putamen_mean",
        "total_sbr",
        "AI_caudate",
        "AI_putamen",
        "AI_abs_caudate",
        "AI_abs_putamen",
        "CP_ratio_mean",
        "ant_putamen_mean",
        "AI_ant_putamen",
        "AI_abs_ant_putamen",
        "DATSCAN_CAUDATE_R",
        "DATSCAN_CAUDATE_L",
        "DATSCAN_PUTAMEN_R",
        "DATSCAN_PUTAMEN_L",
        "DATSCAN_PUTAMEN_R_ANT",
        "DATSCAN_PUTAMEN_L_ANT",
    ]
    cols = [c for c in cols if c in dat.columns]
    return dat[["PATNO"] + cols].copy(), cols


def load_mri_features() -> Tuple[pd.DataFrame, List[str]]:
    aseg_path = find_one(PPMI, "FS7_ASEG_VOL_*.csv")
    aseg = read_csv(aseg_path)
    if "EVENT_ID" in aseg.columns:
        bl = aseg[aseg["EVENT_ID"].isin(["BL", "SC"])].copy()
        bl["_event_rank"] = bl["EVENT_ID"].map({"BL": 0, "SC": 1}).fillna(2)
        aseg = bl.sort_values(["PATNO", "_event_rank"]).groupby("PATNO").first().reset_index()
    else:
        aseg = aseg.groupby("PATNO").first().reset_index()

    rois = [
        "Left_Putamen",
        "Right_Putamen",
        "Left_Caudate",
        "Right_Caudate",
        "Left_Pallidum",
        "Right_Pallidum",
        "Left_Hippocampus",
        "Right_Hippocampus",
        "Left_Amygdala",
        "Right_Amygdala",
        "Left_Thalamus",
        "Right_Thalamus",
        "Brain_Stem",
        "Left_Lateral_Ventricle",
        "Right_Lateral_Ventricle",
        "3rd_Ventricle",
        "4th_Ventricle",
        "Left_VentralDC",
        "Right_VentralDC",
        "Left_Accumbens_area",
        "Right_Accumbens_area",
    ]
    valid = [r for r in rois if r in aseg.columns]
    etiv = (
        pd.to_numeric(aseg["EstimatedTotalIntraCranialVol"], errors="coerce")
        if "EstimatedTotalIntraCranialVol" in aseg.columns
        else None
    )
    feature_cols = []
    for roi in valid:
        aseg[roi] = pd.to_numeric(aseg[roi], errors="coerce")
        col = f"{roi}_norm"
        if etiv is not None:
            aseg[col] = aseg[roi] / etiv * 1000.0
        else:
            aseg[col] = aseg[roi]
        feature_cols.append(col)
    return aseg[["PATNO"] + feature_cols].copy(), feature_cols


def patient_level_targets(df: pd.DataFrame, posterior_cols: List[str]) -> pd.DataFrame:
    rows = []
    for patno, group in df.groupby("PATNO"):
        post = normalize_simplex(group[posterior_cols].to_numpy(float).mean(axis=0, keepdims=True))[0]
        row = {
            "PATNO": int(patno),
            "hard_state": int(group["cluster_label"].mode().iloc[0]),
            "severity": float(group["severity"].mean()),
            "age_years": float(pd.to_numeric(group["age_years"], errors="coerce").mean()),
            "disease_duration_years": float(pd.to_numeric(group["disease_duration_years"], errors="coerce").mean()),
        }
        for i, col in enumerate(posterior_cols):
            row[f"pi_{col}"] = float(post[i])
        rows.append(row)
    return pd.DataFrame(rows)


def _fit_predict_soft_fold(
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    X: pd.DataFrame,
    y: np.ndarray,
    seed: int,
    n_estimators: int,
    rf_n_jobs: int,
) -> Tuple[np.ndarray, np.ndarray]:
    pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=n_estimators,
                    min_samples_leaf=3,
                    random_state=seed,
                    n_jobs=rf_n_jobs,
                ),
            ),
        ]
    )
    pipe.fit(X.iloc[train_idx], y[train_idx])
    return test_idx, pipe.predict(X.iloc[test_idx])


def evaluate_soft_posterior_model(
    name: str,
    data: pd.DataFrame,
    feature_cols: List[str],
    target_cols: List[str],
    folds: int,
    seed: int,
    n_estimators: int,
    workers: int,
    rf_workers: int,
    backend: str,
) -> Tuple[dict, pd.DataFrame]:
    data = data.dropna(subset=target_cols).copy()
    X = data[feature_cols].replace([np.inf, -np.inf], np.nan)
    y = normalize_simplex(data[target_cols].to_numpy(float))
    hard = np.argmax(y, axis=1)

    kf = KFold(n_splits=min(folds, len(data)), shuffle=True, random_state=seed)
    splits = list(kf.split(X))
    fold_jobs = max(1, min(workers, len(splits)))
    rf_n_jobs = max(1, min(rf_workers, max(1, workers // fold_jobs)))
    logging.info(
        "Imaging model %s: n=%d features=%d folds=%d fold_jobs=%d rf_n_jobs_per_fold=%d",
        name,
        len(data),
        len(feature_cols),
        len(splits),
        fold_jobs,
        rf_n_jobs,
    )
    pred = np.zeros_like(y)
    fold_predictions = Parallel(n_jobs=fold_jobs, backend=backend)(
        delayed(_fit_predict_soft_fold)(train_idx, test_idx, X, y, seed, n_estimators, rf_n_jobs)
        for train_idx, test_idx in splits
    )
    for test_idx, fold_pred in fold_predictions:
        pred[test_idx] = fold_pred
    pred = normalize_simplex(pred)

    try:
        macro_auc = roc_auc_score(hard, pred, multi_class="ovr", average="macro")
    except Exception:
        macro_auc = np.nan

    pred_class = np.argmax(pred, axis=1)
    confidence = pred.max(axis=1)
    soft_accuracy = y[np.arange(len(y)), pred_class]
    ece = 0.0
    for lo, hi in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:]):
        mask = (confidence >= lo) & (confidence < hi if hi < 1 else confidence <= hi)
        if mask.any():
            ece += float(mask.mean()) * abs(float(confidence[mask].mean()) - float(soft_accuracy[mask].mean()))

    final_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=n_estimators,
                    min_samples_leaf=3,
                    random_state=seed,
                    n_jobs=max(1, min(rf_workers, workers)),
                ),
            ),
        ]
    )
    final_pipe.fit(X, y)
    model = final_pipe.named_steps["model"]
    importances = np.mean([est.feature_importances_ for est in model.estimators_], axis=0)
    importance_df = pd.DataFrame({"feature": feature_cols, "importance": importances}).sort_values(
        "importance", ascending=False
    )
    importance_df["model"] = name

    summary = {
        "n_samples": int(len(data)),
        "n_features": int(len(feature_cols)),
        "cv_folds": int(min(folds, len(data))),
        "macro_auc_hard_state": float(macro_auc) if not np.isnan(macro_auc) else None,
        "mean_kl": float(kl_divergence_rows(y, pred).mean()),
        "mean_jsd": float(js_divergence_rows(y, pred).mean()),
        "brier_soft": float(np.mean(np.sum((y - pred) ** 2, axis=1))),
        "ece_soft": float(ece),
    }
    return summary, importance_df


def imaging_to_posterior(df: pd.DataFrame, posterior_cols: List[str], cfg: RunConfig) -> None:
    logging.info("Running imaging-to-posterior soft-label models")
    targets = patient_level_targets(df, posterior_cols)
    target_cols = [f"pi_{c}" for c in posterior_cols]
    dat, dat_cols = load_datscan_features()
    mri, mri_cols = load_mri_features()

    datasets = {}
    datasets["DaTSCAN"] = (targets.merge(dat, on="PATNO", how="inner"), dat_cols)
    datasets["MRI"] = (targets.merge(mri, on="PATNO", how="inner"), mri_cols)
    fusion = targets.merge(dat, on="PATNO", how="inner").merge(mri, on="PATNO", how="inner")
    datasets["DaTSCAN_MRI_Fusion"] = (fusion, dat_cols + mri_cols)

    summaries = {}
    importance_frames = []
    for name, (table, features) in datasets.items():
        summary, imp = evaluate_soft_posterior_model(
            name,
            table,
            features,
            target_cols,
            cfg.cv_folds,
            cfg.seed,
            cfg.n_estimators,
            cfg.workers,
            cfg.rf_workers,
            cfg.parallel_backend,
        )
        summaries[name] = summary
        importance_frames.append(imp)

    importance = pd.concat(importance_frames, ignore_index=True)
    importance.to_csv(RESULTS / "imaging_to_posterior_feature_importance.csv", index=False)
    write_json(RESULTS / "imaging_to_posterior_summary.json", summaries)

    metric = "mean_jsd"
    fig, ax = plt.subplots(figsize=(6.5, 4.4), dpi=180)
    names = list(summaries.keys())
    vals = [summaries[n][metric] for n in names]
    ax.bar(names, vals, color=["#2b6cb0", "#805ad5", "#2f855a"])
    ax.set_ylabel("Mean Jensen-Shannon Divergence")
    ax.set_title("Imaging-to-Motor Posterior Prediction")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(FIGURES / "imaging_to_posterior_performance.png", bbox_inches="tight")
    plt.close(fig)

    top = importance.groupby("model").head(10).copy()
    fig, ax = plt.subplots(figsize=(7.4, 5.2), dpi=180)
    y_labels = [f"{m}: {f}" for m, f in zip(top["model"], top["feature"])]
    ax.barh(range(len(top)), top["importance"], color="#2b6cb0")
    ax.set_yticks(range(len(top)), y_labels, fontsize=6)
    ax.invert_yaxis()
    ax.set_xlabel("Random Forest Importance")
    ax.set_title("Top Imaging Features for Soft Posterior Prediction")
    fig.tight_layout()
    fig.savefig(FIGURES / "imaging_to_posterior_feature_importance.png", bbox_inches="tight")
    plt.close(fig)


def covariate_adjusted_delta_r2(y: pd.Series, base: pd.DataFrame, state: pd.Series) -> Tuple[float, float, float]:
    table = pd.concat([y.rename("y"), base, state.rename("state")], axis=1).replace([np.inf, -np.inf], np.nan)
    table = table.dropna(subset=["y"])
    if table["state"].nunique() < 2 or len(table) < 20:
        return np.nan, np.nan, np.nan
    x_base = pd.get_dummies(table.drop(columns=["y", "state"]), dummy_na=True)
    x_state = pd.get_dummies(table.drop(columns=["y"]), columns=["state"], dummy_na=True)
    x_base = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(x_base), columns=x_base.columns)
    x_state = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(x_state), columns=x_state.columns)
    yy = table["y"].to_numpy(float)
    r2_base = LinearRegression().fit(x_base, yy).score(x_base, yy)
    r2_state = LinearRegression().fit(x_state, yy).score(x_state, yy)
    groups = [g["y"].to_numpy(float) for _, g in table.groupby("state")]
    h = stats.kruskal(*groups).statistic if len(groups) > 1 else np.nan
    eta = (h - len(groups) + 1) / (len(table) - len(groups)) if len(groups) > 1 and len(table) > len(groups) else np.nan
    return float(r2_state - r2_base), float(r2_base), float(eta)


def effect_size_imaging_validation(df: pd.DataFrame) -> None:
    logging.info("Running effect-size-first imaging validation")
    targets = patient_level_targets(df, [str(i) for i in range(5)])
    targets = targets[["PATNO", "hard_state", "severity", "age_years", "disease_duration_years"]]
    dat, dat_cols = load_datscan_features()
    mri, mri_cols = load_mri_features()
    table = targets.merge(dat, on="PATNO", how="left").merge(mri, on="PATNO", how="left")
    base = table[["severity", "age_years", "disease_duration_years"]]

    selected = [
        "putamen_mean",
        "caudate_mean",
        "AI_putamen",
        "AI_caudate",
        "Left_Hippocampus_norm",
        "Right_Hippocampus_norm",
        "3rd_Ventricle_norm",
        "Left_Thalamus_norm",
        "Right_Thalamus_norm",
    ]
    selected = [c for c in selected if c in table.columns]
    rows = []
    for feature in selected:
        y = pd.to_numeric(table[feature], errors="coerce")
        delta_r2, base_r2, eta = covariate_adjusted_delta_r2(y, base, table["hard_state"])
        m2 = table[table["hard_state"] == 2][feature].dropna().astype(float)
        m3 = table[table["hard_state"] == 3][feature].dropna().astype(float)
        pooled = math.sqrt(((len(m2) - 1) * m2.var() + (len(m3) - 1) * m3.var()) / max(len(m2) + len(m3) - 2, 1))
        d_m3_m2 = (m3.mean() - m2.mean()) / pooled if pooled and not np.isnan(pooled) else np.nan
        rows.append(
            {
                "feature": feature,
                "n_nonmissing": int(y.notna().sum()),
                "delta_r2_phenotype_over_covariates": delta_r2,
                "base_covariate_r2": base_r2,
                "eta_squared_kruskal": eta,
                "cohens_d_M3_vs_M2": float(d_m3_m2) if not np.isnan(d_m3_m2) else np.nan,
            }
        )
    out = pd.DataFrame(rows).sort_values("delta_r2_phenotype_over_covariates", ascending=False)
    out.to_csv(RESULTS / "imaging_effect_size_validation.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=180)
    ax.barh(out["feature"], out["delta_r2_phenotype_over_covariates"], color="#2f855a")
    ax.invert_yaxis()
    ax.set_xlabel("Adjusted Delta R2 Beyond Severity/Covariates")
    ax.set_title("Effect-Size-First Imaging Validation")
    fig.tight_layout()
    fig.savefig(FIGURES / "imaging_effect_size_forest.png", bbox_inches="tight")
    plt.close(fig)


def fit_transfer_bgmm(
    df: pd.DataFrame,
    score_cols: List[str],
    seed: int,
) -> Tuple[SimpleImputer, RobustScaler, BayesianGaussianMixture, Dict[int, int]]:
    train = numeric_frame(df, score_cols).replace([np.inf, -np.inf], np.nan)
    imputer = SimpleImputer(strategy="median")
    scaler = RobustScaler()
    X = scaler.fit_transform(imputer.fit_transform(train))
    model = make_bgmm(seed)
    raw_labels = model.fit_predict(X)
    original = df["cluster_label"].to_numpy(int)
    confusion = np.zeros((5, 5), dtype=int)
    for o, p in zip(original, raw_labels):
        if 0 <= int(o) < 5 and 0 <= int(p) < 5:
            confusion[int(o), int(p)] += 1
    row_ind, col_ind = linear_sum_assignment(-confusion)
    mapping = {int(raw): int(orig) for orig, raw in zip(row_ind, col_ind)}
    return imputer, scaler, model, mapping


def predict_transfer_bgmm(
    table: pd.DataFrame,
    score_cols: List[str],
    imputer: SimpleImputer,
    scaler: RobustScaler,
    model: BayesianGaussianMixture,
    mapping: Dict[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    aligned = ensure_np3_total(table, score_cols)
    for col in score_cols:
        if col not in aligned.columns:
            aligned[col] = np.nan
    X = scaler.transform(imputer.transform(numeric_frame(aligned, score_cols).replace([np.inf, -np.inf], np.nan)))
    raw_proba = model.predict_proba(X)
    mapped = np.zeros((len(aligned), 5), dtype=float)
    for raw_idx in range(raw_proba.shape[1]):
        target = mapping.get(raw_idx, raw_idx if raw_idx < 5 else int(np.argmax(raw_proba[:, raw_idx])))
        if 0 <= target < 5:
            mapped[:, target] += raw_proba[:, raw_idx]
    mapped = normalize_simplex(mapped)
    return np.argmax(mapped, axis=1), mapped


def sum_prefix_endpoint(frame: pd.DataFrame, prefix: str, endpoint_name: str) -> pd.DataFrame:
    cols = [c for c in frame.columns if c.startswith(prefix) and c.upper() not in {"PATNO", "EVENT_ID"}]
    cols = [c for c in cols if frame[c].dtype != object or frame[c].astype(str).str.match(r"^-?\d+(\.\d+)?$").mean() > 0.5]
    if not cols:
        return pd.DataFrame(columns=["PATNO", endpoint_name])
    out = frame[["PATNO"]].copy()
    out[endpoint_name] = numeric_frame(frame, cols).sum(axis=1, min_count=max(1, len(cols) // 2))
    return out.groupby("PATNO", as_index=False)[endpoint_name].mean()


def first_biofind(pattern: str) -> Optional[Path]:
    files = sorted(BIOFIND.glob(pattern))
    return files[0] if files else None


def biofind_endpoint_validation(
    df: pd.DataFrame,
    score_cols: List[str],
    posterior_cols: List[str],
    cfg: RunConfig,
) -> None:
    logging.info("Running BioFIND external endpoint validation")
    motor_path = first_biofind("MDS_UPDRS_Part_III__Post_Dose__*.csv")
    if motor_path is None:
        write_json(RESULTS / "biofind_assignment_summary.json", {"error": "BioFIND motor Part III file not found."})
        return

    imputer, scaler, model, mapping = fit_transfer_bgmm(df, score_cols, cfg.seed)
    motor = ensure_np3_total(read_csv(motor_path), score_cols)
    motor["PATNO"] = pd.to_numeric(motor["PATNO"], errors="coerce")
    motor = motor.dropna(subset=["PATNO"]).copy()
    motor["PATNO"] = motor["PATNO"].astype(int)
    motor["biofind_severity"] = pd.to_numeric(motor.get("NP3TOT", np.nan), errors="coerce")
    if "NHY" in motor.columns:
        motor["NHY"] = pd.to_numeric(motor["NHY"], errors="coerce")
    biofind_domain_cols = []
    for domain, items in MOTOR_DOMAINS.items():
        valid = [c for c in items if c in motor.columns]
        if valid:
            col = f"biofind_domain_{domain}"
            motor[col] = numeric_frame(motor, valid).mean(axis=1)
            biofind_domain_cols.append(col)
    if "biofind_domain_Tremor" in motor.columns and "biofind_domain_Axial" in motor.columns:
        motor["biofind_profile_tremor_axial"] = motor["biofind_domain_Tremor"] - motor["biofind_domain_Axial"]
        biofind_domain_cols.append("biofind_profile_tremor_axial")
    labels, proba = predict_transfer_bgmm(motor, score_cols, imputer, scaler, model, mapping)
    motor["assigned_state"] = labels
    motor["assigned_label"] = [PHENO_LABELS.get(int(x), str(x)) for x in labels]
    motor["max_posterior"] = proba.max(axis=1)
    motor["posterior_entropy"] = entropy_rows(proba)
    for idx in range(5):
        motor[f"pi_{idx}"] = proba[:, idx]
    motor.to_csv(RESULTS / "biofind_visit_assignments.csv", index=False)

    patient = (
        motor.groupby("PATNO")
        .agg(
            assigned_state=("assigned_state", lambda x: int(pd.Series(x).mode().iloc[0])),
            n_biofind_motor_visits=("assigned_state", "size"),
            biofind_severity=("biofind_severity", "mean"),
            biofind_nhy=("NHY", "mean") if "NHY" in motor.columns else ("biofind_severity", "size"),
            max_posterior=("max_posterior", "mean"),
            posterior_entropy=("posterior_entropy", "mean"),
        )
        .reset_index()
    )
    for idx in range(5):
        patient[f"pi_{idx}"] = motor.groupby("PATNO")[f"pi_{idx}"].mean().reindex(patient["PATNO"]).to_numpy()
    if "NHY" not in motor.columns:
        patient = patient.rename(columns={"biofind_nhy": "motor_visit_count_placeholder"})
        patient["biofind_nhy"] = np.nan
        patient = patient.drop(columns=["motor_visit_count_placeholder"])
    if biofind_domain_cols:
        domain_summary = motor.groupby("PATNO", as_index=False)[biofind_domain_cols].mean()
        patient = patient.merge(domain_summary, on="PATNO", how="left")

    endpoint_tables = []
    part2_path = first_biofind("MDS_UPDRS_Part_II__Patient_Questionnaire_*.csv")
    if part2_path:
        endpoint_tables.append(sum_prefix_endpoint(read_csv(part2_path), "NP2", "mds_updrs_part2_total"))
    part1_path = first_biofind("MDS_UPDRS_Part_I_*.csv")
    if part1_path:
        endpoint_tables.append(sum_prefix_endpoint(read_csv(part1_path), "NP1", "mds_updrs_part1_clinician_total"))
    part1q_path = first_biofind("MDS_UPDRS_Part_I__Patient_Questionnaire_*.csv")
    if part1q_path:
        endpoint_tables.append(sum_prefix_endpoint(read_csv(part1q_path), "NP1", "mds_updrs_part1_patient_total"))
    moca_path = first_biofind("Montreal_Cognitive_Assessment__MoCA__*.csv")
    if moca_path:
        moca = read_csv(moca_path)
        if "MCATOT" in moca.columns:
            moca["MCATOT"] = pd.to_numeric(moca["MCATOT"], errors="coerce")
            endpoint_tables.append(moca.groupby("PATNO", as_index=False)["MCATOT"].mean().rename(columns={"MCATOT": "moca_total"}))
    rbd_path = first_biofind("REM_Sleep_Disorder_Questionnaire_*.csv")
    if rbd_path:
        rbd = read_csv(rbd_path)
        score_col = next((c for c in rbd.columns if "RBDSQ" in c.upper() and "TOTAL" in c.upper()), None)
        if score_col:
            rbd[score_col] = pd.to_numeric(rbd[score_col], errors="coerce")
            endpoint_tables.append(
                rbd.groupby("PATNO", as_index=False)[score_col].mean().rename(columns={score_col: "rbd_total"})
            )
    features_path = first_biofind("PD_Features_*.csv")
    if features_path:
        features = read_csv(features_path)
        cols = [c for c in ["PATNO", "SXYEAR", "PDDXDT", "DXTREMOR", "DXRIGID", "DXBRADY", "DXPOSINS", "DOMSIDE"] if c in features.columns]
        feat = features[cols].drop_duplicates("PATNO").copy()
        if "SXYEAR" in feat.columns:
            feat["symptom_onset_year"] = pd.to_numeric(feat["SXYEAR"], errors="coerce")
        endpoint_tables.append(feat.drop(columns=[c for c in ["SXYEAR", "PDDXDT"] if c in feat.columns], errors="ignore"))

    endpoint = patient.copy()
    for table in endpoint_tables:
        if "PATNO" in table.columns:
            table["PATNO"] = pd.to_numeric(table["PATNO"], errors="coerce")
            table = table.dropna(subset=["PATNO"]).copy()
            table["PATNO"] = table["PATNO"].astype(int)
            endpoint = endpoint.merge(table, on="PATNO", how="left")

    endpoint.to_csv(RESULTS / "biofind_patient_endpoint_table.csv", index=False)
    endpoint_cols = [
        c
        for c in endpoint.columns
        if c
        not in {
            "PATNO",
            "assigned_state",
            "n_biofind_motor_visits",
            "max_posterior",
            "posterior_entropy",
        }
        and not c.startswith("pi_")
    ]

    rows = []
    for col in endpoint_cols:
        values = pd.to_numeric(endpoint[col], errors="coerce")
        valid = endpoint.assign(_value=values).dropna(subset=["_value"])
        groups = [g["_value"].to_numpy(float) for _, g in valid.groupby("assigned_state") if len(g) > 1]
        if len(groups) < 2 or len(valid) < 10:
            h_stat, p_val, eta = np.nan, np.nan, np.nan
        else:
            h_stat = float(stats.kruskal(*groups).statistic)
            p_val = float(stats.kruskal(*groups).pvalue)
            eta = float((h_stat - len(groups) + 1) / max(len(valid) - len(groups), 1))
        base_cols = [
            c
            for c in ["biofind_severity", "biofind_nhy", "n_biofind_motor_visits"]
            if c in endpoint.columns and c != col
        ]
        if base_cols:
            delta_r2, base_r2, _ = covariate_adjusted_delta_r2(
                values,
                endpoint[base_cols],
                endpoint["assigned_state"],
            )
        else:
            delta_r2, base_r2 = np.nan, np.nan
        row = {
            "endpoint": col,
            "n_nonmissing": int(values.notna().sum()),
            "kruskal_h": h_stat,
            "p_value": p_val,
            "eta_squared": eta,
            "delta_r2_assigned_state_over_severity_covariates": delta_r2,
            "base_covariate_r2": base_r2,
        }
        for state in range(5):
            state_values = values[endpoint["assigned_state"] == state].dropna()
            row[f"mean_state_{state}"] = float(state_values.mean()) if len(state_values) else np.nan
            row[f"n_state_{state}"] = int(len(state_values))
        rows.append(row)
    stats_df = pd.DataFrame(rows)
    if not stats_df.empty:
        stats_df["p_fdr_bh"] = benjamini_hochberg(stats_df["p_value"].tolist())
        stats_df = stats_df.sort_values("eta_squared", ascending=False, na_position="last")
    stats_df.to_csv(RESULTS / "biofind_endpoint_validation.csv", index=False)

    ppmi_patients = patient_level_targets(df, posterior_cols)
    ppmi_dist = phenotype_distribution(ppmi_patients["hard_state"].to_numpy(int))
    bio_dist = phenotype_distribution(endpoint["assigned_state"].to_numpy(int))

    if not stats_df.empty:
        fig, ax = plt.subplots(figsize=(7.0, 4.8), dpi=180)
        top = stats_df.dropna(subset=["eta_squared"]).head(10).sort_values("eta_squared")
        if not top.empty:
            ax.barh(top["endpoint"], top["eta_squared"], color="#dd6b20")
        ax.set_xlabel("Kruskal Eta Squared")
        ax.set_title("BioFIND Endpoint Gradients by Assigned State")
        fig.tight_layout()
        fig.savefig(FIGURES / "biofind_endpoint_gradients.png", bbox_inches="tight")
        plt.close(fig)

    write_json(
        RESULTS / "biofind_assignment_summary.json",
        {
            "motor_file": str(motor_path),
            "n_biofind_motor_visits": int(len(motor)),
            "n_biofind_patients": int(endpoint["PATNO"].nunique()),
            "mean_max_posterior": float(endpoint["max_posterior"].mean()),
            "mean_posterior_entropy": float(endpoint["posterior_entropy"].mean()),
            "ppmi_patient_state_distribution": ppmi_dist.tolist(),
            "biofind_state_distribution": bio_dist.tolist(),
            "ppmi_vs_biofind_distribution_jsd": float(jensenshannon(ppmi_dist, bio_dist, base=2.0) ** 2),
            "n_endpoints_tested": int(len(stats_df)),
            "top_endpoint_by_eta_squared": stats_df.head(5).to_dict(orient="records") if not stats_df.empty else [],
        },
    )


def _fit_multiview_bgmm(
    name: str,
    table: pd.DataFrame,
    feature_cols: List[str],
    reference_col: str,
    seed: int,
) -> Tuple[dict, pd.DataFrame]:
    data = table.dropna(subset=[reference_col]).copy()
    if data.empty:
        return {"view": name, "error": "empty_table"}, pd.DataFrame()
    X_raw = data[feature_cols].replace([np.inf, -np.inf], np.nan)
    X = RobustScaler().fit_transform(SimpleImputer(strategy="median").fit_transform(X_raw))
    bgmm = make_bgmm(seed)
    labels = bgmm.fit_predict(X)
    proba = bgmm.predict_proba(X)
    hard = data[reference_col].to_numpy(int)
    unique_labels = np.unique(labels)
    if len(unique_labels) > 1 and len(unique_labels) < len(data):
        try:
            sil = float(silhouette_score(X, labels))
        except Exception:
            sil = np.nan
    else:
        sil = np.nan
    assignment = data[["PATNO", reference_col]].copy()
    assignment["view"] = name
    assignment["latent_label"] = labels
    assignment["latent_max_posterior"] = proba.max(axis=1)
    assignment["latent_entropy"] = entropy_rows(normalize_simplex(proba))
    summary = {
        "view": name,
        "n_patients": int(len(data)),
        "n_features": int(len(feature_cols)),
        "effective_k_weight_gt_0p01": int(np.sum(bgmm.weights_ > 0.01)),
        "ari_vs_motor_state": float(adjusted_rand_score(hard, labels)),
        "nmi_vs_motor_state": float(normalized_mutual_info_score(hard, labels)),
        "silhouette": sil if not np.isnan(sil) else None,
        "mean_latent_max_posterior": float(assignment["latent_max_posterior"].mean()),
        "mean_latent_entropy": float(assignment["latent_entropy"].mean()),
    }
    return summary, assignment


def multimodal_latent_mixture_ablation(
    df: pd.DataFrame,
    posterior_cols: List[str],
    cfg: RunConfig,
) -> None:
    logging.info("Running multimodal latent-variable mixture ablation")
    targets = patient_level_targets(df, posterior_cols)[["PATNO", "hard_state"]]
    motor_features = [f"z_{d}" for d in MOTOR_DOMAINS]
    motor_patient = (
        df.groupby("PATNO")
        .agg({f: "mean" for f in motor_features} | {"cluster_label": lambda x: int(pd.Series(x).mode().iloc[0])})
        .reset_index()
        .rename(columns={"cluster_label": "hard_state"})
    )
    dat, dat_cols = load_datscan_features()
    mri, mri_cols = load_mri_features()

    views = [
        ("Motor_only", motor_patient, motor_features),
        ("DaTSCAN_only", targets.merge(dat, on="PATNO", how="inner"), dat_cols),
        ("MRI_only", targets.merge(mri, on="PATNO", how="inner"), mri_cols),
        (
            "Motor_DaTSCAN_MRI_inner",
            motor_patient.merge(dat, on="PATNO", how="inner").merge(mri, on="PATNO", how="inner"),
            motor_features + dat_cols + mri_cols,
        ),
    ]

    missing_aware = motor_patient.merge(dat, on="PATNO", how="left").merge(mri, on="PATNO", how="left")
    missing_feature_cols = motor_features + dat_cols + mri_cols
    for col in missing_feature_cols:
        missing_aware[f"{col}_missing"] = missing_aware[col].isna().astype(int)
    views.append(
        (
            "Motor_DaTSCAN_MRI_missing_aware",
            missing_aware,
            missing_feature_cols + [f"{c}_missing" for c in missing_feature_cols],
        )
    )

    summaries = []
    assignments = []
    for name, table, features in views:
        valid_features = [c for c in features if c in table.columns]
        if not valid_features:
            summaries.append({"view": name, "error": "no_valid_features"})
            continue
        summary, assignment = _fit_multiview_bgmm(name, table, valid_features, "hard_state", cfg.seed)
        summaries.append(summary)
        if not assignment.empty:
            assignments.append(assignment)

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(RESULTS / "multimodal_latent_mixture_ablation.csv", index=False)
    if assignments:
        pd.concat(assignments, ignore_index=True).to_csv(RESULTS / "multimodal_latent_assignments.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.4, 4.8), dpi=180)
    plot_df = summary_df.dropna(subset=["ari_vs_motor_state"])
    if not plot_df.empty:
        ax.barh(plot_df["view"], plot_df["ari_vs_motor_state"], color="#2b6cb0")
    ax.set_xlabel("ARI vs Motor Posterior State")
    ax.set_title("Multimodal Latent Mixture Agreement")
    fig.tight_layout()
    fig.savefig(FIGURES / "multimodal_latent_ablation.png", bbox_inches="tight")
    plt.close(fig)


def _fit_baseline_labels(method: str, seed: int, X: np.ndarray) -> Tuple[str, int, np.ndarray]:
    if method == "kmeans":
        labels = KMeans(n_clusters=5, n_init=20, random_state=seed).fit_predict(X)
    elif method == "gmm":
        labels = GaussianMixture(n_components=5, covariance_type="full", random_state=seed, n_init=5).fit_predict(X)
    elif method == "bgmm":
        labels = make_bgmm(seed).fit_predict(X)
    else:
        raise ValueError(f"Unknown baseline method: {method}")
    return method, seed, labels


def baseline_seed_stability(df: pd.DataFrame, cfg: RunConfig) -> None:
    logging.info("Running baseline seed stability benchmark")
    patient = (
        df.groupby("PATNO")
        .agg({f"z_{d}": "mean" for d in MOTOR_DOMAINS} | {"cluster_label": lambda x: int(pd.Series(x).mode().iloc[0])})
        .reset_index()
    )
    feature_cols = [f"z_{d}" for d in MOTOR_DOMAINS]
    X = patient[feature_cols].to_numpy(float)
    seeds = list(range(cfg.seed, cfg.seed + cfg.baseline_seeds))
    tasks = [(method, s) for method in ["kmeans", "gmm", "bgmm"] for s in seeds]
    n_jobs = max(1, min(cfg.workers, len(tasks)))
    logging.info("Baseline seed benchmark: tasks=%d n_jobs=%d seeds_per_method=%d", len(tasks), n_jobs, len(seeds))
    fitted = Parallel(n_jobs=n_jobs, backend=cfg.parallel_backend)(
        delayed(_fit_baseline_labels)(method, s, X) for method, s in tasks
    )

    algorithms = {"kmeans": [], "gmm": [], "bgmm": []}
    for method, _, labels in fitted:
        algorithms[method].append(labels)
    algorithms["agglomerative"] = [AgglomerativeClustering(n_clusters=5).fit_predict(X)]

    rows = []
    reference = patient["cluster_label"].to_numpy(int)
    for name, labels_list in algorithms.items():
        pair_ari = []
        pair_nmi = []
        for i in range(len(labels_list)):
            for j in range(i + 1, len(labels_list)):
                pair_ari.append(adjusted_rand_score(labels_list[i], labels_list[j]))
                pair_nmi.append(normalized_mutual_info_score(labels_list[i], labels_list[j]))
        rows.append(
            {
                "method": name,
                "n_runs": len(labels_list),
                "ari_vs_original_mean": float(np.mean([adjusted_rand_score(reference, l) for l in labels_list])),
                "ari_vs_original_sd": float(np.std([adjusted_rand_score(reference, l) for l in labels_list])),
                "pairwise_ari_mean": float(np.mean(pair_ari)) if pair_ari else np.nan,
                "pairwise_ari_sd": float(np.std(pair_ari)) if pair_ari else np.nan,
                "pairwise_nmi_mean": float(np.mean(pair_nmi)) if pair_nmi else np.nan,
                "pairwise_nmi_sd": float(np.std(pair_nmi)) if pair_nmi else np.nan,
            }
        )
    pd.DataFrame(rows).to_csv(RESULTS / "baseline_seed_stability.csv", index=False)


def run_pipeline(cfg: RunConfig) -> None:
    start = time.time()
    df, score_cols, posterior_cols = load_motor_with_assignments()
    df = add_patient_covariates(df)

    steps = set(cfg.steps)
    if "all" in steps or "cohort" in steps:
        with timed_step("cohort_inclusion_audit", cfg):
            cohort_inclusion_audit(df)
    if "all" in steps or "patient" in steps:
        with timed_step("patient_state_analysis", cfg):
            patient_state_analysis(df, posterior_cols)
    if "all" in steps or "severity" in steps:
        with timed_step("severity_deconfounding", cfg):
            severity_deconfounding(df, cfg.seed)
    if "all" in steps or "longitudinal" in steps:
        with timed_step("longitudinal_predictive_model", cfg, cv_folds=cfg.cv_folds, n_estimators=cfg.n_estimators):
            longitudinal_predictive_model(df, posterior_cols, cfg)
    if "all" in steps or "calibration" in steps:
        with timed_step("posterior_calibration", cfg, bootstrap_reps=cfg.bootstrap_reps):
            posterior_calibration(
                df,
                score_cols,
                cfg.bootstrap_reps,
                cfg.workers,
                cfg.seed,
                cfg.parallel_backend,
            )
    if "all" in steps or "imaging" in steps:
        with timed_step("imaging_to_posterior", cfg, cv_folds=cfg.cv_folds, n_estimators=cfg.n_estimators):
            imaging_to_posterior(df, posterior_cols, cfg)
    if "all" in steps or "effects" in steps:
        with timed_step("effect_size_imaging_validation", cfg):
            effect_size_imaging_validation(df)
    if "all" in steps or "biofind" in steps:
        with timed_step("biofind_endpoint_validation", cfg):
            biofind_endpoint_validation(df, score_cols, posterior_cols, cfg)
    if "all" in steps or "multimodal" in steps:
        with timed_step("multimodal_latent_mixture_ablation", cfg):
            multimodal_latent_mixture_ablation(df, posterior_cols, cfg)
    if "all" in steps or "baselines" in steps:
        with timed_step("baseline_seed_stability", cfg, baseline_seeds=cfg.baseline_seeds):
            baseline_seed_stability(df, cfg)

    write_json(
        RESULTS / "revision_pipeline_run_summary.json",
        {
            "mode": cfg.mode,
            "steps": cfg.steps,
            "seed": cfg.seed,
            "workers": cfg.workers,
            "rf_workers": cfg.rf_workers,
            "parallel_backend": cfg.parallel_backend,
            "bootstrap_reps": cfg.bootstrap_reps,
            "cv_folds": cfg.cv_folds,
            "n_estimators": cfg.n_estimators,
            "baseline_seeds": cfg.baseline_seeds,
            "n_score_cols": len(score_cols),
            "n_visits": int(len(df)),
            "n_patients": int(df["PATNO"].nunique()),
            "wall_time_seconds": round(time.time() - start, 2),
        },
    )


def parse_args() -> RunConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--steps", default="all", help="Comma-separated steps or all")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--rf-workers", type=int, default=None)
    parser.add_argument("--baseline-seeds", type=int, default=None)
    parser.add_argument("--parallel-backend", choices=["loky", "threading"], default=None)
    parser.add_argument("--bootstrap-reps", type=int, default=None)
    parser.add_argument("--cv-folds", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    config = load_json(CONFIG_PATH)
    seed = args.seed if args.seed is not None else int(config.get("random_seed", 42))
    workers = args.workers if args.workers is not None else int(config.get("workers_default", 120))
    rf_workers = args.rf_workers if args.rf_workers is not None else int(config.get("rf_workers_default", 96))
    backend = args.parallel_backend if args.parallel_backend is not None else str(config.get("parallel_backend_default", "loky"))
    if args.mode == "smoke":
        reps = args.bootstrap_reps if args.bootstrap_reps is not None else int(config.get("bootstrap_reps_smoke", 8))
        folds = args.cv_folds if args.cv_folds is not None else int(config.get("cv_folds_smoke", 3))
        baseline_seeds = (
            args.baseline_seeds if args.baseline_seeds is not None else int(config.get("baseline_seeds_smoke", 10))
        )
        n_estimators = 80
    else:
        reps = args.bootstrap_reps if args.bootstrap_reps is not None else int(config.get("bootstrap_reps_full", 250))
        folds = args.cv_folds if args.cv_folds is not None else int(config.get("cv_folds_full", 5))
        baseline_seeds = (
            args.baseline_seeds if args.baseline_seeds is not None else int(config.get("baseline_seeds_full", 75))
        )
        n_estimators = 500

    steps = [s.strip() for s in args.steps.split(",") if s.strip()]
    return RunConfig(args.mode, steps, seed, workers, rf_workers, backend, reps, folds, n_estimators, baseline_seeds)


def main() -> None:
    setup_logging()
    cfg = parse_args()
    logging.info("Starting revision pipeline: %s", cfg)
    run_pipeline(cfg)
    logging.info("Finished revision pipeline")


if __name__ == "__main__":
    main()
