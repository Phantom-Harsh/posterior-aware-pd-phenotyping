#!/usr/bin/env python3
"""Build manuscript cohort-sensitivity table from saved May 2026 outputs.

This script intentionally uses only saved per-patient/per-visit outputs for
metrics that can be recomputed without rerunning model fitting. It does not
invent calibration or imaging-posterior subgroup metrics, because those require
saved bootstrap consistency or soft-posterior predictions.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
DATA = PROJECT / "data" / "PPMI_Full"
RESULTS = ROOT / "results"
LOGS = ROOT / "logs"


def setup_logging() -> Path:
    LOGS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = LOGS / f"cohort_sensitivity_table_{stamp}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(path), logging.StreamHandler()],
    )
    return path


def find_one(pattern: str) -> Path | None:
    files = sorted(DATA.glob(pattern))
    return files[0] if files else None


def load_patient_sets() -> dict[str, set[int]]:
    audit = pd.read_csv(RESULTS / "cohort_inclusion_audit.csv")
    full_n = int(audit.loc[audit["criterion"].eq("all_complete_np3_rows_aligned_to_original_bgmm_assignments"), "aligned_patients"].iloc[0])
    logging.info("Audit full aligned patients: %d", full_n)

    patient_state = pd.read_csv(RESULTS / "patient_state_summary.csv", usecols=["PATNO"])
    full = set(patient_state["PATNO"].astype(int))

    sets: dict[str, set[int]] = {"Full aligned table": full}

    status_path = find_one("Participant_Status_*.csv")
    if status_path:
        status = pd.read_csv(status_path, low_memory=False)
        status["PATNO"] = pd.to_numeric(status["PATNO"], errors="coerce")
        status = status.dropna(subset=["PATNO"]).copy()
        status["PATNO"] = status["PATNO"].astype(int)
        cohort_col = next((c for c in status.columns if c.upper() == "COHORT_DEFINITION"), None)
        enroll_col = next((c for c in status.columns if c.upper() == "ENROLL_STATUS"), None)
        if cohort_col:
            cohort_text = status[cohort_col].astype(str).str.strip().str.lower()
            pd_mask = cohort_text.isin({"parkinson's disease", "parkinson disease"})
            sets["PD cohort"] = set(status.loc[pd_mask, "PATNO"].astype(int)) & full
        if cohort_col and enroll_col:
            cohort_text = status[cohort_col].astype(str).str.strip().str.lower()
            enroll_text = status[enroll_col].astype(str).str.lower()
            pd_mask = cohort_text.isin({"parkinson's disease", "parkinson disease"})
            enrolled = enroll_text.str.contains("enrolled", na=False)
            sets["PD cohort + enrolled"] = set(status.loc[pd_mask & enrolled, "PATNO"].astype(int)) & full

    diag_path = find_one("Primary_Clinical_Diagnosis_*.csv")
    if diag_path:
        diag = pd.read_csv(diag_path, low_memory=False)
        if "PATNO" in diag.columns and "PRIMDIAG" in diag.columns:
            diag["PATNO"] = pd.to_numeric(diag["PATNO"], errors="coerce")
            diag["PRIMDIAG"] = pd.to_numeric(diag["PRIMDIAG"], errors="coerce")
            diag = diag.dropna(subset=["PATNO"]).copy()
            diag["PATNO"] = diag["PATNO"].astype(int)
            sets["PRIMDIAG == 1"] = set(diag.loc[diag["PRIMDIAG"].eq(1), "PATNO"].astype(int)) & full

    return sets


def subgroup_auc(frame: pd.DataFrame) -> float:
    y = frame["state_transition"].to_numpy(int)
    p = frame["cv_proba_state_transition"].to_numpy(float)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def main() -> None:
    log_path = setup_logging()
    logging.info("Building cohort sensitivity table from saved outputs")

    patient_sets = load_patient_sets()
    state = pd.read_csv(RESULTS / "patient_state_summary.csv")
    long = pd.read_csv(
        RESULTS / "longitudinal_prediction_rows.csv",
        usecols=["PATNO", "state_transition", "cv_proba_state_transition"],
    )
    residual = pd.read_csv(
        RESULTS / "severity_residualized_visit_profiles.csv",
        usecols=["PATNO", "residual_cluster"],
    )
    audit = pd.read_csv(RESULTS / "cohort_inclusion_audit.csv")

    audit_lookup = {
        "Full aligned table": "all_complete_np3_rows_aligned_to_original_bgmm_assignments",
        "PD cohort": "COHORT_DEFINITION == Parkinson's Disease",
        "PRIMDIAG == 1": "PRIMDIAG == 1 (current_primary_clinical_diagnosis_PD)",
        "PD cohort + enrolled": "Parkinson's Disease cohort and enrolled",
    }

    rows = []
    for label in ["Full aligned table", "PRIMDIAG == 1", "PD cohort", "PD cohort + enrolled"]:
        pats = patient_sets.get(label, set())
        s = state[state["PATNO"].isin(pats)]
        l = long[long["PATNO"].isin(pats)]
        r = residual[residual["PATNO"].isin(pats)]
        audit_row = audit[audit["criterion"].eq(audit_lookup[label])]
        if audit_row.empty:
            aligned_visits = int(r.shape[0])
            aligned_patients = int(len(pats))
        else:
            aligned_visits = int(audit_row["aligned_visits"].iloc[0])
            aligned_patients = int(audit_row["aligned_patients"].iloc[0])
        rows.append(
            {
                "cohort": label,
                "aligned_visits": aligned_visits,
                "aligned_patients": aligned_patients,
                "modal_family_fraction_mean": float(s["modal_family_fraction"].mean()),
                "ever_transitioned_pct": float(100.0 * s["ever_transitioned_state"].mean()),
                "residual_active_components_observed": int(r["residual_cluster"].nunique()),
                "transition_auc_using_saved_patient_heldout_predictions": subgroup_auc(l),
                "calibration_ece_status": "requires subgroup bootstrap rerun",
                "fused_imaging_jsd_status": "requires subgroup imaging CV rerun",
            }
        )

    out = pd.DataFrame(rows)
    csv_path = RESULTS / "cohort_sensitivity_available_metrics.csv"
    json_path = RESULTS / "cohort_sensitivity_available_metrics.json"
    out.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(rows, indent=2))
    logging.info("Wrote %s", csv_path)
    logging.info("Wrote %s", json_path)
    logging.info("Run log: %s", log_path)


if __name__ == "__main__":
    main()
