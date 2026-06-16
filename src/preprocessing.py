#!/usr/bin/env python3
"""
preprocessing.py — MICCAI 2026 PD Phenotyping Pipeline
Load, clean, impute, aggregate, and scale PPMI MDS-UPDRS-III data.

Authors: Tirhekar, Yadav & Bajaj (UT Austin / NIT Raipur)
"""

import argparse
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

# ── Domain definitions (MDS-UPDRS-III) ────────────────────────────────────────
DOMAIN_MAP = {
    "Tremor": [
        "NP3TRMRU", "NP3TRMRL", "NP3TRMLU", "NP3TRMLL",
        "NP3TMFNR", "NP3TMFNL", "NP3TDNSL", "NP3TDNSR",
        "NP3PTRMR", "NP3PTMRL",
    ],
    "Bradykinesia": [
        "NP3FTAPR", "NP3FTAPL", "NP3HMOVR", "NP3HMOVL",
        "NP3RTRMR", "NP3RTMRL", "NP3KTRMR", "NP3KTMRL",
        "NP3BRADY", "NP3FNFNR",
    ],
    "Rigidity": [
        "NP3RIGN", "NP3RIGRU", "NP3RIGLU", "NP3RIGRL", "NP3RIGLL",
    ],
    "Axial": [
        "NP3PSTBL", "NP3GAIT", "NP3FRZGT", "NP3POSTR",
    ],
    "Bulbar": [
        "NP3SPCH", "NP3FACXP", "NP3DYSTN",
    ],
}

ALL_ITEMS = [item for items in DOMAIN_MAP.values() for item in items]
MISSING_THRESHOLD = 0.20   # exclude visit if >20% items missing


def setup_logger(name: str) -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger(name)


log = setup_logger("preprocessing")


def load_ppmi_updrs(path: str) -> pd.DataFrame:
    """Load and validate PPMI MDS-UPDRS-III CSV."""
    log.info(f"Loading PPMI data from {path}")
    df = pd.read_csv(path, low_memory=False)
    # Normalise column names
    df.columns = df.columns.str.strip().str.upper()

    required = {"PATNO", "EVENT_ID"}
    missing_cols = required - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Keep only columns we need
    keep = list(required | {"INFODT"}) + [c for c in ALL_ITEMS if c in df.columns]
    df = df[keep].copy()

    # Cast score columns to numeric
    for col in ALL_ITEMS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    log.info(f"  Loaded {len(df):,} rows, {df['PATNO'].nunique():,} patients")
    return df


def impute_visits(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-visit median imputation after filtering visits with >20% missing.
    Returns filtered + imputed DataFrame.
    """
    item_cols = [c for c in ALL_ITEMS if c in df.columns]
    n_items = len(item_cols)
    missing_frac = df[item_cols].isna().mean(axis=1)

    n_before = len(df)
    df = df[missing_frac <= MISSING_THRESHOLD].copy()
    log.info(f"  Removed {n_before - len(df):,} visits (>{MISSING_THRESHOLD*100:.0f}% missing)")

    # Per-visit median imputation
    df[item_cols] = df[item_cols].apply(
        lambda row: row.fillna(row.median()), axis=1
    )
    log.info(f"  After imputation: {len(df):,} visits remain")
    return df


def aggregate_domains(df: pd.DataFrame) -> pd.DataFrame:
    """Sum items into 5 clinical domain scores."""
    for domain, items in DOMAIN_MAP.items():
        available = [c for c in items if c in df.columns]
        if available:
            df[domain] = df[available].sum(axis=1)
        else:
            log.warning(f"Domain '{domain}': no items found in data")
            df[domain] = np.nan
    return df


def fit_and_scale(df: pd.DataFrame, scaler_path: str) -> tuple[pd.DataFrame, RobustScaler]:
    """Fit RobustScaler on domain scores, save scaler, return scaled DataFrame."""
    domain_cols = list(DOMAIN_MAP.keys())
    X = df[domain_cols].values

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    df_scaled = df.copy()
    for i, col in enumerate(domain_cols):
        df_scaled[f"{col}_scaled"] = X_scaled[:, i]

    Path(scaler_path).parent.mkdir(parents=True, exist_ok=True)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    log.info(f"  Saved RobustScaler → {scaler_path}")
    return df_scaled, scaler


def run(input_path: str, output_path: str, scaler_path: str):
    df = load_ppmi_updrs(input_path)
    df = impute_visits(df)
    df = aggregate_domains(df)
    df, _ = fit_and_scale(df, scaler_path)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    log.info(f"Saved processed data ({len(df):,} visits) → {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess PPMI MDS-UPDRS-III data")
    parser.add_argument("--input",  required=True, help="Raw PPMI CSV path")
    parser.add_argument("--output", required=True, help="Output parquet path")
    parser.add_argument("--scaler_path", default="results/robust_scaler.pkl")
    args = parser.parse_args()
    run(args.input, args.output, args.scaler_path)


if __name__ == "__main__":
    main()
