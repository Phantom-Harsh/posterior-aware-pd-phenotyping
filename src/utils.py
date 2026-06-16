#!/usr/bin/env python3
"""
utils.py — Shared utilities for the MICCAI 2026 PD Phenotyping Pipeline.
Authors: Tirhekar, Yadav & Bajaj
"""

import json
import logging
from pathlib import Path
from typing import Any, Callable, List

import numpy as np
import scipy.stats as stats
from joblib import Parallel, delayed
from tqdm import tqdm


# ── Logging ───────────────────────────────────────────────────────────────────

def setup_logger(name: str, output_dir: str = None) -> logging.Logger:
    handlers = [logging.StreamHandler()]
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(Path(output_dir) / f"{name}.log"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
        force=True,
    )
    return logging.getLogger(name)


# ── Parallelism ───────────────────────────────────────────────────────────────

def parallel_apply(
    func: Callable,
    items: List[Any],
    n_workers: int = -1,
    desc: str = "Processing",
    backend: str = "loky",
) -> List[Any]:
    """joblib-parallel map with tqdm progress bar."""
    results = Parallel(n_jobs=n_workers, backend=backend)(
        delayed(func)(item) for item in tqdm(items, desc=desc, unit="item")
    )
    return results


# ── Statistics ────────────────────────────────────────────────────────────────

def fisher_combine_pvalues(pvalues: np.ndarray) -> tuple[float, float]:
    """
    Fisher's method to combine independent p-values.
    Returns (chi2_statistic, combined_p_value).
    chi2 = -2 * sum(ln(p_i)) ~ chi2(2k)
    """
    pvalues = np.asarray(pvalues, dtype=float)
    pvalues = np.clip(pvalues, 1e-300, 1.0)  # avoid log(0)
    chi2 = -2.0 * np.sum(np.log(pvalues))
    df = 2 * len(pvalues)
    p_combined = 1.0 - stats.chi2.cdf(chi2, df=df)
    return float(chi2), float(p_combined)


def cramers_v(contingency: np.ndarray) -> float:
    """
    Cramér's V association measure for contingency table.
    V = sqrt(chi2 / (n * (min(r,c) - 1)))
    """
    chi2, _, _, _ = stats.chi2_contingency(contingency, correction=False)
    n = contingency.sum()
    r, c = contingency.shape
    v = np.sqrt(chi2 / (n * (min(r, c) - 1)))
    return float(v)


def fdr_correct(pvalues: np.ndarray, alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """
    Benjamini-Hochberg FDR correction.
    Returns (rejected boolean array, adjusted p-values).
    """
    pvalues = np.asarray(pvalues, dtype=float)
    n = len(pvalues)
    order = np.argsort(pvalues)
    ranked = np.empty(n)
    ranked[order] = np.arange(1, n + 1)
    adjusted = np.minimum(1.0, pvalues * n / ranked)
    # Make monotone
    adjusted = np.minimum.accumulate(adjusted[order][::-1])[::-1]
    adjusted_sorted = np.empty(n)
    adjusted_sorted[order] = adjusted
    rejected = adjusted_sorted <= alpha
    return rejected, adjusted_sorted


def eta_squared(H: float, n: int, k: int) -> float:
    """Effect size η² from Kruskal-Wallis H statistic."""
    return float((H - k + 1) / (n - k))


def bootstrap_ci(
    data: np.ndarray,
    stat_fn: Callable = np.mean,
    B: int = 1000,
    alpha: float = 0.05,
    rng: np.random.Generator = None,
) -> tuple[float, float, float]:
    """
    Percentile bootstrap confidence interval.
    Returns (point_estimate, lower_ci, upper_ci).
    """
    if rng is None:
        rng = np.random.default_rng(42)
    point = stat_fn(data)
    boots = [stat_fn(rng.choice(data, size=len(data), replace=True)) for _ in range(B)]
    lo = float(np.percentile(boots, 100 * alpha / 2))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return float(point), lo, hi


# ── I/O ───────────────────────────────────────────────────────────────────────

class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def save_json(obj: Any, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, cls=_NumpyEncoder)
