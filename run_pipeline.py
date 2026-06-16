#!/usr/bin/env python3
"""
End-to-end pipeline runner for Posterior-Aware PD Motor Phenotyping.
MICCAI 2026 — Tirhekar, Yadav & Bajaj.

Usage:
  python run_pipeline.py --data data/MDS_UPDRS_Part_III.csv --n_workers 120
  python run_pipeline.py --data data/sample/synthetic_5patients.csv --demo
"""

import argparse
import logging
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("pipeline")

STAGES = [
    "preprocessing",
    "bgmm_sweep",
    "posterior_triage",
    "granger_analysis",
    "hierarchy",
    "imaging_validation",
    "biofind_transfer",
]


def run_stage(stage: str, args: argparse.Namespace, output_dir: Path):
    """Import and run a single pipeline stage."""
    import importlib
    import sys
    sys.path.insert(0, str(Path(__file__).parent / "src"))

    log.info(f"{'='*60}")
    log.info(f"  Stage: {stage.upper()}")
    log.info(f"{'='*60}")
    t0 = time.time()

    if stage == "preprocessing":
        mod = importlib.import_module("preprocessing")
        mod.run(
            input_path=args.data,
            output_path=str(output_dir / "processed.parquet"),
            scaler_path=str(output_dir / "robust_scaler.pkl"),
        )

    elif stage == "bgmm_sweep":
        mod = importlib.import_module("bgmm_sweep")
        mod.run(
            input_path=str(output_dir / "processed.parquet"),
            n_configs=args.n_configs,
            n_workers=args.n_workers,
            output_dir=str(output_dir),
            seed=args.seed,
            demo=args.demo,
        )

    elif stage == "posterior_triage":
        mod = importlib.import_module("posterior_triage")
        mod.run(
            model_path=str(output_dir / "best_bgmm.pkl"),
            data_path=str(output_dir / "processed.parquet"),
            output_dir=str(output_dir),
        )

    elif stage == "granger_analysis":
        mod = importlib.import_module("granger_analysis")
        mod.run(
            data_path=str(output_dir / "triage_results.csv"),
            n_workers=args.n_workers,
            output_dir=str(output_dir),
        )

    elif stage == "hierarchy":
        mod = importlib.import_module("hierarchy")
        mod.run(
            data_path=str(output_dir / "processed.parquet"),
            model_k5_path=str(output_dir / "best_bgmm.pkl"),
            output_dir=str(output_dir),
        )

    elif stage == "imaging_validation":
        if args.datscan and args.mri:
            mod = importlib.import_module("imaging_validation")
            mod.run(
                datscan_path=args.datscan,
                mri_path=args.mri,
                labels_path=str(output_dir / "triage_results.csv"),
                output_dir=str(output_dir),
                n_workers=args.n_workers,
            )
        else:
            log.warning("Skipping imaging_validation — provide --datscan and --mri.")

    elif stage == "biofind_transfer":
        if args.biofind:
            mod = importlib.import_module("biofind_transfer")
            mod.run(
                biofind_path=args.biofind,
                scaler_path=str(output_dir / "robust_scaler.pkl"),
                model_path=str(output_dir / "best_bgmm.pkl"),
                output_dir=str(output_dir),
            )
        else:
            log.warning("Skipping biofind_transfer — provide --biofind.")

    elapsed = time.time() - t0
    log.info(f"  ✓ {stage} completed in {elapsed:.1f}s")


def main():
    parser = argparse.ArgumentParser(
        description="Posterior-Aware PD Motor Phenotyping Pipeline (MICCAI 2026)"
    )
    parser.add_argument("--data", required=True,
                        help="Path to MDS-UPDRS-III CSV (PPMI format)")
    parser.add_argument("--datscan", default=None,
                        help="Path to DaTSCAN SBR CSV (optional)")
    parser.add_argument("--mri", default=None,
                        help="Path to FreeSurfer ASEG CSV (optional)")
    parser.add_argument("--biofind", default=None,
                        help="Path to BioFIND MDS-UPDRS-III CSV (optional)")
    parser.add_argument("--output_dir", default="results",
                        help="Output directory (default: results/)")
    parser.add_argument("--n_workers", type=int, default=120,
                        help="Number of parallel workers (default: 120)")
    parser.add_argument("--n_configs", type=int, default=2912,
                        help="Number of BGMM configs to sweep (default: 2912)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--stages", nargs="+", default=STAGES,
                        choices=STAGES, help="Stages to run")
    parser.add_argument("--demo", action="store_true",
                        help="Demo mode: reduced sweep (100 configs)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.demo:
        args.n_configs = 100
        args.n_workers = min(args.n_workers, 8)
        log.info("Demo mode: n_configs=100, n_workers≤8")

    log.info(f"Pipeline started | stages={args.stages}")
    log.info(f"Output: {output_dir.resolve()}")
    t_total = time.time()

    for stage in args.stages:
        run_stage(stage, args, output_dir)

    log.info(f"\n{'='*60}")
    log.info(f"  All stages complete in {time.time()-t_total:.1f}s")
    log.info(f"  Results at: {output_dir.resolve()}")
    log.info(f"{'='*60}")


if __name__ == "__main__":
    main()
