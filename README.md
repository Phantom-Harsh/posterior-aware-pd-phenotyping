# Posterior-Aware Motor Phenotyping with Multimodal Imaging Validation in Parkinson's Disease

[![MICCAI 2026](https://img.shields.io/badge/MICCAI-2026-blue)](https://conferences.miccai.org/2026/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![PPMI Data](https://img.shields.io/badge/Data-PPMI-orange)](https://www.ppmi-info.org/)
[![DOI](https://img.shields.io/badge/DOI-MICCAI%202026-green)]()

**Harsh Milind Tirhekar¹†, Priyanshi Yadav²†, Chandrajit Bajaj¹³†\***

> ¹ Department of Computer Science, College of Natural Sciences, The University of Texas at Austin  
> ² Department of Biomedical Engineering, National Institute of Technology Raipur  
> ³ Oden Institute for Computational Engineering and Sciences, UT Austin  
> † Equal contribution · \* Correspondence: bajaj@cs.utexas.edu

---

## Overview

This repository contains the full analysis pipeline for our **MICCAI 2026** paper:

> **"Posterior-Aware Motor Phenotyping with Multimodal Imaging Validation in Parkinson's Disease"**

We propose a **posterior-aware Bayesian Gaussian Mixture Model (BGMM)** framework that:

- Sweeps **2,912 hyperparameter configurations** to robustly identify `k_eff = 5` motor phenotypic states from 29,366 PPMI MDS-UPDRS-III assessments
- Introduces **three-tier posterior triage**: Textbook (99.5%) / Phenotypic Chimera (0.5%) / Ambiguous (0%)
- Validates discovered states via **DaTSCAN SPECT** (n=1,839; p<10⁻⁸) and **FreeSurfer 7 MRI** (n=1,706; 13/25 FDR-significant ROIs)
- Achieves **99.7% decisive zero-shot generalization** on external BioFIND cohort (n=310)
- Identifies **bradykinesia as a leading temporal predictor** of axial motor decline via Granger predictability

---

## Pipeline Overview

```
PPMI Raw Data (n=29,366 assessments, 1,847 patients)
        │
        ▼
┌─────────────────────────────┐
│  1. Preprocessing           │  RobustScaler, 34→5 domain aggregation
│     src/preprocessing.py    │  Missing value imputation (<20% threshold)
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  2. BGMM Mega-Sweep         │  2,912 configs × parallel workers
│     src/bgmm_sweep.py       │  Bootstrap B=1,000 | k_eff=5 selected
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  3. Posterior Triage        │  Textbook/Chimera/Ambiguous flags
│     src/posterior_triage.py │  Gap Δ, KL divergence, entropy
└────────────┬────────────────┘
             │
        ┌────┴─────┐
        ▼          ▼
┌──────────────┐  ┌──────────────────────┐
│ 4. Granger   │  │  5. Multi-Scale      │
│    Analysis  │  │     Hierarchy        │
│ src/granger  │  │  src/hierarchy.py    │
│   .py        │  │  Cramér's V=0.945    │
└──────┬───────┘  └──────────┬───────────┘
       └─────────┬───────────┘
                 ▼
┌─────────────────────────────┐
│  6. Imaging Validation      │  DaTSCAN SPECT + FreeSurfer 7 MRI
│     src/imaging_val.py      │  Kruskal-Wallis + FDR (B=10,000)
└────────────┬────────────────┘
             ▼
┌─────────────────────────────┐
│  7. BioFIND Zero-Shot       │  External generalization (n=310)
│     src/biofind_transfer.py │  No refitting — 99.7% decisive
└─────────────────────────────┘
```

---

## Repository Structure

```
posterior-aware-pd-phenotyping/
├── src/
│   ├── preprocessing.py          # Data loading, imputation, RobustScaler
│   ├── bgmm_sweep.py             # 2,912-config parallel BGMM mega-sweep
│   ├── posterior_triage.py       # Textbook/Chimera/Ambiguous triage + diagnostics
│   ├── granger_analysis.py       # Granger predictability, Fisher aggregation
│   ├── hierarchy.py              # Multi-scale k=5↔k=8 nesting, Cramér's V
│   ├── imaging_validation.py     # DaTSCAN SPECT + FreeSurfer MRI validation
│   ├── biofind_transfer.py       # Zero-shot BioFIND generalization
│   └── utils.py                  # Shared utilities, parallel helpers
├── configs/
│   └── sweep_config.yaml         # Full 2,912-configuration sweep specification
├── figures/
│   ├── MICCAI_Pipeline.png       # Fig 1 — Pipeline overview
│   ├── fig2_explainability.png   # Fig 2 — Domain profiles, k_eff dist, Granger matrix
│   ├── fig3_chimera_hierarchy.png # Fig 3 — Posterior triage + nesting heatmap
│   └── fig4_imaging_validation.png # Fig 4 — DaTSCAN + MRI validation
├── data/
│   └── sample/                   # Toy synthetic sample (5 patients, 10 visits)
├── docs/
│   └── data_access.md            # Instructions for PPMI & BioFIND data access
├── results/                      # Output directory (gitignored for real results)
├── run_pipeline.py               # End-to-end pipeline runner
├── requirements.txt              # Exact pinned dependencies
├── environment.yml               # Conda environment specification
├── LICENSE                       # MIT License
└── README.md                     # This file
```

---

## Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/Phantom-Harsh/posterior-aware-pd-phenotyping.git
cd posterior-aware-pd-phenotyping

# Option A: pip
pip install -r requirements.txt

# Option B: conda
conda env create -f environment.yml
conda activate pd-phenotyping
```

### 2. Data Access

**PPMI data** requires registration at [ppmi-info.org](https://www.ppmi-info.org/).  
**BioFIND data** is available through AMP-PD at [amp-pd.org](https://amp-pd.org/).

After downloading, place the PPMI MDS-UPDRS-III file at:
```
data/MDS_UPDRS_Part_III.csv
```
See [`docs/data_access.md`](docs/data_access.md) for detailed instructions.

### 3. Run Full Pipeline

```bash
# Full pipeline (all 7 stages)
python run_pipeline.py --data data/MDS_UPDRS_Part_III.csv --n_workers 120

# Individual stages
python src/bgmm_sweep.py --n_configs 2912 --n_workers 120
python src/imaging_validation.py --datscan data/DaTSCAN_SBR.csv --mri data/FreeSurfer_ASEG.csv
python src/biofind_transfer.py --biofind data/BioFIND_UPDRS.csv
```

### 4. Run on Toy Sample (No PPMI Access Required)

```bash
python run_pipeline.py --data data/sample/synthetic_5patients.csv --demo
```

---

## Reproducing Paper Results

All figures and statistics from the paper can be reproduced:

```bash
# Reproduce all 4 main figures
python src/bgmm_sweep.py             # → k_eff=5, Silhouette, bootstrap
python src/posterior_triage.py       # → 99.5% Textbook, 0.5% Chimera
python src/granger_analysis.py       # → Brady→Axial 20.3%
python src/imaging_validation.py     # → DaTSCAN p<1e-8, 13/25 MRI FDR-sig
python src/biofind_transfer.py       # → 99.7% decisive BioFIND
```

Expected outputs are logged to `results/` and figures saved to `figures/`.

---

## Computational Requirements

All analyses were run on a **TACC Vista** node:
- **CPU:** 144-core ARM Neoverse-V2, 243 GB RAM
- **Parallelism:** 120 workers via `joblib` / `ProcessPoolExecutor`
- **Runtime:** BGMM mega-sweep ~4h; imaging validation ~2h; Granger analysis ~6h

For smaller machines, reduce `--n_workers` (minimum ~8 cores recommended).

---

## Key Results

| Metric | Value |
|---|---|
| Optimal clusters (k_eff) | **5** (bootstrap-stable: 5.0±0.0) |
| Textbook decisive posteriors | **99.5%** (mean gap Δ=0.995) |
| Phenotypic Chimeras | **0.5%** (M2↔M3: 62%, M1↔M3: 38%) |
| Multi-scale nesting (Cramér's V) | **0.945** (k=5↔k=8) |
| DaTSCAN SPECT significance | **p=1.07×10⁻⁸** (putamen Kruskal-Wallis H=42.9) |
| FreeSurfer MRI significant ROIs | **13/25** subcortical ROIs (FDR-corrected) |
| M3 hippocampal atrophy | **−8%** bilateral (p_FDR=0.004) |
| BioFIND zero-shot generalization | **99.7%** decisive (JSD=0.192) |
| Granger top direction | **Brady→Axial 20.3%** significant |

---

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{tirhekar2026posterior,
  title     = {Posterior-Aware Motor Phenotyping with Multimodal Imaging Validation
               in {Parkinson's} Disease},
  author    = {Tirhekar, Harsh Milind and Yadav, Priyanshi and Bajaj, Chandrajit},
  booktitle = {Medical Image Computing and Computer Assisted Intervention -- MICCAI 2026},
  series    = {Lecture Notes in Computer Science},
  publisher = {Springer, Cham},
  year      = {2026}
}
```

---

## License

This code is released under the [MIT License](LICENSE).  
The PPMI and BioFIND datasets are subject to their respective data use agreements.

---

## Acknowledgments

- **PPMI** is funded by the Michael J. Fox Foundation for Parkinson's Research.
- Compute resources provided by **TACC Vista** .
- We thank the PPMI and BioFIND participants and investigators.
