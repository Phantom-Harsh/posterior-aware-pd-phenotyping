# Data Access Instructions

## PPMI — Parkinson's Progression Markers Initiative

The primary dataset used in this paper is from PPMI
([ppmi-info.org](https://www.ppmi-info.org/)).

### Steps to Access

1. **Register** at [ppmi-info.org](https://www.ppmi-info.org/) and complete the
   online Data Use Agreement (DUA).
2. **Log in** to the PPMI data portal.
3. **Download** the following files under *Study Data → Motor Assessments*:
   - `MDS_UPDRS_Part_III.csv` — Motor examination scores (34 items, all visits)
4. **Download** imaging data (under *Study Data → Biospecimen / Imaging*):
   - `DaTSCAN_SPECT_Analysis.csv` — Striatal binding ratios (SBR)
   - `FreeSurfer_7_ASEG.csv` — Subcortical segmentation volumes
5. Place downloaded files in `data/`:
   ```
   data/
   ├── MDS_UPDRS_Part_III.csv
   ├── DaTSCAN_SPECT_Analysis.csv
   └── FreeSurfer_7_ASEG.csv
   ```

### Expected Columns

**MDS_UPDRS_Part_III.csv** (minimum required):
```
PATNO, EVENT_ID, INFODT,
NP3SPCH, NP3FACXP, NP3RIGN, NP3RIGRU, NP3RIGLU, NP3RIGRL, NP3RIGLL,
NP3FTAPR, NP3FTAPL, NP3HMOVR, NP3HMOVL, NP3TRMRU, NP3TRMRL,
NP3TRMLU, NP3TRMLL, NP3TMFNR, NP3TMFNL, NP3TDNSL, NP3TDNSR,
NP3PSTBL, NP3GAIT, NP3FRZGT, NP3DYSTN, NP3POSTR, ...
```

---

## BioFIND — Fox Investigation for New Discovery of Biomarkers

BioFIND is available via AMP-PD ([amp-pd.org](https://amp-pd.org/)).

### Steps to Access

1. **Register** at [amp-pd.org](https://amp-pd.org/).
2. Navigate to *Data → BioFIND* and apply for access.
3. Download `BioFIND_MDS_UPDRS_Part_III.csv`.
4. Place it at `data/BioFIND_MDS_UPDRS_Part_III.csv`.

---

## Sample Synthetic Data

A **small synthetic dataset** (5 patients, 10 visits each, realistic MDS-UPDRS-III
ranges) is provided at `data/sample/synthetic_5patients.csv` for testing the
pipeline without real data.

Run with: `python run_pipeline.py --data data/sample/synthetic_5patients.csv --demo`

> ⚠️ Results on the synthetic sample will NOT match the paper's statistics.
> The synthetic data is only for verifying that the pipeline runs end-to-end.
