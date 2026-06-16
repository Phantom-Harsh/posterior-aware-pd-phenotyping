#!/usr/bin/env python3
"""Generate synthetic sample dataset for pipeline testing (no PPMI access required)."""

import numpy as np
import pandas as pd
from pathlib import Path

RNG = np.random.default_rng(42)
N_PATIENTS = 50
VISITS_PER_PATIENT = 10

# MDS-UPDRS-III items (34 items, scored 0-4)
UPDRS_ITEMS = [
    "NP3SPCH", "NP3FACXP",
    "NP3RIGN", "NP3RIGRU", "NP3RIGLU", "NP3RIGRL", "NP3RIGLL",
    "NP3FTAPR", "NP3FTAPL", "NP3HMOVR", "NP3HMOVL",
    "NP3TRMRU", "NP3TRMRL", "NP3TRMLU", "NP3TRMLL", "NP3TMFNR", "NP3TMFNL",
    "NP3TDNSL", "NP3TDNSR",
    "NP3PSTBL", "NP3GAIT", "NP3FRZGT",
    "NP3LGAGR", "NP3LGAGL",
    "NP3DYSTN",
    "NP3POSTR",
    "NP3BRADY",
    "NP3PTRMR", "NP3KTRMR", "NP3RTRMR",
    "NP3PTMRL", "NP3KTMRL", "NP3RTMRL",
    "NP3FNFNR",
]

records = []
events = [f"V{str(i).zfill(2)}" for i in range(VISITS_PER_PATIENT)]

for pat in range(1, N_PATIENTS + 1):
    base = RNG.integers(0, 3, size=len(UPDRS_ITEMS))
    for v, evt in enumerate(events):
        noise = RNG.integers(-1, 2, size=len(UPDRS_ITEMS))
        scores = np.clip(base + noise + v // 3, 0, 4)
        row = {"PATNO": pat, "EVENT_ID": evt, "INFODT": f"2019-{(v%12)+1:02d}-01"}
        for item, score in zip(UPDRS_ITEMS, scores):
            row[item] = int(score)
        records.append(row)

df = pd.DataFrame(records)
out = Path(__file__).parent / "data" / "sample" / "synthetic_5patients.csv"
out.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out, index=False)
print(f"Saved {len(df)} rows ({N_PATIENTS} patients × {VISITS_PER_PATIENT} visits) → {out}")
