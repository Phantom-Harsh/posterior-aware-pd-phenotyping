#!/usr/bin/env python3
"""
VERIFY TRI-MODAL COHORT: Is n=204 real?
"""

import pandas as pd
from pathlib import Path

BASE_DIR = Path("/home1/11021/harshtirhekar/WORK/GaitAnalysis/Parkinson-Disease-using-LRRK2")

print("="*80)
print("VERIFYING TRI-MODAL COHORT (MoCA + UPSIT + Gait)")
print("="*80)

# REPRODUCE EXACT CODE from comprehensive_multimodal.py
curated = pd.read_excel(BASE_DIR / "data/PPMI_Gait/PPMI_Curated_Data_Cut_Public_20241211.xlsx")
gait = pd.read_csv(BASE_DIR / "data/PPMI_Gait/Gait_Data_with_Selected_Features.csv")

print(f"\nCurated shape: {curated.shape}")
print(f"Gait shape: {gait.shape}")

# EXACT reproduction of multimodal code
merged = pd.merge(curated, gait, on='PATNO', how='inner')
print(f"\nAfter merge on PATNO: {len(merged)} records")

available_cols = [col for col in ['moca', 'upsit', 'SP_U'] if col in merged.columns]
print(f"Available columns: {available_cols}")

multi_bio = merged[available_cols].dropna()

print(f"\n{'='*80}")
print(f"RESULT FROM ORIGINAL CODE:")
print(f"{'='*80}")
print(f"len(multi_bio) = {len(multi_bio)} ← This is what was logged")
print(f"Unique patients = {merged.loc[multi_bio.index, 'PATNO'].nunique()}")

print(f"\n{'='*80}")
print(f"VERDICT:")
print(f"{'='*80}")

if len(multi_bio) == 204:
    print(f"✅ n=204 is CORRECT (represents {len(multi_bio)} records/assessments)")
    print(f"   But unique patients = {merged.loc[multi_bio.index, 'PATNO'].nunique()}")
    print(f"   ⚠️  Should clarify: assessments vs unique patients")
else:
    print(f"❌ n=204 does NOT match actual: {len(multi_bio)}")

print("="*80)
