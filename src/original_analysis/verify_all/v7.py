#!/usr/bin/env python3
"""
INVESTIGATE MULTIMODAL COHORT DISCREPANCY
=========================================

Why does verification show n=93 but claim was n=204?

Possible explanations:
1. Different definition (UPSIT + Gait + Motor?)
2. Assessments vs unique patients
3. Different gait dataset used
4. Different inclusion criteria
"""

import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path("/home1/11021/harshtirhekar/WORK/GaitAnalysis/Parkinson-Disease-using-LRRK2")

def header(title):
    print("\n" + "="*90)
    print(f"  {title}")
    print("="*90)

header("INVESTIGATING MULTIMODAL COHORT SIZE: 204 vs 93")

# Load data
curated = pd.read_excel(BASE_DIR / "data/PPMI_Gait/PPMI_Curated_Data_Cut_Public_20241211.xlsx")
gait_features = pd.read_csv(BASE_DIR / "data/PPMI_Gait/Gait_Data_with_Selected_Features.csv")
gait_arm = pd.read_csv(BASE_DIR / "data/PPMI_Gait/Gait_Data___Arm_swing_06Jan2025.csv")
gait_mobility = pd.read_csv(BASE_DIR / "data/PPMI_Gait/Gait_Substudy_Gait_Mobility_Assessment_and_Measurement_06Jan2025.csv")

print("\nData loaded:")
print(f"  Curated: {len(curated):,} records, {curated['PATNO'].nunique():,} unique patients")
print(f"  Gait Features: {len(gait_features):,} records, {gait_features['PATNO'].nunique():,} unique patients")
print(f"  Gait Arm Swing: {len(gait_arm):,} records, {gait_arm['PATNO'].nunique():,} unique patients")
print(f"  Gait Mobility: {len(gait_mobility):,} records, {gait_mobility['PATNO'].nunique():,} unique patients")

# ============================================================================
# HYPOTHESIS 1: UPSIT + Gait (unique patients)
# ============================================================================
header("HYPOTHESIS 1: UPSIT + Gait Features (Unique Patients)")

upsit_patients = curated[curated['upsit'].notna()]['PATNO'].unique()
gait_patients = gait_features['PATNO'].unique()
overlap1 = set(upsit_patients) & set(gait_patients)

print(f"\nUPSIT patients: {len(upsit_patients):,}")
print(f"Gait Features patients: {len(gait_patients):,}")
print(f"Overlap: {len(overlap1):,}")

# ============================================================================
# HYPOTHESIS 2: UPSIT + ANY Gait Dataset
# ============================================================================
header("HYPOTHESIS 2: UPSIT + ANY Gait Dataset (Union)")

all_gait_patients = set(gait_features['PATNO'].unique()) | \
                    set(gait_arm['PATNO'].unique()) | \
                    set(gait_mobility['PATNO'].unique())

overlap2 = set(upsit_patients) & all_gait_patients

print(f"\nUPSIT patients: {len(upsit_patients):,}")
print(f"All gait datasets combined: {len(all_gait_patients):,}")
print(f"Overlap: {len(overlap2):,}")

# ============================================================================
# HYPOTHESIS 3: UPSIT + Gait (Assessments, not patients)
# ============================================================================
header("HYPOTHESIS 3: UPSIT + Gait (Total Assessments)")

# Merge and count paired assessments
merged = pd.merge(
    curated[curated['upsit'].notna()][['PATNO', 'EVENT_ID', 'upsit']],
    gait_features[['PATNO', 'EVENT_ID', 'SP_U']],
    on=['PATNO', 'EVENT_ID'],
    how='inner'
)

print(f"\nPaired UPSIT+Gait assessments: {len(merged):,}")
print(f"Unique patients in pairs: {merged['PATNO'].nunique():,}")

# ============================================================================
# HYPOTHESIS 4: UPSIT + Gait with Speed data
# ============================================================================
header("HYPOTHESIS 4: UPSIT + Gait with non-null speed")

gait_with_speed = gait_features[gait_features['SP_U'].notna()]
overlap4 = set(upsit_patients) & set(gait_with_speed['PATNO'].unique())

print(f"\nUPSIT patients: {len(upsit_patients):,}")
print(f"Gait patients with speed data: {gait_with_speed['PATNO'].nunique():,}")
print(f"Overlap: {len(overlap4):,}")

# ============================================================================
# HYPOTHESIS 5: Three-way (UPSIT + Gait + Motor)
# ============================================================================
header("HYPOTHESIS 5: Three-way Multimodal (UPSIT + Gait + Motor)")

updrs = pd.read_csv(BASE_DIR / "data/PPMI_Gait/MDS-UPDRS_Part_III_06Jan2025.csv")
baseline = updrs[updrs['EVENT_ID'] == 'BL']
motor_items = [col for col in baseline.columns if col.startswith('NP3') and col != 'NP3TOT']
complete_motor = baseline[['PATNO'] + motor_items].dropna()
motor_patients = complete_motor['PATNO'].unique()

three_way = set(upsit_patients) & set(gait_patients) & set(motor_patients)

print(f"\nUPSIT patients: {len(upsit_patients):,}")
print(f"Gait patients: {len(gait_patients):,}")
print(f"Motor patients: {len(motor_patients):,}")
print(f"Three-way overlap: {len(three_way):,}")

# ============================================================================
# HYPOTHESIS 6: Check curated data for gait columns
# ============================================================================
header("HYPOTHESIS 6: Gait Speed in Curated Data")

gait_cols_in_curated = [c for c in curated.columns if 'gait' in c.lower() or 'speed' in c.lower() or 'walk' in c.lower()]
print(f"\nGait-related columns in curated data: {len(gait_cols_in_curated)}")
if gait_cols_in_curated:
    print(f"  Columns: {gait_cols_in_curated[:5]}")
    
    # Try first gait column
    if gait_cols_in_curated:
        gait_curated = curated[curated[gait_cols_in_curated[0]].notna()]['PATNO'].unique()
        overlap6 = set(upsit_patients) & set(gait_curated)
        
        print(f"  Patients with {gait_cols_in_curated[0]}: {len(gait_curated):,}")
        print(f"  Overlap with UPSIT: {len(overlap6):,}")

# ============================================================================
# HYPOTHESIS 7: Check other gait files
# ============================================================================
header("HYPOTHESIS 7: Using Gait Arm Swing or Mobility Data")

upsit_gait_arm = set(upsit_patients) & set(gait_arm['PATNO'].unique())
upsit_gait_mob = set(upsit_patients) & set(gait_mobility['PATNO'].unique())

print(f"\nUPSIT + Gait Arm Swing: {len(upsit_gait_arm):,}")
print(f"UPSIT + Gait Mobility: {len(upsit_gait_mob):,}")

# ============================================================================
# SUMMARY
# ============================================================================
header("SUMMARY: ALL POSSIBLE INTERPRETATIONS")

print(f"""
Possible ways to get different multimodal cohort sizes:

1. UPSIT + Gait Features (unique patients):      {len(overlap1):,}  {'← Verified result' if len(overlap1) == 93 else ''}
2. UPSIT + ANY gait dataset:                     {len(overlap2):,}
3. UPSIT + Gait (total assessments):             {len(merged):,}  {'← POSSIBLE if counting assessments!' if len(merged) == 204 else ''}
4. UPSIT + Gait with speed:                      {len(overlap4):,}
5. UPSIT + Gait + Motor (three-way):             {len(three_way):,}
6. UPSIT + Gait Arm Swing:                       {len(upsit_gait_arm):,}
7. UPSIT + Gait Mobility:                        {len(upsit_gait_mob):,}
""")

header("MOST LIKELY EXPLANATION")

if len(merged) == 204:
    print(f"""
✅ FOUND IT! The n=204 comes from counting ASSESSMENTS, not unique patients!

Your original analysis counted:
  - Paired UPSIT + Gait assessments: {len(merged):,}
  - But unique patients in those pairs: {merged['PATNO'].nunique():,}

So n=204 is ASSESSMENTS, while n=93 is UNIQUE PATIENTS.

RECOMMENDATION:
  Either:
  a) Say "204 paired assessments from 93 unique patients"
  b) Change to "93 patients with multimodal data"
""")
else:
    print(f"\n⚠️  Could not identify source of n=204")
    print(f"Check your original multimodal analysis code to see what it counted")

print("="*90 + "\n")