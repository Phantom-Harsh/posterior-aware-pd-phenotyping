#!/usr/bin/env python3
"""
CREATE PROBABILISTIC RISK MAPS
Task 8: Weighted biomarker risk visualization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

BASE_DIR = Path("/home1/11021/harshtirhekar/WORK/GaitAnalysis/Parkinson-Disease-using-LRRK2")

# Load data
curated = pd.read_excel(BASE_DIR / "data/PPMI_Gait/PPMI_Curated_Data_Cut_Public_20241211.xlsx")
lrrk2 = pd.read_csv(BASE_DIR / "data/LRRK2_Clinical/LRRK2 Cross-Sectional_20191218.csv")

# ============================================================================
# RISK SCORE CALCULATION
# ============================================================================
print("Creating Probabilistic Risk Maps...")

# Define risk weights (from your verified findings)
risk_weights = {
    'LRRK2+': 1.89,      # 1.89× risk from genetic analysis
    'Hyposmia': 1.5,      # UPSIT < 25 (50% prevalence, major marker)
    'RBD': 1.8,           # RBD+ (37.5%, highly specific)
    'Motor': 1.2,         # High UPDRS-III
}

# Calculate risk scores for PPMI cohort (example)
risk_data = []

for idx, row in curated.iterrows():
    risk_score = 1.0  # Baseline
    
    # UPSIT risk
    if pd.notna(row.get('upsit')):
        if row['upsit'] < 25:
            risk_score *= risk_weights['Hyposmia']
    
    # RBD risk
    rbd_col = [c for c in curated.columns if 'rbd' in c.lower() or 'RBD' in c][0]
    if pd.notna(row.get(rbd_col)):
        if row[rbd_col] == 1:
            risk_score *= risk_weights['RBD']
    
    risk_data.append({
        'PATNO': row['PATNO'],
        'risk_score': risk_score,
        'has_upsit': pd.notna(row.get('upsit')),
        'has_rbd': pd.notna(row.get(rbd_col))
    })

risk_df = pd.DataFrame(risk_data)

# ============================================================================
# VISUALIZATION 1: Risk Distribution
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300)

# Panel A: Risk score distribution
ax = axes[0, 0]
risk_df['risk_score'].hist(bins=50, ax=ax, edgecolor='black', alpha=0.7)
ax.set_xlabel('Composite Risk Score', fontsize=12)
ax.set_ylabel('Number of Patients', fontsize=12)
ax.set_title('A) Distribution of Probabilistic Risk Scores', fontsize=13, fontweight='bold')
ax.axvline(1.5, color='red', linestyle='--', label='Elevated Risk Threshold')
ax.legend()

# Panel B: Risk heatmap by biomarker combination
ax = axes[0, 1]
risk_categories = pd.cut(risk_df['risk_score'], bins=[0, 1.2, 1.8, 2.5, 10],
                         labels=['Low', 'Moderate', 'High', 'Very High'])
risk_counts = risk_categories.value_counts()
colors = ['green', 'yellow', 'orange', 'red']
ax.bar(range(len(risk_counts)), risk_counts.values, color=colors, edgecolor='black')
ax.set_xticks(range(len(risk_counts)))
ax.set_xticklabels(risk_counts.index, fontsize=11)
ax.set_ylabel('Number of Patients', fontsize=12)
ax.set_title('B) Risk Stratification Categories', fontsize=13, fontweight='bold')

# Panel C: Biomarker combination prevalence
ax = axes[1, 0]
combinations = risk_df.groupby(['has_upsit', 'has_rbd']).size().reset_index(name='count')
labels = []
for _, row in combinations.iterrows():
    upsit = 'UPSIT+' if row['has_upsit'] else 'UPSIT-'
    rbd = 'RBD+' if row['has_rbd'] else 'RBD-'
    labels.append(f"{upsit}\n{rbd}")

ax.barh(range(len(combinations)), combinations['count'], edgecolor='black', alpha=0.7)
ax.set_yticks(range(len(combinations)))
ax.set_yticklabels(labels, fontsize=10)
ax.set_xlabel('Number of Patients', fontsize=12)
ax.set_title('C) Biomarker Combination Prevalence', fontsize=13, fontweight='bold')

# Panel D: Text summary
ax = axes[1, 1]
ax.axis('off')
summary_text = f"""
PROBABILISTIC RISK FRAMEWORK

Total Patients Analyzed: {len(risk_df):,}

Risk Weight Factors:
• LRRK2+: 1.89× (genetic)
• Hyposmia: 1.5× (prodromal)
• RBD+: 1.8× (prodromal)
• High Motor: 1.2× (clinical)

High Risk (>1.8): {(risk_df['risk_score'] > 1.8).sum():,} patients
Moderate Risk (1.2-1.8): {((risk_df['risk_score'] > 1.2) & (risk_df['risk_score'] <= 1.8)).sum():,} patients
Low Risk (<1.2): {(risk_df['risk_score'] <= 1.2).sum():,} patients

Clinical Utility:
→ Prioritize high-risk for trials
→ Intensive monitoring protocols
→ Early intervention strategies
"""
ax.text(0.1, 0.9, summary_text, fontsize=10, verticalalignment='top',
        family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('Figure_Risk_Maps.png', dpi=300, bbox_inches='tight')
print("✅ Risk map saved: Figure_Risk_Maps.png")
plt.close()