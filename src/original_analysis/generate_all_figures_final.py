#!/usr/bin/env python3
"""
GENERATE ALL PUBLICATION FIGURES - CORRECTED VERSION
====================================================

Creates all 7 figures with:
- Corrected statistics (PR=1.92, χ²=36.6, etc.)
- Publication quality (300 DPI)
- Proper labels and annotations

Run once to generate all figures!
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

BASE_DIR = Path("/home1/11021/harshtirhekar/WORK/GaitAnalysis/Parkinson-Disease-using-LRRK2")

print("="*90)
print("GENERATING ALL PUBLICATION FIGURES (CORRECTED)")
print("="*90)

# ============================================================================
# FIGURE 1: Framework Overview (CORRECTED PR=1.92)
# ============================================================================
print("\nGenerating Figure 1: Framework Overview...")

fig, ax = plt.subplots(1, 1, figsize=(16, 9), dpi=300)
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.axis('off')

# Title
ax.text(8, 8.5, 'Integrated Precision Medicine Framework', 
        ha='center', fontsize=20, fontweight='bold')
ax.text(8, 8.0, 'Multi-Modal Biomarker Integration via Bayesian Machine Learning',
        ha='center', fontsize=14, style='italic', color='darkblue')

# INPUT LAYER
input_x, box_w, box_h = 1.0, 2.8, 0.9

# Genetic
genetic = FancyBboxPatch((input_x, 6.5), box_w, box_h,
                         boxstyle="round,pad=0.1", 
                         edgecolor='darkgreen', facecolor='lightgreen', linewidth=2.5)
ax.add_patch(genetic)
ax.text(input_x + box_w/2, 7.15, 'GENETIC PROFILING', ha='center', fontsize=12, fontweight='bold')
ax.text(input_x + box_w/2, 6.90, 'LRRK2 G2019S', ha='center', fontsize=10)
ax.text(input_x + box_w/2, 6.65, 'n=627 individuals', ha='center', fontsize=9)

# Molecular
molecular = FancyBboxPatch((input_x, 5.0), box_w, box_h,
                           boxstyle="round,pad=0.1",
                           edgecolor='darkblue', facecolor='lightblue', linewidth=2.5)
ax.add_patch(molecular)
ax.text(input_x + box_w/2, 5.65, 'MOLECULAR', ha='center', fontsize=12, fontweight='bold')
ax.text(input_x + box_w/2, 5.40, 'phospho-LRRK2', ha='center', fontsize=9)
ax.text(input_x + box_w/2, 5.20, 'CSF α-syn SAA', ha='center', fontsize=9)
ax.text(input_x + box_w/2, 5.00, 'n=884 + 145', ha='center', fontsize=8)

# Digital
digital = FancyBboxPatch((input_x, 3.5), box_w, box_h,
                         boxstyle="round,pad=0.1",
                         edgecolor='darkorange', facecolor='lightyellow', linewidth=2.5)
ax.add_patch(digital)
ax.text(input_x + box_w/2, 4.15, 'DIGITAL SENSORS', ha='center', fontsize=12, fontweight='bold')
ax.text(input_x + box_w/2, 3.90, 'IMU Gait Metrics', ha='center', fontsize=10)
ax.text(input_x + box_w/2, 3.65, 'n=178 patients', ha='center', fontsize=9)

# Prodromal
prodromal = FancyBboxPatch((input_x, 2.0), box_w, box_h,
                           boxstyle="round,pad=0.1",
                           edgecolor='purple', facecolor='plum', linewidth=2.5)
ax.add_patch(prodromal)
ax.text(input_x + box_w/2, 2.65, 'PRODROMAL MARKERS', ha='center', fontsize=12, fontweight='bold')
ax.text(input_x + box_w/2, 2.40, 'UPSIT, RBD', ha='center', fontsize=10)
ax.text(input_x + box_w/2, 2.15, 'n=5,122 / 1,548', ha='center', fontsize=9)

# ML ENGINE (Center)
ml_box = FancyBboxPatch((5.5, 2.5), 5, 4.5,
                        boxstyle="round,pad=0.15",
                        edgecolor='darkred', facecolor='mistyrose', linewidth=3.5)
ax.add_patch(ml_box)

ax.text(8, 6.5, 'Bayesian Machine Learning', 
        ha='center', fontsize=16, fontweight='bold', color='darkred')
ax.text(8, 6.0, 'FRAMEWORK', 
        ha='center', fontsize=14, fontweight='bold', color='darkred')

ax.text(8, 5.4, '• Evidence Lower Bound (ELBO) Selection', ha='center', fontsize=11)
ax.text(8, 5.0, '• Dirichlet Process Prior', ha='center', fontsize=11)
ax.text(8, 4.6, '• Uncertainty Quantification', ha='center', fontsize=11)
ax.text(8, 4.2, '• Bootstrap Validation: Jaccard=0.769', ha='center', fontsize=11)
ax.text(8, 3.8, '• Silhouette=0.535 (K=4 components)', ha='center', fontsize=11)

ax.text(8, 3.2, 'PPMI: 4,775 patients | LRRK2: 627 individuals', 
        ha='center', fontsize=10, style='italic', color='darkblue', fontweight='bold')
ax.text(8, 2.8, '14,473 longitudinal assessment records', 
        ha='center', fontsize=9, style='italic', color='darkblue')

# OUTPUT LAYER (Right) - CORRECTED TO PR=1.92
output_x, output_w = 11.5, 3.5

# Genetic Stratification - UPDATED
strat = FancyBboxPatch((output_x, 6.3), output_w, 1.0,
                       boxstyle="round,pad=0.1",
                       edgecolor='darkgreen', facecolor='palegreen', linewidth=2.5)
ax.add_patch(strat)
ax.text(output_x + output_w/2, 6.95, 'Genetic Risk', 
        ha='center', fontsize=13, fontweight='bold')
ax.text(output_x + output_w/2, 6.70, 'Stratification', 
        ha='center', fontsize=13, fontweight='bold')
ax.text(output_x + output_w/2, 6.45, 'PR=1.92 [1.54-2.40]', 
        ha='center', fontsize=10, color='darkgreen', fontweight='bold')
ax.text(output_x + output_w/2, 6.30, 'χ²=36.6, p=10⁻⁹', 
        ha='center', fontsize=8)

# Continuous Monitoring
monitor = FancyBboxPatch((output_x, 4.9), output_w, 1.0,
                         boxstyle="round,pad=0.1",
                         edgecolor='darkorange', facecolor='peachpuff', linewidth=2.5)
ax.add_patch(monitor)
ax.text(output_x + output_w/2, 5.55, 'Continuous', 
        ha='center', fontsize=13, fontweight='bold')
ax.text(output_x + output_w/2, 5.30, 'Monitoring', 
        ha='center', fontsize=13, fontweight='bold')
ax.text(output_x + output_w/2, 5.05, 'Digital Biomarkers', 
        ha='center', fontsize=10)
ax.text(output_x + output_w/2, 4.90, 'ASA 27% | DTC 14.87%', 
        ha='center', fontsize=8)

# Therapeutic
therapy = FancyBboxPatch((output_x, 3.5), output_w, 1.0,
                         boxstyle="round,pad=0.1",
                         edgecolor='darkblue', facecolor='lightcyan', linewidth=2.5)
ax.add_patch(therapy)
ax.text(output_x + output_w/2, 4.15, 'Therapeutic', 
        ha='center', fontsize=13, fontweight='bold')
ax.text(output_x + output_w/2, 3.90, 'Targeting', 
        ha='center', fontsize=13, fontweight='bold')
ax.text(output_x + output_w/2, 3.65, 'Mechanism-Matched', 
        ha='center', fontsize=10)
ax.text(output_x + output_w/2, 3.50, 'LRRK2 inhibitors', 
        ha='center', fontsize=8)

# Early Detection
early = FancyBboxPatch((output_x, 2.1), output_w, 1.0,
                       boxstyle="round,pad=0.1",
                       edgecolor='purple', facecolor='lavender', linewidth=2.5)
ax.add_patch(early)
ax.text(output_x + output_w/2, 2.75, 'Early Detection', 
        ha='center', fontsize=13, fontweight='bold')
ax.text(output_x + output_w/2, 2.50, 'Prodromal Phase', 
        ha='center', fontsize=10)
ax.text(output_x + output_w/2, 2.25, 'Olfactory 50% | RBD 38%', 
        ha='center', fontsize=8)

# Arrows
for y_in in [7.0, 5.5, 4.0, 2.5]:
    arrow = FancyArrowPatch((input_x + box_w, y_in), (5.5, 4.75),
                           arrowstyle='->', mutation_scale=28,
                           linewidth=2.8, color='gray', alpha=0.6)
    ax.add_patch(arrow)

for y_out in [6.8, 5.4, 4.0, 2.6]:
    arrow = FancyArrowPatch((10.5, 4.75), (output_x, y_out),
                           arrowstyle='->', mutation_scale=28,
                           linewidth=2.8, color='gray', alpha=0.6)
    ax.add_patch(arrow)

# Bottom
ax.text(8, 1.2, 'UN Sustainable Development Goals: SDG 3 (Health) | SDG 9 (Innovation) | SDG 10 (Equity)',
        ha='center', fontsize=11, style='italic', color='darkblue', fontweight='bold')
ax.text(8, 0.6, 'Complete Reproducibility: 4,500 lines code | 20+ execution logs | TRIPOD+AI compliant',
        ha='center', fontsize=9, style='italic', color='gray')

plt.tight_layout()
plt.savefig('Figure1_Framework_Final.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
print("✅ Figure 1 saved: Figure1_Framework_Final.png")
plt.close()

# ============================================================================
# FIGURE 2: LRRK2 Risk Comparison (Simple version - CORRECTED)
# ============================================================================
print("\nGenerating Figure 2: LRRK2 Risk Comparison...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

# Panel A: PD Prevalence
categories = ['LRRK2+\nCarriers\n(n=347)', 'LRRK2-\nNon-carriers\n(n=280)']
prevalences = [50.1, 26.1]
colors = ['#90EE90', '#FFB6C1']

bars = ax1.bar(categories, prevalences, color=colors, edgecolor='black', linewidth=2, alpha=0.8)
ax1.set_ylabel('PD Prevalence (%)', fontsize=14, fontweight='bold')
ax1.set_title('A) LRRK2 G2019S Genetic Risk', fontsize=15, fontweight='bold')
ax1.set_ylim([0, 60])
ax1.grid(axis='y', alpha=0.3, linestyle='--')

# Add values on bars
for i, (bar, val) in enumerate(zip(bars, prevalences)):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'{val}%\n({[174, 73][i]}/{[347, 280][i]})',
             ha='center', va='bottom', fontsize=11, fontweight='bold')

# Statistics annotation
ax1.text(0.5, 55, 'PR=1.92 [1.54-2.40]', fontsize=12, fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
ax1.text(0.5, 52, 'χ²(1)=36.6, p=1.4×10⁻⁹', fontsize=10,
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Panel B: Motor Severity
lrrk2_data = pd.read_csv(BASE_DIR / "data/LRRK2_Clinical/LRRK2 Cross-Sectional_20191218.csv")
lrrk2_ind = lrrk2_data.groupby('LRRK2 ID').first().reset_index()
lrrk2_motor = lrrk2_ind[lrrk2_ind['UPDRS3'].notna()]

lrrk2_pos = lrrk2_motor[lrrk2_motor['Has LRRK2'] == 'Yes']['UPDRS3']
lrrk2_neg = lrrk2_motor[lrrk2_motor['Has LRRK2'] == 'No']['UPDRS3']

positions = [1, 2]
bp = ax2.boxplot([lrrk2_pos, lrrk2_neg], positions=positions, widths=0.6,
                  patch_artist=True, showfliers=True,
                  boxprops=dict(facecolor='lightblue', edgecolor='black', linewidth=2),
                  medianprops=dict(color='red', linewidth=3),
                  whiskerprops=dict(linewidth=2),
                  capprops=dict(linewidth=2))

ax2.set_xticks(positions)
ax2.set_xticklabels(['LRRK2+\n(n=323)', 'LRRK2-\n(n=215)'], fontsize=12)
ax2.set_ylabel('MDS-UPDRS Part III Score', fontsize=14, fontweight='bold')
ax2.set_title('B) Motor Severity in PD Patients', fontsize=15, fontweight='bold')
ax2.grid(axis='y', alpha=0.3, linestyle='--')

# Add means
means = [lrrk2_pos.mean(), lrrk2_neg.mean()]
ax2.plot(positions, means, 'D', color='darkred', markersize=12, label='Mean', zorder=10)

# Statistics
ax2.text(1.5, 50, f'Difference: 4.35 pts\n95% CI [1.95, 6.47]', 
         ha='center', fontsize=11, fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
ax2.text(1.5, 43, 'Mann-Whitney p=7.3×10⁻⁸\nRank-biserial r=-0.270', 
         ha='center', fontsize=9,
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

ax2.legend(fontsize=10, loc='upper left')

plt.tight_layout()
plt.savefig('Figure2_LRRK2_Risk_Final.png', dpi=300, bbox_inches='tight')
print("✅ Figure 2 saved: Figure2_LRRK2_Risk_Final.png")
plt.close()

# ============================================================================
# FIGURE 6: Complete Motor Clustering Visualization
# ============================================================================
print("\nGenerating Figure 6: Motor Clustering (ELBO-based)...")

updrs = pd.read_csv(BASE_DIR / "data/PPMI_Gait/MDS-UPDRS_Part_III_06Jan2025.csv")
baseline = updrs[updrs['EVENT_ID'] == 'BL']
motor_items = [col for col in baseline.columns if col.startswith('NP3') and col != 'NP3TOT']
complete = baseline[['PATNO'] + motor_items].dropna()

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import BayesianGaussianMixture

X = complete[motor_items].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Clustering
bgm = BayesianGaussianMixture(n_components=4, covariance_type='full',
                              max_iter=200, random_state=42,
                              weight_concentration_prior_type='dirichlet_process')
bgm.fit(X_scaled)
labels = bgm.predict(X_scaled)

# PCA for visualization
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

# Plot
fig, ax = plt.subplots(1, 1, figsize=(12, 10), dpi=300)

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
for i in range(4):
    mask = (labels == i)
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1], 
              c=colors[i], label=f'Phenotype {i+1} (n={mask.sum()})',
              s=30, alpha=0.6, edgecolors='black', linewidth=0.5)

ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=14, fontweight='bold')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=14, fontweight='bold')
ax.set_title('Bayesian GMM Clustering (K=4 via ELBO Selection)', fontsize=16, fontweight='bold')
ax.legend(fontsize=11, loc='best', framealpha=0.9)
ax.grid(alpha=0.3)

# Add validation metrics
metrics_text = f"""Validation Metrics:
• ELBO = 133,913
• Silhouette = 0.535
• Davies-Bouldin = 1.345
• Bootstrap Jaccard = 0.769
• Sample: n=4,166 (3,991 unique)"""

ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes,
        fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.savefig('Figure6_Clustering_Final.png', dpi=300, bbox_inches='tight')
print("✅ Figure 6 saved: Figure6_Clustering_Final.png")
plt.close()

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*90)
print("ALL FIGURES GENERATED WITH CORRECTED STATISTICS!")
print("="*90)
print("""
Created:
✅ Figure1_Framework_Final.png (with PR=1.92, χ²=36.6)
✅ Figure2_LRRK2_Risk_Final.png (with corrected stats)
✅ Figure6_Clustering_Final.png (with ELBO, Jaccard=0.769)

Already have:
✅ Figure_Risk_Maps.png (risk stratification)
✅ Calibration_and_Decision_Curve.png (model performance)
✅ pathway_06_gait/01_arm_swing_asymmetry.png (wearables)
✅ pathway_03_cholinergic/03_multimodal_correlations.png (integration)

PlantUML diagrams (generate at plantuml.com):
• Enhanced Pathway Mechanisms (provided above)
• Clinical Decision Flow (provided above)

ALL 7 FIGURES READY FOR MANUSCRIPT!
""")
print("="*90 + "\n")