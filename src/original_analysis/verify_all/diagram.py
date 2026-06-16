#!/usr/bin/env python3
"""
Figure 1: Framework Diagram - CORRECTED VERSION
With updated PR = 1.92 (not 1.89)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(1, 1, figsize=(16, 9), dpi=300)
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.axis('off')

# Title
ax.text(8, 8.5, 'Integrated Precision Medicine Framework', 
        ha='center', fontsize=20, fontweight='bold')
ax.text(8, 8.0, 'Bayesian Machine Learning for Mechanism-Based Stratification',
        ha='center', fontsize=14, style='italic', color='darkblue')

# ============================================================================
# INPUT LAYER (Left)
# ============================================================================
input_x = 1.0
box_width = 2.8
box_height = 0.9

# Genetic
genetic = FancyBboxPatch((input_x, 6.5), box_width, box_height,
                         boxstyle="round,pad=0.1", 
                         edgecolor='darkgreen', facecolor='lightgreen', linewidth=2.5)
ax.add_patch(genetic)
ax.text(input_x + box_width/2, 7.0, 'GENETIC', ha='center', fontsize=11, fontweight='bold')
ax.text(input_x + box_width/2, 6.75, 'LRRK2 G2019S', ha='center', fontsize=10)
ax.text(input_x + box_width/2, 6.55, 'n=627 individuals', ha='center', fontsize=8)

# Molecular
molecular = FancyBboxPatch((input_x, 5.0), box_width, box_height,
                           boxstyle="round,pad=0.1",
                           edgecolor='darkblue', facecolor='lightblue', linewidth=2.5)
ax.add_patch(molecular)
ax.text(input_x + box_width/2, 5.5, 'MOLECULAR', ha='center', fontsize=11, fontweight='bold')
ax.text(input_x + box_width/2, 5.25, 'phospho-LRRK2', ha='center', fontsize=9)
ax.text(input_x + box_width/2, 5.05, 'CSF α-syn SAA', ha='center', fontsize=9)

# Digital Wearable
digital = FancyBboxPatch((input_x, 3.5), box_width, box_height,
                         boxstyle="round,pad=0.1",
                         edgecolor='darkorange', facecolor='lightyellow', linewidth=2.5)
ax.add_patch(digital)
ax.text(input_x + box_width/2, 4.0, 'DIGITAL SENSORS', ha='center', fontsize=11, fontweight='bold')
ax.text(input_x + box_width/2, 3.75, 'IMU Gait Metrics', ha='center', fontsize=9)
ax.text(input_x + box_width/2, 3.55, 'n=178 patients', ha='center', fontsize=8)

# Prodromal
prodromal = FancyBboxPatch((input_x, 2.0), box_width, box_height,
                           boxstyle="round,pad=0.1",
                           edgecolor='purple', facecolor='plum', linewidth=2.5)
ax.add_patch(prodromal)
ax.text(input_x + box_width/2, 2.5, 'PRODROMAL', ha='center', fontsize=11, fontweight='bold')
ax.text(input_x + box_width/2, 2.25, 'UPSIT, RBD', ha='center', fontsize=9)
ax.text(input_x + box_width/2, 2.05, 'n=5,122 / 1,548', ha='center', fontsize=8)

# ============================================================================
# ML ENGINE (Center) - WITH CORRECTED NUMBERS
# ============================================================================
ml_box = FancyBboxPatch((5.5, 2.5), 5, 4.5,
                        boxstyle="round,pad=0.15",
                        edgecolor='darkred', facecolor='mistyrose', linewidth=3.5)
ax.add_patch(ml_box)
ax.text(8, 6.5, 'Bayesian Machine Learning Engine', 
        ha='center', fontsize=15, fontweight='bold', color='darkred')

ax.text(8, 5.8, '• Evidence Lower Bound (ELBO) Selection', ha='center', fontsize=10)
ax.text(8, 5.4, '• Dirichlet Process Prior', ha='center', fontsize=10)
ax.text(8, 5.0, '• Uncertainty Quantification', ha='center', fontsize=10)
ax.text(8, 4.6, '• Bootstrap Validation (Jaccard=0.769)', ha='center', fontsize=10)
ax.text(8, 4.2, '• Silhouette=0.535 (K=4 clusters)', ha='center', fontsize=10)
ax.text(8, 3.6, 'PPMI: 4,775 patients | LRRK2: 627 individuals', 
        ha='center', fontsize=9, style='italic', color='darkblue')
ax.text(8, 3.2, 'Total: 14,473 longitudinal records', 
        ha='center', fontsize=9, style='italic', color='darkblue')

# ============================================================================
# OUTPUT LAYER (Right) - WITH CORRECTED PR=1.92
# ============================================================================
output_x = 11.5
output_width = 3.5

# Genetic Stratification - CORRECTED TO 1.92
strat = FancyBboxPatch((output_x, 6.3), output_width, 1.0,
                       boxstyle="round,pad=0.1",
                       edgecolor='darkgreen', facecolor='palegreen', linewidth=2.5)
ax.add_patch(strat)
ax.text(output_x + output_width/2, 6.9, 'Genetic Risk Stratification', 
        ha='center', fontsize=12, fontweight='bold')
ax.text(output_x + output_width/2, 6.6, 'PR=1.92 [1.54-2.40]', 
        ha='center', fontsize=9, color='darkgreen', fontweight='bold')
ax.text(output_x + output_width/2, 6.35, 'χ²=36.6, p=1.4×10⁻⁹', 
        ha='center', fontsize=8)

# Continuous Monitoring
monitor = FancyBboxPatch((output_x, 4.9), output_width, 1.0,
                         boxstyle="round,pad=0.1",
                         edgecolor='darkorange', facecolor='peachpuff', linewidth=2.5)
ax.add_patch(monitor)
ax.text(output_x + output_width/2, 5.5, 'Continuous Monitoring', 
        ha='center', fontsize=12, fontweight='bold')
ax.text(output_x + output_width/2, 5.2, 'Digital Biomarkers', 
        ha='center', fontsize=9)
ax.text(output_x + output_width/2, 4.95, 'ASA 27% | Dual-Task 14.87%', 
        ha='center', fontsize=8)

# Therapeutic Targeting
therapy = FancyBboxPatch((output_x, 3.5), output_width, 1.0,
                         boxstyle="round,pad=0.1",
                         edgecolor='darkblue', facecolor='lightcyan', linewidth=2.5)
ax.add_patch(therapy)
ax.text(output_x + output_width/2, 4.1, 'Therapeutic Targeting', 
        ha='center', fontsize=12, fontweight='bold')
ax.text(output_x + output_width/2, 3.8, 'Mechanism-Matched Rx', 
        ha='center', fontsize=9)
ax.text(output_x + output_width/2, 3.55, 'LRRK2 kinase inhibitors', 
        ha='center', fontsize=8)

# Early Detection
early = FancyBboxPatch((output_x, 2.1), output_width, 1.0,
                       boxstyle="round,pad=0.1",
                       edgecolor='purple', facecolor='lavender', linewidth=2.5)
ax.add_patch(early)
ax.text(output_x + output_width/2, 2.7, 'Early Detection', 
        ha='center', fontsize=12, fontweight='bold')
ax.text(output_x + output_width/2, 2.4, 'Prodromal Markers', 
        ha='center', fontsize=9)
ax.text(output_x + output_width/2, 2.15, 'UPSIT 50.2% | RBD 37.5%', 
        ha='center', fontsize=8)

# Arrows - Input to ML
for y_pos in [7.0, 5.5, 4.0, 2.5]:
    arrow = FancyArrowPatch((input_x + box_width, y_pos), (5.5, 4.75),
                           arrowstyle='->', mutation_scale=25,
                           linewidth=2.5, color='gray', alpha=0.7)
    ax.add_patch(arrow)

# Arrows - ML to Output
for y_pos in [6.8, 5.4, 4.0, 2.6]:
    arrow = FancyArrowPatch((10.5, 4.75), (output_x, y_pos),
                           arrowstyle='->', mutation_scale=25,
                           linewidth=2.5, color='gray', alpha=0.7)
    ax.add_patch(arrow)

# Bottom annotation
ax.text(8, 0.8, 'Aligned with UN SDGs: SDG 3 (Health) | SDG 9 (Innovation) | SDG 10 (Equity)',
        ha='center', fontsize=10, style='italic', color='darkblue')
ax.text(8, 0.3, 'Complete code-data-result traceability | TRIPOD+AI compliant',
        ha='center', fontsize=9, style='italic', color='gray')

plt.tight_layout()
plt.savefig('Figure1_Framework_CORRECTED.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
print("✅ Figure 1 (CORRECTED) saved!")
plt.close()