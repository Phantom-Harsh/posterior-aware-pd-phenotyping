#!/usr/bin/env python3
"""
Create Figure 1: Multimodal Precision Medicine Framework Diagram
Publication quality (300 DPI)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

fig, ax = plt.subplots(1, 1, figsize=(14, 8), dpi=300)
ax.set_xlim(0, 14)
ax.set_ylim(0, 8)
ax.axis('off')

# Title
ax.text(7, 7.5, 'Integrated Precision Medicine Framework', 
        ha='center', fontsize=18, fontweight='bold')

# ============================================================================
# INPUT LAYER (Left side)
# ============================================================================
input_y = 6
box_height = 0.8
box_width = 2.5

# Genetic Input
genetic_box = FancyBboxPatch((0.5, input_y-0.4), box_width, box_height,
                             boxstyle="round,pad=0.1", 
                             edgecolor='darkgreen', facecolor='lightgreen',
                             linewidth=2)
ax.add_patch(genetic_box)
ax.text(1.75, input_y, 'Genetic\nLRRK2 G2019S\nn=627', 
        ha='center', va='center', fontsize=10, fontweight='bold')

# Molecular Input
molecular_box = FancyBboxPatch((0.5, input_y-2.2), box_width, box_height,
                               boxstyle="round,pad=0.1",
                               edgecolor='darkblue', facecolor='lightblue',
                               linewidth=2)
ax.add_patch(molecular_box)
ax.text(1.75, input_y-1.8, 'Molecular\nphospho-LRRK2\nCSF α-syn SAA\nn=1,029',
        ha='center', va='center', fontsize=10, fontweight='bold')

# Digital/Wearable Input
digital_box = FancyBboxPatch((0.5, input_y-4), box_width, box_height,
                             boxstyle="round,pad=0.1",
                             edgecolor='darkorange', facecolor='lightyellow',
                             linewidth=2)
ax.add_patch(digital_box)
ax.text(1.75, input_y-3.6, 'Digital Sensors\nIMU Gait\nn=178',
        ha='center', va='center', fontsize=10, fontweight='bold')

# Clinical Input
clinical_box = FancyBboxPatch((0.5, input_y-5.8), box_width, box_height,
                              boxstyle="round,pad=0.1",
                              edgecolor='purple', facecolor='plum',
                              linewidth=2)
ax.add_patch(clinical_box)
ax.text(1.75, input_y-5.4, 'Clinical\nUPSIT, RBD\nn=5,122',
        ha='center', va='center', fontsize=10, fontweight='bold')

# ============================================================================
# AI/ML ENGINE (Center)
# ============================================================================
ml_box = FancyBboxPatch((5, 2), 4, 4,
                        boxstyle="round,pad=0.15",
                        edgecolor='red', facecolor='mistyrose',
                        linewidth=3)
ax.add_patch(ml_box)
ax.text(7, 5.5, 'Bayesian Machine Learning', 
        ha='center', fontsize=14, fontweight='bold', color='darkred')
ax.text(7, 4.8, 'Framework', 
        ha='center', fontsize=14, fontweight='bold', color='darkred')
ax.text(7, 4, '• Bayesian GMM', ha='center', fontsize=10)
ax.text(7, 3.5, '• Uncertainty Quantification', ha='center', fontsize=10)
ax.text(7, 3, '• Dirichlet Process Prior', ha='center', fontsize=10)
ax.text(7, 2.5, '• BIC Model Selection', ha='center', fontsize=10)

# ============================================================================
# ARROWS (Input -> ML)
# ============================================================================
for y_pos in [input_y, input_y-1.8, input_y-3.6, input_y-5.4]:
    arrow = FancyArrowPatch((3.2, y_pos), (5, 4),
                           arrowstyle='->', mutation_scale=20,
                           linewidth=2, color='gray')
    ax.add_patch(arrow)

# ============================================================================
# OUTPUT LAYER (Right side)
# ============================================================================
output_x = 10.5

# Precision Stratification
strat_box = FancyBboxPatch((output_x, 5.5), 3, 1,
                           boxstyle="round,pad=0.1",
                           edgecolor='darkviolet', facecolor='lavender',
                           linewidth=2)
ax.add_patch(strat_box)
ax.text(12, 6, 'Precision Stratification', ha='center', fontsize=11, fontweight='bold')
ax.text(12, 5.7, 'Risk: 1.89× LRRK2+', ha='center', fontsize=8)

# Continuous Monitoring
monitor_box = FancyBboxPatch((output_x, 4), 3, 1,
                             boxstyle="round,pad=0.1",
                             edgecolor='darkorange', facecolor='peachpuff',
                             linewidth=2)
ax.add_patch(monitor_box)
ax.text(12, 4.5, 'Continuous Monitoring', ha='center', fontsize=11, fontweight='bold')
ax.text(12, 4.2, 'Digital Biomarkers', ha='center', fontsize=8)

# Therapeutic Targeting
therapy_box = FancyBboxPatch((output_x, 2.5), 3, 1,
                             boxstyle="round,pad=0.1",
                             edgecolor='darkgreen', facecolor='honeydew',
                             linewidth=2)
ax.add_patch(therapy_box)
ax.text(12, 3, 'Therapeutic Targeting', ha='center', fontsize=11, fontweight='bold')
ax.text(12, 2.7, 'Mechanism-Based', ha='center', fontsize=8)

# Early Detection
early_box = FancyBboxPatch((output_x, 1), 3, 1,
                           boxstyle="round,pad=0.1",
                           edgecolor='darkblue', facecolor='lightcyan',
                           linewidth=2)
ax.add_patch(early_box)
ax.text(12, 1.5, 'Early Detection', ha='center', fontsize=11, fontweight='bold')
ax.text(12, 1.2, 'Prodromal Markers', ha='center', fontsize=8)

# ============================================================================
# ARROWS (ML -> Output)
# ============================================================================
for y_pos in [6, 4.5, 3, 1.5]:
    arrow = FancyArrowPatch((9, 4), (output_x, y_pos),
                           arrowstyle='->', mutation_scale=20,
                           linewidth=2, color='gray')
    ax.add_patch(arrow)

# ============================================================================
# Bottom annotation
# ============================================================================
ax.text(7, 0.3, 'PPMI: 4,775 patients | LRRK2: 627 individuals | Total: 14,473 longitudinal records',
        ha='center', fontsize=9, style='italic')

plt.tight_layout()
plt.savefig('Figure1_Framework_Diagram.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
print("✅ Figure 1 saved: Figure1_Framework_Diagram.png")
plt.close()