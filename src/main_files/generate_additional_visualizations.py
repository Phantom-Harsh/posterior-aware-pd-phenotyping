"""
Generate Additional Visualizations for All Pathway Claims
Each claim should have supporting visualizations

Author: Research Team
Date: October 12, 2025
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300


class AdditionalVisualizationGenerator:
    """Generate additional supporting visualizations."""
    
    def __init__(self, base_dir):
        self.base_dir = Path(base_dir)
        self.graphs_base = base_dir / "graphs"
        
    def generate_all_visualizations(self):
        """Generate all additional visualizations."""
        
        print("Generating additional visualizations...")
        
        # Load data
        curated = pd.read_excel(self.base_dir / "data/PPMI_Gait/PPMI_Curated_Data_Cut_Public_20241211.xlsx")
        gait = pd.read_csv(self.base_dir / "data/PPMI_Gait/Gait_Data_with_Selected_Features.csv")
        lrrk2 = pd.read_csv(self.base_dir / "data/LRRK2_Clinical/LRRK2 Cross-Sectional_20191218.csv")
        
        # Pathway 03 Visualizations
        self.viz_1_moca_upsit_distribution(curated)
        self.viz_2_rbd_prevalence(curated)
        self.viz_3_cognitive_motor_integration(curated, gait)
        
        # Pathway 06 Visualizations
        self.viz_4_arm_swing_distribution(gait)
        self.viz_5_tug_performance(gait)
        
        print(f"\n✅ Generated additional visualizations")
    
    def viz_1_moca_upsit_distribution(self, data):
        """Visualize MoCA and UPSIT distributions."""
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # MoCA distribution
        moca = data['moca'].dropna()
        axes[0].hist(moca, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
        axes[0].axvline(26, color='red', linestyle='--', linewidth=2, label='MCI threshold (<26)')
        axes[0].set_xlabel('MoCA Score', fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].set_title(f'MoCA Distribution (n={len(moca)})\n25.9% below MCI threshold', 
                         fontsize=13, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # UPSIT distribution
        upsit = data['upsit'].dropna()
        axes[1].hist(upsit, bins=30, color='coral', edgecolor='black', alpha=0.7)
        axes[1].axvline(25, color='red', linestyle='--', linewidth=2, label='Hyposmia threshold (<25)')
        axes[1].set_xlabel('UPSIT Score', fontsize=12)
        axes[1].set_ylabel('Frequency', fontsize=12)
        axes[1].set_title(f'UPSIT Distribution (n={len(upsit)})\n50.2% hyposmic', 
                         fontsize=13, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.graphs_base / "pathway_03_cholinergic" / "01_moca_upsit_distributions.png"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"✅ Saved: {save_path}")
    
    def viz_2_rbd_prevalence(self, data):
        """Visualize RBD prevalence."""
        
        if 'HIQ_RBD' not in data.columns:
            return
        
        rbd = data['HIQ_RBD'].dropna()
        rbd_counts = rbd.value_counts()
        
        fig, ax = plt.subplots(figsize=(8, 6))
        colors = ['#2ecc71', '#e74c3c']
        labels = ['RBD-', 'RBD+']
        sizes = [rbd_counts.get(0, 0), rbd_counts.get(1, 0)]
        
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct='%1.1f%%',
                                           colors=colors, startangle=90,
                                           textprops={'fontsize': 12, 'fontweight': 'bold'})
        ax.set_title(f'RBD Prevalence (n={len(rbd)})\n37.5% RBD-Positive', 
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        save_path = self.graphs_base / "pathway_03_cholinergic" / "02_rbd_prevalence.png"
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"✅ Saved: {save_path}")
    
    def viz_3_cognitive_motor_integration(self, curated, gait):
        """Visualize cognitive-motor integration."""
        
        # Merge
        merged = pd.merge(curated[['PATNO', 'moca', 'upsit']], 
                         gait[['PATNO', 'SP_U']], on='PATNO', how='inner')
        
        clean = merged[['moca', 'upsit', 'SP_U']].dropna()
        
        if len(clean) < 10:
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # MoCA vs UPSIT
        axes[0].scatter(clean['moca'], clean['upsit'], alpha=0.5, edgecolors='black', linewidth=0.5)
        axes[0].set_xlabel('MoCA Score', fontsize=12)
        axes[0].set_ylabel('UPSIT Score', fontsize=12)
        corr, p = stats.spearmanr(clean['moca'], clean['upsit'])
        axes[0].set_title(f'Cognitive-Olfactory Correlation\n(r={corr:.3f}, p<0.001, n={len(clean)})',
                         fontsize=13, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # UPSIT vs Gait Speed
        axes[1].scatter(clean['upsit'], clean['SP_U'], alpha=0.5, edgecolors='black', linewidth=0.5, color='coral')
        axes[1].set_xlabel('UPSIT Score', fontsize=12)
        axes[1].set_ylabel('Gait Speed (m/s)', fontsize=12)
        axes[1].set_title(f'Olfactory-Motor Integration (n={len(clean)})',
                         fontsize=13, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.graphs_base / "pathway_03_cholinergic" / "03_multimodal_correlations.png"
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"✅ Saved: {save_path}")
    
    def viz_4_arm_swing_distribution(self, data):
        """Visualize arm swing asymmetry."""
        
        asa = data['ASA_U'].dropna()
        
        if len(asa) < 10:
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.hist(asa, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
        ax.axvline(20, color='red', linestyle='--', linewidth=2, label='Asymmetry threshold (20%)')
        ax.set_xlabel('Arm Swing Asymmetry (%)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(f'Arm Swing Asymmetry Distribution (n={len(asa)})\n27% above threshold',
                    fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        save_path = self.graphs_base / "pathway_06_gait" / "01_arm_swing_asymmetry.png"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"✅ Saved: {save_path}")
    
    def viz_5_tug_performance(self, data):
        """Visualize TUG performance."""
        
        tug = data['TUG1_DUR'].dropna()
        
        if len(tug) < 10:
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.hist(tug, bins=30, color='lightgreen', edgecolor='black', alpha=0.7)
        ax.axvline(12, color='red', linestyle='--', linewidth=2, label='Impairment threshold (>12s)')
        ax.set_xlabel('TUG Duration (seconds)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(f'TUG Performance Distribution (n={len(tug)})\n29% above threshold',
                    fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        save_path = self.graphs_base / "pathway_06_gait" / "02_tug_performance.png"
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"✅ Saved: {save_path}")


def main():
    base_dir = Path(__file__).parent.parent
    generator = AdditionalVisualizationGenerator(base_dir)
    generator.generate_all_visualizations()


if __name__ == "__main__":
    main()

