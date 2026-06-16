"""
Visualization Module
Author: Research Team
Date: October 2025

This module provides comprehensive visualization capabilities for:
- Exploratory data analysis
- Clustering results
- Pathway analysis
- Statistical test results
- Longitudinal tracking
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Set publication-quality style
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.figsize'] = (10, 6)


class Visualizer:
    """
    Comprehensive visualization toolkit for PD mechanism research.
    """
    
    def __init__(self, output_dir, logger):
        """
        Initialize the visualizer.
        
        Args:
            output_dir: Directory to save plots
            logger: Logger instance
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger
        self.saved_plots = []
        
    def plot_cluster_scatter(self, X_2d: np.ndarray, 
                            labels: np.ndarray,
                            uncertainties: Optional[np.ndarray] = None,
                            title: str = "Cluster Analysis",
                            filename: str = "cluster_scatter.png",
                            subdir: str = "clustering") -> str:
        """
        Create a scatter plot of clustered data in 2D space.
        
        Args:
            X_2d: 2D array of data points (PCA or t-SNE reduced)
            labels: Cluster labels
            uncertainties: Optional uncertainty values for each point
            title: Plot title
            filename: Output filename
            subdir: Subdirectory for plot
            
        Returns:
            Path to saved plot
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        unique_labels = np.unique(labels)
        colors = sns.color_palette("husl", len(unique_labels))
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            if uncertainties is not None:
                # Size points by uncertainty (inverse)
                sizes = 50 + 100 * (1 - uncertainties[mask])
                scatter = ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                                   c=[colors[i]], s=sizes, alpha=0.6,
                                   label=f'Cluster {label}', edgecolors='black', linewidth=0.5)
            else:
                scatter = ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                                   c=[colors[i]], s=100, alpha=0.6,
                                   label=f'Cluster {label}', edgecolors='black', linewidth=0.5)
        
        ax.set_xlabel('Principal Component 1', fontsize=12)
        ax.set_ylabel('Principal Component 2', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='best', frameon=True)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        save_path = self.output_dir / subdir / filename
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        
        self.saved_plots.append(str(save_path))
        self.logger.info(f"Saved cluster scatter plot: {save_path}")
        
        return str(save_path)
    
    def plot_uncertainty_distribution(self, uncertainties: np.ndarray,
                                     labels: np.ndarray,
                                     filename: str = "uncertainty_distribution.png",
                                     subdir: str = "clustering") -> str:
        """
        Plot distribution of uncertainty scores by cluster.
        
        Args:
            uncertainties: Array of uncertainty values
            labels: Cluster labels
            filename: Output filename
            subdir: Subdirectory
            
        Returns:
            Path to saved plot
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Histogram of uncertainties
        axes[0].hist(uncertainties, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
        axes[0].axvline(0.7, color='red', linestyle='--', linewidth=2, label='High Uncertainty Threshold')
        axes[0].set_xlabel('Uncertainty Score', fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].set_title('Distribution of Uncertainty Scores', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Box plot by cluster
        uncertainty_df = pd.DataFrame({
            'Uncertainty': uncertainties,
            'Cluster': labels
        })
        sns.boxplot(data=uncertainty_df, x='Cluster', y='Uncertainty', ax=axes[1], palette='Set2')
        axes[1].axhline(0.7, color='red', linestyle='--', linewidth=2, label='High Uncertainty')
        axes[1].set_xlabel('Cluster', fontsize=12)
        axes[1].set_ylabel('Uncertainty Score', fontsize=12)
        axes[1].set_title('Uncertainty by Cluster', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        save_path = self.output_dir / subdir / filename
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        
        self.saved_plots.append(str(save_path))
        self.logger.info(f"Saved uncertainty distribution plot: {save_path}")
        
        return str(save_path)
    
    def plot_cluster_heatmap(self, characterization_df: pd.DataFrame,
                            filename: str = "cluster_heatmap.png",
                            subdir: str = "clustering") -> str:
        """
        Create a heatmap of cluster characteristics.
        
        Args:
            characterization_df: DataFrame with cluster feature means
            filename: Output filename
            subdir: Subdirectory
            
        Returns:
            Path to saved plot
        """
        # Select only mean columns
        mean_cols = [col for col in characterization_df.columns if col.endswith('_mean')]
        
        if len(mean_cols) == 0:
            self.logger.warning("No mean columns found for heatmap")
            return ""
        
        # Create matrix for heatmap
        heatmap_data = characterization_df[['cluster_id'] + mean_cols].set_index('cluster_id')
        heatmap_data.columns = [col.replace('_mean', '') for col in heatmap_data.columns]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.heatmap(heatmap_data.T, annot=True, fmt='.2f', cmap='RdYlBu_r', 
                   center=0, cbar_kws={'label': 'Standardized Value'},
                   linewidths=0.5, ax=ax)
        ax.set_xlabel('Cluster', fontsize=12)
        ax.set_ylabel('Feature', fontsize=12)
        ax.set_title('Cluster Feature Profiles', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        save_path = self.output_dir / subdir / filename
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        
        self.saved_plots.append(str(save_path))
        self.logger.info(f"Saved cluster heatmap: {save_path}")
        
        return str(save_path)
    
    def plot_statistical_results(self, results_df: pd.DataFrame,
                                filename: str = "statistical_results.png",
                                subdir: str = "pathway_analysis") -> str:
        """
        Visualize statistical test results.
        
        Args:
            results_df: DataFrame with statistical test results
            filename: Output filename
            subdir: Subdirectory
            
        Returns:
            Path to saved plot
        """
        if len(results_df) == 0:
            self.logger.warning("No statistical results to plot")
            return ""
        
        # Select top 20 most significant features
        top_results = results_df.nsmallest(20, 'p_fdr_corrected')
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot 1: P-values
        y_pos = np.arange(len(top_results))
        axes[0].barh(y_pos, -np.log10(top_results['p_fdr_corrected']), color='steelblue')
        axes[0].set_yticks(y_pos)
        axes[0].set_yticklabels(top_results['feature'], fontsize=10)
        axes[0].axvline(-np.log10(0.05), color='red', linestyle='--', label='p=0.05')
        axes[0].set_xlabel('-log10(FDR-corrected p-value)', fontsize=12)
        axes[0].set_title('Top Significant Features', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Effect sizes
        axes[1].barh(y_pos, top_results['effect_size'], color='coral')
        axes[1].set_yticks(y_pos)
        axes[1].set_yticklabels(top_results['feature'], fontsize=10)
        axes[1].set_xlabel('Effect Size', fontsize=12)
        axes[1].set_title('Effect Sizes', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        save_path = self.output_dir / subdir / filename
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        
        self.saved_plots.append(str(save_path))
        self.logger.info(f"Saved statistical results plot: {save_path}")
        
        return str(save_path)
    
    def plot_correlation_matrix(self, corr_matrix: pd.DataFrame,
                               filename: str = "correlation_matrix.png",
                               subdir: str = "exploratory",
                               top_n: int = 30) -> str:
        """
        Plot correlation matrix heatmap.
        
        Args:
            corr_matrix: Correlation matrix DataFrame
            filename: Output filename
            subdir: Subdirectory
            top_n: Number of top features to include
            
        Returns:
            Path to saved plot
        """
        if len(corr_matrix) == 0:
            self.logger.warning("Empty correlation matrix")
            return ""
        
        # Select top N features if matrix is too large
        if len(corr_matrix) > top_n:
            # Select features with highest average absolute correlation
            avg_corr = corr_matrix.abs().mean(axis=1).sort_values(ascending=False)
            top_features = avg_corr.head(top_n).index
            corr_matrix = corr_matrix.loc[top_features, top_features]
        
        fig, ax = plt.subplots(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
                   annot=False, ax=ax)
        ax.set_title(f'Feature Correlation Matrix (Top {len(corr_matrix)} Features)', 
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        save_path = self.output_dir / subdir / filename
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        
        self.saved_plots.append(str(save_path))
        self.logger.info(f"Saved correlation matrix: {save_path}")
        
        return str(save_path)
    
    def plot_feature_distributions(self, data: pd.DataFrame,
                                   features: List[str],
                                   group_col: Optional[str] = None,
                                   filename: str = "feature_distributions.png",
                                   subdir: str = "exploratory") -> str:
        """
        Plot distributions of multiple features.
        
        Args:
            data: DataFrame with features
            features: List of features to plot
            group_col: Optional column for grouping
            filename: Output filename
            subdir: Subdirectory
            
        Returns:
            Path to saved plot
        """
        n_features = min(len(features), 9)  # Limit to 9 features
        features = features[:n_features]
        
        n_rows = (n_features + 2) // 3
        n_cols = min(3, n_features)
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        axes = axes.flatten() if n_features > 1 else [axes]
        
        for idx, feature in enumerate(features):
            if feature not in data.columns:
                continue
            
            ax = axes[idx]
            
            if group_col and group_col in data.columns:
                for group in data[group_col].unique():
                    group_data = data[data[group_col] == group][feature].dropna()
                    ax.hist(group_data, bins=30, alpha=0.5, label=f'{group}', edgecolor='black')
                ax.legend()
            else:
                ax.hist(data[feature].dropna(), bins=30, color='steelblue', 
                       edgecolor='black', alpha=0.7)
            
            ax.set_xlabel(feature, fontsize=10)
            ax.set_ylabel('Frequency', fontsize=10)
            ax.set_title(f'Distribution of {feature}', fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(n_features, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        
        save_path = self.output_dir / subdir / filename
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        
        self.saved_plots.append(str(save_path))
        self.logger.info(f"Saved feature distributions: {save_path}")
        
        return str(save_path)
    
    def plot_pathway_coverage(self, pathway_matches: Dict,
                             filename: str = "pathway_coverage.png",
                             subdir: str = "pathway_analysis") -> str:
        """
        Visualize pathway-biomarker mapping coverage.
        
        Args:
            pathway_matches: Dictionary of pathway matches
            filename: Output filename
            subdir: Subdirectory
            
        Returns:
            Path to saved plot
        """
        # Count pathways by group
        group_counts = {}
        for key, value in pathway_matches.items():
            group = value['pathway_group']
            group_counts[group] = group_counts.get(group, 0) + 1
        
        groups = list(group_counts.keys())
        counts = list(group_counts.values())
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(groups, counts, color=sns.color_palette("viridis", len(groups)))
        
        # Add value labels
        for i, (bar, count) in enumerate(zip(bars, counts)):
            ax.text(count + 0.1, i, str(count), va='center', fontsize=11, fontweight='bold')
        
        ax.set_xlabel('Number of Matched Pathways', fontsize=12)
        ax.set_ylabel('Pathway Group', fontsize=12)
        ax.set_title('Pathway-Biomarker Mapping Coverage', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        save_path = self.output_dir / subdir / filename
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        
        self.saved_plots.append(str(save_path))
        self.logger.info(f"Saved pathway coverage plot: {save_path}")
        
        return str(save_path)
    
    def get_saved_plots_summary(self) -> pd.DataFrame:
        """
        Generate summary of all saved plots.
        
        Returns:
            DataFrame with plot information
        """
        summary = []
        for plot_path in self.saved_plots:
            path_obj = Path(plot_path)
            summary.append({
                'plot_path': plot_path,
                'filename': path_obj.name,
                'subdirectory': path_obj.parent.name,
                'type': path_obj.suffix
            })
        
        return pd.DataFrame(summary)


if __name__ == "__main__":
    # Test the visualizer
    from logger_config import setup_logging
    
    logger_system = setup_logging()
    logger = logger_system.get_logger("visualization_test", "analysis")
    
    output_dir = "/home1/11021/harshtirhekar/WORK/GaitAnalysis/Attempt2/graphs"
    viz = Visualizer(output_dir, logger)
    
    # Test scatter plot
    X_2d = np.random.randn(100, 2)
    labels = np.random.randint(0, 3, 100)
    uncertainties = np.random.rand(100)
    
    viz.plot_cluster_scatter(X_2d, labels, uncertainties)
    print("Test plot saved!")



