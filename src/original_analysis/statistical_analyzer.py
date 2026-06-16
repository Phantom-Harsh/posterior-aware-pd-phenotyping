"""
Statistical Analysis Module
Author: Research Team
Date: October 2025

This module provides comprehensive statistical validation methods including:
- Hypothesis testing (t-tests, Mann-Whitney, Kruskal-Wallis)
- Multiple testing correction (FDR, Bonferroni)
- Effect size calculations
- Distribution analysis
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import mannwhitneyu, kruskal, chi2_contingency, ttest_ind
from statsmodels.stats.multitest import multipletests
from typing import Dict, List, Tuple, Optional


class StatisticalAnalyzer:
    """
    Comprehensive statistical analysis toolkit for PD mechanism research.
    """
    
    def __init__(self, logger):
        """
        Initialize the statistical analyzer.
        
        Args:
            logger: Logger instance for tracking operations
        """
        self.logger = logger
        self.results = {}
        
    def test_group_differences(self, data: pd.DataFrame, 
                               feature_cols: List[str],
                               group_col: str,
                               test_type: str = 'auto') -> pd.DataFrame:
        """
        Test for significant differences between groups across multiple features.
        
        Args:
            data: DataFrame containing features and group labels
            feature_cols: List of feature column names to test
            group_col: Column name containing group labels
            test_type: 'auto', 'parametric', 'nonparametric', or 'categorical'
            
        Returns:
            DataFrame with test results and corrected p-values
        """
        self.logger.info(f"Testing group differences for {len(feature_cols)} features...")
        
        results = []
        groups = data[group_col].unique()
        
        for feature in feature_cols:
            try:
                # Extract non-null values for each group
                group_data = [data[data[group_col] == g][feature].dropna().values 
                             for g in groups]
                
                # Skip if any group has insufficient data
                if any(len(gd) < 3 for gd in group_data):
                    continue
                
                # Determine test type
                if test_type == 'auto':
                    # Check if data is normally distributed
                    is_normal = all(stats.shapiro(gd)[1] > 0.05 for gd in group_data if len(gd) >= 3)
                    actual_test = 'parametric' if is_normal else 'nonparametric'
                else:
                    actual_test = test_type
                
                # Perform appropriate test
                if len(groups) == 2:
                    if actual_test == 'parametric':
                        statistic, p_value = ttest_ind(group_data[0], group_data[1])
                        test_name = 't-test'
                    else:
                        statistic, p_value = mannwhitneyu(group_data[0], group_data[1])
                        test_name = 'Mann-Whitney U'
                else:
                    if actual_test == 'parametric':
                        statistic, p_value = stats.f_oneway(*group_data)
                        test_name = 'ANOVA'
                    else:
                        statistic, p_value = kruskal(*group_data)
                        test_name = 'Kruskal-Wallis'
                
                # Calculate effect size (Cohen's d for 2 groups, eta-squared for >2)
                if len(groups) == 2:
                    effect_size = self._cohens_d(group_data[0], group_data[1])
                    effect_name = "Cohen's d"
                else:
                    effect_size = self._eta_squared(group_data, statistic)
                    effect_name = "Eta-squared"
                
                # Calculate means for each group
                means = {f"mean_{g}": np.mean(gd) for g, gd in zip(groups, group_data)}
                
                result = {
                    'feature': feature,
                    'test': test_name,
                    'statistic': statistic,
                    'p_value': p_value,
                    'effect_size': effect_size,
                    'effect_size_type': effect_name,
                    **means
                }
                
                results.append(result)
                
            except Exception as e:
                self.logger.warning(f"Could not test {feature}: {e}")
                continue
        
        results_df = pd.DataFrame(results)
        
        if len(results_df) > 0:
            # Apply multiple testing correction
            results_df = self._apply_multiple_testing_correction(results_df)
        
        self.logger.info(f"Completed statistical tests for {len(results_df)} features")
        
        return results_df
    
    def _cohens_d(self, group1: np.ndarray, group2: np.ndarray) -> float:
        """Calculate Cohen's d effect size for two groups."""
        n1, n2 = len(group1), len(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        return (np.mean(group1) - np.mean(group2)) / pooled_std if pooled_std > 0 else 0.0
    
    def _eta_squared(self, groups: List[np.ndarray], statistic: float) -> float:
        """Calculate eta-squared effect size for multiple groups."""
        n_total = sum(len(g) for g in groups)
        k = len(groups)
        return statistic / (statistic + (n_total - k))
    
    def _apply_multiple_testing_correction(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """Apply FDR and Bonferroni corrections to p-values."""
        p_values = results_df['p_value'].values
        
        # FDR correction (Benjamini-Hochberg)
        _, p_fdr, _, _ = multipletests(p_values, method='fdr_bh')
        results_df['p_fdr_corrected'] = p_fdr
        
        # Bonferroni correction
        _, p_bonf, _, _ = multipletests(p_values, method='bonferroni')
        results_df['p_bonferroni'] = p_bonf
        
        # Mark significant results
        results_df['significant_fdr'] = results_df['p_fdr_corrected'] < 0.05
        results_df['significant_bonf'] = results_df['p_bonferroni'] < 0.05
        
        return results_df
    
    def correlation_analysis(self, data: pd.DataFrame, 
                            features: List[str],
                            method: str = 'spearman') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Perform correlation analysis between features.
        
        Args:
            data: DataFrame containing features
            features: List of feature names
            method: 'pearson' or 'spearman'
            
        Returns:
            Tuple of (correlation matrix, p-value matrix)
        """
        self.logger.info(f"Computing {method} correlations for {len(features)} features...")
        
        # Select only numeric features that exist in data
        valid_features = [f for f in features if f in data.columns and pd.api.types.is_numeric_dtype(data[f])]
        
        if len(valid_features) == 0:
            self.logger.warning("No valid numeric features found for correlation")
            return pd.DataFrame(), pd.DataFrame()
        
        feature_data = data[valid_features].dropna(axis=1, how='all')
        
        # Compute correlation matrix
        if method == 'pearson':
            corr_matrix = feature_data.corr(method='pearson')
        else:
            corr_matrix = feature_data.corr(method='spearman')
        
        # Compute p-values
        p_matrix = pd.DataFrame(np.zeros_like(corr_matrix), 
                               columns=corr_matrix.columns, 
                               index=corr_matrix.index)
        
        for i, col1 in enumerate(corr_matrix.columns):
            for j, col2 in enumerate(corr_matrix.columns):
                if i < j:
                    data1 = feature_data[col1].dropna()
                    data2 = feature_data[col2].dropna()
                    common_idx = data1.index.intersection(data2.index)
                    
                    if len(common_idx) > 3:
                        if method == 'pearson':
                            _, p_val = stats.pearsonr(data1[common_idx], data2[common_idx])
                        else:
                            _, p_val = stats.spearmanr(data1[common_idx], data2[common_idx])
                        p_matrix.loc[col1, col2] = p_val
                        p_matrix.loc[col2, col1] = p_val
        
        self.logger.info(f"Correlation analysis completed")
        
        return corr_matrix, p_matrix
    
    def categorical_association_test(self, data: pd.DataFrame,
                                     var1: str, var2: str) -> Dict:
        """
        Test association between two categorical variables using Chi-square.
        
        Args:
            data: DataFrame containing variables
            var1: First categorical variable name
            var2: Second categorical variable name
            
        Returns:
            Dictionary with test results
        """
        self.logger.info(f"Testing association between {var1} and {var2}...")
        
        # Create contingency table
        contingency = pd.crosstab(data[var1], data[var2])
        
        # Perform chi-square test
        chi2, p_value, dof, expected = chi2_contingency(contingency)
        
        # Calculate Cramér's V (effect size)
        n = contingency.sum().sum()
        min_dim = min(contingency.shape[0] - 1, contingency.shape[1] - 1)
        cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0.0
        
        result = {
            'var1': var1,
            'var2': var2,
            'chi2_statistic': chi2,
            'p_value': p_value,
            'degrees_of_freedom': dof,
            'cramers_v': cramers_v,
            'contingency_table': contingency
        }
        
        self.logger.info(f"Chi-square test: χ²={chi2:.4f}, p={p_value:.4e}, V={cramers_v:.4f}")
        
        return result
    
    def distribution_summary(self, data: pd.DataFrame, 
                            features: List[str]) -> pd.DataFrame:
        """
        Generate comprehensive distribution summaries for features.
        
        Args:
            data: DataFrame containing features
            features: List of feature names
            
        Returns:
            DataFrame with distribution statistics
        """
        self.logger.info(f"Computing distribution summaries for {len(features)} features...")
        
        summaries = []
        
        for feature in features:
            if feature not in data.columns:
                continue
                
            feature_data = data[feature].dropna()
            
            if len(feature_data) == 0:
                continue
            
            summary = {
                'feature': feature,
                'n': len(feature_data),
                'mean': np.mean(feature_data),
                'std': np.std(feature_data),
                'median': np.median(feature_data),
                'q25': np.percentile(feature_data, 25),
                'q75': np.percentile(feature_data, 75),
                'min': np.min(feature_data),
                'max': np.max(feature_data),
                'skewness': stats.skew(feature_data),
                'kurtosis': stats.kurtosis(feature_data)
            }
            
            # Test for normality if sample size is adequate
            if len(feature_data) >= 3:
                _, normality_p = stats.shapiro(feature_data[:5000])  # Limit to 5000 samples
                summary['normality_p'] = normality_p
                summary['is_normal'] = normality_p > 0.05
            
            summaries.append(summary)
        
        summary_df = pd.DataFrame(summaries)
        self.logger.info(f"Distribution summaries completed for {len(summary_df)} features")
        
        return summary_df
    
    def identify_outliers(self, data: pd.DataFrame, 
                         feature: str, 
                         method: str = 'iqr') -> Tuple[np.ndarray, Dict]:
        """
        Identify outliers in a feature using specified method.
        
        Args:
            data: DataFrame containing feature
            feature: Feature name
            method: 'iqr' (interquartile range) or 'zscore'
            
        Returns:
            Tuple of (outlier_mask, statistics)
        """
        feature_data = data[feature].dropna()
        
        if method == 'iqr':
            q25 = np.percentile(feature_data, 25)
            q75 = np.percentile(feature_data, 75)
            iqr = q75 - q25
            lower_bound = q25 - 1.5 * iqr
            upper_bound = q75 + 1.5 * iqr
            outliers = (feature_data < lower_bound) | (feature_data > upper_bound)
            stats_dict = {
                'method': 'IQR',
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'n_outliers': outliers.sum()
            }
        else:  # zscore
            z_scores = np.abs(stats.zscore(feature_data))
            outliers = z_scores > 3
            stats_dict = {
                'method': 'Z-score',
                'threshold': 3,
                'n_outliers': outliers.sum()
            }
        
        self.logger.info(f"Identified {outliers.sum()} outliers in {feature} using {method}")
        
        return outliers.values, stats_dict


if __name__ == "__main__":
    # Test the statistical analyzer
    from logger_config import setup_logging
    
    logger_system = setup_logging()
    logger = logger_system.get_logger("statistical_test", "analysis")
    
    analyzer = StatisticalAnalyzer(logger)
    
    # Generate test data
    np.random.seed(42)
    test_data = pd.DataFrame({
        'feature1': np.concatenate([np.random.normal(0, 1, 50), np.random.normal(1, 1, 50)]),
        'feature2': np.concatenate([np.random.normal(0, 1, 50), np.random.normal(0.5, 1, 50)]),
        'group': ['A'] * 50 + ['B'] * 50
    })
    
    # Test group differences
    results = analyzer.test_group_differences(test_data, ['feature1', 'feature2'], 'group')
    print("\nStatistical Test Results:")
    print(results)



