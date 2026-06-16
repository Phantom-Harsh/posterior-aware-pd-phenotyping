"""
Data Imputation Module
Author: Research Team
Date: October 2025

Handles sparse data through multiple imputation strategies:
- KNN Imputation
- MICE (Multiple Imputation by Chained Equations)
- Iterative Imputer
- Mean/Median/Mode fallback

Required for Bayesian clustering with complete data.
"""

import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class DataImputer:
    """
    Comprehensive imputation system for handling sparse PD datasets.
    """
    
    def __init__(self, logger):
        """
        Initialize imputer.
        
        Args:
            logger: Logger instance
        """
        self.logger = logger
        self.imputation_stats = {}
        
    def analyze_missingness(self, data: pd.DataFrame) -> Dict:
        """
        Analyze missing data patterns.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Dictionary with missingness statistics
        """
        self.logger.info("Analyzing missing data patterns...")
        
        missing_counts = data.isnull().sum()
        total_cells = len(data) * len(data.columns)
        total_missing = missing_counts.sum()
        
        stats = {
            'total_rows': len(data),
            'total_columns': len(data.columns),
            'total_cells': total_cells,
            'total_missing': int(total_missing),
            'missing_percentage': round(100 * total_missing / total_cells, 2),
            'columns_with_missing': {},
            'rows_with_missing': int((data.isnull().sum(axis=1) > 0).sum()),
            'completely_empty_columns': []
        }
        
        for col in data.columns:
            if missing_counts[col] > 0:
                stats['columns_with_missing'][col] = {
                    'count': int(missing_counts[col]),
                    'percentage': round(100 * missing_counts[col] / len(data), 2)
                }
            
            if missing_counts[col] == len(data):
                stats['completely_empty_columns'].append(col)
        
        self.logger.info(f"Missing data: {stats['missing_percentage']}% of total cells")
        self.logger.info(f"Columns with missing: {len(stats['columns_with_missing'])}/{len(data.columns)}")
        
        return stats
    
    def impute_knn(self, data: pd.DataFrame, n_neighbors: int = 5) -> pd.DataFrame:
        """
        KNN imputation for numeric features.
        
        Args:
            data: Input DataFrame
            n_neighbors: Number of neighbors for KNN
            
        Returns:
            Imputed DataFrame
        """
        self.logger.info(f"Performing KNN imputation (n_neighbors={n_neighbors})...")
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) == 0:
            self.logger.warning("No numeric columns for KNN imputation")
            return data
        
        imputer = KNNImputer(n_neighbors=n_neighbors)
        data_imputed = data.copy()
        
        try:
            data_imputed[numeric_cols] = imputer.fit_transform(data[numeric_cols])
            self.logger.info(f"✓ KNN imputation completed for {len(numeric_cols)} numeric columns")
        except Exception as e:
            self.logger.error(f"KNN imputation failed: {e}")
            return data
        
        return data_imputed
    
    def impute_iterative(self, data: pd.DataFrame, max_iter: int = 10) -> pd.DataFrame:
        """
        Iterative (MICE-like) imputation.
        
        Args:
            data: Input DataFrame
            max_iter: Maximum iterations
            
        Returns:
            Imputed DataFrame
        """
        self.logger.info(f"Performing iterative imputation (max_iter={max_iter})...")
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) == 0:
            self.logger.warning("No numeric columns for iterative imputation")
            return data
        
        imputer = IterativeImputer(max_iter=max_iter, random_state=42)
        data_imputed = data.copy()
        
        try:
            data_imputed[numeric_cols] = imputer.fit_transform(data[numeric_cols])
            self.logger.info(f"✓ Iterative imputation completed for {len(numeric_cols)} columns")
        except Exception as e:
            self.logger.error(f"Iterative imputation failed: {e}")
            return data
        
        return data_imputed
    
    def impute_simple(self, data: pd.DataFrame, strategy: str = 'median') -> pd.DataFrame:
        """
        Simple imputation (mean, median, mode).
        
        Args:
            data: Input DataFrame
            strategy: 'mean', 'median', or 'most_frequent'
            
        Returns:
            Imputed DataFrame
        """
        self.logger.info(f"Performing simple imputation (strategy={strategy})...")
        
        data_imputed = data.copy()
        
        # Numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) > 0:
            num_imputer = SimpleImputer(strategy=strategy if strategy != 'most_frequent' else 'median')
            data_imputed[numeric_cols] = num_imputer.fit_transform(data[numeric_cols])
        
        # Categorical columns
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
        if len(categorical_cols) > 0:
            cat_imputer = SimpleImputer(strategy='most_frequent')
            data_imputed[categorical_cols] = cat_imputer.fit_transform(data[categorical_cols])
        
        self.logger.info(f"✓ Simple imputation completed")
        
        return data_imputed
    
    def smart_impute(self, data: pd.DataFrame, 
                     features: List[str],
                     method: str = 'auto') -> Tuple[pd.DataFrame, Dict]:
        """
        Smart imputation that chooses best strategy based on data characteristics.
        
        Args:
            data: Input DataFrame
            features: List of features to impute
            method: 'auto', 'knn', 'iterative', or 'simple'
            
        Returns:
            Tuple of (imputed data, imputation statistics)
        """
        self.logger.info("="*60)
        self.logger.info("SMART IMPUTATION SYSTEM")
        self.logger.info("="*60)
        
        # Filter to requested features that exist
        valid_features = [f for f in features if f in data.columns]
        self.logger.info(f"Imputing {len(valid_features)} features")
        
        if len(valid_features) == 0:
            self.logger.warning("No valid features to impute")
            return data, {}
        
        subset = data[valid_features].copy()
        
        # Analyze missingness
        miss_stats = self.analyze_missingness(subset)
        
        # Remove completely empty columns
        if miss_stats['completely_empty_columns']:
            self.logger.warning(f"Removing {len(miss_stats['completely_empty_columns'])} completely empty columns")
            subset = subset.drop(columns=miss_stats['completely_empty_columns'])
            valid_features = [f for f in valid_features if f not in miss_stats['completely_empty_columns']]
        
        # Choose imputation strategy
        if method == 'auto':
            missing_pct = miss_stats['missing_percentage']
            n_samples = len(subset)
            
            if missing_pct < 5:
                chosen_method = 'simple'
                self.logger.info("Auto: Using simple imputation (low missingness)")
            elif n_samples > 100 and missing_pct < 30:
                chosen_method = 'knn'
                self.logger.info("Auto: Using KNN imputation (moderate missingness, sufficient samples)")
            elif n_samples > 50 and missing_pct < 40:
                chosen_method = 'iterative'
                self.logger.info("Auto: Using iterative imputation (higher missingness)")
            else:
                chosen_method = 'simple'
                self.logger.info("Auto: Using simple imputation (high missingness or low samples)")
        else:
            chosen_method = method
        
        # Perform imputation
        if chosen_method == 'knn':
            subset_imputed = self.impute_knn(subset, n_neighbors=min(5, len(subset)-1))
        elif chosen_method == 'iterative':
            subset_imputed = self.impute_iterative(subset, max_iter=10)
        else:
            subset_imputed = self.impute_simple(subset, strategy='median')
        
        # Update original dataframe
        data_imputed = data.copy()
        data_imputed[valid_features] = subset_imputed[valid_features]
        
        # Verify imputation
        remaining_missing = data_imputed[valid_features].isnull().sum().sum()
        
        stats = {
            'method_used': chosen_method,
            'features_imputed': valid_features,
            'original_missing': miss_stats['total_missing'],
            'remaining_missing': int(remaining_missing),
            'imputation_success': remaining_missing == 0
        }
        
        if stats['imputation_success']:
            self.logger.info(f"✓ Imputation SUCCESS: All {stats['original_missing']} missing values filled")
        else:
            self.logger.warning(f"⚠ Imputation PARTIAL: {remaining_missing} missing values remain")
        
        self.imputation_stats = stats
        
        return data_imputed, stats


if __name__ == "__main__":
    from logger_config import setup_logging
    
    logger_system = setup_logging()
    logger = logger_system.get_logger("imputation_test", "preprocessing")
    
    imputer = DataImputer(logger)
    
    # Test with sample data
    test_data = pd.DataFrame({
        'feature1': [1, 2, np.nan, 4, 5],
        'feature2': [np.nan, 2, 3, np.nan, 5],
        'feature3': [1, np.nan, 3, 4, np.nan]
    })
    
    print("Original data:")
    print(test_data)
    
    imputed, stats = imputer.smart_impute(test_data, test_data.columns.tolist())
    
    print("\nImputed data:")
    print(imputed)
    print("\nStats:", stats)



