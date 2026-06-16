"""
Multimodal Data Integration Module
Author: Research Team
Date: October 2025

This module integrates multimodal data sources for comprehensive PD analysis:
- Clinical assessments (UPDRS, MoCA, etc.)
- Gait/kinematic features (IMU sensors)
- Biomarkers (LRRK2, alpha-synuclein, etc.)
- Longitudinal tracking
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class MultimodalIntegrator:
    """
    Integrates and aligns multimodal PD datasets for mechanism inference.
    """
    
    def __init__(self, logger):
        """
        Initialize the multimodal integrator.
        
        Args:
            logger: Logger instance
        """
        self.logger = logger
        self.integrated_data = None
        self.feature_sources = {}
        
    def integrate_ppmi_gait_data(self, ppmi_datasets: Dict[str, pd.DataFrame],
                                 patient_id_col: str = 'PATNO') -> pd.DataFrame:
        """
        Integrate PPMI gait-related datasets into unified DataFrame.
        Uses baseline/first visit data to avoid cartesian product from longitudinal records.
        
        Args:
            ppmi_datasets: Dictionary of PPMI DataFrames
            patient_id_col: Column name for patient ID
            
        Returns:
            Integrated DataFrame
        """
        self.logger.info("Integrating PPMI gait datasets...")
        self.logger.info("Using baseline/first visit strategy to avoid longitudinal explosion")
        
        integrated = None
        
        # Start with demographics as base (typically one row per patient)
        if 'demographics' in ppmi_datasets:
            integrated = ppmi_datasets['demographics'].copy()
            # Keep only first record per patient if duplicates exist
            if patient_id_col in integrated.columns:
                integrated = integrated.groupby(patient_id_col).first().reset_index()
            self.logger.info(f"Base: Demographics with {len(integrated)} unique patients")
            self._track_features(integrated, 'demographics')
        
        # Helper function to get baseline/first visit
        def get_baseline_data(df, id_col, event_col='EVENT_ID'):
            if id_col not in df.columns:
                return df
            # Try to filter baseline events
            if event_col in df.columns:
                baseline = df[df[event_col] == 'BL'].copy()
                if len(baseline) > 0:
                    return baseline.groupby(id_col).first().reset_index()
            # Otherwise take first record per patient
            return df.groupby(id_col).first().reset_index()
        
        # Merge motor assessments (baseline only)
        motor_keys = [key for key in ppmi_datasets.keys() if 'UPDRS' in key or 'MDS' in key]
        for key in motor_keys:
            if patient_id_col in ppmi_datasets[key].columns:
                baseline_data = get_baseline_data(ppmi_datasets[key], patient_id_col)
                if integrated is None:
                    integrated = baseline_data
                else:
                    integrated = pd.merge(integrated, baseline_data,
                                        on=patient_id_col, how='outer', suffixes=('', f'_{key}'))
                self.logger.info(f"Merged {key} (baseline): {len(integrated)} unique patients")
                self._track_features(baseline_data, key)
        
        # Merge gait features (baseline only)
        if 'gait_features' in ppmi_datasets:
            gait_df = get_baseline_data(ppmi_datasets['gait_features'], patient_id_col)
            if patient_id_col in gait_df.columns:
                if integrated is None:
                    integrated = gait_df
                else:
                    integrated = pd.merge(integrated, gait_df,
                                        on=patient_id_col, how='outer', suffixes=('', '_gait'))
                self.logger.info(f"Merged gait features (baseline): {len(integrated)} unique patients")
                self._track_features(gait_df, 'gait_features')
        
        # Merge arm swing data (baseline only)
        if 'arm_swing' in ppmi_datasets:
            arm_df = get_baseline_data(ppmi_datasets['arm_swing'], patient_id_col)
            if patient_id_col in arm_df.columns:
                if integrated is None:
                    integrated = arm_df
                else:
                    integrated = pd.merge(integrated, arm_df,
                                        on=patient_id_col, how='outer', suffixes=('', '_arm'))
                self.logger.info(f"Merged arm swing (baseline): {len(integrated)} unique patients")
                self._track_features(arm_df, 'arm_swing')
        
        # Merge medical history (baseline only)
        hist_keys = ['features_of_parkinsonism', 'neurological_exam', 'other_clinical_features']
        for key in hist_keys:
            if key in ppmi_datasets:
                hist_df = get_baseline_data(ppmi_datasets[key], patient_id_col)
                if patient_id_col in hist_df.columns:
                    if integrated is None:
                        integrated = hist_df
                    else:
                        integrated = pd.merge(integrated, hist_df,
                                            on=patient_id_col, how='outer', suffixes=('', f'_{key}'))
                    self.logger.info(f"Merged {key} (baseline): {len(integrated)} unique patients")
                    self._track_features(hist_df, key)
        
        if integrated is not None:
            self.logger.info(f"Final integrated PPMI dataset: {integrated.shape}")
        else:
            self.logger.warning("No PPMI data could be integrated")
            integrated = pd.DataFrame()
        
        return integrated
    
    def integrate_lrrk2_data(self, clinical_datasets: Dict[str, pd.DataFrame],
                            biomarker_datasets: Dict[str, pd.DataFrame],
                            patient_id_col: str = 'SUBJID') -> pd.DataFrame:
        """
        Integrate LRRK2 clinical and biomarker datasets.
        
        Args:
            clinical_datasets: Dictionary of clinical DataFrames
            biomarker_datasets: Dictionary of biomarker DataFrames
            patient_id_col: Column name for patient ID
            
        Returns:
            Integrated DataFrame
        """
        self.logger.info("Integrating LRRK2 datasets...")
        
        integrated = None
        
        # Integrate clinical data
        for key, df in clinical_datasets.items():
            # Try to find patient ID column
            id_col = self._find_id_column(df, patient_id_col)
            
            if id_col:
                if integrated is None:
                    integrated = df.copy()
                    self.logger.info(f"Base: {key} with {len(integrated)} patients")
                else:
                    integrated = pd.merge(integrated, df,
                                        on=id_col, how='outer', suffixes=('', f'_{key}'))
                    self.logger.info(f"Merged {key}: {len(integrated)} total patients")
                self._track_features(df, key)
        
        # Integrate biomarker data
        for key, df in biomarker_datasets.items():
            # Skip if it's not a DataFrame
            if not isinstance(df, pd.DataFrame):
                continue
            
            id_col = self._find_id_column(df, patient_id_col)
            
            if id_col:
                if integrated is None:
                    integrated = df.copy()
                    self.logger.info(f"Base: {key} with {len(integrated)} patients")
                else:
                    # Use outer merge to keep all patients
                    integrated = pd.merge(integrated, df,
                                        on=id_col, how='outer', suffixes=('', f'_{key}'))
                    self.logger.info(f"Merged {key}: {len(integrated)} total patients")
                self._track_features(df, key)
        
        if integrated is not None:
            self.logger.info(f"Final integrated LRRK2 dataset: {integrated.shape}")
        else:
            self.logger.warning("No LRRK2 data could be integrated")
            integrated = pd.DataFrame()
        
        return integrated
    
    def _find_id_column(self, df: pd.DataFrame, preferred_col: str) -> Optional[str]:
        """Find the patient ID column in a DataFrame."""
        if preferred_col in df.columns:
            return preferred_col
        
        # Try common ID column names
        id_variants = ['SUBJID', 'PATNO', 'SubjectID', 'PatientID', 'ID', 'subject_id', 'patient_id']
        for variant in id_variants:
            if variant in df.columns:
                return variant
        
        return None
    
    def _track_features(self, df: pd.DataFrame, source: str):
        """Track which features come from which source."""
        for col in df.columns:
            if col not in self.feature_sources:
                self.feature_sources[col] = []
            self.feature_sources[col].append(source)
    
    def create_unified_dataset(self, ppmi_data: pd.DataFrame,
                              lrrk2_data: pd.DataFrame) -> pd.DataFrame:
        """
        Create a unified dataset from PPMI and LRRK2 sources.
        
        Args:
            ppmi_data: Integrated PPMI DataFrame
            lrrk2_data: Integrated LRRK2 DataFrame
            
        Returns:
            Unified DataFrame with source column
        """
        self.logger.info("Creating unified dataset from PPMI and LRRK2...")
        
        # Add source column
        ppmi_data['data_source'] = 'PPMI'
        lrrk2_data['data_source'] = 'LRRK2'
        
        # Find common columns
        common_cols = list(set(ppmi_data.columns).intersection(set(lrrk2_data.columns)))
        self.logger.info(f"Found {len(common_cols)} common columns")
        
        # Combine datasets
        unified = pd.concat([ppmi_data[common_cols], lrrk2_data[common_cols]], 
                           ignore_index=True, sort=False)
        
        self.logger.info(f"Unified dataset shape: {unified.shape}")
        self.logger.info(f"PPMI patients: {(unified['data_source'] == 'PPMI').sum()}")
        self.logger.info(f"LRRK2 patients: {(unified['data_source'] == 'LRRK2').sum()}")
        
        self.integrated_data = unified
        
        return unified
    
    def extract_longitudinal_data(self, data: pd.DataFrame,
                                  patient_id_col: str,
                                  time_col: str,
                                  visit_col: Optional[str] = None) -> Dict:
        """
        Extract and organize longitudinal data by patient.
        
        Args:
            data: DataFrame with longitudinal records
            patient_id_col: Column with patient IDs
            time_col: Column with time/date information
            visit_col: Optional visit identifier column
            
        Returns:
            Dictionary organized by patient with temporal data
        """
        self.logger.info("Extracting longitudinal data...")
        
        if patient_id_col not in data.columns:
            self.logger.error(f"Patient ID column '{patient_id_col}' not found")
            return {}
        
        longitudinal = {}
        
        for patient_id in data[patient_id_col].unique():
            patient_data = data[data[patient_id_col] == patient_id].copy()
            
            # Sort by time if available
            if time_col in patient_data.columns:
                patient_data = patient_data.sort_values(time_col)
            elif visit_col and visit_col in patient_data.columns:
                patient_data = patient_data.sort_values(visit_col)
            
            longitudinal[patient_id] = {
                'n_visits': len(patient_data),
                'data': patient_data
            }
        
        self.logger.info(f"Extracted longitudinal data for {len(longitudinal)} patients")
        
        avg_visits = np.mean([v['n_visits'] for v in longitudinal.values()])
        self.logger.info(f"Average visits per patient: {avg_visits:.2f}")
        
        return longitudinal
    
    def calculate_data_density(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate data density (non-missing rate) for each feature.
        
        Args:
            data: Input DataFrame
            
        Returns:
            DataFrame with density statistics
        """
        self.logger.info("Calculating data density...")
        
        if len(data) == 0:
            self.logger.warning("Empty DataFrame - cannot calculate density")
            return pd.DataFrame(columns=['feature', 'non_missing', 'total', 'density_percent', 'source'])
        
        density_stats = []
        
        for col in data.columns:
            non_missing = data[col].notna().sum()
            total = len(data)
            density = 100 * non_missing / total if total > 0 else 0
            
            density_stats.append({
                'feature': col,
                'non_missing': non_missing,
                'total': total,
                'density_percent': density,
                'source': ', '.join(self.feature_sources.get(col, ['unknown']))
            })
        
        density_df = pd.DataFrame(density_stats)
        
        if len(density_df) > 0:
            density_df = density_df.sort_values('density_percent', ascending=False)
            avg_density = density_df['density_percent'].mean()
            self.logger.info(f"Average data density: {avg_density:.2f}%")
        
        return density_df
    
    def select_complete_cases(self, data: pd.DataFrame,
                             required_features: List[str],
                             min_density: float = 0.8) -> pd.DataFrame:
        """
        Select patients with sufficient data completeness.
        
        Args:
            data: Input DataFrame
            required_features: List of required feature names
            min_density: Minimum required data density per patient
            
        Returns:
            DataFrame with complete cases
        """
        self.logger.info(f"Selecting complete cases with min density {min_density}...")
        
        # Filter to required features that exist
        valid_features = [f for f in required_features if f in data.columns]
        self.logger.info(f"Using {len(valid_features)} valid features")
        
        if len(valid_features) == 0:
            self.logger.warning("No valid features for completeness check")
            return data
        
        # Calculate per-patient density
        patient_density = data[valid_features].notna().mean(axis=1)
        
        # Select patients meeting threshold
        complete_mask = patient_density >= min_density
        complete_data = data[complete_mask].copy()
        
        self.logger.info(f"Selected {len(complete_data)} / {len(data)} patients with ≥{min_density*100}% completeness")
        
        return complete_data
    
    def get_feature_source_summary(self) -> pd.DataFrame:
        """
        Generate summary of feature sources.
        
        Returns:
            DataFrame mapping features to their data sources
        """
        summary = []
        for feature, sources in self.feature_sources.items():
            summary.append({
                'feature': feature,
                'sources': ', '.join(set(sources)),
                'n_sources': len(set(sources))
            })
        
        return pd.DataFrame(summary)


if __name__ == "__main__":
    # Test the multimodal integrator
    from logger_config import setup_logging
    
    logger_system = setup_logging()
    logger = logger_system.get_logger("multimodal_test", "data_loading")
    
    integrator = MultimodalIntegrator(logger)
    
    # Create test data
    test_ppmi = {
        'demographics': pd.DataFrame({
            'PATNO': [1, 2, 3],
            'AGE': [60, 65, 70],
            'SEX': [1, 0, 1]
        }),
        'gait_features': pd.DataFrame({
            'PATNO': [1, 2, 3],
            'SP_U': [1.2, 1.0, 0.8],
            'STR_CV_U': [0.05, 0.07, 0.09]
        })
    }
    
    integrated = integrator.integrate_ppmi_gait_data(test_ppmi)
    print("\nIntegrated PPMI Data:")
    print(integrated)

