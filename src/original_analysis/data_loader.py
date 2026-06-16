"""
Data Loader Module
Author: Research Team
Date: October 2025

This module handles loading of all datasets from PPMI Gait and LRRK2C cohorts.
Supports multiple file formats: CSV, Excel, MAT files.
Tracks all loaded files for documentation in CLAIMS.md
"""

import pandas as pd
import numpy as np
from pathlib import Path
import scipy.io as sio
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class DataLoader:
    """
    Comprehensive data loading system for multimodal PD datasets.
    """
    
    def __init__(self, data_dir, logger):
        """
        Initialize the data loader.
        
        Args:
            data_dir: Base data directory
            logger: Logger instance for tracking operations
        """
        self.data_dir = Path(data_dir)
        self.logger = logger
        self.loaded_files = []
        self.datasets = {}
        
    def load_ppmi_gait_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load all PPMI Gait-related datasets.
        
        Returns:
            Dictionary of DataFrames with descriptive keys
        """
        self.logger.info("Loading PPMI Gait datasets...")
        ppmi_dir = self.data_dir / "PPMI_Gait"
        datasets = {}
        
        try:
            # Load primary gait features
            gait_features_file = ppmi_dir / "Gait_Data_with_Selected_Features.csv"
            if gait_features_file.exists():
                datasets['gait_features'] = pd.read_csv(gait_features_file)
                self.loaded_files.append(str(gait_features_file))
                self.logger.info(f"Loaded gait features: {datasets['gait_features'].shape}")
            
            # Load motor assessments
            for file in ppmi_dir.glob("*.csv"):
                if "MDS" in file.name or "UPDRS" in file.name:
                    key = file.stem
                    datasets[key] = pd.read_csv(file)
                    self.loaded_files.append(str(file))
                    self.logger.info(f"Loaded {key}: {datasets[key].shape}")
                elif "Gait" in file.name and "Arm_swing" in file.name:
                    datasets['arm_swing'] = pd.read_csv(file)
                    self.loaded_files.append(str(file))
                    self.logger.info(f"Loaded arm swing: {datasets['arm_swing'].shape}")
                elif "Mobility" in file.name:
                    datasets['mobility'] = pd.read_csv(file)
                    self.loaded_files.append(str(file))
                    self.logger.info(f"Loaded mobility: {datasets['mobility'].shape}")
            
            # Load demographics
            demo_file = ppmi_dir / "Demographics_08Jan2025.csv"
            if demo_file.exists():
                datasets['demographics'] = pd.read_csv(demo_file)
                self.loaded_files.append(str(demo_file))
                self.logger.info(f"Loaded demographics: {datasets['demographics'].shape}")
            
            # Load medical history
            for hist_type in ['Features_of_Parkinsonism', 'Neurological_Exam', 'Other_Clinical_Features']:
                hist_file = list(ppmi_dir.glob(f"{hist_type}*.csv"))
                if hist_file:
                    key = hist_type.lower()
                    datasets[key] = pd.read_csv(hist_file[0])
                    self.loaded_files.append(str(hist_file[0]))
                    self.logger.info(f"Loaded {key}: {datasets[key].shape}")
            
            # Load participant status
            status_file = ppmi_dir / "Participant_Status_08Jan2025.csv"
            if status_file.exists():
                datasets['participant_status'] = pd.read_csv(status_file)
                self.loaded_files.append(str(status_file))
                self.logger.info(f"Loaded participant status: {datasets['participant_status'].shape}")
            
            # Load curated data cut
            curated_file = ppmi_dir / "PPMI_Curated_Data_Cut_Public_20241211.xlsx"
            if curated_file.exists():
                datasets['curated_data'] = pd.read_excel(curated_file)
                self.loaded_files.append(str(curated_file))
                self.logger.info(f"Loaded curated data: {datasets['curated_data'].shape}")
            
            # Load digital sensor data
            datasets['digital_sensor_patients'] = []
            for patient_file in sorted(ppmi_dir.glob("Patient-*.xlsx")):
                try:
                    patient_data = pd.read_excel(patient_file)
                    patient_id = patient_file.stem.replace("Patient-", "")
                    datasets['digital_sensor_patients'].append({
                        'patient_id': patient_id,
                        'data': patient_data
                    })
                    self.loaded_files.append(str(patient_file))
                except Exception as e:
                    self.logger.warning(f"Could not load {patient_file}: {e}")
            
            self.logger.info(f"Loaded {len(datasets['digital_sensor_patients'])} patient sensor files")
            
        except Exception as e:
            self.logger.error(f"Error loading PPMI gait data: {e}", exc_info=True)
            
        return datasets
    
    def load_lrrk2_clinical_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load LRRK2 clinical and longitudinal datasets.
        
        Returns:
            Dictionary of DataFrames with clinical data
        """
        self.logger.info("Loading LRRK2 clinical datasets...")
        lrrk2_dir = self.data_dir / "LRRK2_Clinical"
        datasets = {}
        
        try:
            for csv_file in lrrk2_dir.glob("*.csv"):
                key = csv_file.stem
                datasets[key] = pd.read_csv(csv_file)
                self.loaded_files.append(str(csv_file))
                self.logger.info(f"Loaded {key}: {datasets[key].shape}")
                
        except Exception as e:
            self.logger.error(f"Error loading LRRK2 clinical data: {e}", exc_info=True)
            
        return datasets
    
    def load_lrrk2_biomarker_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load LRRK2 biomarker datasets from biologic studies.
        
        Returns:
            Dictionary of DataFrames with biomarker data
        """
        self.logger.info("Loading LRRK2 biomarker datasets...")
        biomarker_dir = self.data_dir / "LRRK2_Biomarkers"
        datasets = {}
        
        try:
            # Load CSV files
            for csv_file in biomarker_dir.glob("*.csv"):
                key = csv_file.stem
                try:
                    datasets[key] = pd.read_csv(csv_file)
                    self.loaded_files.append(str(csv_file))
                    self.logger.info(f"Loaded {key}: {datasets[key].shape}")
                except Exception as e:
                    self.logger.warning(f"Could not load {csv_file}: {e}")
            
            # Load Excel files
            for xlsx_file in biomarker_dir.glob("*.xlsx"):
                key = xlsx_file.stem
                try:
                    datasets[key] = pd.read_excel(xlsx_file)
                    self.loaded_files.append(str(xlsx_file))
                    self.logger.info(f"Loaded {key}: {datasets[key].shape}")
                except Exception as e:
                    self.logger.warning(f"Could not load {xlsx_file}: {e}")
                    
        except Exception as e:
            self.logger.error(f"Error loading LRRK2 biomarker data: {e}", exc_info=True)
            
        return datasets
    
    def load_synapse_weargait_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load Synapse Wear-Gait PD datasets.
        
        Returns:
            Dictionary with control and PD participant data
        """
        self.logger.info("Loading Synapse Wear-Gait datasets...")
        synapse_dir = self.data_dir / "Synapse_WearGait"
        datasets = {}
        
        try:
            # Load demographics
            control_demo = synapse_dir / "CONTROLS - Demographic+Clinical - datasetV1.csv"
            if control_demo.exists():
                datasets['control_demographics'] = pd.read_csv(control_demo)
                self.loaded_files.append(str(control_demo))
                self.logger.info(f"Loaded control demographics: {datasets['control_demographics'].shape}")
            
            pd_demo = synapse_dir / "PD - Demographic+Clinical - datasetV1.csv"
            if pd_demo.exists():
                datasets['pd_demographics'] = pd.read_csv(pd_demo)
                self.loaded_files.append(str(pd_demo))
                self.logger.info(f"Loaded PD demographics: {datasets['pd_demographics'].shape}")
            
            # Load MAT files
            datasets['control_mat_files'] = []
            datasets['pd_mat_files'] = []
            
            for mat_file in synapse_dir.glob("*.mat"):
                try:
                    mat_data = sio.loadmat(mat_file)
                    file_info = {
                        'filename': mat_file.name,
                        'data': mat_data
                    }
                    
                    if 'HC' in mat_file.name or 'WHC' in mat_file.name or 'control' in mat_file.name:
                        datasets['control_mat_files'].append(file_info)
                    else:
                        datasets['pd_mat_files'].append(file_info)
                    
                    self.loaded_files.append(str(mat_file))
                except Exception as e:
                    self.logger.warning(f"Could not load {mat_file}: {e}")
            
            self.logger.info(f"Loaded {len(datasets['control_mat_files'])} control MAT files")
            self.logger.info(f"Loaded {len(datasets['pd_mat_files'])} PD MAT files")
            
        except Exception as e:
            self.logger.error(f"Error loading Synapse data: {e}", exc_info=True)
            
        return datasets
    
    def load_mendeley_gait_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load Mendeley Gait assessment datasets.
        
        Returns:
            Dictionary with summary tables
        """
        self.logger.info("Loading Mendeley Gait datasets...")
        mendeley_dir = self.data_dir / "Mendeley_Gait"
        datasets = {}
        
        try:
            for table_file in mendeley_dir.glob("Table*.xlsx"):
                key = table_file.stem.replace(" ", "_").lower()
                datasets[key] = pd.read_excel(table_file)
                self.loaded_files.append(str(table_file))
                self.logger.info(f"Loaded {key}: {datasets[key].shape}")
                
        except Exception as e:
            self.logger.error(f"Error loading Mendeley data: {e}", exc_info=True)
            
        return datasets
    
    def load_bioclite_data(self) -> Dict:
        """
        Load Bioclite restricted arm swing data.
        
        Returns:
            Dictionary with MAT file data
        """
        self.logger.info("Loading Bioclite datasets...")
        bioclite_dir = self.data_dir / "Bioclite"
        datasets = {}
        
        try:
            mat_file = bioclite_dir / "data_restricted_arm_swing.mat"
            if mat_file.exists():
                datasets['arm_swing_data'] = sio.loadmat(mat_file)
                self.loaded_files.append(str(mat_file))
                self.logger.info(f"Loaded Bioclite arm swing data")
                
        except Exception as e:
            self.logger.error(f"Error loading Bioclite data: {e}", exc_info=True)
            
        return datasets
    
    def load_all_datasets(self) -> Dict[str, Dict]:
        """
        Load all available datasets from all sources.
        
        Returns:
            Nested dictionary containing all datasets organized by source
        """
        self.logger.info("="*80)
        self.logger.info("LOADING ALL DATASETS")
        self.logger.info("="*80)
        
        all_data = {
            'ppmi_gait': self.load_ppmi_gait_data(),
            'lrrk2_clinical': self.load_lrrk2_clinical_data(),
            'lrrk2_biomarkers': self.load_lrrk2_biomarker_data(),
            'synapse_weargait': self.load_synapse_weargait_data(),
            'mendeley_gait': self.load_mendeley_gait_data(),
            'bioclite': self.load_bioclite_data()
        }
        
        self.datasets = all_data
        
        self.logger.info("="*80)
        self.logger.info(f"TOTAL FILES LOADED: {len(self.loaded_files)}")
        self.logger.info("="*80)
        
        return all_data
    
    def get_loaded_files_summary(self) -> pd.DataFrame:
        """
        Generate a summary of all loaded files for documentation.
        
        Returns:
            DataFrame with file paths and metadata
        """
        summary_data = []
        for file_path in self.loaded_files:
            path_obj = Path(file_path)
            summary_data.append({
                'file_path': file_path,
                'filename': path_obj.name,
                'directory': path_obj.parent.name,
                'file_type': path_obj.suffix
            })
        
        return pd.DataFrame(summary_data)


if __name__ == "__main__":
    # Test the data loader
    from logger_config import setup_logging
    
    logger_system = setup_logging()
    logger = logger_system.get_logger("data_loader_test", "data_loading")
    
    loader = DataLoader("/home1/11021/harshtirhekar/WORK/GaitAnalysis/Attempt2/data", logger)
    all_data = loader.load_all_datasets()
    
    summary = loader.get_loaded_files_summary()
    print("\nLoaded Files Summary:")
    print(summary)



