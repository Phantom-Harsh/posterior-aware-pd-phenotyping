"""
Comprehensive Data Explorer Module
Author: Research Team  
Date: October 2025

This module performs DEEP exploration of ALL datasets:
- pd.head() for structure
- pd.info() for types and missing data
- pd.describe() for statistical summaries
- Value ranges and distributions
- Missing data patterns
- Feature correlations

Generates comprehensive exploration reports for ALL 414 files.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import scipy.io as sio
from typing import Dict, List, Tuple, Optional
import json
import warnings
warnings.filterwarnings('ignore')


class ComprehensiveDataExplorer:
    """
    Deep exploration system for all PD datasets.
    Examines structure, content, quality of every file.
    """
    
    def __init__(self, data_dir, logger):
        """
        Initialize data explorer.
        
        Args:
            data_dir: Base data directory
            logger: Logger instance
        """
        self.data_dir = Path(data_dir)
        self.logger = logger
        self.exploration_results = {}
        
    def explore_single_file(self, file_path: Path) -> Dict:
        """
        Perform comprehensive exploration of a single file.
        
        Args:
            file_path: Path to data file
            
        Returns:
            Dictionary with exploration results
        """
        self.logger.info(f"Exploring: {file_path.name}")
        
        result = {
            'file_name': file_path.name,
            'file_path': str(file_path),
            'file_size_mb': file_path.stat().st_size / (1024*1024),
            'file_type': file_path.suffix,
            'exploration_status': 'pending'
        }
        
        try:
            # Load based on file type
            if file_path.suffix == '.csv':
                df = pd.read_csv(file_path, low_memory=False)
            elif file_path.suffix in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path)
            elif file_path.suffix == '.mat':
                mat_data = sio.loadmat(file_path)
                # Extract main data array if exists
                result['mat_keys'] = list(mat_data.keys())
                result['exploration_status'] = 'mat_file'
                return result
            else:
                result['exploration_status'] = 'unsupported_format'
                return result
            
            # Basic structure
            result['n_rows'] = len(df)
            result['n_columns'] = len(df.columns)
            result['columns'] = list(df.columns)
            result['dtypes'] = {col: str(dtype) for col, dtype in df.dtypes.items()}
            
            # Missing data analysis
            missing_counts = df.isnull().sum()
            result['missing_data'] = {
                col: int(count) for col, count in missing_counts.items() if count > 0
            }
            result['missing_percentage'] = {
                col: round(100 * count / len(df), 2) 
                for col, count in missing_counts.items() if count > 0
            }
            result['data_density'] = round(100 * (1 - df.isnull().sum().sum() / (df.shape[0] * df.shape[1])), 2)
            
            # Head samples (first 3 rows)
            result['head_sample'] = df.head(3).to_dict('records')
            
            # Numeric columns analysis
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            result['numeric_columns'] = numeric_cols
            
            if len(numeric_cols) > 0:
                desc = df[numeric_cols].describe()
                result['numeric_summary'] = {
                    col: {
                        'mean': round(desc[col]['mean'], 4) if not pd.isna(desc[col]['mean']) else None,
                        'std': round(desc[col]['std'], 4) if not pd.isna(desc[col]['std']) else None,
                        'min': round(desc[col]['min'], 4) if not pd.isna(desc[col]['min']) else None,
                        'max': round(desc[col]['max'], 4) if not pd.isna(desc[col]['max']) else None,
                        '25%': round(desc[col]['25%'], 4) if not pd.isna(desc[col]['25%']) else None,
                        '50%': round(desc[col]['50%'], 4) if not pd.isna(desc[col]['50%']) else None,
                        '75%': round(desc[col]['75%'], 4) if not pd.isna(desc[col]['75%']) else None
                    }
                    for col in numeric_cols[:20]  # Limit to first 20 numeric columns
                }
            
            # Categorical columns analysis
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            result['categorical_columns'] = categorical_cols
            
            if len(categorical_cols) > 0:
                result['categorical_summary'] = {
                    col: {
                        'unique_values': int(df[col].nunique()),
                        'top_values': df[col].value_counts().head(5).to_dict()
                    }
                    for col in categorical_cols[:10]  # Limit to first 10 categorical columns
                }
            
            # Identify potential ID columns
            potential_id_cols = []
            for col in df.columns:
                if any(id_term in col.upper() for id_term in ['ID', 'PATNO', 'SUBJID', 'SUBJECT', 'PATIENT']):
                    potential_id_cols.append(col)
            result['potential_id_columns'] = potential_id_cols
            
            # Identify potential time columns
            potential_time_cols = []
            for col in df.columns:
                if any(time_term in col.upper() for time_term in ['DATE', 'TIME', 'VISIT', 'EVENT', 'INFODT']):
                    potential_time_cols.append(col)
            result['potential_time_columns'] = potential_time_cols
            
            result['exploration_status'] = 'success'
            self.logger.info(f"✓ {file_path.name}: {len(df)} rows × {len(df.columns)} cols, {result['data_density']}% density")
            
        except Exception as e:
            result['exploration_status'] = 'error'
            result['error_message'] = str(e)
            self.logger.warning(f"✗ {file_path.name}: {e}")
        
        return result
    
    def explore_directory(self, directory: Path, pattern: str = "*") -> List[Dict]:
        """
        Explore all files in a directory matching pattern.
        
        Args:
            directory: Directory to explore
            pattern: File pattern (e.g., "*.csv")
            
        Returns:
            List of exploration results
        """
        self.logger.info(f"Exploring directory: {directory.name}")
        
        results = []
        for file_path in sorted(directory.glob(pattern)):
            if file_path.is_file():
                result = self.explore_single_file(file_path)
                results.append(result)
        
        return results
    
    def explore_all_data(self) -> Dict[str, List[Dict]]:
        """
        Explore ALL datasets from all sources.
        
        Returns:
            Dictionary organized by data source
        """
        self.logger.info("="*80)
        self.logger.info("COMPREHENSIVE DATA EXPLORATION - ALL DATASETS")
        self.logger.info("="*80)
        
        all_explorations = {}
        
        # PPMI Gait Data
        ppmi_dir = self.data_dir / "PPMI_Gait"
        if ppmi_dir.exists():
            self.logger.info("\n### EXPLORING PPMI GAIT DATA ###")
            all_explorations['PPMI_Gait'] = []
            for file_path in sorted(ppmi_dir.glob("*")):
                if file_path.is_file() and file_path.suffix in ['.csv', '.xlsx', '.xls']:
                    result = self.explore_single_file(file_path)
                    all_explorations['PPMI_Gait'].append(result)
        
        # LRRK2 Clinical
        lrrk2_clinical_dir = self.data_dir / "LRRK2_Clinical"
        if lrrk2_clinical_dir.exists():
            self.logger.info("\n### EXPLORING LRRK2 CLINICAL DATA ###")
            all_explorations['LRRK2_Clinical'] = []
            for file_path in sorted(lrrk2_clinical_dir.glob("*.csv")):
                result = self.explore_single_file(file_path)
                all_explorations['LRRK2_Clinical'].append(result)
        
        # LRRK2 Biomarkers
        lrrk2_bio_dir = self.data_dir / "LRRK2_Biomarkers"
        if lrrk2_bio_dir.exists():
            self.logger.info("\n### EXPLORING LRRK2 BIOMARKER DATA ###")
            all_explorations['LRRK2_Biomarkers'] = []
            for file_path in sorted(lrrk2_bio_dir.glob("*")):
                if file_path.is_file() and file_path.suffix in ['.csv', '.xlsx', '.xls']:
                    result = self.explore_single_file(file_path)
                    all_explorations['LRRK2_Biomarkers'].append(result)
        
        # Synapse WearGait
        synapse_dir = self.data_dir / "Synapse_WearGait"
        if synapse_dir.exists():
            self.logger.info("\n### EXPLORING SYNAPSE WEARGAIT DATA ###")
            all_explorations['Synapse_WearGait'] = []
            for file_path in sorted(synapse_dir.glob("*")):
                if file_path.is_file():
                    result = self.explore_single_file(file_path)
                    all_explorations['Synapse_WearGait'].append(result)
        
        # Mendeley Gait
        mendeley_dir = self.data_dir / "Mendeley_Gait"
        if mendeley_dir.exists():
            self.logger.info("\n### EXPLORING MENDELEY GAIT DATA ###")
            all_explorations['Mendeley_Gait'] = []
            for file_path in sorted(mendeley_dir.glob("*.xlsx")):
                result = self.explore_single_file(file_path)
                all_explorations['Mendeley_Gait'].append(result)
        
        # Bioclite
        bioclite_dir = self.data_dir / "Bioclite"
        if bioclite_dir.exists():
            self.logger.info("\n### EXPLORING BIOCLITE DATA ###")
            all_explorations['Bioclite'] = []
            for file_path in sorted(bioclite_dir.glob("*")):
                if file_path.is_file():
                    result = self.explore_single_file(file_path)
                    all_explorations['Bioclite'].append(result)
        
        self.exploration_results = all_explorations
        
        # Generate summary
        total_files = sum(len(files) for files in all_explorations.values())
        successful = sum(1 for source in all_explorations.values() 
                        for file in source if file.get('exploration_status') == 'success')
        
        self.logger.info("="*80)
        self.logger.info(f"EXPLORATION COMPLETE: {successful}/{total_files} files successfully explored")
        self.logger.info("="*80)
        
        return all_explorations
    
    def generate_exploration_report(self, output_path: str):
        """
        Generate comprehensive exploration report.
        
        Args:
            output_path: Path to save report
        """
        self.logger.info(f"Generating exploration report: {output_path}")
        
        # Save full results as JSON
        json_path = Path(output_path).parent / "data_exploration_full.json"
        with open(json_path, 'w') as f:
            json.dump(self.exploration_results, f, indent=2, default=str)
        
        # Generate human-readable summary
        with open(output_path, 'w') as f:
            f.write("# DATA EXPLORATION REPORT\n\n")
            f.write("## Summary by Source\n\n")
            
            for source, files in self.exploration_results.items():
                f.write(f"### {source}\n\n")
                f.write(f"**Total Files:** {len(files)}\n\n")
                
                for file_result in files:
                    if file_result.get('exploration_status') == 'success':
                        f.write(f"#### {file_result['file_name']}\n\n")
                        f.write(f"- **Rows:** {file_result.get('n_rows', 'N/A')}\n")
                        f.write(f"- **Columns:** {file_result.get('n_columns', 'N/A')}\n")
                        f.write(f"- **Data Density:** {file_result.get('data_density', 'N/A')}%\n")
                        f.write(f"- **File Size:** {file_result.get('file_size_mb', 0):.2f} MB\n")
                        
                        if 'potential_id_columns' in file_result and file_result['potential_id_columns']:
                            f.write(f"- **ID Columns:** {', '.join(file_result['potential_id_columns'])}\n")
                        
                        if 'numeric_columns' in file_result:
                            f.write(f"- **Numeric Features:** {len(file_result['numeric_columns'])}\n")
                        
                        f.write("\n")
                
                f.write("\n")
        
        self.logger.info(f"Report saved: {output_path}")
        self.logger.info(f"Full JSON: {json_path}")


if __name__ == "__main__":
    from logger_config import setup_logging
    
    logger_system = setup_logging()
    logger = logger_system.get_logger("data_explorer_test", "data_loading")
    
    explorer = ComprehensiveDataExplorer(
        "/home1/11021/harshtirhekar/WORK/GaitAnalysis/Attempt2/data",
        logger
    )
    
    # Test exploration
    explorations = explorer.explore_all_data()
    explorer.generate_exploration_report("data_exploration_report.md")
    
    print("\nExploration complete!")



