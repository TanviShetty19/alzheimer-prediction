import pandas as pd
from pathlib import Path

def merge_clinical_data(longitudinal_path: Path, cross_sectional_path: Path, output_path: Path) -> pd.DataFrame:
    """Combine longitudinal and cross-sectional datasets"""
    long = pd.read_csv(longitudinal_path)
    cross = pd.read_csv(cross_sectional_path)
    
    # Standardize column names if needed
    cross.rename(columns={"ID": "MRI_ID"}, inplace=True)
    
    # Add dataset source marker
    long["data_source"] = "longitudinal"
    cross["data_source"] = "cross_sectional"
    
    combined = pd.concat([long, cross], ignore_index=True)
    combined.to_csv(output_path, index=False)
    
    print(f"Merged data saved to {output_path}")
    print(f"Shape: {combined.shape}")
    print("Unique subjects:", combined["Subject ID"].nunique())
    
    return combined