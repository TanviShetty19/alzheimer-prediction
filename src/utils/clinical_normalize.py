import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
from typing import Dict, Tuple

def validate_input_data(df: pd.DataFrame, numerical_cols: list) -> None:
    """Validate input data structure and content"""
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")
    
    if len(df) == 0:
        raise ValueError("Input DataFrame is empty")
    
    missing_cols = [col for col in numerical_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    for col in numerical_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"Column {col} must be numeric")

def clean_data(df: pd.DataFrame, numerical_cols: list) -> pd.DataFrame:
    """Clean and validate numerical data"""
    # Replace infinities with NaN
    df[numerical_cols] = df[numerical_cols].replace([np.inf, -np.inf], np.nan)
    
    # Count NA values before cleaning
    na_counts = df[numerical_cols].isna().sum()
    
    # Remove rows with NA values in numerical columns
    cleaned = df.dropna(subset=numerical_cols).copy()
    
    if len(cleaned) == 0:
        raise ValueError("All rows contained invalid values after cleaning")
    
    # Log cleaning results
    print(f"Removed {len(df) - len(cleaned)} rows with invalid values")
    for col, count in na_counts.items():
        if count > 0:
            print(f" - {col}: {count} NA values removed")
    
    return cleaned

def normalize_clinical_data(input_path: Path, output_path: Path, metadata_path: Path) -> pd.DataFrame:
    """
    Normalize clinical data with comprehensive validation
    
    Args:
        input_path: Path to cleaned input CSV
        output_path: Path to save normalized CSV
        metadata_path: Path to save scaler metadata
        
    Returns:
        Normalized DataFrame
        
    Raises:
        ValueError: If validation checks fail
    """
    try:
        # 1. Load and validate input
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
            
        df = pd.read_csv(input_path)
        original_samples = len(df)
        
        numerical_cols = ["Age", "EDUC", "SES", "MMSE", "eTIV", "nWBV", "ASF"]
        validate_input_data(df, numerical_cols)
        
        # 2. Clean data
        df = clean_data(df, numerical_cols)
        
        # 3. Normalize with validation
        scaler = MinMaxScaler()
        normalized_values = scaler.fit_transform(df[numerical_cols])
        
        if normalized_values.shape != df[numerical_cols].shape:
            raise ValueError(
                f"Shape mismatch: Input {df[numerical_cols].shape} vs "
                f"Output {normalized_values.shape}"
            )
            
        df[numerical_cols] = normalized_values
        
        # 4. Save outputs with metadata
        scaler_meta = {
            "n_samples": len(df),
            "features": numerical_cols,
            "min_values": scaler.data_min_.tolist(),
            "max_values": scaler.data_max_.tolist(),
            "original_samples": original_samples,
            "dropped_samples": original_samples - len(df)
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(scaler_meta, f, indent=2)
            
        df.to_csv(output_path, index=False)
        print(f"Successfully normalized {len(df)} samples")
        return df
        
    except Exception as e:
        # Clean up any partial outputs
        if 'output_path' in locals() and output_path.exists():
            output_path.unlink()
        if 'metadata_path' in locals() and metadata_path.exists():
            metadata_path.unlink()
        raise ValueError(f"Normalization failed: {str(e)}")