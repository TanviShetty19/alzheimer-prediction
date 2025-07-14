import pandas as pd
from pathlib import Path

def clean_clinical_data(input_path: Path, output_path: Path) -> pd.DataFrame:
    """Handle missing values and outliers"""
    df = pd.read_csv(input_path)
    
    # Missing data handling
    df["SES"].fillna(df["SES"].median(), inplace=True)
    df["MMSE"].fillna(df["MMSE"].median(), inplace=True)
    
    # Convert CDR to categorical
    df["CDR"] = df["CDR"].astype('category')
    
    # Handle edge cases
    df = df[df["Age"] >= 18]  # Remove pediatric cases if any
    
    df.to_csv(output_path, index=False)
    
    print(f"Cleaned data saved to {output_path}")
    print("Missing values after cleaning:")
    print(df.isnull().sum())
    
    return df