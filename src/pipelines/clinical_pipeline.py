import pandas as pd
from pathlib import Path
from typing import Dict, List
from clinical_normalize import normalize_clinical_data

def load_and_merge_data(raw_dir: Path) -> pd.DataFrame:
    """Load and merge raw data files with validation"""
    raw_files = list(raw_dir.glob("*.csv"))
    if not raw_files:
        raise FileNotFoundError(f"No CSV files found in {raw_dir}")
    
    dfs = []
    required_cols = {"Age", "Gender", "Group"}
    
    for f in raw_files:
        try:
            df = pd.read_csv(f)
            missing_cols = required_cols - set(df.columns)
            if missing_cols:
                print(f"Warning: Missing columns {missing_cols} in {f.name}")
                continue
            dfs.append(df)
            print(f"Loaded {len(df)} samples from {f.name}")
        except Exception as e:
            print(f"Error loading {f.name}: {str(e)}")
            continue
    
    if not dfs:
        raise ValueError("No valid data files found")
    
    merged = pd.concat(dfs, ignore_index=True)
    print(f"\nMerged {len(merged)} total samples")
    return merged

def clean_merged_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean merged data with validation"""
    # Basic cleaning
    cleaned = df.dropna(subset=["Age", "Gender", "Group"]).copy()
    cleaned = cleaned[cleaned["Age"] > 0]  # Remove invalid ages
    
    # Additional validation
    valid_groups = ["CN", "MCI", "AD"]
    cleaned = cleaned[cleaned["Group"].isin(valid_groups)]
    
    if len(cleaned) == 0:
        raise ValueError("No valid samples after cleaning")
    
    print(f"After cleaning: {len(cleaned)} valid samples")
    print(f"Removed {len(df) - len(cleaned)} invalid records")
    return cleaned

def split_data(df: pd.DataFrame, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into train/test sets with validation"""
    if len(df) < 10:
        raise ValueError(f"Insufficient samples ({len(df)}) for splitting")
    
    try:
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=42,
            stratify=df["Group"]
        )
    except ValueError:
        # Fallback if stratification fails
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=42
        )
    
    print(f"\nSplit results:")
    print(f"- Training set: {len(train_df)} samples")
    print(f"- Test set: {len(test_df)} samples")
    return train_df, test_df

def run_clinical_pipeline(raw_dir: Path, processed_dir: Path) -> Dict[str, int]:
    """
    Full clinical data processing pipeline
    
    Args:
        raw_dir: Directory containing raw data
        processed_dir: Directory for processed outputs
        
    Returns:
        Dictionary of processing statistics
    """
    stats = {}
    try:
        # Create directories if needed
        processed_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Load and merge raw data
        print("\n=== Loading and Merging Data ===")
        merged = load_and_merge_data(raw_dir)
        stats["initial_samples"] = len(merged)
        
        # 2. Clean data
        print("\n=== Cleaning Data ===")
        cleaned = clean_merged_data(merged)
        stats["cleaned_samples"] = len(cleaned)
        
        cleaned_path = processed_dir / "2_cleaned.csv"
        cleaned.to_csv(cleaned_path, index=False)
        
        # 3. Normalize data
        print("\n=== Normalizing Data ===")
        normalized = normalize_clinical_data(
            input_path=cleaned_path,
            output_path=processed_dir/"3_normalized.csv",
            metadata_path=processed_dir/"scaler_metadata.json"
        )
        stats["normalized_samples"] = len(normalized)
        
        # 4. Split data
        print("\n=== Splitting Data ===")
        train_df, test_df = split_data(normalized)
        stats["train_samples"] = len(train_df)
        stats["test_samples"] = len(test_df)
        
        # Save splits
        train_df.to_csv(processed_dir/"4_train.csv", index=False)
        test_df.to_csv(processed_dir/"4_test.csv", index=False)
        
        return stats
        
    except Exception as e:
        print(f"\n💥 Pipeline failed: {str(e)}")
        
        # Clean up partial outputs
        for step in range(2, 5):
            for f in processed_dir.glob(f"{step}_*"):
                try:
                    f.unlink()
                except:
                    pass
                    
        if (processed_dir/"scaler_metadata.json").exists():
            (processed_dir/"scaler_metadata.json").unlink()
            
        raise

if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    
    try:
        raw_data = Path("data/raw")
        processed_data = Path("data/processed")
        
        print("🚀 Starting Clinical Data Pipeline")
        results = run_clinical_pipeline(raw_dir=raw_data, processed_dir=processed_data)
        
        print("\n📊 Pipeline Results:")
        for k, v in results.items():
            print(f"{k.replace('_', ' ').title()}: {v}")
            
    except Exception as e:
        print(f"\n❌ Critical Error: {str(e)}")
        print("Please check:")
        print("- Input files exist and are readable")
        print("- Required columns are present")
        print("- Data values are valid")
        exit(1)