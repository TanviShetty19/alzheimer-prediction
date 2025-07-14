import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

def create_splits(input_path: Path, output_dir: Path, test_size: float, random_state: int):
    """Create reproducible train/test splits"""
    df = pd.read_csv(input_path)
    
    # Get unique subjects for splitting
    subjects = df["Subject ID"].unique()
    
    # Stratify by diagnosis group if available
    stratify = df.groupby("Subject ID")["Group"].first() if "Group" in df else None
    
    train_subjects, test_subjects = train_test_split(
        subjects,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify
    )
    
    # Save subject IDs
    output_dir.mkdir(exist_ok=True)
    pd.Series(train_subjects).to_csv(output_dir/"train_ids.csv", index=False, header=False)
    pd.Series(test_subjects).to_csv(output_dir/"test_ids.csv", index=False, header=False)
    
    print(f"Train/test splits saved to {output_dir}")
    print(f"Train subjects: {len(train_subjects)}")
    print(f"Test subjects: {len(test_subjects)}")