import json
import pandas as pd
from pathlib import Path

def create_metadata(csv_path: str, output_json: str) -> None:
    """Generate metadata JSON from clinical CSV"""
    df = pd.read_csv(csv_path)
    
    # Auto-detect column properties
    metadata = {
        "columns": {},
        "stats": {
            "num_subjects": df['Subject ID'].nunique(),
            "num_scans": len(df),
            "date_range": {
                "min_age": int(df['Age'].min()),
                "max_age": int(df['Age'].max())
            }
        }
    }
    
    # Column rules (extend as needed)
    column_rules = {
        "Group": {
            "type": "categorical",
            "values": ["Nondemented", "Demented", "Converted"],
            "description": "Diagnosis classification"
        },
        "MMSE": {
            "type": "numerical",
            "range": [0, 30],
            "description": "Mini-Mental State Examination score"
        }
    }
    
    # Auto-populate remaining columns
    for col in df.columns:
        if col not in column_rules:
            dtype = "categorical" if df[col].nunique() < 20 else "numerical"
            metadata["columns"][col] = {"type": dtype}
    
    metadata["columns"].update(column_rules)
    
    with open(output_json, 'w') as f:
        json.dump(metadata, f, indent=2)

def validate_data(df: pd.DataFrame, metadata_path: str) -> dict:
    """Validate DataFrame against metadata rules"""
    with open(metadata_path) as f:
        meta = json.load(f)
    
    errors = {}
    
    for col, specs in meta["columns"].items():
        if col not in df.columns:
            errors[col] = "Missing column"
            continue
            
        # Check categorical values
        if "values" in specs:
            invalid = set(df[col].dropna().unique()) - set(specs["values"])
            if invalid:
                errors[col] = f"Invalid values: {invalid}"
                
        # Check numerical ranges
        if "range" in specs:
            out_of_range = ~df[col].between(*specs["range"])
            if out_of_range.any():
                errors[col] = f"Values outside {specs['range']}"
    
    return errors