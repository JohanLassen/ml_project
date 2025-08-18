# scripts/prepare_data.py - Simple data preparation
import argparse
import pandas as pd
import json
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Input CSV file')
    parser.add_argument('--version', required=True, help='Data version')
    args = parser.parse_args()
    
    # Load data
    df = pd.read_csv(args.input)
    
    # Dynamic column mapping - could be configured via args if needed
    feature_prefix = 'M'
    target_column = 'age'  # Source column name
    target_name = 'outcome'  # Standardized name for processing
    
    # Rename target column for consistent processing
    if target_column in df.columns and target_column != target_name:
        df = df.rename(columns={target_column: target_name})
    
    # Extract features
    feature_cols = [col for col in df.columns if col.startswith(feature_prefix)]
    
    # Save as parquet
    df.to_parquet('train_data.parquet')
    
    # Save metadata
    metadata = {
        'data_version': args.version,
        'n_samples': len(df),
        'n_features': len(feature_cols),
        'target_col': target_name,
        'original_target_col': target_column,
        'source_file': args.input
    }
    
    with open('data_info.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Prepared data v{args.version}: {len(df)} samples, {len(feature_cols)} features")

if __name__ == "__main__":
    main()