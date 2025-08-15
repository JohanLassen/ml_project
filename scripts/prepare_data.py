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
    
    # Extract M features
    m_features = [col for col in df.columns if col.startswith('M')]
    
    # Save as parquet
    df.to_parquet('train_data.parquet')
    
    # Save metadata
    metadata = {
        'data_version': args.version,
        'n_samples': len(df),
        'n_features': len(m_features),
        'target_col': 'outcome',
        'source_file': args.input
    }
    
    with open('data_info.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Prepared data v{args.version}: {len(df)} samples, {len(m_features)} features")

if __name__ == "__main__":
    main()