# src/prepare_data.py - Simple data preparation (moved from scripts/)
import argparse
import pandas as pd
import json
from pathlib import Path
import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Input CSV file')
    parser.add_argument('--version', required=True, help='Data version')
    parser.add_argument('--config', default='config/config.yaml', help='Path to config.yaml')
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    data_cfg = config.get('data', {})
    feature_prefix = data_cfg.get('feature_prefix')
    target_column = data_cfg.get('target_column')
    target_name = data_cfg.get('target_name')
    exclude_columns = data_cfg.get('exclude_columns', [])

    # Load data
    df = pd.read_csv(args.input)

    # Rename target column for consistent processing
    if target_column in df.columns and target_column != target_name:
        df = df.rename(columns={target_column: target_name})

    # Extract features
    feature_cols = [col for col in df.columns if col.startswith(feature_prefix)] if feature_prefix else [col for col in df.columns if col != target_name and col not in exclude_columns]

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
