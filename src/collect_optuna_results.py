#!/usr/bin/env python3
"""
Collect results from distributed Optuna trials and find the best parameters
"""
import argparse
import json
import glob
from pathlib import Path
from distributed_optuna_trainer import DistributedOptunaTrainer

def main():
    parser = argparse.ArgumentParser(description='Collect distributed Optuna results')
    parser.add_argument('--preprocessing', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--search', required=True)
    parser.add_argument('--trial_files', nargs='+', required=True)
    parser.add_argument('--data_version', required=True)
    
    args = parser.parse_args()
    
    # Load configuration to get study info
    import os
    from omegaconf import OmegaConf
    project_dir = os.environ.get('PROJECT_DIR', Path('.').resolve())
    
    config = OmegaConf.load(f"{project_dir}/config/config.yaml")
    config.preprocessing = OmegaConf.load(f"{project_dir}/config/preprocessing/preprocessing.yaml")[args.preprocessing]
    config.model = OmegaConf.load(f"{project_dir}/config/model/{args.model}.yaml")
    config.search = OmegaConf.load(f"{project_dir}/config/search/{args.search}.yaml")
    config = OmegaConf.to_container(config, resolve=True)
    config = OmegaConf.create(config)
    
    # Get best results from the shared Optuna database
    distributed_trainer = DistributedOptunaTrainer("sqlite:///shared_optuna_studies.db")
    best_params, best_rmse = distributed_trainer.get_best_results(config)
    
    # Count completed trials
    completed_trials = []
    failed_trials = []
    
    for trial_file in args.trial_files:
        try:
            with open(trial_file, 'r') as f:
                trial_result = json.load(f)
                if trial_result.get('status') == 'completed':
                    completed_trials.append(trial_result)
                else:
                    failed_trials.append(trial_result)
        except Exception as e:
            print(f"Error reading {trial_file}: {e}")
            failed_trials.append({'trial_file': trial_file, 'error': str(e)})
    
    # Create summary result
    experiment_name = f"{args.preprocessing}_{args.model}_{args.search}"
    result = {
        'experiment': experiment_name,
        'best_params': best_params,
        'best_rmse': best_rmse,
        'n_completed_trials': len(completed_trials),
        'n_failed_trials': len(failed_trials),
        'status': 'completed' if best_params else 'failed',
        'distributed_optuna': True
    }
    
    # Save result
    output_file = f'result_{experiment_name}.json'
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"Experiment {experiment_name}:")
    print(f"  Best RMSE: {best_rmse:.4f}")
    print(f"  Completed trials: {len(completed_trials)}")
    print(f"  Failed trials: {len(failed_trials)}")
    print(f"  Best params: {best_params}")

if __name__ == "__main__":
    main()
