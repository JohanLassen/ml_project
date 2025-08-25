#!/usr/bin/env python3
"""
Run a single Optuna trial as part of distributed optimization
"""
import argparse
import json
import os
import mlflow
from pathlib import Path
from data_processing import DataProcessor
from models import ModelTrainer
from distributed_optuna_trainer import DistributedOptunaTrainer

def setup_mlflow(data_version, experiment_name):
    """Setup MLflow with PostgreSQL storage"""
    if mlflow.active_run():
        mlflow.end_run()

    # Setup MLflow - use environment variable for PostgreSQL URI
    # Format: postgresql://username:password@host:port/database_name
    mlflow_uri = os.environ.get('MLFLOW_TRACKING_URI', 'postgresql://mlflow_user:mlflow_password@localhost:5432/mlflow_db')
    mlflow.set_tracking_uri(mlflow_uri)
    
    mlflow.set_experiment(f"tabular_regression_v{data_version}")
    return mlflow.start_run(run_name=experiment_name)

def main():
    parser = argparse.ArgumentParser(description='Run single Optuna trial')
    parser.add_argument('--preprocessing', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--search', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--data_version', required=True)
    parser.add_argument('--trial_id', type=int, required=True)
    parser.add_argument('--max_trials', type=int, default=100)
    parser.add_argument('--cpus', type=int, default=4)
    parser.add_argument('--test_mode', action='store_true')
    
    args = parser.parse_args()
    
    # Load configuration
    from omegaconf import OmegaConf
    project_dir = os.environ.get('PROJECT_DIR', '.')
    
    OmegaConf.set_struct(OmegaConf.create({}), False)
    
    config = OmegaConf.load(f"{project_dir}/config/config.yaml")
    config.preprocessing = OmegaConf.load(f"{project_dir}/config/preprocessing/preprocessing.yaml")[args.preprocessing]
    config.model = OmegaConf.load(f"{project_dir}/config/model/{args.model}.yaml")
    config.search = OmegaConf.load(f"{project_dir}/config/search/{args.search}.yaml")
    
    # Optimize for test mode
    if args.test_mode:
        if args.model == 'xgboost':
            config.model.search_space.n_estimators = {"type": "choice", "values": [50, 100]}
            config.model.search_space.max_depth = {"type": "randint", "lower": 3, "upper": 5}
            config.model.params.early_stopping_rounds = 10
        elif args.model == 'linear':
            config.model.search_space.alpha = {"type": "loguniform", "lower": 0.01, "upper": 10.0}
        elif args.model == 'random_forest':
            config.model.search_space.n_estimators = {"type": "choice", "values": [50, 100]}
            config.model.search_space.max_depth = {"type": "randint", "lower": 3, "upper": 8}
    
    # Resolve config
    config = OmegaConf.to_container(config, resolve=True) 
    config = OmegaConf.create(config)
    config.data_version = args.data_version
    
    # Setup MLflow
    experiment_name = f"{args.preprocessing}_{args.model}_{args.search}_trial_{args.trial_id}"
    
    try:
        with setup_mlflow(args.data_version, experiment_name) as run:
            # Log trial info
            mlflow.log_params({
                "preprocessing": args.preprocessing,
                "model": args.model,
                "search_algorithm": args.search,
                "data_version": args.data_version,
                "trial_id": args.trial_id,
                "max_trials": args.max_trials,
                "test_mode": args.test_mode
            })
            
            # Load and process data
            data_processor = DataProcessor(config)
            X, y = data_processor.load_data_from_file(args.data)
            
            # Subset data for testing
            if args.test_mode:
                if args.model == 'xgboost':
                    X, y = data_processor.subset_data_for_testing(X, y, max_features=100, max_samples=20)
                else:
                    X, y = data_processor.subset_data_for_testing(X, y, max_features=500, max_samples=50)
            
            # Log data info
            mlflow.log_params({
                "n_features": X.shape[1],
                "n_samples": X.shape[0]
            })
            
            # Build preprocessing pipeline
            data_processor.build_pipeline_from_yaml(config.preprocessing.steps)
            X_processed = data_processor.fit_transform(X, y)
            
            mlflow.log_param("n_features_after_preprocessing", X_processed.shape[1])
            
            # Run single trial with distributed Optuna
            distributed_trainer = DistributedOptunaTrainer("sqlite:///shared_optuna_studies.db")
            trial_params, trial_score = distributed_trainer.run_single_trial(
                config, X_processed, y, args.max_trials
            )
            
            # Log trial results
            for param, value in trial_params.items():
                mlflow.log_param(f"trial_{param}", value)
            mlflow.log_metric("trial_rmse", trial_score)
            
            # Save trial result
            result = {
                'trial_id': args.trial_id,
                'experiment': f"{args.preprocessing}_{args.model}_{args.search}",
                'trial_params': trial_params,
                'trial_rmse': trial_score,
                'status': 'completed'
            }
            
            with open(f'trial_result_{args.trial_id}_{args.preprocessing}_{args.model}_{args.search}.json', 'w') as f:
                json.dump(result, f)
            
            print(f"Completed trial {args.trial_id}: RMSE={trial_score:.4f}")
            
    except Exception as e:
        # Save failed trial result
        result = {
            'trial_id': args.trial_id,
            'experiment': f"{args.preprocessing}_{args.model}_{args.search}",
            'error': str(e),
            'status': 'failed'
        }
        
        with open(f'trial_result_{args.trial_id}_{args.preprocessing}_{args.model}_{args.search}.json', 'w') as f:
            json.dump(result, f)
        
        print(f"Trial {args.trial_id} failed: {e}")
        raise

if __name__ == "__main__":
    main()
