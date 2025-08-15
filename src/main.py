# src/main.py - Clean and simple
import argparse
import json
import os
import mlflow
from pathlib import Path
from data_processing import DataProcessor
from models import ModelTrainer
from ray_trainer import run_hyperparameter_search

def setup_mlflow(data_version, experiment_name):
    """Setup MLflow with data version"""
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment(f"tabular_regression_v{data_version}")
    return mlflow.start_run(run_name=experiment_name)

def setup_gpu_if_needed(model_name, cpus):
    """Configure GPU settings for XGBoost"""
    if model_name == 'xgboost' and 'CUDA_VISIBLE_DEVICES' in os.environ:
        return {
            'ray': {
                'num_cpus': cpus,
                'num_gpus': 1,
                'num_samples': 1000
            }
        }
    else:
        return {
            'ray': {
                'num_cpus': cpus,
                'num_samples': 500
            }
        }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preprocessing', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--search', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--data_version', required=True)
    parser.add_argument('--cpus', type=int, default=4)
    args = parser.parse_args()
    
    # Load configuration
    from omegaconf import OmegaConf
    config = OmegaConf.load(f"config/{args.preprocessing}.yaml")
    config.model = OmegaConf.load(f"config/model/{args.model}.yaml")
    config.search = OmegaConf.load(f"config/search/{args.search}.yaml")
    
    # Add GPU settings
    config.update(setup_gpu_if_needed(args.model, args.cpus))
    config.data_version = args.data_version
    
    # Setup MLflow
    experiment_name = f"{args.preprocessing}_{args.model}_{args.search}"
    with setup_mlflow(args.data_version, experiment_name):
        
        # Log basic info
        mlflow.log_params({
            'data_version': args.data_version,
            'preprocessing': args.preprocessing,
            'model': args.model,
            'search': args.search,
            'cpus': args.cpus,
            'gpu_enabled': args.model == 'xgboost' and 'CUDA_VISIBLE_DEVICES' in os.environ
        })
        
        # Load and process data
        data_processor = DataProcessor(config)
        X, y = data_processor.load_data_from_file(args.data)
        X_processed = data_processor.fit_transform(X, y)
        
        # Run hyperparameter search
        analysis, best_params = run_hyperparameter_search(config, X_processed, y)
        
        # Train final model
        trainer = ModelTrainer({'n_splits': 5, 'random_state': 42})
        final_model = trainer.train_final_model(config.model, best_params, X_processed, y)
        
        # Log results
        best_rmse = analysis.get_best_trial('loss', 'min').last_result['loss']
        mlflow.log_metric("best_rmse", best_rmse)
        mlflow.sklearn.log_model(final_model, "model")
        
        # Save result for Nextflow
        result = {
            'experiment': experiment_name,
            'best_rmse': best_rmse,
            'status': 'completed'
        }
        
        with open(f'result_{experiment_name}.json', 'w') as f:
            json.dump(result, f)
        
        print(f"Completed: {experiment_name}, RMSE: {best_rmse:.4f}")

if __name__ == "__main__":
    main()