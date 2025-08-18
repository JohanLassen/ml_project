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

    if mlflow.active_run():
        print("MLflow run already active, ending previous run.")
        mlflow.end_run()

    """Setup MLflow with PostgreSQL backend"""
    # PostgreSQL connection string - customize as needed
    db_uri = os.environ.get(
        'MLFLOW_TRACKING_URI', 
        'postgresql://mlflow_user:mlflow_pass@localhost:5432/mlflow_db'
    )
    mlflow.set_tracking_uri(db_uri)
    mlflow.set_experiment(f"tabular_regression_v{data_version}")
    return mlflow.start_run(run_name=experiment_name)

def setup_gpu_if_needed(model_name, cpus):
    """Configure GPU settings for XGBoost"""
    if model_name == 'xgboost' and 'CUDA_VISIBLE_DEVICES' in os.environ:
        return {
            'ray': {
                'num_cpus': cpus,
                'num_gpus': 1,
                'num_samples': 1000,
                'max_concurrent': 16  # HPC with GPU
            }
        }
    else:
        return {
            'ray': {
                'num_cpus': cpus,
                'num_samples': 500,
                'max_concurrent': 4   # Local or HPC CPU
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
    project_dir = os.environ.get('PROJECT_DIR', '.')
    
    # Set seed first so it can be interpolated
    OmegaConf.set_struct(OmegaConf.create({}), False)
    
    config = OmegaConf.load(f"{project_dir}/config/config.yaml")
    config.preprocessing = OmegaConf.load(f"{project_dir}/config/preprocessing/preprocessing.yaml")[args.preprocessing]
    config.model = OmegaConf.load(f"{project_dir}/config/model/{args.model}.yaml")
    config.search = OmegaConf.load(f"{project_dir}/config/search/{args.search}.yaml")
    
    # Resolve ${seed} interpolations in model config # freezes the config but makes values easier to access
    config = OmegaConf.to_container(config, resolve=True) 
    config = OmegaConf.create(config)
    
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
        
        # Start MLflow run
        experiment_name = f"{args.preprocessing}_{args.model}_{args.search}"
        run = setup_mlflow(args.data_version, experiment_name)
        
        with run:
            # Log experiment configuration
            mlflow.log_params({
                "preprocessing": args.preprocessing,
                "model": args.model,
                "search_algorithm": args.search,
                "data_version": args.data_version
            })
            
            
            # Load and process data
            data_processor = DataProcessor(config)
            X, y = data_processor.load_data_from_file(args.data)
            
            # Log data info
            mlflow.log_params({
                "n_features": X.shape[1],
                "n_samples": X.shape[0],
            })
            
            # Build preprocessing pipeline and transform data
            data_processor.build_pipeline_from_yaml(config.preprocessing.steps)
            X_processed = data_processor.fit_transform(X, y)
            
            # Log preprocessing info
            mlflow.log_param("n_features_after_preprocessing", X_processed.shape[1])
            
            # Run hyperparameter search
            analysis, best_params = run_hyperparameter_search(config, X_processed, y)
            
            # Log best hyperparameters
            for param, value in best_params.items():
                mlflow.log_param(f"best_{param}", value)
            
            # Train final model
            trainer = ModelTrainer(config.cv)
            final_model = trainer.train_final_model(config.model, best_params, X_processed, y)
            
            # Log results
            best_rmse = analysis.get_best_trial('loss', 'min').last_result['loss']
            mlflow.log_metric("best_rmse", best_rmse)
            mlflow.log_metric("cv_rmse", best_rmse)  # Alias for easier searching
            
            # Log model
            mlflow.sklearn.log_model(final_model, "model")
            
            # Log all trials as metrics for comparison
            all_trials = analysis.trials
            for i, trial in enumerate(all_trials):
                if trial.last_result:
                    mlflow.log_metric(f"trial_{i}_rmse", trial.last_result['loss'])
                    
            print(f"MLflow run completed: {run.info.run_id}")
            mlflow.end_run()  # End any existing run
        
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