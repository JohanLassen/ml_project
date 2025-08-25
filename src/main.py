# src/main.py - Clean and simple
import argparse
import json
import os
import mlflow
from pathlib import Path
from data_processing import DataProcessor
from models import ModelTrainer
from optuna_trainer import OptunaTrainer


def setup_mlflow(data_version, experiment_name):

    if mlflow.active_run():
        print("MLflow run already active, ending previous run.")
        mlflow.end_run()

    """Setup MLflow with PostgreSQL storage"""
    # PostgreSQL MLflow tracking - use environment variable for flexibility
    mlflow_uri = os.environ.get('MLFLOW_TRACKING_URI', 'postgresql://mlflow_user:mlflow_password@localhost:5432/mlflow_db')
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(f"tabular_regression_v{data_version}")
    return mlflow.start_run(run_name=experiment_name)

def setup_optimization(model_name, cpus, test_mode=False):
    """Configure optimization settings"""
    # Reduce trials significantly in test mode
    n_trials = 4 if test_mode else (100 if model_name == 'xgboost' else 50)
    
    return {
        'optuna': {
            'n_trials': n_trials,
            'storage_path': 'sqlite:///optuna_studies.db'
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
    parser.add_argument('--test_mode', action='store_true', help='Use subset of data for faster testing')
    args = parser.parse_args()
    
    # Load configuration
    from omegaconf import OmegaConf
    project_dir = os.environ.get('PROJECT_DIR', '.')
    
    # Set seed first so it can be interpolated
    OmegaConf.set_struct(OmegaConf.create({}), False)
    
    config = OmegaConf.load(f"{project_dir}/config/config.yaml")
    config.preprocessing = OmegaConf.load(f"{project_dir}/config/preprocessing/preprocessing.yaml")[args.preprocessing]
    config.model = OmegaConf.load(f"{project_dir}/config/model/{args.model}.yaml")
    
    # Optimize models for test mode
    if args.test_mode:
        if args.model == 'xgboost':
            # Override search space for faster testing
            config.model.search_space.n_estimators = {"type": "choice", "values": [50, 100]}
            config.model.search_space.max_depth = {"type": "randint", "lower": 3, "upper": 5}
            config.model.params.early_stopping_rounds = 10
        elif args.model == 'linear':
            # Reduce alpha search space for linear models
            config.model.search_space.alpha = {"type": "loguniform", "lower": 0.01, "upper": 10.0}
        elif args.model == 'random_forest':
            # Reduce forest complexity for testing
            config.model.search_space.n_estimators = {"type": "choice", "values": [50, 100]}
            config.model.search_space.max_depth = {"type": "randint", "lower": 3, "upper": 8}
    config.search = OmegaConf.load(f"{project_dir}/config/search/{args.search}.yaml")
    
    # Resolve ${seed} interpolations in model config # freezes the config but makes values easier to access
    config = OmegaConf.to_container(config, resolve=True) 
    config = OmegaConf.create(config)
    
    # Add optimization settings
    config.update(setup_optimization(args.model, args.cpus, args.test_mode))
    config.data_version = args.data_version

        
    # Setup MLflow
    experiment_name = f"{args.preprocessing}_{args.model}_{args.search}"
    with setup_mlflow(args.data_version, experiment_name) as run:
            # Log experiment configuration
            mlflow.log_params({
                "preprocessing": args.preprocessing,
                "model": args.model,
                "search_algorithm": args.search,
                "data_version": args.data_version,
                "cpus": args.cpus,
                "test_mode": args.test_mode
            })
            
            
            # Load and process data
            data_processor = DataProcessor(config)
            X, y = data_processor.load_data_from_file(args.data)
            
            # Subset data for testing if requested
            if args.test_mode:
                # Use even smaller datasets for complex models like XGBoost
                if args.model == 'xgboost':
                    X, y = data_processor.subset_data_for_testing(X, y, max_features=100, max_samples=20)
                else:
                    X, y = data_processor.subset_data_for_testing(X, y, max_features=500, max_samples=50)
            
            # Log data info
            mlflow.log_params({
                "n_features": X.shape[1],
                "n_samples": X.shape[0],
                "test_mode": args.test_mode,
            })
            
            # Build preprocessing pipeline and transform data
            data_processor.build_pipeline_from_yaml(config.preprocessing.steps)
            X_processed = data_processor.fit_transform(X, y)
            
            # Log preprocessing info
            mlflow.log_param("n_features_after_preprocessing", X_processed.shape[1])
            
            # Run hyperparameter search
            optuna_trainer = OptunaTrainer(config['optuna']['storage_path'])
            best_params, best_rmse = optuna_trainer.run_optimization(config, X_processed, y, config['optuna']['n_trials'])
            
            # Log best hyperparameters
            for param, value in best_params.items():
                mlflow.log_param(f"best_{param}", value)
            
            # Train final model
            trainer = ModelTrainer(config.cv)
            final_model = trainer.train_final_model(config.model, best_params, X_processed, y)
            
            # Log results
            mlflow.log_metric("best_rmse", best_rmse)
            mlflow.log_metric("cv_rmse", best_rmse)  # Alias for easier searching
            
            # Log model
            mlflow.sklearn.log_model(final_model, "model")
            
            # Note: Individual trial logging could be added here if needed
            # by accessing the Optuna study database
                    
            print(f"MLflow run completed: {run.info.run_id}")
        
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