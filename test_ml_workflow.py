#!/usr/bin/env python3
"""Test ML workflow with MLflow logging"""

import os
import sys
import mlflow
from omegaconf import OmegaConf
from data_processing import DataProcessor
from models import ModelTrainer
from ray_trainer import run_hyperparameter_search

def setup_mlflow(data_version, experiment_name):
    if mlflow.active_run():
        print("MLflow run already active, ending previous run.")
        mlflow.end_run()

    # PostgreSQL connection string
    db_uri = 'postgresql://mlflow_user:mlflow_pass@localhost:5432/mlflow_db'
    mlflow.set_tracking_uri(db_uri)
    mlflow.set_experiment(f"test_workflow_v{data_version}")
    return mlflow.start_run(run_name=experiment_name)

def test_workflow():
    # Load minimal configuration
    config = OmegaConf.create({
        'name': 'test_workflow',
        'data': {
            'feature_prefix': '',
            'exclude_columns': ['name', 'age', 'outcome', 'year', 'yearmonth', 'pseudo_id', 'batch', 'sex'],
            'target_column': 'age',
            'target_name': 'outcome'
        },
        'preprocessing': {
            'steps': {
                'imputer': {
                    'module': 'sklearn.impute',
                    'class': 'SimpleImputer',
                    'params': {'strategy': 'median'}
                },
                'scaler': {
                    'module': 'sklearn.preprocessing',
                    'class': 'StandardScaler',
                    'params': {}
                }
            }
        },
        'model': {
            'name': 'linear',
            'class_path': 'sklearn.linear_model.Ridge',
            'params': {'random_state': 42},
            'search_space': {
                'alpha': {
                    'type': 'loguniform',
                    'lower': 0.001,
                    'upper': 100.0
                }
            }
        },
        'search': {
            'name': 'random',
            'algorithm': 'random',
            'params': {}
        },
        'ray': {
            'num_cpus': 2,
            'num_samples': 3,  # Very small for testing
            'max_concurrent': 2
        },
        'cv': {
            'n_splits': 3,  # Smaller for testing
            'random_state': 42
        },
        'seed': 42
    })
    
    # Setup MLflow
    experiment_name = "test_linear_workflow"
    with setup_mlflow("test", experiment_name):
        
        # Log experiment configuration
        mlflow.log_params({
            "preprocessing": "basic",
            "model": "linear", 
            "search_algorithm": "random",
            "data_version": "test"
        })
        
        # Load and process data
        data_processor = DataProcessor(config)
        X, y = data_processor.load_data_from_file("data/processed/v1.0.0/train_data.parquet")
        
        # Take only first 100 samples for quick test
        X_small = X.iloc[:100]
        y_small = y.iloc[:100]
        
        # Log data info
        mlflow.log_params({
            "n_features": X_small.shape[1],
            "n_samples": X_small.shape[0],
        })
        
        # Build preprocessing pipeline and transform data
        data_processor.build_pipeline_from_yaml(config.preprocessing.steps)
        X_processed = data_processor.fit_transform(X_small, y_small)
        
        # Log preprocessing info
        mlflow.log_param("n_features_after_preprocessing", X_processed.shape[1])
        
        print(f"Running hyperparameter search with {config.ray.num_samples} samples...")
        
        # Run hyperparameter search
        analysis, best_params = run_hyperparameter_search(config, X_processed, y_small)
        
        # Log best hyperparameters
        for param, value in best_params.items():
            mlflow.log_param(f"best_{param}", value)
        
        # Train final model
        trainer = ModelTrainer(config.cv)
        final_model = trainer.train_final_model(config.model, best_params, X_processed, y_small)
        
        # Log results
        best_rmse = analysis.get_best_trial('loss', 'min').last_result['loss']
        mlflow.log_metric("best_rmse", best_rmse)
        mlflow.log_metric("cv_rmse", best_rmse)
        
        # Log model
        mlflow.sklearn.log_model(final_model, "model")
        
        print(f"✅ Workflow completed successfully!")
        print(f"   - Best RMSE: {best_rmse:.4f}")
        print(f"   - Best params: {best_params}")
        print(f"   - Run ID: {mlflow.active_run().info.run_id}")
        
        return best_rmse, best_params

if __name__ == "__main__":
    test_workflow()