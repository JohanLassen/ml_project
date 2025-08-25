import optuna
import mlflow
import logging
from typing import Dict, Any, Tuple
import json
import os
from models import ModelTrainer
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)

class OptunaTrainer:
    """Optuna-based hyperparameter optimization that works well with SLURM and Nextflow"""
    
    def __init__(self, storage_path: str = "sqlite:///optuna_studies.db"):
        """
        Initialize Optuna trainer
        
        Args:
            storage_path: Database path for storing study results (shared across SLURM jobs)
        """
        self.storage_path = storage_path
        # Ensure database directory exists
        if storage_path.startswith("sqlite:///"):
            db_path = Path(storage_path.replace("sqlite:///", ""))
            db_path.parent.mkdir(parents=True, exist_ok=True)
    
    def create_objective(self, config: Dict, X, y) -> callable:
        """Create objective function for Optuna optimization"""
        
        def objective(trial: optuna.Trial) -> float:
            # Generate hyperparameters based on search space
            hyperparams = self._suggest_hyperparams(trial, config['model']['search_space'])
            
            # Create trainer
            trainer = ModelTrainer(config['cv'])
            
            # Evaluate model
            rmse = trainer.evaluate_model(
                config['model'],
                hyperparams,
                X,
                y
            )
            
            # Log to MLflow
            self._log_to_mlflow(config, hyperparams, rmse, trial.number)
            
            return rmse
        
        return objective
    
    def _suggest_hyperparams(self, trial: optuna.Trial, search_space: Dict) -> Dict:
        """Convert search space config to Optuna suggestions"""
        hyperparams = {}
        
        for param, config in search_space.items():
            if config['type'] == "choice":
                hyperparams[param] = trial.suggest_categorical(param, config['values'])
            elif config['type'] == "randint":
                hyperparams[param] = trial.suggest_int(param, config['lower'], config['upper'])
            elif config['type'] == "uniform":
                hyperparams[param] = trial.suggest_float(param, config['lower'], config['upper'])
            elif config['type'] == "loguniform":
                hyperparams[param] = trial.suggest_float(
                    param, config['lower'], config['upper'], log=True
                )
        
        return hyperparams
    
    def _log_to_mlflow(self, config: Dict, hyperparams: Dict, rmse: float, trial_number: int):
        """Log results to MLflow"""
        try:
            with mlflow.start_run(nested=True, run_name=f"trial_{trial_number}") as run:
                mlflow.log_params(hyperparams)
                mlflow.log_metric("rmse", rmse)
                mlflow.log_param("model_type", config['model']['name'])
                mlflow.log_param("preprocessing", config.get('name', 'unknown'))
                mlflow.log_param("trial_number", trial_number)
        except Exception as e:
            logger.warning(f"MLflow logging failed: {e}")
    
    def run_optimization(self, config: Dict, X, y, n_trials: int = None) -> Tuple[Dict, float]:
        """
        Run hyperparameter optimization
        
        Args:
            config: Configuration dictionary
            X, y: Training data
            n_trials: Number of trials (if None, uses config value)
            
        Returns:
            Tuple of (best_params, best_score)
        """
        # Create study name based on experiment configuration
        study_name = f"{config.get('name', 'study')}_{config['model']['name']}_{config['search']['name']}"
        
        # Determine number of trials
        if n_trials is None:
            n_trials = config.get('search', {}).get('n_trials', 50)
        
        # Create or load study
        study = optuna.create_study(
            study_name=study_name,
            storage=self.storage_path,
            load_if_exists=True,
            direction='minimize',
            sampler=self._get_sampler(config['search'])
        )
        
        # Create objective function
        objective = self.create_objective(config, X, y)
        
        # Run optimization
        logger.info(f"Starting Optuna optimization with {n_trials} trials")
        study.optimize(objective, n_trials=n_trials)
        
        # Get best results
        best_params = study.best_params
        best_score = study.best_value
        
        logger.info(f"Best RMSE: {best_score:.4f}")
        logger.info(f"Best params: {best_params}")
        
        return best_params, best_score
    
    def _get_sampler(self, search_config: Dict) -> optuna.samplers.BaseSampler:
        """Get appropriate sampler based on search configuration"""
        algorithm = search_config.get('algorithm', 'tpe')
        
        if algorithm == 'tpe':
            return optuna.samplers.TPESampler()
        elif algorithm == 'cmaes':
            return optuna.samplers.CmaEsSampler()
        elif algorithm == 'random':
            return optuna.samplers.RandomSampler()
        elif algorithm == 'bohb' or algorithm == 'hyperband':
            # Use TPE with pruning for BOHB-like behavior
            return optuna.samplers.TPESampler()
        else:
            logger.warning(f"Unknown algorithm {algorithm}, using TPE")
            return optuna.samplers.TPESampler()
    
    def get_study_summary(self, study_name: str) -> Dict:
        """Get summary of completed study"""
        try:
            study = optuna.load_study(
                study_name=study_name,
                storage=self.storage_path
            )
            
            return {
                'study_name': study_name,
                'n_trials': len(study.trials),
                'best_value': study.best_value,
                'best_params': study.best_params,
                'best_trial_number': study.best_trial.number
            }
        except Exception as e:
            logger.error(f"Failed to load study {study_name}: {e}")
            return {}

def run_single_trial_optuna(config: Dict, X, y, trial_params: Dict = None) -> Dict:
    """
    Run a single trial with given parameters (useful for Nextflow parallelization)
    
    Args:
        config: Configuration dictionary
        X, y: Training data  
        trial_params: Specific parameters to evaluate (if None, suggests new ones)
        
    Returns:
        Dictionary with trial results
    """
    trainer = ModelTrainer(config['cv'])
    
    if trial_params is None:
        # This would need to be integrated with a study to suggest parameters
        raise NotImplementedError("Parameter suggestion needs study integration")
    
    # Evaluate model with given parameters
    rmse = trainer.evaluate_model(
        config['model'],
        trial_params,
        X,
        y
    )
    
    return {
        'params': trial_params,
        'rmse': rmse,
        'model_type': config['model']['name'],
        'preprocessing': config.get('name', 'unknown')
    }
