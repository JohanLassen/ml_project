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

class DistributedOptunaTrainer:
    """Distributed Optuna trainer that works across multiple SLURM jobs"""
    
    def __init__(self, storage_path: str = "sqlite:///optuna_studies.db"):
        """
        Initialize distributed Optuna trainer
        
        Args:
            storage_path: Shared database path (accessible from all SLURM nodes)
        """
        self.storage_path = storage_path
        
        # Ensure database directory exists and is accessible across SLURM nodes
        if storage_path.startswith("sqlite:///"):
            db_path = Path(storage_path.replace("sqlite:///", ""))
            db_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Make sure the database file has proper permissions for HPC
            if db_path.exists():
                import stat
                db_path.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IWGRP)
    
    def run_single_trial(self, config: Dict, X, y, max_trials: int = 100) -> Tuple[Dict, float]:
        """
        Run a single trial (designed to be called from separate SLURM jobs)
        
        Args:
            config: Configuration dictionary
            X, y: Training data
            max_trials: Maximum total trials for the study
            
        Returns:
            Tuple of (trial_params, trial_score) for this specific trial
        """
        # Create study name based on experiment configuration
        study_name = f"{config.get('name', 'study')}_{config['model']['name']}_{config['search']['name']}"
        
        # Create or load study (shared across all jobs)
        study = optuna.create_study(
            study_name=study_name,
            storage=self.storage_path,
            load_if_exists=True,
            direction='minimize',
            sampler=self._get_sampler(config['search'])
        )
        
        # Check if we've already completed enough trials
        if len(study.trials) >= max_trials:
            logger.info(f"Study {study_name} already has {len(study.trials)} trials, skipping")
            return study.best_params, study.best_value
        
        # Create objective function for this trial
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
        
        # Run exactly ONE trial in this job
        study.optimize(objective, n_trials=1)
        
        # Get the trial we just completed
        last_trial = study.trials[-1]
        trial_params = last_trial.params
        trial_score = last_trial.value
        
        logger.info(f"Completed trial {last_trial.number}: RMSE={trial_score:.4f}")
        logger.info(f"Study {study_name} now has {len(study.trials)} total trials")
        
        return trial_params, trial_score
    
    def get_best_results(self, config: Dict) -> Tuple[Dict, float]:
        """Get the best results from completed study"""
        study_name = f"{config.get('name', 'study')}_{config['model']['name']}_{config['search']['name']}"
        
        try:
            study = optuna.load_study(
                study_name=study_name,
                storage=self.storage_path
            )
            return study.best_params, study.best_value
        except Exception as e:
            logger.error(f"Failed to load study {study_name}: {e}")
            return {}, float('inf')
    
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
            return optuna.samplers.TPESampler()
        else:
            logger.warning(f"Unknown algorithm {algorithm}, using TPE")
            return optuna.samplers.TPESampler()
