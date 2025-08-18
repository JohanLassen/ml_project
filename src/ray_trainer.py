import os
import ray
from ray import tune
try:
    from ray.tune.schedulers import BOHBScheduler
    from ray.tune.search.bohb import TuneBOHB
except ImportError:
    # Fallback for different Ray versions
    from ray.tune.schedulers import HyperBandScheduler as BOHBScheduler
    from ray.tune.search.hyperopt import HyperOptSearch as TuneBOHB
import mlflow
from models import ModelTrainer
from typing import Dict
import logging

logger = logging.getLogger(__name__)

class RayTuneTrainable(tune.Trainable):
    def setup(self, config: Dict):
        # Set CUDA environment variables in worker process
        import os
        if 'CUDA_VISIBLE_DEVICES' not in os.environ:
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        
        self.tune_config = config
        self.hydra_config = config['hydra_config']
        self.X = config['X']
        self.y = config['y']
        self.trainer = ModelTrainer(self.hydra_config.cv)
    
    def step(self) -> Dict[str, float]:
        # Extract hyperparameters (remove non-param keys)
        hyperparams = {k: v for k, v in self.tune_config.items() 
                      if k not in ['hydra_config', 'X', 'y']}
        
        # Evaluate model
        rmse = self.trainer.evaluate_model(
            self.hydra_config.model,
            hyperparams,
            self.X,
            self.y
        )
        
        # Log to MLflow (create nested run for each trial)
        try:
            import mlflow
            with mlflow.start_run(nested=True, run_name=f"trial_{self.trial_id}") as nested_run:
                mlflow.log_params(hyperparams)
                mlflow.log_metric("rmse", rmse)
                mlflow.log_param("model_type", self.hydra_config.model['name'])
                mlflow.log_param("preprocessing", self.hydra_config['name'])
        except Exception as e:
            # Don't fail the trial if MLflow logging fails
            print(f"MLflow logging failed: {e}")
        
        return {"loss": rmse}

def create_search_space(model_config: Dict) -> Dict:
    """Convert model config to Ray Tune search space"""
    search_space = {}
    
    for param, config in model_config['search_space'].items():
        if config['type'] == "choice":
            search_space[param] = tune.choice(config['values'])
        elif config['type'] == "randint":
            search_space[param] = tune.randint(config['lower'], config['upper'])
        elif config['type'] == "uniform":
            search_space[param] = tune.uniform(config['lower'], config['upper'])
        elif config['type'] == "loguniform":
            search_space[param] = tune.loguniform(config['lower'], config['upper'])
    
    return search_space

def run_hyperparameter_search(config: Dict, X, y) -> tuple:
    """Run Ray Tune hyperparameter search"""
    
    # Initialize Ray with GPU support if needed
    ray_init_config = {"ignore_reinit_error": True}
    if 'device' in config.get('model', {}).get('params', {}) and 'cuda' in str(config['model']['params']['device']):
        ray_init_config['num_gpus'] = 1
    
    ray.init(**ray_init_config)
    
    try:
        # Create search space
        search_space = create_search_space(config['model'])
        search_space.update({'hydra_config': config, 'X': X, 'y': y})
        
        # Set up scheduler and search algorithm
        scheduler = None
        search_alg = None
        
        if config['search']['algorithm'] == "bohb":
            scheduler = BOHBScheduler(max_t=100, reduction_factor=2)
            search_alg = TuneBOHB()
        
        # Run search
        # Check if this is an XGBoost GPU model
        is_gpu_model = 'device' in config.get('model', {}).get('params', {}) and 'cuda' in str(config['model']['params']['device'])
        
        # Set resources based on model type
        if is_gpu_model:
            resources_per_trial = {"cpu": 1, "gpu": 0.25}  # Share GPU among trials
        else:
            resources_per_trial = {"cpu": 1}

        analysis = tune.run(
            RayTuneTrainable,
            config=search_space,
            scheduler=scheduler,
            search_alg=search_alg,
            num_samples=config['ray']['num_samples'],
            max_concurrent_trials=config['ray']['max_concurrent'],
            resources_per_trial=resources_per_trial,
            metric="loss",
            mode="min",
            name=f"{config['name']}_{config['model']['name']}_{config['search']['name']}",
            storage_path=os.path.abspath("./ray_results")
        )
        
        # Get best parameters
        best_trial = analysis.get_best_trial("loss", "min")
        best_params = {k: v for k, v in best_trial.config.items() 
                      if k not in ['hydra_config', 'X', 'y']}
        
        logger.info(f"Best RMSE: {best_trial.last_result['loss']:.4f}")
        
        return analysis, best_params
        
    finally:
        ray.shutdown()
