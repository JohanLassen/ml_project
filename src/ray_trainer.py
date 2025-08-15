import ray
from ray import tune
from ray.tune.schedulers import BOHBScheduler
from ray.tune.search.bohb import TuneBOHB
import mlflow
from models import ModelTrainer
from typing import Dict
import logging

logger = logging.getLogger(__name__)

class RayTuneTrainable(tune.Trainable):
    def setup(self, config: Dict):
        self.tune_config = config
        self.hydra_config = config['hydra_config']
        self.X = config['X']
        self.y = config['y']
        self.trainer = ModelTrainer(self.hydra_config['cv'])
    
    def step(self) -> Dict[str, float]:
        # Extract hyperparameters (remove non-param keys)
        hyperparams = {k: v for k, v in self.tune_config.items() 
                      if k not in ['hydra_config', 'X', 'y']}
        
        # Evaluate model
        rmse = self.trainer.evaluate_model(
            self.hydra_config['model'],
            hyperparams,
            self.X,
            self.y
        )
        
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
    
    ray.init(ignore_reinit_error=True)
    
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
        analysis = tune.run(
            RayTuneTrainable,
            config=search_space,
            scheduler=scheduler,
            search_alg=search_alg,
            num_samples=config['ray']['num_samples'],
            max_concurrent_trials=config['ray']['max_concurrent'],
            resources_per_trial={"cpu": 1},
            metric="loss",
            mode="min",
            name=f"{config['preprocessing']['name']}_{config['model']['name']}_{config['search']['name']}",
            local_dir="./ray_results"
        )
        
        # Get best parameters
        best_trial = analysis.get_best_trial("loss", "min")
        best_params = {k: v for k, v in best_trial.config.items() 
                      if k not in ['hydra_config', 'X', 'y']}
        
        logger.info(f"Best RMSE: {best_trial.last_result['loss']:.4f}")
        
        return analysis, best_params
        
    finally:
        ray.shutdown()
