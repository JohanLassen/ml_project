import numpy as np
from sklearn.model_selection import cross_val_score, KFold
import importlib
from typing import Dict

class ModelFactory:
    @staticmethod
    def create_model(model_config: Dict, hyperparams: Dict = None):
        """Create model from config and hyperparameters"""
        module_path, class_name = model_config['class_path'].rsplit('.', 1)
        module = importlib.import_module(module_path)
        model_class = getattr(module, class_name)
        
        params = model_config['params'].copy()
        if hyperparams:
            params.update(hyperparams)
            
        return model_class(**params)

class ModelTrainer:
    def __init__(self, cv_config: Dict):
        self.cv_splitter = KFold(
            n_splits=cv_config['n_splits'],
            shuffle=True,
            random_state=cv_config['random_state']
        )
    
    def evaluate_model(self, model_config: Dict, hyperparams: Dict, X: np.ndarray, y: np.ndarray) -> float:
        """Evaluate model with cross-validation"""
        model = ModelFactory.create_model(model_config, hyperparams)
        
        cv_scores = cross_val_score(
            model, X, y, 
            cv=self.cv_splitter,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        
        return float(np.sqrt(-cv_scores.mean()))  # Return RMSE
    
    def train_final_model(self, model_config: Dict, best_params: Dict, X: np.ndarray, y: np.ndarray):
        """Train final model with best parameters"""
        model = ModelFactory.create_model(model_config, best_params)
        model.fit(X, y)
        return model
