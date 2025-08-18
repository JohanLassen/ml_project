import numpy as np
from sklearn.model_selection import cross_val_score, KFold, train_test_split
import importlib
from typing import Dict
import xgboost as xgb

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
        """Evaluate model with cross-validation and early stopping for XGBoost"""
        model = ModelFactory.create_model(model_config, hyperparams)
        
        # Use early stopping for XGBoost models
        if 'XGB' in model_config['class_path']:
            return self._evaluate_xgboost_with_early_stopping(model, X, y)
        else:
            # Standard cross-validation for other models
            cv_scores = cross_val_score(
                model, X, y, 
                cv=self.cv_splitter,
                scoring='neg_mean_squared_error',
                n_jobs=-1
            )
            return float(np.sqrt(-cv_scores.mean()))  # Return RMSE
    
    def _evaluate_xgboost_with_early_stopping(self, model, X: np.ndarray, y: np.ndarray) -> float:
        """Evaluate XGBoost with early stopping using validation split"""
        scores = []
        
        for train_idx, test_idx in self.cv_splitter.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Create 10% validation split from training data
            X_train_fit, X_val, y_train_fit, y_val = train_test_split(
                X_train, y_train, test_size=0.1, random_state=42
            )
            
            # Fit with early stopping
            model.fit(
                X_train_fit, y_train_fit,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
            
            # Predict on test set
            y_pred = model.predict(X_test)
            rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
            scores.append(rmse)
        
        return float(np.mean(scores))
    
    def train_final_model(self, model_config: Dict, best_params: Dict, X: np.ndarray, y: np.ndarray):
        """Train final model with best parameters and early stopping for XGBoost"""
        model = ModelFactory.create_model(model_config, best_params)
        
        # Use early stopping for XGBoost models
        if 'XGB' in model_config['class_path']:
            # Create validation split
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.1, random_state=42
            )
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
        else:
            model.fit(X, y)
            
        return model
