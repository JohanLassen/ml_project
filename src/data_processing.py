import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_regression
from sklearn.pipeline import Pipeline
from typing import Dict, Tuple

class DataProcessor:
    def __init__(self, config: Dict):
        self.config = config
        self.pipeline = None
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Load data and extract M features"""
        df = pd.read_csv(self.config['data']['path'])
        
        # Extract M features and target
        m_features = [col for col in df.columns if col.startswith(self.config['data']['feature_prefix'])]
        X = df[m_features]
        y = df[self.config['data']['target_col']]
        
        return X, y
    
    def build_pipeline(self, preprocessing_config: Dict) -> Pipeline:
        """Build preprocessing pipeline from config"""
        steps = []
        
        for step_name, step_config in preprocessing_config['steps'].items():
            if step_name == 'imputer':
                steps.append(('imputer', SimpleImputer(strategy=step_config['strategy'])))
            elif step_name == 'variance_selector':
                steps.append(('variance_selector', VarianceThreshold(threshold=step_config['threshold'])))
            elif step_name == 'k_best_selector':
                steps.append(('k_best_selector', SelectKBest(score_func=f_regression, k=step_config['k'])))
            elif step_name == 'scaler':
                scaler = StandardScaler() if step_config['method'] == 'standard' else RobustScaler()
                steps.append(('scaler', scaler))
        
        self.pipeline = Pipeline(steps)
        return self.pipeline
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series = None) -> np.ndarray:
        """Fit and transform data"""
        return self.pipeline.fit_transform(X, y)
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform data using fitted pipeline"""
        return self.pipeline.transform(X)
