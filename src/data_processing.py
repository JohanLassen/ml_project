import importlib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_regression
from sklearn.pipeline import Pipeline
from typing import Dict, Tuple
from omegaconf import OmegaConf

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
    
    def load_data_from_file(self, file_path: str) -> Tuple[pd.DataFrame, pd.Series]:
        """Load data from specific file path with dynamic target mapping"""
        if file_path.endswith('.parquet'):
            df = pd.read_parquet(file_path)
        else:
            df = pd.read_csv(file_path)
        
        print(df)
        # Get data config
        feature_prefix = self.config.data.feature_prefix
        print(feature_prefix)
        exclude_columns = self.config.data.exclude_columns
        print(exclude_columns)
        target_column = self.config.data.target_column
        target_name = self.config.data.target_name
        
        # Extract features and target
        if feature_prefix:
            # Use prefix matching (original behavior)
            feature_cols = [col for col in df.columns if col.startswith(feature_prefix)]
        else:
            # Use exclusion list (new behavior for numeric features)
            feature_cols = [col for col in df.columns if col not in exclude_columns]
        
        X = df[feature_cols]
        print(f"Selected {len(feature_cols)} feature columns")
        
        # Use the configured target column
        if target_column in df.columns:
            y = df[target_column]
        elif target_name in df.columns:
            y = df[target_name]  # Fallback if already renamed
        else:
            raise ValueError(f"Target column '{target_column}' not found in data")
        
        return X, y
    
    def subset_data_for_testing(self, X: pd.DataFrame, y: pd.Series, max_features: int = 500, max_samples: int = 50) -> Tuple[pd.DataFrame, pd.Series]:
        """Subset data for faster testing"""
        # Sample features if we have too many
        if X.shape[1] > max_features:
            # Select features randomly but deterministically
            np.random.seed(42)
            selected_features = np.random.choice(X.columns, size=max_features, replace=False)
            X = X[selected_features]
            print(f"Subsetted features from {X.shape[1]} to {len(selected_features)}")
        
        # Sample rows if we have too many
        if X.shape[0] > max_samples:
            # Sample rows randomly but deterministically
            np.random.seed(42)
            selected_indices = np.random.choice(X.index, size=max_samples, replace=False)
            X = X.loc[selected_indices].reset_index(drop=True)
            y = y.loc[selected_indices].reset_index(drop=True)
            print(f"Subsetted samples from {X.shape[0]} to {max_samples}")
        
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
                if step_config['method'] == 'standard':
                    scaler = StandardScaler()
                elif step_config['method'] == 'robust':
                    scaler = RobustScaler()
                elif step_config['method'] == 'minmax':
                    scaler = MinMaxScaler()
                else:
                    scaler = StandardScaler()  # Default fallback
                steps.append(('scaler', scaler))
        

    
    def build_pipeline_from_yaml(self, steps_config):
        steps = []
        for step_name, step_info in steps_config.items():
            module = importlib.import_module(step_info['module'])
            cls = getattr(module, step_info['class'])
            steps.append((step_name, cls(**step_info.get('params', {}))))
        self.pipeline = Pipeline(steps)
        return self.pipeline
        
    def fit_transform(self, X: pd.DataFrame, y: pd.Series = None) -> np.ndarray:
        """Fit and transform data"""
        return self.pipeline.fit_transform(X, y)
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform data using fitted pipeline"""
        return self.pipeline.transform(X)
