#!/usr/bin/env python3
"""Simple test to verify MLflow logging works with PostgreSQL"""

import os
import mlflow
import mlflow.sklearn
from sklearn.datasets import make_regression
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

def test_mlflow_logging():
    # Setup MLflow with PostgreSQL
    db_uri = 'postgresql://mlflow_user:mlflow_pass@localhost:5432/mlflow_db'
    mlflow.set_tracking_uri(db_uri)
    mlflow.set_experiment("test_experiment")
    
    # Generate simple test data
    X, y = make_regression(n_samples=100, n_features=5, noise=0.1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Start MLflow run
    with mlflow.start_run(run_name="simple_test"):
        # Log parameters
        alpha = 1.0
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("model_type", "Ridge")
        mlflow.log_param("n_features", X.shape[1])
        mlflow.log_param("n_samples", X.shape[0])
        
        # Train model
        model = Ridge(alpha=alpha, random_state=42)
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        # Log metrics
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("test_rmse", rmse)
        
        # Log model artifact
        mlflow.sklearn.log_model(model, "model")
        
        # Log a simple text artifact
        with open("test_results.txt", "w") as f:
            f.write(f"Test RMSE: {rmse:.4f}\n")
            f.write(f"Model: Ridge with alpha={alpha}\n")
        mlflow.log_artifact("test_results.txt")
        
        print(f"✅ MLflow logging completed successfully!")
        print(f"   - Parameters logged: alpha={alpha}, model_type=Ridge, n_features={X.shape[1]}, n_samples={X.shape[0]}")
        print(f"   - Metrics logged: rmse={rmse:.4f}, test_rmse={rmse:.4f}")
        print(f"   - Artifacts logged: model, test_results.txt")
        print(f"   - Run ID: {mlflow.active_run().info.run_id}")
        
    # Clean up
    os.remove("test_results.txt")

if __name__ == "__main__":
    test_mlflow_logging()