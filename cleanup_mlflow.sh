#!/bin/bash

# Cleanup and reset MLflow database
# Use this script when you need to start fresh

echo "=== MLflow Database Cleanup ==="

# Load configuration if it exists
if [ -f "mlflow_config.env" ]; then
    source mlflow_config.env
    echo "Loaded existing configuration"
else
    echo "No mlflow_config.env found. Nothing to clean up."
    exit 0
fi

# Stop MLflow server if running
echo "Stopping any running MLflow processes..."
pkill -f "mlflow server" || echo "No MLflow server processes found"

# Stop PostgreSQL if running
echo "Stopping PostgreSQL..."
pg_ctl -D $HOME/postgres_data stop || echo "PostgreSQL not running or already stopped"

# Remove existing database data
if [ -d "$HOME/postgres_data" ]; then
    echo "Removing PostgreSQL data directory..."
    rm -rf $HOME/postgres_data
fi

# Remove MLflow artifacts
if [ -d "/faststorage/project/forensics_TOFscreenings/01_aging_study/modeling/ml_project/mlflow_artifacts" ]; then
    echo "Removing MLflow artifacts..."
    rm -rf /faststorage/project/forensics_TOFscreenings/01_aging_study/modeling/ml_project/mlflow_artifacts
fi

# Remove old mlruns directory if it exists
if [ -d "mlruns" ]; then
    echo "Removing old mlruns directory..."
    rm -rf mlruns
fi

# Remove configuration file
if [ -f "mlflow_config.env" ]; then
    echo "Removing old configuration..."
    rm mlflow_config.env
fi

# Clean up log files
rm -f $HOME/postgres.log
rm -f mlflow_server_*.out mlflow_server_*.err

echo "=== Cleanup complete ==="
echo "You can now run: bash setup_mlflow_postgres.sh"
