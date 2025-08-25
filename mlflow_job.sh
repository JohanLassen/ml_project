#!/bin/bash
#SBATCH --account=forensics_TOFscreenings
#SBATCH --job-name=mlflow_server
#SBATCH --time=4:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --output=mlflow_server_%j.out
#SBATCH --error=mlflow_server_%j.err

# Load conda environment
source ~/.bashrc
conda activate ml-gpu-project

# Navigate to project directory to ensure consistent path resolution
cd /home/jklassen/forensics_TOFscreenings/01_aging_study/modeling/ml_project

# Set MLflow tracking URI environment variable for consistency
export MLFLOW_TRACKING_URI="$(readlink -f ./mlruns)"

# Start MLflow server using environment variable
mlflow server \
    --backend-store-uri "file://${MLFLOW_TRACKING_URI}" \
    --default-artifact-root "${MLFLOW_TRACKING_URI}" \
    --host 0.0.0.0 \
    --port 5000 \
    --serve-artifacts
