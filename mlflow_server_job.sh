#!/bin/bash
#SBATCH --job-name=mlflow-server
#SBATCH --account=forensics_TOFscreenings
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --output=mlflow_server_%j.out
#SBATCH --error=mlflow_server_%j.err

echo "Starting MLflow server on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Time: $(date)"

# Load environment
source mlflow_config.env

# Activate conda environment
conda activate ml-hpc-project

# Start MLflow server
echo "Starting MLflow server..."
bash start_mlflow_server.sh
