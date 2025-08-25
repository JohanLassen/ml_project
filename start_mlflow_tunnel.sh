#!/bin/bash
# start_mlflow_tunnel.sh - Easy way to start MLflow with SSH tunnel

echo "🚀 Starting MLflow Server on GenomeDK cluster..."
echo ""

# Update username if needed
USER="jklassen"
echo "👤 Using username: $USER"
echo "📁 Project path: /home/$USER/forensics_TOFscreenings/01_aging_study/modeling/ml_project"
echo ""

# Start the tunnel
./mlflow_cluster.sh -u $USER

echo ""
echo "ℹ️  To stop the MLflow server later:"
echo "   1. Cancel the SLURM job: ssh $USER@login.genome.au.dk 'scancel -n mlflow_server'"
echo "   2. Kill local tunnel: lsof -ti:5000 | xargs kill -9"
