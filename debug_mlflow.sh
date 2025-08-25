#!/bin/bash
# debug_mlflow.sh - Debug MLflow setup

echo "🔍 MLflow Debugging Script"
echo "=========================="
echo ""

# Check if we're in the right directory
echo "📁 Current directory: $(pwd)"
echo "📁 Expected: /home/jklassen/forensics_TOFscreenings/01_aging_study/modeling/ml_project"
echo ""

# Check if mlruns directory exists and what's in it
echo "📂 MLruns directory contents:"
if [ -d "mlruns" ]; then
    echo "✅ mlruns directory exists"
    ls -la mlruns/
    echo ""
    echo "📊 Experiments in mlruns:"
    find mlruns/ -name "meta.yaml" -exec dirname {} \; | sort
    echo ""
    echo "🏃 Runs found:"
    find mlruns/ -type d -name "*" | grep -E "/[0-9a-f]{32}$" | wc -l
else
    echo "❌ mlruns directory does not exist"
fi
echo ""

# Check if any ML jobs have actually run
echo "📈 Recent result files:"
find results/ -name "*.json" -mtime -1 2>/dev/null | head -5
echo ""

# Check MLflow server status
echo "🖥️  MLflow server status:"
if pgrep -f "mlflow server" > /dev/null; then
    echo "✅ MLflow server is running"
    echo "📊 Server process:"
    pgrep -f "mlflow server" -l
else
    echo "❌ MLflow server is not running"
fi
echo ""

# Check SLURM jobs
echo "🎯 Current SLURM jobs:"
squeue -u $USER
echo ""

# Test MLflow connectivity
echo "🔗 Testing MLflow setup:"
python3 -c "
import mlflow
import os
print(f'MLflow tracking URI: {mlflow.get_tracking_uri()}')
print(f'Current working directory: {os.getcwd()}')
try:
    experiments = mlflow.search_experiments()
    print(f'Found {len(experiments)} experiments:')
    for exp in experiments:
        print(f'  - {exp.name} (ID: {exp.experiment_id})')
except Exception as e:
    print(f'Error connecting to MLflow: {e}')
"
