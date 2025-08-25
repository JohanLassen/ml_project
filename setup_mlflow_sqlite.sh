#!/bin/bash

# Setup MLflow with SQLite backend - simpler alternative to PostgreSQL
# SQLite works well for moderate concurrency and is easier to set up

# MLflow configuration
ARTIFACT_ROOT="/faststorage/project/forensics_TOFscreenings/01_aging_study/modeling/ml_project/mlflow_artifacts"
SQLITE_DB="/faststorage/project/forensics_TOFscreenings/01_aging_study/modeling/ml_project/mlflow.db"
MLFLOW_PORT="5000"

echo "=== Setting up MLflow with SQLite backend ==="

# Create artifact directory
mkdir -p $ARTIFACT_ROOT
echo "Created artifact directory: $ARTIFACT_ROOT"

# Create database directory
DB_DIR=$(dirname $SQLITE_DB)
mkdir -p $DB_DIR

# SQLite connection string
SQLITE_CONNECTION_STRING="sqlite:///$SQLITE_DB"

echo "=== Configuration complete ==="
echo "Database file: $SQLITE_DB"
echo "Artifact root: $ARTIFACT_ROOT"

# Save configuration for later use
cat > mlflow_config.env << EOF
# MLflow SQLite Configuration
export MLFLOW_TRACKING_URI="$SQLITE_CONNECTION_STRING"
export MLFLOW_DEFAULT_ARTIFACT_ROOT="$ARTIFACT_ROOT"
export MLFLOW_SERVER_HOST="0.0.0.0"
export MLFLOW_SERVER_PORT="$MLFLOW_PORT"

# Database details (for reference)
export SQLITE_DB="$SQLITE_DB"
EOF

echo "Configuration saved to mlflow_config.env"
echo ""
echo "=== Next steps ==="
echo "1. Source the configuration: source mlflow_config.env"
echo "2. Start MLflow server: bash start_mlflow_server.sh"
echo "3. In your SLURM jobs, set: export MLFLOW_TRACKING_URI=\"$SQLITE_CONNECTION_STRING\""
echo ""
echo "Note: SQLite has some limitations with high concurrency."
echo "If you experience database locks, consider using PostgreSQL instead."
