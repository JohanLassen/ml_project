#!/bin/bash

# Start MLflow server with PostgreSQL backend
# Run this script to start the MLflow UI server

# Load configuration
if [ -f "mlflow_config.env" ]; then
    source mlflow_config.env
    echo "Loaded MLflow configuration"
else
    echo "Error: mlflow_config.env not found. Run setup_mlflow_postgres.sh first."
    exit 1
fi

# Check if PostgreSQL is running
if ! pg_isready -h $DB_HOST -p $DB_PORT -U $DB_USER; then
    echo "Starting PostgreSQL server..."
    pg_ctl -D $HOME/postgres_data -l $HOME/postgres.log start
    sleep 3
fi

echo "=== Starting MLflow Server ==="
echo "Tracking URI: $MLFLOW_TRACKING_URI"
echo "Artifact Root: $MLFLOW_DEFAULT_ARTIFACT_ROOT"
echo "Server will be available at: http://$(hostname):$MLFLOW_SERVER_PORT"
echo ""

# Start MLflow server
mlflow server \
    --backend-store-uri "$MLFLOW_TRACKING_URI" \
    --default-artifact-root "$MLFLOW_DEFAULT_ARTIFACT_ROOT" \
    --host "$MLFLOW_SERVER_HOST" \
    --port "$MLFLOW_SERVER_PORT" \
    --serve-artifacts
