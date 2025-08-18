#!/bin/bash
# setup_mlflow_db.sh - PostgreSQL setup for MLflow tracking

set -e

echo "Setting up PostgreSQL for MLflow tracking..."

# Database configuration
DB_NAME="mlflow_db"
DB_USER="mlflow_user"
DB_PASSWORD="mlflow_pass"

# Initialize PostgreSQL if not already initialized
if [ ! -d "$CONDA_PREFIX/var/postgres" ]; then
    echo "Initializing PostgreSQL database..."
    initdb -D "$CONDA_PREFIX/var/postgres"
fi

# Start PostgreSQL server
echo "Starting PostgreSQL server..."
pg_ctl -D "$CONDA_PREFIX/var/postgres" -l "$CONDA_PREFIX/var/postgres/server.log" start

# Wait for server to start
sleep 3

# Create database and user
echo "Creating MLflow database and user..."
createdb $DB_NAME || echo "Database $DB_NAME already exists"

# Create user with password (if doesn't exist)
psql -d postgres -c "CREATE USER $DB_USER WITH PASSWORD '$DB_PASSWORD';" || echo "User $DB_USER already exists"
psql -d postgres -c "GRANT ALL PRIVILEGES ON DATABASE $DB_NAME TO $DB_USER;"

# Set environment variable
export MLFLOW_TRACKING_URI="postgresql://$DB_USER:$DB_PASSWORD@localhost:5432/$DB_NAME"

echo "PostgreSQL setup complete!"
echo "MLflow tracking URI: $MLFLOW_TRACKING_URI"
echo ""
echo "To start MLflow UI, run:"
echo "  mlflow ui --backend-store-uri postgresql://$DB_USER:$DB_PASSWORD@localhost:5432/$DB_NAME"
echo ""
echo "To stop PostgreSQL server:"
echo "  pg_ctl -D \$CONDA_PREFIX/var/postgres stop"