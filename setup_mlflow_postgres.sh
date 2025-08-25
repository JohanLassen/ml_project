#!/bin/bash

# Setup MLflow with PostgreSQL backend for SLURM cluster
# This script should be run once to initialize the database

# Database configuration
DB_NAME="mlflow_db"
DB_USER="mlflow_user"
DB_PASSWORD="mlflow_password_$(date +%s)"  # Generate unique password
DB_HOST="localhost"
DB_PORT="5432"

# MLflow server configuration
MLFLOW_PORT="5000"
ARTIFACT_ROOT="/faststorage/project/forensics_TOFscreenings/01_aging_study/modeling/ml_project/mlflow_artifacts"

echo "=== Setting up MLflow with PostgreSQL backend ==="

# Create artifact directory
mkdir -p $ARTIFACT_ROOT
echo "Created artifact directory: $ARTIFACT_ROOT"

# Load PostgreSQL module (adjust for your cluster)
module load PostgreSQL/13.7-GCCcore-11.3.0 || echo "PostgreSQL module not available, assuming system install"

# Initialize PostgreSQL database if needed
if [ ! -d "$HOME/postgres_data" ]; then
    echo "Initializing PostgreSQL database..."
    initdb -D $HOME/postgres_data
fi

# Start PostgreSQL server
echo "Starting PostgreSQL server..."
pg_ctl -D $HOME/postgres_data -l $HOME/postgres.log start

# Wait for server to start
sleep 5

# Create database and user (handle existing database gracefully)
echo "Setting up database and user..."

# Check if database exists, create if not
if psql -lqt | cut -d \| -f 1 | grep -qw $DB_NAME; then
    echo "Database $DB_NAME already exists, skipping creation"
else
    echo "Creating database $DB_NAME..."
    createdb $DB_NAME
fi

# Check if user exists, create if not
if psql -d $DB_NAME -tAc "SELECT 1 FROM pg_roles WHERE rolname='$DB_USER'" | grep -q 1; then
    echo "User $DB_USER already exists, updating permissions..."
else
    echo "Creating user $DB_USER..."
    psql -d $DB_NAME -c "CREATE USER $DB_USER WITH ENCRYPTED PASSWORD '$DB_PASSWORD';"
fi

# Always update permissions (idempotent operations)
psql -d $DB_NAME -c "GRANT ALL PRIVILEGES ON DATABASE $DB_NAME TO $DB_USER;"
psql -d $DB_NAME -c "GRANT ALL ON SCHEMA public TO $DB_USER;"
psql -d $DB_NAME -c "GRANT CREATE ON SCHEMA public TO $DB_USER;"
psql -d $DB_NAME -c "ALTER USER $DB_USER CREATEDB;"

# Create connection string
DB_CONNECTION_STRING="postgresql://$DB_USER:$DB_PASSWORD@$DB_HOST:$DB_PORT/$DB_NAME"

echo "=== Database setup complete ==="
echo "Database connection string: $DB_CONNECTION_STRING"
echo "Artifact root: $ARTIFACT_ROOT"

# Save configuration for later use
cat > mlflow_config.env << EOF
# MLflow PostgreSQL Configuration
export MLFLOW_TRACKING_URI="$DB_CONNECTION_STRING"
export MLFLOW_DEFAULT_ARTIFACT_ROOT="$ARTIFACT_ROOT"
export MLFLOW_SERVER_HOST="0.0.0.0"
export MLFLOW_SERVER_PORT="$MLFLOW_PORT"

# Database details (for reference)
export DB_NAME="$DB_NAME"
export DB_USER="$DB_USER"
export DB_PASSWORD="$DB_PASSWORD"
export DB_HOST="$DB_HOST"
export DB_PORT="$DB_PORT"
EOF

echo "Configuration saved to mlflow_config.env"
echo ""
echo "=== Next steps ==="
echo "1. Source the configuration: source mlflow_config.env"
echo "2. Install required Python packages: pip install mlflow psycopg2-binary"
echo "3. Start MLflow server: bash start_mlflow_server.sh"
echo "4. In your SLURM jobs, set: export MLFLOW_TRACKING_URI=\"$DB_CONNECTION_STRING\""
