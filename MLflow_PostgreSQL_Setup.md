# MLflow PostgreSQL Setup Guide

This guide explains how to set up MLflow with PostgreSQL backend for distributed SLURM execution.

## Quick Setup

### Option A: PostgreSQL (Recommended for high concurrency)

1. **Clean up any existing setup (if needed):**
   ```bash
   bash cleanup_mlflow.sh
   ```

2. **Initialize PostgreSQL and MLflow configuration:**
   ```bash
   bash setup_mlflow_postgres.sh
   ```

3. **Install Python dependencies:**
   ```bash
   pip install psycopg2-binary
   # or update your conda environment
   conda install psycopg2
   ```

### Option B: SQLite (Simpler setup, moderate concurrency)

1. **Clean up any existing setup (if needed):**
   ```bash
   bash cleanup_mlflow.sh
   ```

2. **Initialize SQLite and MLflow configuration:**
   ```bash
   bash setup_mlflow_sqlite.sh
   ```

### Common Steps (for both options)

3. **Start MLflow server (in a separate terminal or SLURM job):**
   ```bash
   # Option A: Interactive session
   bash start_mlflow_server.sh
   
   # Option B: Submit as SLURM job
   sbatch mlflow_server_job.sh
   ```

4. **Run your Nextflow pipeline:**
   ```bash
   nextflow run main.nf --param-file params.yaml -profile slurm
   ```

## Architecture

- **PostgreSQL Database**: Stores experiment metadata (parameters, metrics, tags)
- **Shared File System**: Stores artifacts (models, plots, etc.) on /faststorage
- **MLflow Server**: Provides web UI and API access
- **SLURM Jobs**: Connect to PostgreSQL from any node

## Benefits

- **Concurrent Access**: Multiple SLURM jobs can write simultaneously
- **Data Integrity**: PostgreSQL handles concurrent transactions
- **Scalability**: Database can handle thousands of experiments
- **Reliability**: No file locking issues across nodes
- **Web UI**: Browse experiments from any node

## Troubleshooting

### Connection Issues
- Check if PostgreSQL is running: `pg_isready -h localhost -p 5432`
- Verify configuration: `source mlflow_config.env && echo $MLFLOW_TRACKING_URI`

### Performance Issues
- Increase PostgreSQL shared_buffers if handling many concurrent writes
- Consider connection pooling for very high concurrency

### Access from Compute Nodes
All SLURM compute nodes should be able to access the PostgreSQL database running on the head node via the cluster network.

## Configuration Files

- `mlflow_config.env`: Environment variables for database connection
- `setup_mlflow_postgres.sh`: One-time PostgreSQL database setup
- `setup_mlflow_sqlite.sh`: One-time SQLite database setup (simpler alternative)
- `cleanup_mlflow.sh`: Clean up and reset all MLflow data
- `start_mlflow_server.sh`: Start MLflow UI server
- `mlflow_server_job.sh`: SLURM job script for MLflow server
