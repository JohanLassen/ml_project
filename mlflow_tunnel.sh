#!/bin/bash
# mlflow_tunnel.sh - Simple MLflow UI tunneling script

set -e  # Exit on any error

# Configuration - MODIFY THESE FOR YOUR SETUP
CLUSTER_HOST="your-cluster.university.edu"
CLUSTER_USER="your_username"
MLFLOW_PORT=5000
LOCAL_PORT=5000
PROJECT_DIR="/path/to/your/ml_project"  # Remote project directory

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to find available local port
find_available_port() {
    local port=$1
    while lsof -i:$port >/dev/null 2>&1; do
        print_warning "Port $port is busy, trying $((port+1))"
        port=$((port+1))
    done
    echo $port
}

# Function to create SSH tunnel
create_ssh_tunnel() {
    local remote_port=$1
    local local_port=$2
    
    print_status "Creating SSH tunnel: localhost:$local_port -> $CLUSTER_HOST:$remote_port"
    
    # Kill existing tunnel if it exists
    pkill -f "ssh.*$local_port:localhost:$remote_port.*$CLUSTER_USER@$CLUSTER_HOST" 2>/dev/null || true
    
    # Create new tunnel in background
    ssh -f -N -L $local_port:localhost:$remote_port "$CLUSTER_USER@$CLUSTER_HOST"
    
    # Wait a moment for tunnel to establish
    sleep 2
    
    # Check if tunnel is working
    if lsof -i:$local_port >/dev/null 2>&1; then
        print_success "SSH tunnel established on port $local_port"
        return 0
    else
        print_error "Failed to establish SSH tunnel"
        return 1
    fi
}

# Function to start MLflow server on cluster
start_mlflow_server() {
    local port=$1
    
    print_status "Starting MLflow server on cluster..."
    
    ssh "$CLUSTER_USER@$CLUSTER_HOST" << EOF &
        set -e
        cd $PROJECT_DIR
        
        # Activate conda environment
        source ~/.bashrc
        conda activate ml-gpu-project
        
        # Kill existing MLflow server if running
        pkill -f "mlflow ui" 2>/dev/null || true
        sleep 2
        
        # Start MLflow server with PostgreSQL backend
        echo "Starting MLflow UI on port $port with PostgreSQL backend..."
        mlflow ui \
            --host 0.0.0.0 \
            --port $port \
            --backend-store-uri postgresql://mlflow_user:mlflow_pass@localhost:5432/mlflow_db
EOF
    
    # Store the SSH process ID for cleanup
    MLFLOW_SSH_PID=$!
    
    # Wait for MLflow to start
    print_status "Waiting for MLflow server to start..."
    sleep 10
    
    # Check if MLflow is accessible through tunnel
    local max_attempts=12
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s "http://localhost:$LOCAL_PORT" >/dev/null 2>&1; then
            print_success "MLflow server is running and accessible"
            return 0
        fi
        
        print_status "Attempt $attempt/$max_attempts: Waiting for MLflow..."
        sleep 5
        attempt=$((attempt+1))
    done
    
    print_error "MLflow server did not start properly"
    return 1
}

# Function to open browser
open_browser() {
    local url="http://localhost:$LOCAL_PORT"
    
    print_status "Opening MLflow UI in browser: $url"
    
    # Detect OS and open browser accordingly
    if command_exists xdg-open; then
        # Linux
        xdg-open "$url" >/dev/null 2>&1 &
    elif command_exists open; then
        # macOS
        open "$url"
    elif command_exists start; then
        # Windows (Git Bash/WSL)
        start "$url"
    else
        print_warning "Could not detect browser opener. Please manually open: $url"
        return 1
    fi
    
    print_success "Browser opened (or attempted to open)"
}

# Function to cleanup
cleanup() {
    print_status "Cleaning up..."
    
    # Kill SSH tunnel
    pkill -f "ssh.*$LOCAL_PORT:localhost:$MLFLOW_PORT.*$CLUSTER_USER@$CLUSTER_HOST" 2>/dev/null || true
    
    # Kill MLflow SSH session
    if [ ! -z "$MLFLOW_SSH_PID" ]; then
        kill $MLFLOW_SSH_PID 2>/dev/null || true
    fi
    
    print_success "Cleanup completed"
}

# Function to show cluster commands
show_cluster_commands() {
    cat << EOF

=== Commands to Run on Cluster ===

1. SSH to your cluster:
   ssh $CLUSTER_USER@$CLUSTER_HOST

2. Navigate to project directory:
   cd $PROJECT_DIR

3. Activate conda environment:
   conda activate ml-gpu-project

4. Run experiments directly:
   # Linear model test
   python src/main.py --preprocessing basic --model linear --search random \\
     --data data/processed/v1.0.0/train_data.parquet --data_version 1.0 --cpus 4

   # XGBoost GPU experiment
   python src/main.py --preprocessing kbest_standard --model xgboost --search bayesian \\
     --data data/processed/v1.0.0/train_data.parquet --data_version 1.0 --cpus 8

   # Full Nextflow pipeline
   nextflow run main.nf --run_mode hpc

5. Monitor results in MLflow UI:
   http://localhost:$LOCAL_PORT

=====================================

EOF
}

# Main execution
main() {
    echo "=============================================="
    echo "  MLflow UI Tunnel Setup"
    echo "=============================================="
    echo ""
    echo "Configuration:"
    echo "  Cluster: $CLUSTER_USER@$CLUSTER_HOST"
    echo "  Project Dir: $PROJECT_DIR"
    echo "  MLflow: $CLUSTER_HOST:$MLFLOW_PORT -> localhost:$LOCAL_PORT"
    echo ""
    
    # Set up signal handler for cleanup
    trap cleanup EXIT INT TERM
    
    # Check prerequisites
    print_status "Checking prerequisites..."
    
    if ! command_exists ssh; then
        print_error "SSH is not installed"
        exit 1
    fi
    
    if ! command_exists curl; then
        print_error "curl is not installed (needed for health checks)"
        exit 1
    fi
    
    # Test SSH connection
    print_status "Testing SSH connection to $CLUSTER_HOST..."
    if ! ssh -o ConnectTimeout=10 -o BatchMode=yes "$CLUSTER_USER@$CLUSTER_HOST" exit 2>/dev/null; then
        print_error "SSH connection failed. Please check your SSH keys and network."
        exit 1
    fi
    print_success "SSH connection successful"
    
    # Find available local port
    LOCAL_PORT=$(find_available_port $LOCAL_PORT)
    
    # Create SSH tunnel
    if ! create_ssh_tunnel $MLFLOW_PORT $LOCAL_PORT; then
        exit 1
    fi
    
    # Start MLflow server
    if ! start_mlflow_server $MLFLOW_PORT; then
        exit 1
    fi
    
    # Open browser
    open_browser
    
    # Show cluster commands
    show_cluster_commands
    
    # Keep script running
    print_success "Setup complete! MLflow UI is available at http://localhost:$LOCAL_PORT"
    print_status "Use the commands above to run experiments on the cluster"
    print_status "Press Ctrl+C to stop the tunnel and cleanup"
    
    # Wait indefinitely (until Ctrl+C)
    while true; do
        sleep 60
        # Check if tunnel is still alive
        if ! lsof -i:$LOCAL_PORT >/dev/null 2>&1; then
            print_warning "SSH tunnel appears to have died, attempting to reconnect..."
            if ! create_ssh_tunnel $MLFLOW_PORT $LOCAL_PORT; then
                print_error "Failed to reconnect tunnel"
                exit 1
            fi
        fi
    done
}

# Show usage if help requested
if [[ $# -gt 0 ]] && [[ "$1" == "-h" || "$1" == "--help" ]]; then
    cat << EOF
Usage: $0

MLflow UI Tunnel Setup

This script will:
1. Test SSH connection to cluster
2. Create SSH tunnel for MLflow UI
3. Start MLflow server on cluster (with PostgreSQL)
4. Open MLflow UI in local browser
5. Show commands for running experiments on cluster

Configuration (edit script variables):
  CLUSTER_HOST     - Your cluster hostname
  CLUSTER_USER     - Your cluster username  
  PROJECT_DIR      - Remote project directory
  MLFLOW_PORT      - MLflow port on cluster (default: 5000)
  LOCAL_PORT       - Local port for tunnel (default: 5000)

Example workflow:
1. Run this script: $0
2. SSH to cluster and run experiments manually
3. View results in local browser at http://localhost:5000

EOF
    exit 0
fi

# Run main function
main "$@"