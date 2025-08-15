#!/bin/bash
# ml_workflow_launcher.sh - Complete ML workflow with SSH tunneling and MLflow UI

set -e  # Exit on any error

# Configuration - MODIFY THESE FOR YOUR SETUP
CLUSTER_HOST="your-cluster.university.edu"
CLUSTER_USER="your_username"
DATA_VERSION="1.0.0"
PROFILE="slurm"  # or "local" for testing
MLFLOW_PORT=5000
LOCAL_PORT=5000
PROJECT_DIR="/path/to/your/ml_project"  # Remote project directory

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
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

# Function to check SSH connection
check_ssh_connection() {
    print_status "Testing SSH connection to $CLUSTER_HOST..."
    if ssh -o ConnectTimeout=10 -o BatchMode=yes "$CLUSTER_USER@$CLUSTER_HOST" exit 2>/dev/null; then
        print_success "SSH connection successful"
        return 0
    else
        print_error "SSH connection failed. Please check your SSH keys and network."
        return 1
    fi
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

# Function to submit workflow
submit_workflow() {
    print_status "Submitting Nextflow workflow on cluster..."
    
    ssh "$CLUSTER_USER@$CLUSTER_HOST" << EOF
        set -e
        cd $PROJECT_DIR
        
        # Activate conda environment
        source ~/.bashrc
        conda activate ml-gpu-project
        
        # Submit workflow
        echo "Submitting workflow with data version $DATA_VERSION..."
        nextflow run main.nf -profile $PROFILE --data_version "$DATA_VERSION" -bg
        
        # Check if submission was successful
        if [ \$? -eq 0 ]; then
            echo "Workflow submitted successfully!"
            nextflow log | tail -5
        else
            echo "Workflow submission failed!"
            exit 1
        fi
EOF
    
    if [ $? -eq 0 ]; then
        print_success "Workflow submitted successfully"
        return 0
    else
        print_error "Workflow submission failed"
        return 1
    fi
}

# Function to start MLflow server
start_mlflow_server() {
    local port=$1
    
    print_status "Starting MLflow server on cluster port $port..."
    
    ssh "$CLUSTER_USER@$CLUSTER_HOST" << EOF &
        set -e
        cd $PROJECT_DIR
        
        # Activate conda environment
        source ~/.bashrc
        conda activate ml-gpu-project
        
        # Kill existing MLflow server if running
        pkill -f "mlflow ui" 2>/dev/null || true
        sleep 2
        
        # Start MLflow server
        echo "Starting MLflow UI on port $port..."
        mlflow ui --host 0.0.0.0 --port $port --backend-store-uri sqlite:///mlflow.db
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

# Function to monitor workflow
monitor_workflow() {
    print_status "Monitoring workflow progress..."
    
    cat << EOF

=== Workflow Monitoring Commands ===
Run these commands in a separate terminal to monitor progress:

# SSH to cluster
ssh $CLUSTER_USER@$CLUSTER_HOST

# Check Nextflow logs
cd $PROJECT_DIR && nextflow log

# Check SLURM queue
squeue -u $CLUSTER_USER

# Check specific job details
scontrol show job <job_id>

# View real-time logs
tail -f work/*/*/.command.log

===================================

EOF
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

# Main execution
main() {
    echo "=================================================="
    echo "  ML Workflow Launcher with SSH Tunneling"
    echo "=================================================="
    echo ""
    echo "Configuration:"
    echo "  Cluster: $CLUSTER_USER@$CLUSTER_HOST"
    echo "  Data Version: $DATA_VERSION"
    echo "  Profile: $PROFILE"
    echo "  Project Dir: $PROJECT_DIR"
    echo "  MLflow Port: $MLFLOW_PORT -> localhost:$LOCAL_PORT"
    echo ""
    
    # Set up signal handler for cleanup
    trap cleanup EXIT INT TERM
    
    # Step 1: Check prerequisites
    print_status "Checking prerequisites..."
    
    if ! command_exists ssh; then
        print_error "SSH is not installed"
        exit 1
    fi
    
    if ! command_exists curl; then
        print_error "curl is not installed (needed for health checks)"
        exit 1
    fi
    
    # Step 2: Test SSH connection
    if ! check_ssh_connection; then
        exit 1
    fi
    
    # Step 3: Find available local port
    LOCAL_PORT=$(find_available_port $LOCAL_PORT)
    
    # Step 4: Submit workflow
    if ! submit_workflow; then
        exit 1
    fi
    
    # Step 5: Create SSH tunnel
    if ! create_ssh_tunnel $MLFLOW_PORT $LOCAL_PORT; then
        exit 1
    fi
    
    # Step 6: Start MLflow server
    if ! start_mlflow_server $MLFLOW_PORT; then
        exit 1
    fi
    
    # Step 7: Open browser
    open_browser
    
    # Step 8: Show monitoring info
    monitor_workflow
    
    # Keep script running
    print_success "Setup complete! MLflow UI should be available at http://localhost:$LOCAL_PORT"
    print_status "Press Ctrl+C to stop the tunnel and cleanup"
    
    # Wait indefinitely (until Ctrl+C)
    while true; do
        sleep 60
        # Optional: Check if tunnel is still alive
        if ! lsof -i:$LOCAL_PORT >/dev/null 2>&1; then
            print_warning "SSH tunnel appears to have died, attempting to reconnect..."
            if ! create_ssh_tunnel $MLFLOW_PORT $LOCAL_PORT; then
                print_error "Failed to reconnect tunnel"
                exit 1
            fi
        fi
    done
}

# Show usage if no arguments or help requested
if [[ $# -gt 0 ]] && [[ "$1" == "-h" || "$1" == "--help" ]]; then
    cat << EOF
Usage: $0 [options]

ML Workflow Launcher with SSH Tunneling

This script will:
1. Test SSH connection to cluster
2. Submit Nextflow workflow
3. Create SSH tunnel for MLflow UI
4. Start MLflow server on cluster
5. Open MLflow UI in local browser

Configuration (edit script variables):
  CLUSTER_HOST     - Your cluster hostname
  CLUSTER_USER     - Your cluster username
  PROJECT_DIR      - Remote project directory
  DATA_VERSION     - Data version to use
  PROFILE          - Nextflow profile (slurm/local)
  MLFLOW_PORT      - MLflow port on cluster
  LOCAL_PORT       - Local port for tunnel

Options:
  -h, --help       Show this help message

Example:
  $0

EOF
    exit 0
fi

# Run main function
main "$@"