# 🚀 ML Regression Pipeline with Nextflow, Optuna & MLflow

A comprehensive machine learning pipeline built for high-performance regression experimentation with distributed hyperparameter optimization and experiment tracking.

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![Nextflow](https://img.shields.io/badge/nextflow-22.0+-green.svg)
![XGBoost](https://img.shields.io/badge/xgboost-regression-orange.svg)
![MLflow](https://img.shields.io/badge/mlflow-file--based-red.svg)

## ✨ Features

- 🔥 **XGBoost Regression** with CPU/GPU support for faster training
- ⚡ **Optuna** distributed hyperparameter optimization (SLURM-compatible)
- 📊 **MLflow** experiment tracking with file-based backend
- 🧬 **Nextflow** pipeline for reproducible workflows
- 🔄 **Dynamic Resource Management** - seamlessly switch between local and HPC execution
- 📈 **Early Stopping** for XGBoost models with validation splits
- 🛠️ **Flexible Preprocessing** pipelines with multiple algorithms
- 🎯 **Cross-Validation** with comprehensive model evaluation

## 🏗️ Architecture

```mermaid
graph TD
    A[Nextflow Pipeline] --> B[Data Processing]
    B --> C[Optuna Hyperparameter Search]
    C --> D[XGBoost Regression]
    C --> E[Linear Regression]
    C --> F[Random Forest Regression]
    D --> G[MLflow Tracking]
    E --> G
    F --> G
    G --> H[File-based MLruns]
    G --> I[Model Artifacts]
```

## 📦 Installation

### Prerequisites

- Python 3.10+
- CUDA-compatible GPU (for XGBoost GPU acceleration)
- PostgreSQL (for MLflow tracking)
- Java 11+ (for Nextflow)

### Quick Setup

1. **Clone the repository:**
```bash
git clone https://github.com/JohanLassen/ml_project.git
cd ml_project
```

2. **Create conda environment:**
```bash
conda env create -f environment.yml
conda activate ml-gpu-project
```

3. **Install Nextflow:**
```bash
curl -s https://get.nextflow.io | bash
sudo mv nextflow /usr/local/bin/
```

4. **Setup MLflow database:**
```bash
chmod +x setup_mlflow_db.sh
./setup_mlflow_db.sh
```

## 🎯 Key Components

### 🔧 Configuration System

The project uses **OmegaConf** for hierarchical configuration management:

- **`config/model/`** - Model configurations (XGBoost, Linear models)
- **`config/preprocessing/`** - Data preprocessing pipelines  
- **`config/search/`** - Hyperparameter search algorithms
- **`params.yaml`** - Main parameter file for Nextflow

### 🚀 Execution Modes

**Local Mode** (Development & Testing):
```yaml
run_mode: "local"
local_config:
  cpus: 4
  memory: "8 GB"
  max_concurrent: 4
```

**HPC Mode** (Production):
```yaml
run_mode: "hpc" 
hpc_config:
  cpus_xgboost: 16
  cpus_other: 8
  memory: "32 GB"
  max_concurrent: 16
```

### 🎛️ GPU Support

Automatic GPU detection and configuration:
- **XGBoost**: Uses `device: "cuda:0"` with modern CUDA API
- **Ray Workers**: Automatic CUDA environment inheritance
- **Resource Management**: GPU-aware trial scheduling

### 📊 MLflow Integration

Comprehensive experiment tracking:
- **PostgreSQL Backend** for scalable storage
- **Nested Runs** for hyperparameter trials
- **Automatic Logging** of parameters, metrics, and models
- **Artifact Storage** for model persistence

## 📁 Project Structure

```
ml_project/
├── config/                 # Configuration files
│   ├── model/              # Model configurations
│   ├── preprocessing/      # Preprocessing pipelines
│   └── search/             # Search algorithms
├── data/                   # Data directory (gitignored)
│   ├── raw/               # Raw datasets
│   └── processed/         # Processed datasets
├── scripts/               # Utility scripts
├── src/                   # Source code
│   ├── main.py           # Main experiment runner
│   ├── data_processing.py # Data preprocessing
│   ├── models.py         # Model definitions
│   └── ray_trainer.py    # Ray Tune integration
├── main.nf               # Nextflow pipeline
├── params.yaml           # Main parameters
└── environment.yml       # Conda environment
```

## 🔬 Model Support

### XGBoost (GPU-Accelerated)
- **Tree Method**: `hist` for optimal GPU performance
- **CUDA Device**: Automatic GPU assignment
- **Early Stopping**: 10% validation split with 50 rounds patience
- **Hyperparameters**: n_estimators, max_depth, learning_rate, regularization

### Linear Models
- **Ridge Regression**: L2 regularization
- **Fast Cross-Validation**: Optimized for quick iterations
- **Alpha Search**: Log-uniform hyperparameter space

### Preprocessing Options
- **Imputation**: Median, mean, or mode strategies
- **Scaling**: Standard, MinMax, or Robust scaling
- **Feature Selection**: Variance threshold, K-best selection
- **Pipeline Composition**: Flexible step combinations

## 🚀 Quick Start Tutorial

### 1. Prepare Your Data

Place your dataset in the `data/raw/` directory:
```bash
# Your data should have:
# - Feature columns (numeric)
# - Target column (numeric for regression)
# - Columns to exclude (metadata, IDs, etc.)
```

### 2. Configure the Experiment

Edit `params.yaml` to match your dataset:
```yaml
data:
  feature_prefix: ""  # Empty for all numeric columns
  exclude_columns: ["id", "name", "metadata_col"]
  target_column: "your_target_column"
  target_name: "outcome"
```

### 3. Run Local Test

Test with a linear model for quick validation:
```bash
# Start MLflow UI
mlflow ui --backend-store-uri postgresql://mlflow_user:mlflow_pass@localhost:5432/mlflow_db &

# Run experiment
python src/main.py \
  --preprocessing basic \
  --model linear \
  --search random \
  --data data/raw/your_dataset.csv \
  --data_version 1.0 \
  --cpus 4
```

### 4. Run GPU XGBoost

Scale up with GPU-accelerated XGBoost:
```bash
# Ensure CUDA is available
nvidia-smi

# Run XGBoost experiment
python src/main.py \
  --preprocessing kbest_standard \
  --model xgboost \
  --search bayesian \
  --data data/processed/your_dataset.parquet \
  --data_version 1.0 \
  --cpus 8
```

### 5. Full Pipeline with Nextflow

Run the complete pipeline:
```bash
# Local mode
nextflow run main.nf --run_mode local

# HPC mode (if available)
nextflow run main.nf --run_mode hpc
```

### 6. Monitor Results

Access your results:

**MLflow UI**: http://localhost:5000
- View experiments and runs
- Compare hyperparameters
- Download model artifacts
- Analyze metrics across trials

**Ray Dashboard**: http://localhost:8265 (when Ray is running)
- Monitor distributed training
- View resource utilization
- Debug worker processes

### 7. Advanced Configuration

**Custom Preprocessing Pipeline:**
```yaml
# config/preprocessing/custom.yaml
custom_pipeline:
  steps:
    imputer:
      module: sklearn.impute
      class: SimpleImputer
      params:
        strategy: median
    selector:
      module: sklearn.feature_selection
      class: SelectKBest
      params:
        k: 100
    scaler:
      module: sklearn.preprocessing
      class: RobustScaler
      params: {}
```

**XGBoost GPU Configuration:**
```yaml
# config/model/xgboost_custom.yaml
name: "xgboost_custom"
class_path: "xgboost.XGBRegressor"
params:
  device: "cuda:0"
  tree_method: "hist"
  early_stopping_rounds: 30
  eval_metric: "rmse"
search_space:
  n_estimators:
    type: "choice"
    values: [500, 1000, 2000]
  max_depth:
    type: "randint"
    lower: 6
    upper: 15
```

## 🎛️ Configuration Options

### Model Selection
```bash
--model linear      # Ridge regression
--model xgboost     # GPU-accelerated XGBoost
```

### Preprocessing
```bash
--preprocessing basic           # Impute + Scale
--preprocessing kbest_standard  # + Feature selection
--preprocessing minmax_pipeline # MinMax scaling
```

### Search Algorithms
```bash
--search random     # Random search
--search bayesian   # Bayesian optimization
--search hyperband  # HyperBand scheduling
```

## 📊 Example Results

After running experiments, you'll see comprehensive results in MLflow:

**Experiment Tracking:**
- 📈 RMSE metrics across all trials
- 🎯 Best hyperparameter combinations
- 📊 Cross-validation scores
- ⏱️ Training times and resource usage

**Model Artifacts:**
- 🤖 Trained model files
- 📋 Preprocessing pipelines
- 📝 Experiment summaries
- 🎨 Visualization plots (if enabled)

## 🛠️ Troubleshooting

**GPU Not Detected:**
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Verify XGBoost GPU
python -c "import xgboost as xgb; print(xgb.XGBRegressor().get_params())"
```

**MLflow Connection Issues:**
```bash
# Test PostgreSQL connection
psql -h localhost -U mlflow_user -d mlflow_db -c "SELECT 1;"

# Restart MLflow UI
pkill -f mlflow
mlflow ui --backend-store-uri postgresql://mlflow_user:mlflow_pass@localhost:5432/mlflow_db
```

**Ray Tune Errors:**
```bash
# Clean Ray processes
ray stop
ray start --head

# Check Ray status
ray status
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Nextflow** for workflow management
- **Ray Tune** for distributed hyperparameter optimization  
- **XGBoost** for gradient boosting algorithms
- **MLflow** for experiment tracking
- **OmegaConf** for configuration management

---

🤖 *This project was developed with assistance from [Claude Code](https://claude.ai/code)*
