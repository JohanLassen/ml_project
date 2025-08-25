# Hyperparameter Optimization Alternatives to Ray Tune

This document outlines better alternatives to Ray Tune that are more compatible with SLURM and Nextflow for your machine learning pipeline.

## Why Replace Ray Tune?

Ray Tune has several limitations in HPC/SLURM environments:
- **Cluster Management Overhead**: Ray requires its own cluster management, which conflicts with SLURM
- **Resource Contention**: Ray's resource allocation doesn't integrate well with SLURM's job scheduling
- **Communication Issues**: Ray workers may have trouble communicating across SLURM nodes
- **Dependency Complexity**: Heavy dependencies and potential version conflicts

## Recommended Alternatives

### 1. **Optuna** (Recommended ⭐)

**Why it's better:**
- ✅ **Database-driven**: Perfect for SLURM's job-based approach
- ✅ **No cluster management**: Each trial is independent
- ✅ **Advanced algorithms**: TPE, CMA-ES, Multi-objective optimization
- ✅ **Fault tolerance**: Can resume interrupted studies
- ✅ **Simple integration**: Easy to replace Ray Tune

**Implementation:**
```python
# Already implemented in src/optuna_trainer.py
from optuna_trainer import OptunaTrainer

trainer = OptunaTrainer("sqlite:///optuna_studies.db")
best_params, best_score = trainer.run_optimization(config, X, y)
```

**Usage:**
```bash
# Install Optuna
pip install optuna

# Your existing Nextflow pipeline will work with minimal changes
nextflow run main.nf -params-file params.yaml
```

### 2. **Native Nextflow Parameter Sweeping**

**Why it's good:**
- ✅ **Perfect SLURM integration**: Each parameter combination = separate SLURM job
- ✅ **Maximum parallelization**: Full use of cluster resources
- ✅ **No external dependencies**: Pure Nextflow
- ✅ **Transparent resource allocation**: Clear resource usage per trial

**Implementation:**
See `main_alternative.nf` for a complete implementation using this approach.

**Usage:**
```bash
# Set hyperparameter mode to grid search
nextflow run main_alternative.nf -params-file params.yaml --hyperparam_mode grid
```

### 3. **Hyperopt**

**Why it's good:**
- ✅ **Lightweight**: Minimal dependencies
- ✅ **Battle-tested**: Mature and stable
- ✅ **Simple API**: Easy to integrate

**Quick Implementation:**
```python
from hyperopt import fmin, tpe, hp, Trials

def objective(params):
    # Your model evaluation code
    return rmse

space = {
    'n_estimators': hp.choice('n_estimators', [50, 100, 200]),
    'max_depth': hp.randint('max_depth', 3, 10)
}

best = fmin(objective, space, algo=tpe.suggest, max_evals=100)
```

### 4. **Scikit-Optimize (skopt)**

**Why it's good:**
- ✅ **Bayesian optimization focused**: Efficient search
- ✅ **Scikit-learn integration**: Works well with your models
- ✅ **Lightweight**: Few dependencies

**Quick Implementation:**
```python
from skopt import gp_minimize

def objective(params):
    n_estimators, max_depth = params
    # Your model evaluation
    return rmse

space = [(50, 500), (3, 15)]  # n_estimators, max_depth
result = gp_minimize(objective, space, n_calls=100)
```

## Configuration Changes Made

### Updated Files:
1. **`src/optuna_trainer.py`** - New Optuna-based hyperparameter optimization
2. **`requirements.txt`** - Replaced Ray with Optuna
3. **`src/main.py`** - Updated to use Optuna instead of Ray
4. **`config/search/*.yaml`** - Updated search configurations for Optuna
5. **`params.yaml`** - Added hyperparameter optimization mode selection

### Search Algorithm Mapping:
- `bayesian` → TPE (Tree-structured Parzen Estimator)
- `random` → Random sampling
- `hyperband` → TPE with pruning

## Performance Comparison

| Method | SLURM Integration | Parallelization | Resource Efficiency | Complexity |
|--------|------------------|-----------------|-------------------|------------|
| **Ray Tune** | ❌ Poor | ⚠️ Limited | ❌ Poor | 🔴 High |
| **Optuna** | ✅ Excellent | ✅ Good | ✅ Excellent | 🟢 Low |
| **Nextflow Native** | ✅ Perfect | ✅ Maximum | ✅ Perfect | 🟡 Medium |
| **Hyperopt** | ✅ Good | ⚠️ Limited | ✅ Good | 🟢 Low |
| **Scikit-Optimize** | ✅ Good | ⚠️ Limited | ✅ Good | 🟢 Low |

## Migration Steps

### Option 1: Use Optuna (Recommended)
1. ✅ **Already done**: Files updated to use Optuna
2. Install Optuna: `pip install optuna`
3. Run your existing pipeline: `nextflow run main.nf`

### Option 2: Use Native Nextflow
1. Use the alternative pipeline: `nextflow run main_alternative.nf`
2. Set `hyperparam_mode: grid` in `params.yaml`
3. Adjust `hyperparam_samples` for desired number of trials

### Option 3: Implement Other Libraries
Use the code examples above to implement Hyperopt or Scikit-Optimize.

## Recommendations by Use Case

- **Best Overall**: Optuna - Great balance of features and SLURM compatibility
- **Maximum Parallelization**: Native Nextflow - Perfect for large parameter spaces
- **Simplest Migration**: Hyperopt - Minimal code changes needed
- **Bayesian Focus**: Scikit-Optimize - If you primarily need Bayesian optimization

## Running on SLURM

All alternatives work better with SLURM because they:
1. Don't require cluster management
2. Work with SLURM's job scheduling
3. Allow proper resource allocation per trial
4. Support fault tolerance and job resumption

Your existing Nextflow configuration for SLURM should work without changes!
