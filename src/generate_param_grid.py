#!/usr/bin/env python3
"""
Generate hyperparameter grid for native Nextflow parameter sweeping
"""
import argparse
import json
import yaml
import numpy as np
from pathlib import Path
import itertools

def load_model_config(model_name: str, project_dir: str) -> dict:
    """Load model configuration from YAML file"""
    config_path = Path(project_dir) / "config" / "model" / f"{model_name}.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)

def generate_grid_search_space(search_space: dict, n_samples: int) -> list:
    """Generate parameter combinations for grid search"""
    param_combinations = []
    
    # For each parameter, generate sample values
    param_grids = {}
    for param, config in search_space.items():
        if config['type'] == "choice":
            param_grids[param] = config['values']
        elif config['type'] == "randint":
            # Generate evenly spaced integers
            n_values = min(n_samples // 2, config['upper'] - config['lower'] + 1)
            param_grids[param] = list(range(config['lower'], config['upper'] + 1, 
                                          max(1, (config['upper'] - config['lower']) // n_values)))
        elif config['type'] == "uniform":
            # Generate evenly spaced values
            param_grids[param] = list(np.linspace(config['lower'], config['upper'], 
                                                min(10, n_samples // 2)))
        elif config['type'] == "loguniform":
            # Generate log-spaced values
            param_grids[param] = list(np.logspace(np.log10(config['lower']), 
                                                np.log10(config['upper']), 
                                                min(10, n_samples // 2)))
    
    # Generate all combinations
    param_names = list(param_grids.keys())
    param_values = list(param_grids.values())
    
    all_combinations = list(itertools.product(*param_values))
    
    # Limit to n_samples if we have too many combinations
    if len(all_combinations) > n_samples:
        # Sample randomly from all combinations
        indices = np.random.choice(len(all_combinations), n_samples, replace=False)
        all_combinations = [all_combinations[i] for i in indices]
    
    # Convert to list of dictionaries
    for combination in all_combinations:
        param_dict = dict(zip(param_names, combination))
        param_combinations.append(param_dict)
    
    return param_combinations

def generate_random_search_space(search_space: dict, n_samples: int) -> list:
    """Generate random parameter combinations"""
    param_combinations = []
    
    for _ in range(n_samples):
        params = {}
        for param, config in search_space.items():
            if config['type'] == "choice":
                params[param] = np.random.choice(config['values'])
            elif config['type'] == "randint":
                params[param] = int(np.random.randint(config['lower'], config['upper'] + 1))
            elif config['type'] == "uniform":
                params[param] = float(np.random.uniform(config['lower'], config['upper']))
            elif config['type'] == "loguniform":
                params[param] = float(np.random.uniform(np.log(config['lower']), 
                                                      np.log(config['upper'])))
                params[param] = np.exp(params[param])
        
        param_combinations.append(params)
    
    return param_combinations

def main():
    parser = argparse.ArgumentParser(description='Generate hyperparameter grid for Nextflow')
    parser.add_argument('--model', required=True, help='Model name')
    parser.add_argument('--search', required=True, help='Search strategy')
    parser.add_argument('--n_samples', type=int, default=20, help='Number of parameter combinations')
    parser.add_argument('--output', required=True, help='Output JSON file')
    parser.add_argument('--project_dir', default='.', help='Project directory')
    
    args = parser.parse_args()
    
    # Load model configuration
    model_config = load_model_config(args.model, args.project_dir)
    search_space = model_config.get('search_space', {})
    
    if not search_space:
        print(f"No search space defined for model {args.model}")
        param_combinations = [{}]  # Empty params
    else:
        # Generate parameter combinations based on search strategy
        if args.search == 'random':
            param_combinations = generate_random_search_space(search_space, args.n_samples)
        else:
            # Use grid search for other strategies
            param_combinations = generate_grid_search_space(search_space, args.n_samples)
    
    # Add parameter IDs
    for i, params in enumerate(param_combinations):
        params['param_id'] = i
    
    # Save to JSON
    with open(args.output, 'w') as f:
        json.dump(param_combinations, f, indent=2)
    
    print(f"Generated {len(param_combinations)} parameter combinations for {args.model}")

if __name__ == "__main__":
    main()
