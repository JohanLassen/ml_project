#!/usr/bin/env python3
"""
Monitor progress of distributed Optuna studies
"""
import optuna
import argparse

def monitor_study(study_name: str, storage_path: str = "sqlite:///shared_optuna_studies.db"):
    """Monitor an Optuna study"""
    try:
        study = optuna.load_study(
            study_name=study_name,
            storage=storage_path
        )
        
        print(f"Study: {study_name}")
        print(f"Total trials: {len(study.trials)}")
        print(f"Best value: {study.best_value:.4f}")
        print(f"Best params: {study.best_params}")
        
        # Show recent trials
        recent_trials = study.trials[-5:]
        print(f"\nLast 5 trials:")
        for trial in recent_trials:
            print(f"  Trial {trial.number}: {trial.value:.4f} - {trial.state}")
            
    except Exception as e:
        print(f"Error loading study {study_name}: {e}")

def list_studies(storage_path: str = "sqlite:///shared_optuna_studies.db"):
    """List all studies in the database"""
    try:
        study_summaries = optuna.get_all_study_summaries(storage=storage_path)
        print("Available studies:")
        for summary in study_summaries:
            print(f"  - {summary.study_name} ({summary.n_trials} trials)")
    except Exception as e:
        print(f"Error listing studies: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Monitor Optuna studies')
    parser.add_argument('--study', help='Study name to monitor')
    parser.add_argument('--list', action='store_true', help='List all studies')
    parser.add_argument('--storage', default='sqlite:///shared_optuna_studies.db', help='Storage path')
    
    args = parser.parse_args()
    
    if args.list:
        list_studies(args.storage)
    elif args.study:
        monitor_study(args.study, args.storage)
    else:
        print("Use --list to see all studies or --study <name> to monitor a specific study")
