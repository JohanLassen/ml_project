// Distributed Optuna Nextflow pipeline
nextflow.enable.dsl=2

workflow {
    // Prepare data
    dataset = Channel.fromPath("data/raw/dataset_v${params.data_version}.csv")
    data_ready = prepare_data(dataset)
    
    // Generate all experiment combinations
    preprocessing_ch = Channel.fromList(params.preprocessing)
    models_ch = Channel.fromList(params.models) 
    search_ch = Channel.fromList(params.search)
    
    experiments = preprocessing_ch
        .combine(models_ch)
        .combine(search_ch)
        .map { prep, model, search -> [prep, model, search] }
    
    // Limit experiments for local testing
    if (params.run_mode == 'local') {
        experiments = experiments.take(params.local_config.max_experiments)
    }
    
    if (params.distributed_optuna == true) {
        // Distributed Optuna approach: Many parallel trials
        trial_ids = Channel.from(1..params.optuna_max_trials)
        
        // Each experiment gets multiple parallel trials
        parallel_trials = experiments
            .combine(trial_ids)
            .map { prep, model, search, trial_id -> 
                [prep, model, search, trial_id] 
            }
        
        // Run trials in parallel across SLURM
        trial_results = run_distributed_optuna_trial(data_ready, parallel_trials)
        
        // Collect best results for each experiment
        best_results = collect_best_optuna_results(
            trial_results
                .groupTuple(by: [0,1,2])  // Group by experiment config
        )
        
    } else {
        // Standard single-job Optuna
        results = run_experiment_optuna(data_ready, experiments)
        best_results = results
    }
    
    // Final collection
    collect_results(best_results.collect())
}

process prepare_data {
    publishDir "data/processed/v${params.data_version}", mode: 'copy'
    
    input:
    path dataset
    
    output:
    path 'train_data.parquet'
    
    script:
    """
    python ${projectDir}/src/prepare_data.py --input ${dataset} --version ${params.data_version} --config ${projectDir}/config/config.yaml
    """
}

process run_distributed_optuna_trial {
    // Each trial runs as a separate SLURM job
    cpus { 
        if (params.run_mode == 'local') {
            params.local_config.cpus
        } else {
            model == 'xgboost' ? params.hpc_config.cpus_xgboost : params.hpc_config.cpus_other
        }
    }
    memory { 
        params.run_mode == 'local' ? params.local_config.memory : params.hpc_config.memory 
    }
    time { 
        if (params.run_mode == 'local') {
            params.local_config.time
        } else {
            // Shorter time per trial since it's just one trial
            model == 'xgboost' ? '30.m' : '20.m'  
        }
    }
    queue { 
        params.run_mode == 'local' ? null : (model == 'xgboost' ? 'gpu' : null) 
    }
    clusterOptions { 
        params.run_mode == 'local' ? null : (model == 'xgboost' ? '--gres=gpu:1' : '') 
    }
    
    // Don't publish individual trials, just return results
    
    input:
    path data_file
    tuple val(prep), val(model), val(search), val(trial_id)
    
    output:
    tuple val(prep), val(model), val(search), path('trial_result_*.json')
    
    script:
    def test_flag = (params.test_mode == true) ? '--test_mode' : ''
    """
    export PROJECT_DIR=${projectDir}
    python ${projectDir}/src/run_distributed_trial.py \\
        --preprocessing ${prep} \\
        --model ${model} \\
        --search ${search} \\
        --data ${data_file} \\
        --data_version ${params.data_version} \\
        --trial_id ${trial_id} \\
        --max_trials ${params.optuna_max_trials} \\
        --cpus ${task.cpus} \\
        ${test_flag}
    """
}

process collect_best_optuna_results {
    publishDir "results/v${params.data_version}", mode: 'copy'
    
    input:
    tuple val(prep), val(model), val(search), path(trial_files)
    
    output:
    path 'result_*.json'
    
    script:
    """
    export PROJECT_DIR=${projectDir}
    python ${projectDir}/src/collect_optuna_results.py \\
        --preprocessing ${prep} \\
        --model ${model} \\
        --search ${search} \\
        --trial_files ${trial_files} \\
        --data_version ${params.data_version}
    """
}

process run_experiment_optuna {
    // Standard single-job Optuna (your current implementation)
    cpus { 
        if (params.run_mode == 'local') {
            params.local_config.cpus
        } else {
            model == 'xgboost' ? params.hpc_config.cpus_xgboost : params.hpc_config.cpus_other
        }
    }
    memory { 
        params.run_mode == 'local' ? params.local_config.memory : params.hpc_config.memory 
    }
    time { 
        if (params.run_mode == 'local') {
            params.local_config.time
        } else {
            model == 'xgboost' ? params.hpc_config.time_xgboost : params.hpc_config.time_other
        }
    }
    queue { 
        params.run_mode == 'local' ? null : (model == 'xgboost' ? 'gpu' : 'compute') 
    }
    clusterOptions { 
        params.run_mode == 'local' ? null : (model == 'xgboost' ? '--gres=gpu:1' : '') 
    }
    
    publishDir "results/v${params.data_version}", mode: 'copy'
    
    input:
    path data_file
    tuple val(prep), val(model), val(search)
    
    output:
    path 'result_*.json'
    
    script:
    def test_flag = (params.test_mode == true) ? '--test_mode' : ''
    """
    export PROJECT_DIR=${projectDir}
    python ${projectDir}/src/main.py \\
        --preprocessing ${prep} \\
        --model ${model} \\
        --search ${search} \\
        --data ${data_file} \\
        --data_version ${params.data_version} \\
        --cpus ${task.cpus} \\
        ${test_flag}
    """
}

process collect_results {
    publishDir "results/v${params.data_version}/summary", mode: 'copy'
    
    input:
    path results
    
    output:
    path 'experiment_summary.json'
    
    script:
    """
    python -c "
import json
import glob

results = []
for f in glob.glob('result_*.json'):
    with open(f) as file:
        results.append(json.load(file))

summary = {
    'total_experiments': len(results),
    'completed': len([r for r in results if r.get('status') == 'completed']),
    'best_experiment': min(results, key=lambda x: x.get('best_rmse', float('inf'))) if results else None,
    'all_results': results
}

with open('experiment_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
    
print(f'Collected {len(results)} experiment results')
"
    """
}
