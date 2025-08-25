// Alternative Nextflow approach with native parameter sweeping
nextflow.enable.dsl=2

// Parameters loaded from nextflow.config and command line

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
    
    // For hyperparameter optimization, we can either:
    // 1. Use Optuna (recommended)
    // 2. Use native Nextflow parameter sweeping (shown below)
    
    if (params.hyperparam_mode == 'optuna') {
        // Use Optuna for hyperparameter optimization
        results = run_experiment_optuna(data_ready, experiments)
    } else {
        // Use native Nextflow parameter sweeping
        param_combinations = generate_hyperparameter_grid(experiments)
        results = run_single_evaluation(data_ready, param_combinations)
        best_results = find_best_hyperparameters(results.groupTuple())
    }
    
    // Collect results
    collect_results(results.collect())
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

process generate_hyperparameter_grid {
    input:
    tuple val(prep), val(model), val(search)
    
    output:
    tuple val(prep), val(model), val(search), path('param_grid.json')
    
    script:
    """
    python ${projectDir}/src/generate_param_grid.py \\
        --model ${model} \\
        --search ${search} \\
        --n_samples ${params.hyperparam_samples ?: 20} \\
        --output param_grid.json
    """
}

process run_experiment_optuna {
    // Dynamic resource allocation based on run mode and model type
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

process run_single_evaluation {
    // Same resource configuration as above
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
    
    input:
    path data_file
    tuple val(prep), val(model), val(search), path(param_file), val(param_id)
    
    output:
    tuple val([prep, model, search]), path('eval_result_*.json')
    
    script:
    """
    export PROJECT_DIR=${projectDir}
    python ${projectDir}/src/evaluate_single.py \\
        --preprocessing ${prep} \\
        --model ${model} \\
        --search ${search} \\
        --data ${data_file} \\
        --params ${param_file} \\
        --param_id ${param_id} \\
        --data_version ${params.data_version}
    """
}

process find_best_hyperparameters {
    publishDir "results/v${params.data_version}", mode: 'copy'
    
    input:
    tuple val(experiment_config), path(eval_results)
    
    output:
    path 'best_result_*.json'
    
    script:
    def prep = experiment_config[0]
    def model = experiment_config[1] 
    def search = experiment_config[2]
    """
    python ${projectDir}/src/find_best_params.py \\
        --results ${eval_results} \\
        --experiment "${prep}_${model}_${search}" \\
        --output "best_result_${prep}_${model}_${search}.json"
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
for f in glob.glob('result_*.json') + glob.glob('best_result_*.json'):
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
