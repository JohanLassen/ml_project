// main.nf - Simple and clean
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
    
    // Create all combinations using combine operators
    experiments = preprocessing_ch
        .combine(models_ch)
        .combine(search_ch)
        .map { prep, model, search -> 
            log.info "🔧 Creating experiment: ${prep} + ${model} + ${search}"
            return [prep, model, search] 
        }
    
    // Print summary information
    log.info "=== Experiment Configuration ==="
    log.info "Preprocessing methods: ${params.preprocessing}"
    log.info "Models: ${params.models}"
    log.info "Search algorithms: ${params.search}"
    log.info "Run mode: ${params.run_mode}"
    log.info "Test mode: ${params.test_mode}"
    log.info "Optuna max trials: ${params.optuna_max_trials}"
    
    // Calculate total combinations
    def total_combinations = params.preprocessing.size() * params.models.size() * params.search.size()
    log.info "Total experiment combinations: ${total_combinations}"  
    log.info "HPC mode: Running all ${total_combinations} experiments in parallel"
    
    // Combine prepared data with each experiment combination
    // Use combine but handle the list structure properly
    experiment_inputs = data_ready
        .combine(experiments)
        .map { data_file, prep, model, search -> 
            log.info "🔗 Pairing data with: ${prep} + ${model} + ${search}"
            return [data_file, prep, model, search]
        }
    
    // Run all experiments - each experiment gets its own SLURM job
    results = run_experiment(experiment_inputs)

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

process run_experiment {
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
        params.run_mode == 'local' ? null : (model == 'xgboost' ? 'gpu' : null) 
    }
    clusterOptions { 
        if (params.run_mode == 'local') {
            null
        } else {
            // Only request GPU for XGBoost if available
            model == 'xgboost' ? '--gres=gpu:1' : null
        }
    }
    
    publishDir "results/v${params.data_version}", mode: 'copy'
    
    input:
    tuple path(data_file), val(prep), val(model), val(search)
    
    output:
    path 'result_*.json'
    
    script:
    def test_flag = (params.test_mode == true) ? '--test_mode' : ''
    """
    export PROJECT_DIR=${projectDir}
    echo "🚀 Starting HPC experiment: ${prep} + ${model} + ${search}"
    echo "Resources: ${task.cpus} CPUs, ${task.memory} memory"
    echo "Queue: ${task.queue ?: 'default'}"
    
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
    'completed': len([r for r in results if r['status'] == 'completed']),
    'best_experiment': min(results, key=lambda x: x['best_rmse']) if results else None,
    'all_results': results
}

with open('experiment_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
    
print(f'Collected {len(results)} experiment results')
"
    """
}