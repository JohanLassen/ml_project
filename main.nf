// main.nf - Simple and clean
nextflow.enable.dsl=2

// Parameters - load from params.yaml
params.data_version = "1.0.0"
params.preprocessing = ['basic', 'minmax_pipeline', 'kbest_standard']
params.models = ['xgboost', 'random_forest', 'linear']
params.search = ['bayesian', 'random']

// Runtime configuration
params.run_mode = 'hpc'  // 'local' or 'hpc'
params.local_config = [
    cpus: 2,
    memory: '8.GB',
    time: '30.m',
    max_experiments: 2  // Limit for local testing
]
params.hpc_config = [
    cpus_xgboost: 4,
    cpus_other: 8,
    memory: '32.GB',
    time_xgboost: '2.h',
    time_other: '4.h'
]

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
    
    // Run all experiments
    results = run_experiment(data_ready, experiments)
    
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
    python ${projectDir}/scripts/prepare_data.py --input ${dataset} --version ${params.data_version}
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
    """
    export PROJECT_DIR=${projectDir}
    python ${projectDir}/src/main.py \\
        --preprocessing ${prep} \\
        --model ${model} \\
        --search ${search} \\
        --data ${data_file} \\
        --data_version ${params.data_version} \\
        --cpus ${task.cpus}
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