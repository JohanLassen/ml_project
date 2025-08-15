// main.nf - Simple and clean
nextflow.enable.dsl=2

// Parameters
params.data_version = "1.0.0"
params.preprocessing = ['basic', 'feature_selection', 'scaling']
params.models = ['xgboost', 'random_forest', 'linear']
params.search = ['bayesian', 'random']

workflow {
    // Prepare data
    dataset = Channel.fromPath("data/raw/dataset_v${params.data_version}.csv")
    data_ready = prepare_data(dataset)
    
    // Generate all experiment combinations
    experiments = Channel.fromList(
        params.preprocessing.collectMany { prep ->
            params.models.collectMany { model ->
                params.search.collect { search ->
                    [prep, model, search]
                }
            }
        }
    )
    
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
    python scripts/prepare_data.py --input ${dataset} --version ${params.data_version}
    """
}

process run_experiment {
    // Dynamic resource allocation based on model type
    cpus { experiment[1] == 'xgboost' ? 4 : 8 }
    memory '32.GB'
    time { experiment[1] == 'xgboost' ? '2.h' : '4.h' }
    queue { experiment[1] == 'xgboost' ? 'gpu' : 'compute' }
    clusterOptions { experiment[1] == 'xgboost' ? '--gres=gpu:1' : '' }
    
    publishDir "results/v${params.data_version}", mode: 'copy'
    
    input:
    path data_file
    tuple val(prep), val(model), val(search)
    
    output:
    path 'result_*.json'
    
    script:
    """
    python src/main.py \\
        --preprocessing ${prep} \\
        --model ${model} \\
        --search ${search} \\
        --data ${data_file} \\
        --data_version ${params.data_version} \\
        --cpus ${task.cpus}
    """
}