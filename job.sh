#!/bin/sh                                                                                                               
#BATCH --mem-per-cpu 6g                                                                                                 
#SBATCH -t 02:00:00                                                                                                     
#SBATCH -o /home/jklassen/jupyter_cluster/jupyter_cluster_moi.out                                                        
#SBATCH -e /home/jklassen/jupyter_cluster/jupyter_cluster_moi.err                                                        
#SBATCH -J mlflow_cluster                                                                                          
#SBATCH -A forensics_TOFscreenings

source ~/.bashrc
conda activate ml-gpu-project
mlflow ui --host 0.0.0.0 --port=5000 #--backend-store-uri postgresql://mlflow_user:mlflow_pass@localhost:5432/mlflow_db

# Tunnel script command:
# bash mlflow_tunnel.sh -u moicoll -c 5000 -l 5000 -s /home/moicoll/mlflow_cluster/mlflow_job.sh
