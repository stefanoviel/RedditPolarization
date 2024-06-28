#!/bin/bash
#SBATCH --job-name=filtered_pipeline
#SBATCH --nodes=1
#SBATCH --gpus=rtx_4090:1
#SBATCH --mem-per-cpu=70GB
#SBATCH --time=08:00:00

module load gcc/8.2.0 python_gpu/3.11.2 cuda/11.2.2

source .venv/bin/activate

# Define the array of subset sizes
subset_sizes=(0.001 0.01 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)

# Loop through each subset size and run the dimensionality reduction script
for size in "${subset_sizes[@]}"
do

    python src/hdbscan.py \
        --DIMENSIONALITY_REDUCTION_FILE "output/testing_pipeline_1/test_percentage_subset/subset_$size.h5" \
        --CLUSTER_FILE "output/testing_pipeline_1/clustering_percentage_subset/subset_$size.h5" \

done
