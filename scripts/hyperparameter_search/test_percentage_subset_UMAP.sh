#!/bin/bash
#SBATCH --job-name=filtered_pipeline
#SBATCH --nodes=1
#SBATCH --gpus=rtx_4090:1
#SBATCH --mem-per-cpu=70GB
#SBATCH --time=08:00:00

module load gcc/8.2.0 python_gpu/3.11.2 cuda/11.2.2

source .venv/bin/activate

# Define the array of subset sizes
subset_sizes=(0.001 0.01 0.1 0.2 0.4 0.6 0.8 0.9 0.9999 1.0)

rm output/testing_different_subsets/coherence.json

# Loop through each subset size and run the dimensionality reduction script
for size in "${subset_sizes[@]}"
do
    echo "Running UMAP with subset size $size"
    # python scripts/start_from_dim_reduction.py --PARTIAL_FIT_DIM_REDUCTION $size --TFIDF_FILE tfids_$size.json
    python scripts/start_from_dim_reduction.py --partial_fit_dim_reduction $size --tfidf_file tfids_$size.json
    
done
