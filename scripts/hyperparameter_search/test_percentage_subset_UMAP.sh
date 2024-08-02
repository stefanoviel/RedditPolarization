#!/bin/bash
#SBATCH --job-name=filtered_pipeline
#SBATCH --nodes=1
#SBATCH --gpus=rtx_4090:1
#SBATCH --mem-per-cpu=60GB
#SBATCH --time=48:00:00
#SBATCH --output=log_scripts/%x-%j.out

# module load stack/.2024-05-silent  gcc/13.2.0 python_cuda/3.11.6
echo "loaded stack"

# source .my_venv/bin/activate
export OMP_NUM_THREADS=16
ulimit -n 4096  # to allow duckdb to open all files

subset_sizes=(0.001 0.1 0.5 1.0)

# Loop through each subset size and run the dimensionality reduction script
for size in "${subset_sizes[@]}"
do
    # echo "Running UMAP with subset size $size"
    python scripts/start_from_dim_reduction.py --partial_fit_dim_reduction $size --tfidf_file tfids_$size.json
    echo $size
done
