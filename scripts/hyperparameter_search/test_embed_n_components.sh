#!/bin/bash
#SBATCH --job-name=filtered_pipeline
#SBATCH --nodes=1
#SBATCH --gpus=rtx_4090:1
#SBATCH --mem-per-cpu=70GB
#SBATCH --time=08:00:00

module load gcc/8.2.0 python_gpu/3.11.2 cuda/11.2.2

source .venv/bin/activate

for i in 2 3 4 5 6 7 8 9 10 15 20 30 50
do
    python src/dimensionality_reduction.py \
        --DIMENSIONALITY_REDUCTION_FILE output/testing_pipeline_1/test_embed_n_components/components_${i}.h5 \
        --PARTIAL_FIT_DIM_REDUCTION 0.2 \
        --UMAP_COMPONENTS ${i}
done
