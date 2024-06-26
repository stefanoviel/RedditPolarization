#!/bin/bash

# Define the array of subset sizes
subset_sizes=(0.001 0.01 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)

# Loop through each subset size and run the dimensionality reduction script
for size in "${subset_sizes[@]}"
do
    echo "Running UMAP with subset size $size"
    python src/dimensionality_reduction.py \
        --DIMENSIONALITY_REDUCTION_FILE "output/testing_pipeline/fit_umap_different_subsets/subset_$size.h5" \
        --PARTIAL_FIT_SAMPLE_SIZE $size \
        --UMAP_COMPONENTS 5
done
