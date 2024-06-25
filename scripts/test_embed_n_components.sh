# test embedding with different number of components

# python src/dimensionality_reduction.py \
#     --DIMENSIONALITY_REDUCTION_FILE output/testing_pipeline/subset_umap_2/subset_0.0001.h5 \
#     --PARTIAL_FIT_SAMPLE_SIZE 0.0001 \
#     --UMAP_COMPONENTS 2

for i in 2 3 4 5 6 7 8 9 10 15 20 30 50
do
    python src/dimensionality_reduction.py \
        --DIMENSIONALITY_REDUCTION_FILE output/testing_pipeline/food_movies_diff_dimensions/components_${i}.h5 \
        --PARTIAL_FIT_SAMPLE_SIZE 0.1 \
        --UMAP_COMPONENTS ${i}
done