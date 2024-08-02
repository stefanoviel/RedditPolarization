import os
import sys

# adding root directory to paths
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from logging_config import configure_get_logger
import config

from cuml.common import logger
logger.set_level(logger.level_error)

if not os.path.exists(config.OUTPUT_DIR):
    os.makedirs(config.OUTPUT_DIR)
logger = configure_get_logger(config.OUTPUT_DIR, config.EXPERIMENT_NAME, executed_file_name = __file__)

import h5py
from tqdm import tqdm
import numpy as np
import cuml
from cuml.manifold import UMAP
from dask_cuda import LocalCUDACluster
from dask.distributed import Client
import dask.array as da
import numexpr
import time

from src.utils.function_runner import run_function_with_overrides, execute_with_gpu_logging
from src.utils.utils import load_h5py, load_with_indices_h5py_efficient, get_indices_for_random_h5py_subset, save_h5py, load_with_indices_h5py

def UMAP_transform_full_fit(
    PROCESSED_REDDIT_DATA: str,
    UMAP_N_Neighbors: int,
    UMAP_COMPONENTS: int,
    UMAP_MINDIST: float,
    DIMENSIONALITY_REDUCTION_DB_NAME: str,
):
    """
    Load embeddings, sample a subset, fit UMAP on the subset, and transform the entire dataset.
    """

    features = load_h5py(PROCESSED_REDDIT_DATA, "embeddings")
    local_model = UMAP(
        n_neighbors=UMAP_N_Neighbors,
        n_components=UMAP_COMPONENTS,
        min_dist=UMAP_MINDIST,
    )
    transformed = local_model.fit_transform(features)
    save_h5py(transformed, PROCESSED_REDDIT_DATA, DIMENSIONALITY_REDUCTION_DB_NAME)


def UMAP_transform_partial_fit(
    PROCESSED_REDDIT_DATA: str,
    UMAP_N_Neighbors: int,
    UMAP_COMPONENTS: int,
    UMAP_MINDIST: float,
    PARTIAL_FIT_DIM_REDUCTION: float,
    NEGATIVE_SAMPLE_RATE: int,
    UMAP_N_EPOCHS: int,
    DIMENSIONALITY_REDUCTION_DB_NAME: str,
) -> None:
    
    """
    Load embeddings, sample a subset, fit UMAP on the subset, and transform the entire dataset.
    """

    print("subset size for partial fit", PARTIAL_FIT_DIM_REDUCTION)
    
    umap_model = UMAP(
        n_neighbors=UMAP_N_Neighbors,
        n_components=UMAP_COMPONENTS,
        min_dist=UMAP_MINDIST,
        negative_sample_rate=NEGATIVE_SAMPLE_RATE,
        n_epochs=UMAP_N_EPOCHS,
    )

    partial_fit_indices, total_samples, num_samples = get_indices_for_random_h5py_subset(PROCESSED_REDDIT_DATA, "embeddings", PARTIAL_FIT_DIM_REDUCTION)
    logger.info(f"Running partial fit on {num_samples} samples out of {total_samples} samples")
    s = time.time()

    # here we only load the necessary indices otherwise oom error
    sampled_features = load_with_indices_h5py(PROCESSED_REDDIT_DATA, "embeddings", partial_fit_indices)  
    print("Time to load data", time.time() - s)
    execute_with_gpu_logging(umap_model.fit, sampled_features)

    # iterate over the rest of the data in chunks of subset_size and transform
    # subset size (derived from PARTIAL_FIT_DIM_REDUCTION) should be the maximum subset of data on which we can fit
    # given a certain GPU memory 
    result = None
    for i in tqdm(range(0, total_samples, num_samples)):
        indices = np.arange(i, min(i + num_samples, total_samples))
        chunk = load_with_indices_h5py_efficient(PROCESSED_REDDIT_DATA, "embeddings", indices)
        transformed_chunk = execute_with_gpu_logging(umap_model.transform, chunk)
        if result is None:
            result = transformed_chunk
        else:
            result = np.concatenate((result, transformed_chunk), axis=0)

    save_h5py(result, PROCESSED_REDDIT_DATA, DIMENSIONALITY_REDUCTION_DB_NAME)


if __name__ == "__main__":
    print("Total running time:", run_function_with_overrides(UMAP_transform_partial_fit, config))
