import os
import sys
import joblib

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
from src.utils.utils import load_h5py, load_with_indices_h5py_efficient, get_indices_for_random_h5py_subset, save_h5py, load_with_indices_h5py, get_number_of_samples_h5py

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

def fit_umap_model(
    PROCESSED_REDDIT_DATA: str,
    UMAP_N_Neighbors: int,
    UMAP_COMPONENTS: int,
    UMAP_MINDIST: float,
    PARTIAL_FIT_DIM_REDUCTION: float,
    NEGATIVE_SAMPLE_RATE: int,
    UMAP_N_EPOCHS: int,
    UMAP_MODEL_SAVE_PATH: str
) -> UMAP:
    """
    Fit UMAP model on a subset of the data and save the model.
    If the model is already present in the designated location, load it instead of training a new one.
    """
    if UMAP_MODEL_SAVE_PATH and os.path.exists(UMAP_MODEL_SAVE_PATH):
        print(f"Loading existing UMAP model from {UMAP_MODEL_SAVE_PATH}")
        logger.info(f"Loading existing UMAP model from {UMAP_MODEL_SAVE_PATH}")
        umap_model = joblib.load(UMAP_MODEL_SAVE_PATH)
    else:
        print("Subset size for partial fit:", PARTIAL_FIT_DIM_REDUCTION)

        umap_model = UMAP(
            n_neighbors=UMAP_N_Neighbors,
            n_components=UMAP_COMPONENTS,
            min_dist=UMAP_MINDIST,
            negative_sample_rate=NEGATIVE_SAMPLE_RATE,
            n_epochs=UMAP_N_EPOCHS,
        )

        partial_fit_indices = get_indices_for_random_h5py_subset(PROCESSED_REDDIT_DATA, "embeddings", PARTIAL_FIT_DIM_REDUCTION)
        total_samples = len(partial_fit_indices)
        logger.info(f"Running partial fit on {total_samples} samples")

        s = time.time()
        sampled_features = load_with_indices_h5py(PROCESSED_REDDIT_DATA, "embeddings", partial_fit_indices)
        logger.info(f"Time to load data: {time.time() - s:.2f} s")

        execute_with_gpu_logging(umap_model.fit, sampled_features)

        if UMAP_MODEL_SAVE_PATH:
            joblib.dump(umap_model, UMAP_MODEL_SAVE_PATH)
            logger.info(f"UMAP model saved at {UMAP_MODEL_SAVE_PATH}")

    return umap_model


def transform_data_chunked(
    umap_model: UMAP,
    PROCESSED_REDDIT_DATA: str,
    DIMENSIONALITY_REDUCTION_DB_NAME: str,
    EMBEDDING_DB_NAME: str,
    PARTIAL_TRANSFORM_DIM_REDUCTION: float
) -> None:
    """
    Transform data using UMAP model in chunks and save the transformed data.
    """

    # The transform can be done in chunks of different dimensions compared to fitting. In the experiments if fitting size is very small
    # trasform would be very slow. That's why we have PARTIAL_TRANSFORM_DIM_REDUCTION
    total_samples, num_samples = get_number_of_samples_h5py(PROCESSED_REDDIT_DATA, EMBEDDING_DB_NAME, PARTIAL_TRANSFORM_DIM_REDUCTION)
    
    with h5py.File(PROCESSED_REDDIT_DATA, 'a') as output_file:
        
        # Remove the existing dataset if it exists
        if DIMENSIONALITY_REDUCTION_DB_NAME in output_file:
            del output_file[DIMENSIONALITY_REDUCTION_DB_NAME]

        for i in tqdm(range(0, total_samples, num_samples)):
            indices = np.arange(i, min(i + num_samples, total_samples))
            chunk = load_with_indices_h5py_efficient(PROCESSED_REDDIT_DATA, EMBEDDING_DB_NAME, indices)
            transformed_chunk = execute_with_gpu_logging(umap_model.transform, chunk)
            
            if DIMENSIONALITY_REDUCTION_DB_NAME not in output_file:
                maxshape = (None, transformed_chunk.shape[1])
                chunks = (num_samples, transformed_chunk.shape[1])  # Define chunk size
                dataset = output_file.create_dataset(
                    DIMENSIONALITY_REDUCTION_DB_NAME, 
                    data=transformed_chunk, 
                    maxshape=maxshape, 
                    chunks=chunks
                )
            else:
                dataset = output_file[DIMENSIONALITY_REDUCTION_DB_NAME]
                dataset.resize((dataset.shape[0] + transformed_chunk.shape[0]), axis=0)
                dataset[-transformed_chunk.shape[0]:] = transformed_chunk


def transform_data_full(
    umap_model: UMAP,
    PROCESSED_REDDIT_DATA: str,
    EMBEDDING_DB_NAME: str,
    DIMENSIONALITY_REDUCTION_DB_NAME: str
) -> None:
    """
    Transform the entire dataset using UMAP model without chunking and save the transformed data.
    """
    total_samples, _ = get_number_of_samples_h5py(PROCESSED_REDDIT_DATA, EMBEDDING_DB_NAME, 1.0)
    indices = np.arange(total_samples)
    full_data = load_with_indices_h5py(PROCESSED_REDDIT_DATA, EMBEDDING_DB_NAME, indices)
    transformed_data = execute_with_gpu_logging(umap_model.transform, full_data)
    save_h5py(transformed_data, PROCESSED_REDDIT_DATA, DIMENSIONALITY_REDUCTION_DB_NAME)

def UMAP_partial_fit_partial_transform(
    PROCESSED_REDDIT_DATA: str,
    UMAP_N_Neighbors: int,
    UMAP_COMPONENTS: int,
    UMAP_MINDIST: float,
    PARTIAL_FIT_DIM_REDUCTION: float,
    NEGATIVE_SAMPLE_RATE: int,
    UMAP_N_EPOCHS: int,
    DIMENSIONALITY_REDUCTION_DB_NAME: str,
    EMBEDDING_DB_NAME: str,
    UMAP_MODEL_SAVE_PATH: str,
    PARTIAL_TRANSFORM_DIM_REDUCTION: float
) -> None:
    """
    Load embeddings, sample a subset, fit UMAP on the subset, save the model, and transform the entire dataset.
    If the model is already present in the designated location, load it instead of training a new one.
    """

    # Fit the UMAP model or load it if already exists
    umap_model = fit_umap_model(
        PROCESSED_REDDIT_DATA,
        UMAP_N_Neighbors,
        UMAP_COMPONENTS,
        UMAP_MINDIST,
        PARTIAL_FIT_DIM_REDUCTION,
        NEGATIVE_SAMPLE_RATE,
        UMAP_N_EPOCHS,
        UMAP_MODEL_SAVE_PATH
    )

    # Transform the data using the fitted UMAP model in chunks
    transform_data_chunked(
        umap_model,
        PROCESSED_REDDIT_DATA,
        DIMENSIONALITY_REDUCTION_DB_NAME,
        EMBEDDING_DB_NAME,
        PARTIAL_TRANSFORM_DIM_REDUCTION
    )


def UMAP_partial_fit_full_transform(
    PROCESSED_REDDIT_DATA: str,
    UMAP_N_Neighbors: int,
    UMAP_COMPONENTS: int,
    UMAP_MINDIST: float,
    PARTIAL_FIT_DIM_REDUCTION: float,
    NEGATIVE_SAMPLE_RATE: int,
    UMAP_N_EPOCHS: int,
    DIMENSIONALITY_REDUCTION_DB_NAME: str,
    EMBEDDING_DB_NAME: str,
    UMAP_MODEL_SAVE_PATH: str,
) -> None:
    """
    Load embeddings, sample a subset, fit UMAP on the subset, save the model, and transform the entire dataset.
    If the model is already present in the designated location, load it instead of training a new one.
    """

    # Fit the UMAP model or load it if already exists
    umap_model = fit_umap_model(
        PROCESSED_REDDIT_DATA,
        UMAP_N_Neighbors,
        UMAP_COMPONENTS,
        UMAP_MINDIST,
        PARTIAL_FIT_DIM_REDUCTION,
        NEGATIVE_SAMPLE_RATE,
        UMAP_N_EPOCHS,
        UMAP_MODEL_SAVE_PATH
    )

    # Transform the data using the fitted UMAP model in chunks
    transform_data_full(
        umap_model,
        PROCESSED_REDDIT_DATA,
        EMBEDDING_DB_NAME,
        DIMENSIONALITY_REDUCTION_DB_NAME
    )


if __name__ == "__main__":
    print("Total running time:", run_function_with_overrides(UMAP_partial_fit_full_transform, config))
