import os
import sys

# adding root directory to paths
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from logging_config import configure_get_logger
import config

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
# from cuml.dask.manifold import UMAP as MNMG_UMAP
os.environ["NUMEXPR_MAX_THREADS"] = "32"
import numexpr

from src.utils.function_runner import run_function_with_overrides
from src.utils.utils import load_embeddings


def UMAP_fit_transform(embedding_filename, n_neighbors, n_components, min_dist):

    features = load_embeddings(embedding_filename, "embeddings")

    reducer = cuml.UMAP(
        n_neighbors=n_neighbors, n_components=n_components, min_dist=min_dist
    )
    coordinates = reducer.fit_transform(features)

    return coordinates


def save_umap_coordinates(coordinates, output_filename):
    """
    Save UMAP coordinates to an HDF5 file.
    """
    with h5py.File(output_filename, "w") as file:
        file.create_dataset("umap_coordinates", data=coordinates)

def random_baseline(EMBEDDINGS_FILE, UMAP_COMPONENTS, DIMENSIONALITY_REDUCTION_FILE):
    print("Running random baseline")
    features = load_embeddings(EMBEDDINGS_FILE, "embeddings")
    random_projection = np.random.rand(features.shape[1], UMAP_COMPONENTS)
    save_umap_coordinates(random_projection, DIMENSIONALITY_REDUCTION_FILE)



def UMAP_transform_full_fit(
    EMBEDDINGS_FILE,
    UMAP_N_Neighbors,
    UMAP_COMPONENTS,
    UMAP_MINDIST,
    DIMENSIONALITY_REDUCTION_FILE,
):
    """
    Load embeddings, sample a subset, fit UMAP on the subset, and transform the entire dataset.
    """

    features = load_embeddings(EMBEDDINGS_FILE, "embeddings")
    local_model = UMAP(
        n_neighbors=UMAP_N_Neighbors,
        n_components=UMAP_COMPONENTS,
        min_dist=UMAP_MINDIST,
    )
    transformed = local_model.fit_transform(features)
    save_umap_coordinates(transformed, DIMENSIONALITY_REDUCTION_FILE)


def UMAP_transform_partial_fit(
    EMBEDDINGS_FILE,
    UMAP_N_Neighbors,
    UMAP_COMPONENTS,
    UMAP_MINDIST,
    PARTIAL_FIT_DIM_REDUCTION,
    DIMENSIONALITY_REDUCTION_FILE,
    NEGATIVE_SAMPLE_RATE,
    UMAP_N_EPOCHS,
):
    """
    Load embeddings, sample a subset, fit UMAP on the subset, and transform the entire dataset.
    """
    
    features = load_embeddings(EMBEDDINGS_FILE, "embeddings")

    local_model = UMAP(
        n_neighbors=UMAP_N_Neighbors,
        n_components=UMAP_COMPONENTS,
        min_dist=UMAP_MINDIST,
        negative_sample_rate=NEGATIVE_SAMPLE_RATE,
        n_epochs=UMAP_N_EPOCHS,
        verbose=True
    )

    sampled_indices = np.random.choice(
        features.shape[0], int(features.shape[0] * PARTIAL_FIT_DIM_REDUCTION), replace=False
    )
    
    logger.info(f"Fitting UMAP on {len(sampled_indices)} samples")
    sampled_features = features[sampled_indices]
    
    local_model.fit(sampled_features)

    subset_size = len(sampled_indices)

    # iterate over the rest of the data in chunks of subset_size and transform
    # subset size (derived from PARTIAL_FIT_DIM_REDUCTION) should be set to be the maximum subset of data on which we can fit
    # given a certain GPU memory

    result = None
    for i in tqdm(range(0, features.shape[0], subset_size)):
        chunk = features[i : i + subset_size]
        transformed_chunk = local_model.transform(chunk)
        if result is None:
            result = transformed_chunk
        else:
            result = np.concatenate((result, transformed_chunk), axis=0)

    save_umap_coordinates(result, DIMENSIONALITY_REDUCTION_FILE)




if __name__ == "__main__":

    run_function_with_overrides(UMAP_transform_partial_fit, config)
