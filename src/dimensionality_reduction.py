import os
import sys

# adding root directory to paths
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from logging_config import configure_get_logger
import config

if not os.path.exists(config.OUTPUT_DIR):
    os.makedirs(config.OUTPUT_DIR)
logger = configure_get_logger(config.OUTPUT_DIR, executed_file_name = __file__)

from src.run_single_step import run_function_with_overrides
import h5py
from tqdm import tqdm
import numpy as np
import cuml
from cuml.manifold import UMAP
from dask_cuda import LocalCUDACluster
from dask.distributed import Client
import dask.array as da
from cuml.dask.manifold import UMAP as MNMG_UMAP


def load_embeddings(file_path: str) -> np.ndarray:
    """
    Load embeddings from an HDF5 file into a NumPy array.
    """
    with h5py.File(file_path, "r") as file:
        # Assume the dataset name in HDF5 file is 'embeddings'
        embeddings = file["embeddings"][:]
    return embeddings


def UMAP_fit_transform(embedding_filename, n_neighbors, n_components, min_dist):

    features = load_embeddings(embedding_filename)

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


def UMAP_transform_partial_fit(
    EMBEDDINGS_FILE,
    UMAP_N_Neighbors,
    UMAP_COMPONENTS,
    UMAP_MINDIST,
    PARTIAL_FIT_SAMPLE_SIZE,
    DIMENSIONALITY_REDUCTION_FILE,
):
    """
    Load embeddings, sample a subset, fit UMAP on the subset, and transform the entire dataset.
    """
    
    features = load_embeddings(EMBEDDINGS_FILE)

    subset_size = int(features.shape[0] * PARTIAL_FIT_SAMPLE_SIZE)
    np.random.shuffle(features)

    local_model = UMAP(
        n_neighbors=UMAP_N_Neighbors,
        n_components=UMAP_COMPONENTS,
        min_dist=UMAP_MINDIST,
    )

    sampled_features = features[:subset_size]
    local_model.fit(sampled_features)

    # iterate over the rest of the data in chunks of subset_size and transform
    result = None
    for i in tqdm(range(subset_size, features.shape[0], subset_size)):
        chunk = features[i : i + subset_size]
        transformed_chunk = local_model.transform(chunk)
        if result is None:
            result = transformed_chunk
        else:
            result = np.concatenate((result, transformed_chunk), axis=0)

    save_umap_coordinates(result, DIMENSIONALITY_REDUCTION_FILE)


if __name__ == "__main__":

    run_function_with_overrides(UMAP_transform_partial_fit, config)
