import os
import sys 
# adding root directory to paths
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from logging_config import configure_get_logger
import config
if not os.path.exists(config.OUTPUT_PATH):
    os.makedirs(config.OUTPUT_PATH)
logger = configure_get_logger(config.OUTPUT_PATH)
os.environ['NUMEXPR_MAX_THREADS'] = '32'  

import h5py
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
    with h5py.File(file_path, 'r') as file:
        # Assume the dataset name in HDF5 file is 'embeddings'
        embeddings = file['embeddings'][:]
    return embeddings


def UMAP_fit_transform(embedding_filename, n_neighbors, n_components, min_dist):
    
    features = load_embeddings(embedding_filename)

    reducer = cuml.UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        min_dist=min_dist
    )
    coordinates = reducer.fit_transform(features)

    return coordinates

def UMAP_transform_partial_fit(embedding_filename, n_neighbors, n_components, min_dist, sample_size):
    """
    Load embeddings, sample a subset, fit UMAP on the subset, and transform the entire dataset.
    """
    cluster = LocalCUDACluster(threads_per_worker=1)
    client = Client(cluster)
    # Load embeddings
    features = load_embeddings(embedding_filename)
    
    # Determine the number of samples for the subset
    total_samples = int(features.shape[0] * sample_size)
    
    # Sample a subset of the embeddings
    sampled_indices = np.random.choice(features.shape[0], total_samples, replace=False)
    sampled_features = features[sampled_indices]
    
    # Initialize and fit UMAP on the sampled subset
    local_model = UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        min_dist=min_dist
    )
    
    local_model.fit(sampled_features)
    
    distributed_model = MNMG_UMAP(model=local_model)

    scattered_features = client.scatter(features, broadcast=True)

    distributed_X = da.from_delayed(scattered_features, shape=features.shape, dtype=features.dtype)
    
    embedding = distributed_model.transform(distributed_X)
    result = embedding.compute()

    client.close()

    print('results', result.shape)

    return result


if __name__ == "__main__":

    umap_coordinates = UMAP_transform_partial_fit(
        config.EMBEDDINGS_FILE,
        config.UMAP_N_Neighbors,
        config.UMAP_COMPONENTS,
        config.UMAP_MINDIST, 
        0.5
    )
    np.save(os.path.join(config.OUTPUT_PATH, 'umap_coordinates.npy'), umap_coordinates)
    logger.info("UMAP coordinates saved.")



