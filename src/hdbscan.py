import os
import sys

# adding root directory to paths
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from logging_config import configure_get_logger
import config

if not os.path.exists(config.OUTPUT_DIR):
    os.makedirs(config.OUTPUT_DIR)
logger = configure_get_logger(config.OUTPUT_DIR, config.EXPERIMENT_NAME, executed_file_name = __file__)

from src.utils.function_runner import run_function_with_overrides
import h5py
from tqdm import tqdm
import numpy as np
import cuml
import time

from cuml.common import logger
logger.set_level(logger.level_error)

os.environ["NUMEXPR_MAX_THREADS"] = "32"
import numexpr


os.environ['CUDA_VISIBLE_DEVICES']=str(1)

from src.utils.utils import load_embeddings


def save_clusters_hdf5(clusters, file_name):
    """Save clusters data to an HDF5 file."""
    print("saving clusters to", file_name)
    with h5py.File(file_name, 'w') as f:
        f.create_dataset('clusters', data=clusters)

def run_dbscan(HDBS_MIN_CLUSTERSIZE:int, HDBS_MIN_SAMPLES:int, DIMENSIONALITY_REDUCTION_FILE:str, CLUSTER_FILE:str):

    data = load_embeddings(DIMENSIONALITY_REDUCTION_FILE, "umap_coordinates")
    scanner = cuml.cluster.hdbscan.HDBSCAN(min_cluster_size=HDBS_MIN_CLUSTERSIZE, min_samples=HDBS_MIN_SAMPLES)

    clusters = scanner.fit_predict(data)
    logger.info(f"Number of clusters: {len(np.unique(clusters))}")
    save_clusters_hdf5(clusters, CLUSTER_FILE)

def run_dbscan_full_data(HDBS_MIN_CLUSTERSIZE: int, HDBS_MIN_SAMPLES: int, DIMENSIONALITY_REDUCTION_FILE: str, CLUSTER_FILE: str):
    # Load the full dataset
    data = load_embeddings(DIMENSIONALITY_REDUCTION_FILE, "umap_coordinates")
    
    # Create an HDBSCAN object with the specified parameters
    scanner = cuml.cluster.hdbscan.HDBSCAN(min_cluster_size=HDBS_MIN_CLUSTERSIZE, min_samples=HDBS_MIN_SAMPLES, verbose=True, prediction_data=True)

    clusters = scanner.fit_transform(data)

    logger.info(f"Number of clusters: {len(np.unique(clusters))}")
    
    # Save the cluster results
    save_clusters_hdf5(clusters, CLUSTER_FILE)




def run_dbscan_partial_fit(HDBS_MIN_CLUSTERSIZE: int, HDBS_MIN_SAMPLES: int, DIMENSIONALITY_REDUCTION_FILE: str, CLUSTER_FILE: str, PARTIAL_FIT_CLUSTER: float):
    # Load the full dataset
    data = load_embeddings(DIMENSIONALITY_REDUCTION_FILE, "umap_coordinates")
    
    scanner = cuml.cluster.hdbscan.HDBSCAN(min_cluster_size=HDBS_MIN_CLUSTERSIZE, min_samples=HDBS_MIN_SAMPLES, verbose=True, prediction_data=True)

    logger.info(f"Fitting on size of data: {int(data.shape[0] * PARTIAL_FIT_CLUSTER):,}")

    batch_size = int(data.shape[0] * PARTIAL_FIT_CLUSTER)
    train_data = data[np.random.choice(data.shape[0], batch_size, replace=False)]
    clusterer = scanner.fit(train_data)

    logger.info("HDBSCAN model fitted successfully")
    
    # Prepare to collect batch results
    all_clusters = []
    all_probs = []
    
    # Process approximate_predict in batches
    for i in range(0, len(data), batch_size):
        print(f"Processing batch {i//batch_size + 1}/{len(data)//batch_size}", end="\r")
        batch_data = data[i:i+batch_size]
        clusters, probs = cuml.cluster.hdbscan.approximate_predict(clusterer, batch_data)
        all_clusters.append(clusters)

    final_clusters = np.concatenate(all_clusters)
    
    # Log the number of unique clusters
    logger.info(f"Number of clusters: {len(np.unique(final_clusters))}")
    
    # Save the cluster results
    save_clusters_hdf5(final_clusters, CLUSTER_FILE)




if __name__ == "__main__":

    run_function_with_overrides(run_dbscan_partial_fit, config)

    
