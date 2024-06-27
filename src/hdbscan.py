import os
import sys

# adding root directory to paths
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from logging_config import configure_get_logger
import config

if not os.path.exists(config.OUTPUT_DIR):
    os.makedirs(config.OUTPUT_DIR)
logger = configure_get_logger(config.OUTPUT_DIR, config.EXPERIMENT_NAME, executed_file_name = __file__)

from src.utils_run_single_step import run_function_with_overrides
import h5py
from tqdm import tqdm
import numpy as np
import cuml
import time


from src.utils import load_embeddings


def save_clusters_hdf5(clusters, file_name):
    """Save clusters data to an HDF5 file."""
    with h5py.File(file_name, 'w') as f:
        f.create_dataset('clusters', data=clusters)


def run_dbscan(HDBS_MIN_CLUSTERSIZE:int, HDBS_MIN_SAMPLES:int, DIMENSIONALITY_REDUCTION_FILE:str, CLUSTER_FILE:str):

    data = load_embeddings(DIMENSIONALITY_REDUCTION_FILE, "umap_coordinates")
    scanner = cuml.cluster.hdbscan.HDBSCAN(min_cluster_size=HDBS_MIN_CLUSTERSIZE, min_samples=HDBS_MIN_SAMPLES)

    clusters = scanner.fit_predict(data)
    save_clusters_hdf5(clusters, CLUSTER_FILE)


if __name__ == "__main__":

    run_function_with_overrides(run_dbscan, config)

    
