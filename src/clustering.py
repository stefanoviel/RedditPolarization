import os
import sys

# adding root directory to paths
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from logging_config import configure_get_logger
import config

from src.utils.function_runner import run_function_with_overrides
import h5py
from tqdm import tqdm
import numpy as np
import cuml
import time
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import dbcv

from src.utils.utils import load_h5py, save_h5py
from src.utils.function_runner import execute_with_gpu_logging
from sklearn.metrics import silhouette_score
from hdbscan import HDBSCAN

from cuml.common import logger
logger.set_level(logger.level_error)

if not os.path.exists(config.OUTPUT_DIR):
    os.makedirs(config.OUTPUT_DIR)
logger = configure_get_logger(config.OUTPUT_DIR, config.EXPERIMENT_NAME, log_level='INFO', executed_file_name = __file__)

import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)

os.environ["NUMEXPR_MAX_THREADS"] = "32"
import numexpr

os.environ['CUDA_VISIBLE_DEVICES']=str(1)


def DBCV(minimum_spanning_tree, labels, alpha=1.0):
        sizes = np.bincount(labels + 1)
        noise_size = sizes[0]
        cluster_size = sizes[1:]
        total = noise_size + np.sum(cluster_size)
        num_clusters = len(cluster_size)
        DSC = np.zeros(num_clusters)
        min_outlier_sep = np.inf  # only required if num_clusters = 1
        correction_const = 2  # only required if num_clusters = 1

        # Unltimately, for each Ci, we only require the
        # minimum of DSPC(Ci, Cj) over all Cj != Ci.
        # So let's call this value DSPC_wrt(Ci), i.e.
        # density separation 'with respect to' Ci.
        DSPC_wrt = np.ones(num_clusters) * np.inf
        max_distance = 0

        mst_df = minimum_spanning_tree.to_pandas()

        for edge in mst_df.iterrows():
            label1 = labels[int(edge[1]["from"])]
            label2 = labels[int(edge[1]["to"])]
            length = edge[1]["distance"]

            max_distance = max(max_distance, length)

            if label1 == -1 and label2 == -1:
                continue
            elif label1 == -1 or label2 == -1:
                # If exactly one of the points is noise
                min_outlier_sep = min(min_outlier_sep, length)
                continue

            if label1 == label2:
                # Set the density sparseness of the cluster
                # to the sparsest value seen so far.
                DSC[label1] = max(length, DSC[label1])
            else:
                # Check whether density separations with
                # respect to each of these clusters can
                # be reduced.
                DSPC_wrt[label1] = min(length, DSPC_wrt[label1])
                DSPC_wrt[label2] = min(length, DSPC_wrt[label2])

        # In case min_outlier_sep is still np.inf, we assign a new value to it.
        # This only makes sense if num_clusters = 1 since it has turned out
        # that the MR-MST has no edges between a noise point and a core point.
        min_outlier_sep = max_distance if min_outlier_sep == np.inf else min_outlier_sep

        # DSPC_wrt[Ci] might be infinite if the connected component for Ci is
        # an "island" in the MR-MST. Whereas for other clusters Cj and Ck, the
        # MR-MST might contain an edge with one point in Cj and ther other one
        # in Ck. Here, we replace the infinite density separation of Ci by
        # another large enough value.
        #
        # TODO: Think of a better yet efficient way to handle this.
        correction = correction_const * (
            max_distance if num_clusters > 1 else min_outlier_sep
        )
        DSPC_wrt[np.where(DSPC_wrt == np.inf)] = correction

        V_index = [
            (DSPC_wrt[i] - DSC[i]) / max(DSPC_wrt[i], DSC[i])
            for i in range(num_clusters)
        ]
        score = np.sum(
            [(cluster_size[i] * V_index[i]) / total for i in range(num_clusters)]
        )

        # penalized for size of noise cluster (as sugested in the paper)
        return score * (total - noise_size) / total


def run_dbscan_full_data(HDBS_MIN_CLUSTERSIZE: int, HDBS_MIN_SAMPLES: int, DIMENSIONALITY_REDUCTION_FILE: str, CLUSTER_FILE: str):
    data = load_h5py(DIMENSIONALITY_REDUCTION_FILE, "data")
    
    scanner = cuml.cluster.hdbscan.HDBSCAN(min_cluster_size=HDBS_MIN_CLUSTERSIZE, min_samples=HDBS_MIN_SAMPLES)
    clusters = scanner.fit_predict(data)
    logger.info(f"Number of clusters: {len(np.unique(clusters))}")

    score = dbcv.dbcv(data, clusters)
    print('SCORE', score)
    
    save_h5py(clusters, CLUSTER_FILE, 'data')


def run_dbscan_partial_fit(HDBS_MIN_CLUSTERSIZE: int, HDBS_MIN_SAMPLES: int, DIMENSIONALITY_REDUCTION_FILE: str, CLUSTER_FILE: str, PARTIAL_FIT_CLUSTER: float):
    # Load the full dataset
    data = load_h5py(DIMENSIONALITY_REDUCTION_FILE, "data")
    
    scanner = cuml.cluster.hdbscan.HDBSCAN(min_cluster_size=HDBS_MIN_CLUSTERSIZE, min_samples=HDBS_MIN_SAMPLES, prediction_data=True)

    logger.info(f"Fitting on size of data: {int(data.shape[0] * PARTIAL_FIT_CLUSTER):,}")

    batch_size = int(data.shape[0] * PARTIAL_FIT_CLUSTER)
    train_data = data[np.random.choice(data.shape[0], batch_size, replace=False)]
    clusterer = execute_with_gpu_logging(scanner.fit, train_data)

    logger.info("HDBSCAN model fitted successfully")
    
    # Prepare to collect batch results
    all_clusters = []
    all_probs = []
    
    # Process approximate_predict in batches
    for i in range(0, len(data), batch_size):
        print(f"Processing batch {i//batch_size + 1}/{len(data)//batch_size}", end="\r")
        batch_data = data[i:i+batch_size]
        clusters, probs = execute_with_gpu_logging(cuml.cluster.hdbscan.approximate_predict, clusterer, batch_data)
        all_clusters.append(clusters)

    final_clusters = np.concatenate(all_clusters)
    
    # Log the number of unique clusters
    logger.info(f"Number of clusters: {len(np.unique(final_clusters))}")
    
    # Save the cluster results
    save_h5py(final_clusters, CLUSTER_FILE, 'data')



def search_best_dbcv(data: np.ndarray, HDBS_MIN_CLUSTERSIZE_PERCENTAGE_SEARCH: list, HDBS_MIN_SAMPLES_SEARCH: list):
    """Seach for the best min_cluster_size and min_samples for HDBSCAN using DBCV"""

    data_size = len(data)
    logger.info(f"Data size: {data_size}")
    best_params = {'min_cluster_size': None, 'min_samples': None, 'dbcv': -1}
    DBCV_scores = []

    for min_cluster_size_percentage in HDBS_MIN_CLUSTERSIZE_PERCENTAGE_SEARCH:
        for min_elements_core_points in HDBS_MIN_SAMPLES_SEARCH:
            min_cluster_size = int(data_size * min_cluster_size_percentage)

            # Check if min_cluster_size violates constraints
            if min_cluster_size < 2 or min_cluster_size > data_size or min_elements_core_points > data_size:
                
                print(f"min_cluster_size: {min_cluster_size}, min_samples: {min_elements_core_points}")
                logger.info(f"Skipping combination: min_cluster_size={min_cluster_size}, min_samples={min_elements_core_points}")
                continue  # Skip this iteration

            scanner = cuml.cluster.hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_elements_core_points, gen_min_span_tree=True)
            clusters = scanner.fit_predict(data)
            
            dbcv = DBCV(scanner.minimum_spanning_tree_, clusters) 
            percentage_non_noise = np.mean(clusters != -1)

            if dbcv > best_params['dbcv']:
                best_params.update({
                    'min_cluster_size': min_cluster_size,
                    'min_samples': min_elements_core_points,
                    'dbcv': dbcv
                })

            logger.info(f"min_cluster_size: {min_cluster_size}, min_samples: {min_elements_core_points}, "
                        f"Number of clusters found: {len(np.unique(clusters))}, dbcv: {dbcv}, "
                        f"percentage_non_noise: {percentage_non_noise}")
            DBCV_scores.append(dbcv)

    logger.info(f"Best min_cluster_size: {best_params['min_cluster_size']}, "
                f"Best min_samples: {best_params['min_samples']}, Best dbcv: {best_params['dbcv']}")

    return best_params, DBCV_scores


def hdbscan_cluster_data(PROCESSED_REDDIT_DATA: str, DIMENSIONALITY_REDUCTION_DB_NAME:str, CLUSTER_DB_NAME: str, HDBS_MIN_CLUSTERSIZE_PERCENTAGE_SEARCH: list, HDBS_MIN_SAMPLES_SEARCH: list):
    data = load_h5py(PROCESSED_REDDIT_DATA, DIMENSIONALITY_REDUCTION_DB_NAME)

    best_params, DBCV_scores = search_best_dbcv(data, HDBS_MIN_CLUSTERSIZE_PERCENTAGE_SEARCH, HDBS_MIN_SAMPLES_SEARCH)

    scanner = cuml.cluster.hdbscan.HDBSCAN(min_cluster_size=best_params['min_cluster_size'], min_samples=best_params['min_samples'])
    clusters = scanner.fit_predict(data)
    save_h5py(clusters, PROCESSED_REDDIT_DATA, CLUSTER_DB_NAME)

    return DBCV_scores


def apply_clustering_existing_clusters(PROCESSED_REDDIT_DATA:str, DIMENSIONALITY_REDUCTION_DB_NAME: str, CLUSTER_DB_NAME: str, SUBCLUSTER_DB_NAME: str):
    data = load_h5py(PROCESSED_REDDIT_DATA, DIMENSIONALITY_REDUCTION_DB_NAME)
    clusters = load_h5py(PROCESSED_REDDIT_DATA, CLUSTER_DB_NAME)

    all_sub_clusters = np.full_like(clusters, -1)  # Initialize with -1 for noise
    max_label_used = -1

    for cluster in np.unique(clusters):
        if cluster == -1:
            continue

        cluster_data = data[clusters == cluster]
        print(f"Cluster {cluster}: {cluster_data.shape[0]}")

        best_params, DBCV_scores = search_best_dbcv(cluster_data)

        scanner = cuml.cluster.hdbscan.HDBSCAN(min_cluster_size=best_params['min_cluster_size'], min_samples=best_params['min_samples'])
        sub_clusters = scanner.fit_predict(cluster_data)

        # Ensure unique labels across different parent clusters
        unique_sub_clusters = sub_clusters + max_label_used + 1
        max_label_used = np.max(unique_sub_clusters)
        all_sub_clusters[clusters == cluster] = unique_sub_clusters

    save_h5py(all_sub_clusters, PROCESSED_REDDIT_DATA, SUBCLUSTER_DB_NAME)




if __name__ == "__main__":

    run_function_with_overrides(hdbscan_cluster_data, config)
    # run_function_with_overrides(apply_clustering_existing_clusters, config)

    
