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
import seaborn as sns
import pandas as pd

from src.utils.utils import load_h5py, save_h5py, get_indices_for_random_h5py_subset, load_with_indices_h5py
from src.utils.function_runner import execute_with_gpu_logging
from sklearn.metrics import silhouette_score
# from hdbscan import HDBSCAN
from cuml.cluster.hdbscan import HDBSCAN

from cuml.common import logger
logger.set_level(logger.level_error)

if not os.path.exists(config.OUTPUT_DIR):
    os.makedirs(config.OUTPUT_DIR)
logger = configure_get_logger(config.OUTPUT_DIR, config.EXPERIMENT_NAME, log_level='INFO', executed_file_name = __file__)


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


def run_dbscan_partial_fit(scanner, PROCESSED_REDDIT_DATA: str, DIMENSIONALITY_REDUCTION_DB_NAME:str, PARTIAL_FIT_CLUSTER: float):
    # Load the full dataset
    data = load_h5py(PROCESSED_REDDIT_DATA, DIMENSIONALITY_REDUCTION_DB_NAME)
    

    logger.info(f"Fitting on size of data: {int(data.shape[0] * PARTIAL_FIT_CLUSTER):,}")

    batch_size = int(data.shape[0] * PARTIAL_FIT_CLUSTER)
    train_data = data[np.random.choice(data.shape[0], batch_size, replace=False)]
    clusterer = execute_with_gpu_logging(scanner.fit, train_data)

    labels = clusterer.labels_

    # Determine the number of clusters (excluding noise, which is labeled as -1)
    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    
    logger.info(f"Number of clusters found: {num_clusters}")

    all_clusters = []
    for i in tqdm(range(0, len(data), batch_size)):
        batch_data = data[i:i+batch_size]
        clusters, probs = execute_with_gpu_logging(cuml.cluster.hdbscan.approximate_predict, clusterer, batch_data)
        all_clusters.append(clusters)

    final_clusters = np.concatenate(all_clusters)
    logger.info(f"Number of clusters: {len(np.unique(final_clusters))}, shape: {final_clusters.shape}")
    
    return final_clusters


def search_best_dbcv(data: np.ndarray, HDBS_MIN_CLUSTERSIZE_SEARCH: list, HDBS_MIN_SAMPLES_SEARCH: list, PARTIAL_FIT_CLUSTER:float, PROCESSED_REDDIT_DATA: str, DIMENSIONALITY_REDUCTION_DB_NAME: str, CLUSTER_DB_NAME: str):
    """Search for the best min_cluster_size and min_samples for HDBSCAN using DBCV"""


    data_size = len(data)
    logger.info(f"Data size: {data_size}")
    best_params = {'min_cluster_size': None, 'min_samples': None, 'dbcv': -1, 'percentage_non_noise': -1}
    DBCV_scores = []

    for min_cluster_size in HDBS_MIN_CLUSTERSIZE_SEARCH:
        for min_elements_core_points in HDBS_MIN_SAMPLES_SEARCH:

            # Check if min_cluster_size violates constraints
            if min_cluster_size < 2 or min_cluster_size > data_size or min_elements_core_points > data_size:
                logger.info(f"Skipping combination: min_cluster_size={min_cluster_size}, min_samples={min_elements_core_points}")
                continue  # Skip this iteration

            scanner = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_elements_core_points, gen_min_span_tree=True, prediction_data=True)
            if PARTIAL_FIT_CLUSTER == 1.0:
                clusters = scanner.fit_predict(data)
            else:
                clusters = run_dbscan_partial_fit(scanner, PROCESSED_REDDIT_DATA, DIMENSIONALITY_REDUCTION_DB_NAME, PARTIAL_FIT_CLUSTER)

            dbcv = DBCV(scanner.minimum_spanning_tree_, clusters) 
            percentage_non_noise = np.mean(clusters != -1)

            if dbcv > best_params['dbcv']:
                best_params.update({
                    'min_cluster_size': min_cluster_size,
                    'min_samples': min_elements_core_points,
                    'dbcv': dbcv, 
                    'percentage_non_noise': percentage_non_noise
                })

            logger.info(f"min_cluster_size: {min_cluster_size}, min_samples: {min_elements_core_points}, "
                        f"Number of clusters found: {len(np.unique(clusters))}, dbcv: {dbcv}, "
                        f"percentage_non_noise: {percentage_non_noise}")
            DBCV_scores.append(dbcv)



    return best_params, DBCV_scores

def save_cluster_centroids(PROCESSED_REDDIT_DATA: str, DIMENSIONALITY_REDUCTION_DB_NAME: str, CLUSTER_DB_NAME: str, CENTROIDS_DB_NAME: str):
    data = load_h5py(PROCESSED_REDDIT_DATA, DIMENSIONALITY_REDUCTION_DB_NAME)
    clusters = load_h5py(PROCESSED_REDDIT_DATA, CLUSTER_DB_NAME)

    max_cluster_id = np.max(clusters)  # Get the maximum cluster ID
    centroids = np.zeros((max_cluster_id + 1, data.shape[1]))  # Create a centroids array with size based on max cluster ID

    for cluster in range(max_cluster_id + 1):
        if cluster == -1:  # Skip noise
            continue

        cluster_data = data[clusters == cluster]
        if cluster_data.size > 0:  # Check if the cluster has any data points
            centroids[cluster] = np.mean(cluster_data, axis=0)

    save_h5py(centroids, PROCESSED_REDDIT_DATA, CENTROIDS_DB_NAME)



def hdbscan_cluster_data(PROCESSED_REDDIT_DATA: str, DIMENSIONALITY_REDUCTION_DB_NAME: str, CLUSTER_DB_NAME: str, HDBS_MIN_CLUSTERSIZE_SEARCH: list, HDBS_MIN_SAMPLES_SEARCH: list, PARTIAL_FIT_CLUSTER: float, CENTROIDS_DB_NAME: str):
    data = load_h5py(PROCESSED_REDDIT_DATA, DIMENSIONALITY_REDUCTION_DB_NAME)
    print("data shape", data.shape)

    if len(HDBS_MIN_CLUSTERSIZE_SEARCH) == 1 and len(HDBS_MIN_SAMPLES_SEARCH) == 1:  # no search needs to be done
        best_params = {'min_cluster_size': int(HDBS_MIN_CLUSTERSIZE_SEARCH[0]), 'min_samples': int(HDBS_MIN_SAMPLES_SEARCH[0])}
    else:
        best_params, DBCV_scores = search_best_dbcv(data, HDBS_MIN_CLUSTERSIZE_SEARCH, HDBS_MIN_SAMPLES_SEARCH, PARTIAL_FIT_CLUSTER, PROCESSED_REDDIT_DATA, DIMENSIONALITY_REDUCTION_DB_NAME, CLUSTER_DB_NAME)

    
    scanner = HDBSCAN(min_cluster_size=best_params['min_cluster_size'], min_samples=best_params['min_samples'], prediction_data=True)
    if PARTIAL_FIT_CLUSTER == 1.0:
        clusters = scanner.fit_predict(data)
    else:
        clusters = run_dbscan_partial_fit(scanner, PROCESSED_REDDIT_DATA, DIMENSIONALITY_REDUCTION_DB_NAME, PARTIAL_FIT_CLUSTER)

    percentage_non_noise = np.mean(clusters != -1)

    logger.info(f"number of clusters: {len(np.unique(clusters))}, percentage of non-noise: {percentage_non_noise}")

    save_h5py(clusters, PROCESSED_REDDIT_DATA, CLUSTER_DB_NAME)

    save_cluster_centroids(PROCESSED_REDDIT_DATA, DIMENSIONALITY_REDUCTION_DB_NAME, CLUSTER_DB_NAME, CENTROIDS_DB_NAME)


def apply_clustering_existing_clusters(PROCESSED_REDDIT_DATA:str, DIMENSIONALITY_REDUCTION_DB_NAME: str, CLUSTER_DB_NAME: str, SUBCLUSTER_DB_NAME: str, HDBS_MIN_CLUSTERSIZE_SEARCH: list, HDBS_MIN_SAMPLES_SEARCH: list, PARTIAL_FIT_CLUSTER: float):
    data = load_h5py(PROCESSED_REDDIT_DATA, DIMENSIONALITY_REDUCTION_DB_NAME)
    clusters = load_h5py(PROCESSED_REDDIT_DATA, CLUSTER_DB_NAME)

    all_sub_clusters = np.full_like(clusters, -1)  # Initialize with -1 for noise
    max_label_used = -1

    for n, cluster in enumerate(np.unique(clusters)):

        if cluster == -1:
            continue

        cluster_data = data[clusters == cluster]
        print(f"Cluster {cluster}: {cluster_data.shape[0]}")

        if len(HDBS_MIN_CLUSTERSIZE_SEARCH) == 1 and len(HDBS_MIN_SAMPLES_SEARCH) == 1:  # no search needs to be done
            best_params = {'min_cluster_size': int(HDBS_MIN_CLUSTERSIZE_SEARCH[0]), 'min_samples': int(HDBS_MIN_SAMPLES_SEARCH[0])}
        else:
            best_params, DBCV_scores = search_best_dbcv(data, HDBS_MIN_CLUSTERSIZE_SEARCH, HDBS_MIN_SAMPLES_SEARCH, PARTIAL_FIT_CLUSTER, PROCESSED_REDDIT_DATA, DIMENSIONALITY_REDUCTION_DB_NAME, CLUSTER_DB_NAME)

        scanner = cuml.cluster.hdbscan.HDBSCAN(min_cluster_size=best_params['min_cluster_size'], min_samples=best_params['min_samples'])
        sub_clusters = scanner.fit_predict(cluster_data)

        print(f"Number of subclusters: {len(np.unique(sub_clusters))}")


        # Ensure unique labels across different parent clusters
        valid_sub_clusters = sub_clusters[sub_clusters != -1]
        unique_valid_sub_clusters = valid_sub_clusters + max_label_used + 1
        max_label_used = np.max(unique_valid_sub_clusters) if unique_valid_sub_clusters.size > 0 else max_label_used
        
        # Apply back only to non-noise points
        all_sub_clusters[(clusters == cluster) & (sub_clusters != -1)] = unique_valid_sub_clusters


    save_h5py(all_sub_clusters, PROCESSED_REDDIT_DATA, SUBCLUSTER_DB_NAME)


if __name__ == "__main__":

    run_function_with_overrides(apply_clustering_existing_clusters, config)


    
