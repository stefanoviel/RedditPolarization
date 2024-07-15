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
logger = configure_get_logger(config.OUTPUT_DIR, config.EXPERIMENT_NAME, log_level='INFO', executed_file_name = __file__)

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

from src.utils.utils import load_embeddings
from src.utils.function_runner import execute_with_gpu_logging
from sklearn.metrics import silhouette_score
from hdbscan import HDBSCAN

import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)

os.environ["NUMEXPR_MAX_THREADS"] = "32"
import numexpr

os.environ['CUDA_VISIBLE_DEVICES']=str(1)


def save_clusters_hdf5(clusters, file_name):
    """Save clusters data to an HDF5 file."""
    print("saving clusters to", file_name)
    with h5py.File(file_name, 'w') as f:
        f.create_dataset('data', data=clusters)


def run_dbscan_full_data(HDBS_MIN_CLUSTERSIZE: int, HDBS_MIN_SAMPLES: int, DIMENSIONALITY_REDUCTION_FILE: str, CLUSTER_FILE: str):
    data = load_embeddings(DIMENSIONALITY_REDUCTION_FILE, "data")
    
    scanner = cuml.cluster.hdbscan.HDBSCAN(min_cluster_size=HDBS_MIN_CLUSTERSIZE, min_samples=HDBS_MIN_SAMPLES)
    clusters = scanner.fit_predict(data)
    logger.info(f"Number of clusters: {len(np.unique(clusters))}")

    score = dbcv.dbcv(data, clusters)
    print('SCORE', score)
    
    save_clusters_hdf5(clusters, CLUSTER_FILE)


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

        # penalized for size of noise cluster
        return score - alpha * noise_size / total

def test_dbcv():
    data = np.random.rand(10000, 2)

    for min_cluster_size in [10, 20]: 
        for min_samples in [5, 10]: 
            scanner = cuml.cluster.hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, gen_min_span_tree=True)
            clusters = scanner.fit_predict(data)
            print(DBCV(scanner.minimum_spanning_tree_, clusters))



def run_dbscan_partial_fit(HDBS_MIN_CLUSTERSIZE: int, HDBS_MIN_SAMPLES: int, DIMENSIONALITY_REDUCTION_FILE: str, CLUSTER_FILE: str, PARTIAL_FIT_CLUSTER: float):
    # Load the full dataset
    data = load_embeddings(DIMENSIONALITY_REDUCTION_FILE, "data")
    
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
    save_clusters_hdf5(final_clusters, CLUSTER_FILE)


def plot_silhouette_heatmap(silhouette_scores):

    # Convert to DataFrame
    print('sil', silhouette_scores)
    df = pd.DataFrame(silhouette_scores, columns=['min_cluster_size', 'min_samples', 'silhouette'])

    # Pivot table for heatmap
    pivot_table = df.pivot(index='min_cluster_size', columns='min_samples', values='silhouette')
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_table, annot=True, cmap="YlGnBu", fmt=".3f")
    plt.title('Silhouette Scores for HDBSCAN Hyperparameter Combinations')
    plt.xlabel('min_samples')
    plt.ylabel('min_cluster_size')
    
    # Save the plot
    plt.savefig('output/silhouette_heatmap.png')
    plt.close()

    

def run_hdbscan_search_best_dbcv(DIMENSIONALITY_REDUCTION_FILE: str, CLUSTER_FILE: str):
    data = load_embeddings(DIMENSIONALITY_REDUCTION_FILE, "umap_coordinates")
    data_size = len(data)
    print('data size', data_size)

    best_min_cluster_size = None
    best_min_samples = None
    best_dbcv = -1
    DBCV_scores = []

    for min_cluster_size in [data_size//100, data_size//1000, data_size//10000]:
        for min_samples in [5, 10, 30]:

            
            scanner = cuml.cluster.hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, gen_min_span_tree=True)
            clusters = scanner.fit_predict(data)
            
            for alpha in [0.1, 0.5, 1.0]:
                dbcv = DBCV(scanner.minimum_spanning_tree_, clusters, alpha=alpha)
                DBCV_scores.append(dbcv)

                percentage_non_noise = len(clusters[clusters != -1]) / len(clusters)
                if dbcv > best_dbcv and percentage_non_noise > 0.8: # only consider if more than 80% of data is not noise
                    best_min_cluster_size = min_cluster_size
                    best_min_samples = min_samples
                    best_dbcv = dbcv

                logger.info(f"min_cluster_size: {min_cluster_size}, min_samples: {min_samples}, Number of clusters found: {len(np.unique(clusters))}, dbcv: {dbcv}, alpha: {alpha}, percentage_non_noise: {percentage_non_noise}")

    logger.info(f"Best min_cluster_size: {best_min_cluster_size}, Best min_samples: {best_min_samples}, Best dbcv: {best_dbcv}")
    scanner = cuml.cluster.hdbscan.HDBSCAN(min_cluster_size=best_min_cluster_size, min_samples=best_min_samples)
    clusters = scanner.fit_predict(data)

    # TODO: plot heatmap of scores
        
    save_clusters_hdf5(clusters, CLUSTER_FILE)

    return DBCV_scores


if __name__ == "__main__":

    # run_dbscan_full_data(config.DIMENSIONALITY_REDUCTION_FILE, config.CLUSTER_FILE)

    run_function_with_overrides(run_hdbscan_search_best_dbcv, config)




    
