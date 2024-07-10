
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
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import pickle
import igraph as ig


def load_data(embeddings_file, clusters_file):
    with h5py.File(embeddings_file, 'r') as f:
        embeddings = np.array(f['data'])
    with h5py.File(clusters_file, 'r') as f:
        clusters = np.array(f['data'])
    return embeddings, clusters

def compute_centroids(embeddings, clusters):
    unique_clusters = np.unique(clusters) 
    centroids = np.array([embeddings[clusters == k].mean(axis=0) for k in unique_clusters])
    return centroids, unique_clusters

def compute_distances(centroids):
    return cdist(centroids, centroids, metric='euclidean')


def plot_and_save_graph(graph, filename):
    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(graph, seed=42)  # Positions for all nodes
    nx.draw_networkx_nodes(graph, pos, node_size=700)
    nx.draw_networkx_edges(graph, pos, width=1.0, alpha=0.5)
    labels = nx.get_edge_attributes(graph, 'weight')
    nx.draw_networkx_edge_labels(graph, pos, edge_labels={key: f"{val:.2f}" for key, val in labels.items()})
    nx.draw_networkx_labels(graph, pos, font_size=20, font_family='sans-serif')
    plt.title("Graph of Clusters with Distances")
    plt.axis('off')  # Turn off the axis
    plt.savefig(filename)
    plt.show()

def save_graph(graph, filename):
    pickle.dump(graph, open(filename, 'wb'))

def load_graph(adjacency_matrix_file):
    # load with h5
    with h5py.File(adjacency_matrix_file, 'r') as f:
        adjacency_matrix = np.array(f['data'])
    
    bool_adjacency = adjacency_matrix.astype(bool)
    graph = ig.Graph.Adjacency(bool_adjacency.tolist(), mode=ig.ADJ_UNDIRECTED)
    edge_weights = adjacency_matrix[bool_adjacency]
    graph.es['weight'] = edge_weights


    return graph

def save_distance_matrix(distances, filename):
    # save with h5
    with h5py.File(filename, 'w') as f:
        f.create_dataset('data', data=distances)

def create_save_graph(DIMENSIONALITY_REDUCTION_FILE, CLUSTER_FILE, ADJACENCY_MATRIX):
    embeddings, clusters = load_data(DIMENSIONALITY_REDUCTION_FILE, CLUSTER_FILE)
    new_clusters = clusters[clusters != -1]
    embeddings = embeddings[clusters != -1]

    centroids, cluster_ids = compute_centroids(embeddings, new_clusters)
    distances = compute_distances(centroids)
    save_distance_matrix(distances, ADJACENCY_MATRIX)


if __name__ == '__main__':
    run_function_with_overrides(create_save_graph, config)




