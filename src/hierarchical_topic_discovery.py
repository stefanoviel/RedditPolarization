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
from gensim.models.coherencemodel import CoherenceModel
from scipy.spatial.distance import cdist
from gensim.test.utils import common_corpus, common_dictionary
import gensim.downloader as api
from gensim.corpora import Dictionary
from gensim.utils import simple_preprocess

from src.create_graph import load_graph
from src.tf_idf import get_cluster_posts, get_important_words
from src.utils.utils import create_database_connection, load_json, load_h5py
import leidenalg
from tqdm import tqdm
import cuml


def apply_leiden(graph, resolution_parameter: float) -> list[list[int]]:
    """Apply the Leiden algorithm to partition a graph."""
    partition = leidenalg.find_partition(graph, leidenalg.RBConfigurationVertexPartition, resolution_parameter=resolution_parameter)
    return partition


def apply_different_resolution_parameters(graph, resolution_parameters: list[float]) -> list[list[int]]:
    """Apply the Leiden algorithm to partition a graph with different resolution parameters."""
    partitions = []
    for resolution_parameter in resolution_parameters:
        partition = apply_leiden(graph, resolution_parameter)
        partitions.append(partition)
    return partitions


def merge_clusters(partitions, clusters):
    """Merge clusters that are part of the same partition."""

    for n_partition, partition in enumerate(partitions):
        clusters = [cluster if cluster not in partition else n_partition for cluster in clusters]

    return clusters


def main(ADJACENCY_MATRIX: str, REDDIT_DATA_DIR:str , TABLE_NAME:str, TFIDF_MAX_FEATURES:str, IDS_FILE:str, CLUSTER_FILE:str, RESOLUTION_PARAMETER:list) -> None:
    graph = load_graph(ADJACENCY_MATRIX)

    db_connection = create_database_connection(REDDIT_DATA_DIR, TABLE_NAME,  ["id", "title", "selftext"])
    wiki_corpus = api.load('wiki-english-20171001')

    texts = []
    for n, text in tqdm(enumerate(wiki_corpus)): 
        for string in text['section_texts']:
            texts.append(simple_preprocess(string))
        if n > 10: # TODO: remove, I'm considering only the first 10 corpus from wikipedia to speed up the process
            break
    
    dictionary = Dictionary(texts)
    
    for resolution_parameter in RESOLUTION_PARAMETER:
        print(f"Resolution parameter: {resolution_parameter}")
        partitions = apply_leiden(graph, resolution_parameter=resolution_parameter)  
        with h5py.File(CLUSTER_FILE, 'r') as cluster_file:
            clusters = cluster_file['data'][:]
        ids = load_json(IDS_FILE)

        merged_clusters = merge_clusters(partitions, clusters)

        topics = []
        for cluster, posts in get_cluster_posts(db_connection, ids, merged_clusters, TABLE_NAME):
            important_words = [word for word, _ in get_important_words(posts, max_features=TFIDF_MAX_FEATURES)]
            topics.append(important_words)
            print(f"Cluster {cluster}: {important_words[:10]}")


        cm = CoherenceModel(topics=topics, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence = cm.get_coherence()
        print(f"resolution_parameter: {resolution_parameter}, coherence: {coherence}")  

def cluster_centroids(CENTROID_POISITION: str, TFIDF_FILE : str) -> None:

    centroids = load_h5py(CENTROID_POISITION, "centroids")
    unique_clusters = load_h5py(CENTROID_POISITION, "unique_clusters")

    reducer = cuml.UMAP(
        n_neighbors=3, n_components=2, min_dist=0
    )
    coordinates = reducer.fit_transform(centroids)

    plt.scatter(coordinates[:, 0], coordinates[:, 1])
    for i, txt in enumerate(unique_clusters):
        plt.annotate(txt, (coordinates[i, 0], coordinates[i, 1]))
    #save the plot
    plt.savefig('output/centroid_clusters.png')

    scanner = cuml.cluster.hdbscan.HDBSCAN(min_cluster_size=1, min_samples=1)
    clusters = scanner.fit_predict(coordinates)

    for original_cluster, new_cluster in zip(unique_clusters, clusters):
        print(f"Cluster {original_cluster} -> {new_cluster}")
        

if __name__ == "__main__":
    run_function_with_overrides(cluster_centroids, config)
