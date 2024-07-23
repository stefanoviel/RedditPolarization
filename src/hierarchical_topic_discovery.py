import os
import sys

# adding root directory to paths
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from logging_config import configure_get_logger
import config
from collections import Counter
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

from src.tf_idf import prepare_documents, TF_IDF_matrix, extract_top_words, get_cluster_posts
from src.utils.utils import create_database_connection, load_json, load_h5py
import igraph as ig
import leidenalg
from tqdm import tqdm
import cuml
import pickle
from gensim import corpora

import numpy as np
import duckdb
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import fcluster

from sklearn.metrics import silhouette_score

def hierarchical_topics_from_similarity(similarity_matrix, topic_labels, linkage_method='average'):
    """Generate a hierarchy of topics based on a cosine similarity matrix.

    Arguments:
        similarity_matrix (np.ndarray): A square matrix where each element represents the cosine similarity between topics.
        topic_labels (list of str): Labels for each topic corresponding to the indices of the similarity matrix.
        linkage_method (str): The linkage algorithm to use ('single', 'complete', 'average', 'weighted', 'centroid', 'median', 'ward').

    Returns:
        pd.DataFrame: A DataFrame that contains a hierarchy of topics represented by their parents and their children.
    """
    distance_matrix = 1 - similarity_matrix
    np.fill_diagonal(distance_matrix, 0)
    condensed_distance_matrix = squareform(distance_matrix)
    Z = linkage(condensed_distance_matrix, method=linkage_method)

    return Z

def get_all_posts(con: duckdb.DuckDBPyConnection, ids:list, clusters:list, TABLE_NAME:str):
    query = f"SELECT title, selftext FROM {TABLE_NAME}"
    cursor = con.execute(query)
    posts = cursor.fetchall()
    return posts

def get_topics(REDDIT_DATA_DIR, TABLE_NAME, TFIDF_MAX_FEATURES, IDS_FILE, cluster_of_all_posts):
    
    db_connection = create_database_connection(REDDIT_DATA_DIR, TABLE_NAME,  ["id", "title", "selftext"])
    ids = load_json(IDS_FILE)

    iterator_cluster_documents = get_cluster_posts(db_connection, ids, cluster_of_all_posts, TABLE_NAME)
    documents, unique_clusters = prepare_documents(iterator_cluster_documents)
    tfidf_matrix, feature_names = TF_IDF_matrix(documents, TFIDF_MAX_FEATURES)
    print(tfidf_matrix.shape)
    for row in tfidf_matrix:
        print(len(row.indices))

    top_words_per_document = extract_top_words(tfidf_matrix, feature_names, unique_clusters)

    # extract the top words for each cluster and create a list of string 
    topics_word = []
    for cluster in top_words_per_document:
        topics_word.append(top_words_per_document[cluster])
    return topics_word


def compute_global_topic_diversity(topics):
    all_words = [word for topic in topics for word in topic]
    word_counts = Counter(all_words)
    top_25_words = word_counts.most_common(25)
    top_words = [word for word, count in top_25_words]
    unique_top_words = len(set(top_words))
    diversity_score = unique_top_words / 25
    return diversity_score


def compute_topic_coherence(topics, con, ids, clusters, TABLE_NAME):

    texts = get_all_posts(con, ids, clusters, TABLE_NAME)
    processed_texts = map(simple_preprocess, texts)
    dictionary = Dictionary(processed_texts)
    
    coherence_model = CoherenceModel(topics=topics, dictionary=dictionary, texts=topics, coherence='c_v')
    coherence_score = coherence_model.get_coherence()
    return coherence_score


def find_optimal_threshold( Z, REDDIT_DATA_DIR, TABLE_NAME, TFIDF_MAX_FEATURES, IDS_FILE, max_clusters=10):
    """
    Find the optimal threshold to cut the dendrogram for hierarchical clustering by maximizing the silhouette score.
    
    Arguments:
    data -- np.ndarray, the original dataset used for clustering.
    Z -- np.ndarray, the linkage matrix obtained from hierarchical clustering.
    max_clusters -- int, maximum number of clusters to consider (defaults to 10).
    
    Returns:
    float, the threshold that maximizes the silhouette score.
    """
    ids = load_json(IDS_FILE)
    db_connection = create_database_connection(REDDIT_DATA_DIR, TABLE_NAME,  ["id", "title", "selftext"])
    clusters = load_h5py(config.CLUSTER_FILE, 'data')
    
    # Test a range of possible number of clusters from 2 up to max_clusters

    # start from half of the max cluster
    for num_clusters in range(10, min(max_clusters, 40)):
        # Obtain the flat clusters for the current number of clusters
        clusters = fcluster(Z, num_clusters, criterion='maxclust')
        print(f"Number of clusters: {num_clusters}")
        print(f"Clusters: {clusters}")

        topics = get_topics(REDDIT_DATA_DIR, TABLE_NAME, TFIDF_MAX_FEATURES, IDS_FILE, clusters)
        diversity_score = compute_global_topic_diversity(topics)
        # coherence_score = compute_topic_coherence(topics, db_connection, ids, clusters, TABLE_NAME)
        print(f"Number of clusters: {num_clusters}, Diversity Score: {diversity_score}")



def plot_dendrogram_and_elbow(Z):
    """
    Plot a dendrogram and the elbow curve to help identify the optimal clustering cutoff.
    
    Arguments:
    Z -- Output from the linkage function (linkage matrix).
    """
    # Create a figure with two subplots
    fig, ax = plt.subplots(2, 1, figsize=(12, 12))
    
    # Plot dendrogram
    optimal_threshold, largest_jump_index = find_optimal_threshold(Z)
    dendrogram(Z, ax=ax[0], color_threshold=optimal_threshold)
    ax[0].set_title('Hierarchical Clustering Dendrogram')
    ax[0].set_xlabel('Sample index')
    ax[0].set_ylabel('Distance')
    
    # Extract the distances from the linkage matrix
    distances = Z[:, 2]
    # distances = distances[::-1]  # reverse to have the largest merge at the start
    ax[1].plot(distances)
    ax[1].set_title('Elbow Plot for Choosing Optimal Cluster Cutoff')
    ax[1].set_xlabel('Step (Number of merges)')
    ax[1].set_ylabel('Linkage Distance')
    ax[1].grid(True)
    
    # Improve layout
    plt.tight_layout()
    plt.show()
    plt.savefig('elbow.png')



if __name__ == "__main__":
    # run_function_with_overrides(main, config)

    adjacency_matrix_file = load_h5py(config.ADJACENCY_MATRIX, 'data') 
    cluster_order = load_json(config.CLUSTER_ORDER)['cluster_order']
    cluster_id_per_post = load_h5py(config.CLUSTER_FILE, 'data')
    Z = hierarchical_topics_from_similarity(adjacency_matrix_file, cluster_order)
    thresholds = find_optimal_threshold(Z, config.REDDIT_DATA_DIR, config.TABLE_NAME, config.TFIDF_MAX_FEATURES, config.IDS_FILE, len(cluster_order))


