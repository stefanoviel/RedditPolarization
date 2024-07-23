import os
import sys

# adding root directory to paths
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from logging_config import configure_get_logger
import config
import h5py
import duckdb
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text
import numpy as np
import cuml
import pandas as pd
from tqdm import tqdm
from src.utils.function_runner import run_function_with_overrides, execute_with_gpu_logging
from src.utils.utils import create_database_connection, load_json, save_h5py, load_h5py, save_json

from cuml.common import logger
logger.set_level(logger.level_error)

if not os.path.exists(config.OUTPUT_DIR):
    os.makedirs(config.OUTPUT_DIR)
logger = configure_get_logger(config.OUTPUT_DIR, config.EXPERIMENT_NAME, executed_file_name = __file__, log_level='INFO')


def get_cluster_posts(con: duckdb.DuckDBPyConnection, ids:list, cluster_of_all_posts:list, TABLE_NAME:str):
    """
    Yield the title and selftext for all the posts in each cluster by executing a database query for each cluster.
    """
    # Map IDs to their clusters, assuming clusters and IDs are in the same order
    cluster_to_ids = {}
    for id, cluster in zip(ids, cluster_of_all_posts):
        if cluster not in cluster_to_ids:
            cluster_to_ids[cluster] = []
        cluster_to_ids[cluster].append(id)

    # Execute a query for each cluster and yield results
    for cluster, cluster_ids in cluster_to_ids.items():
        placeholders = ','.join(['?'] * len(cluster_ids))  # Prepare placeholders for SQL query
        query = f"SELECT title, selftext FROM {TABLE_NAME} WHERE id IN ({placeholders})"
        cursor = con.execute(query, cluster_ids)
        posts = cursor.fetchall()
        yield cluster, posts

def extract_top_words(tfidf_matrix, feature_names, unique_clusters, top_n=10):
    """Extract top words for each document (cluster in our case) from the tfidf matrix."""
    top_words_per_document = {}
    for cluster_index in tqdm(range(tfidf_matrix.shape[0])):
        row = tfidf_matrix.getrow(cluster_index)
        indices = row.indices
        data = row.data
        top_indices = np.argpartition(data, -top_n)[-top_n:]
        top_indices_sorted = top_indices[np.argsort(data[top_indices])[::-1]]
        original_indices = indices[top_indices_sorted]
        top_features = [feature_names[ind] for ind in original_indices]
        cluster_key = str(unique_clusters[cluster_index])
        top_words_per_document[cluster_key] = top_features

    return top_words_per_document


def compute_adjacency_matrix(tfidf_matrix, all_clusters):
    """Compute the adjacency matrix fo the cosine similarity between all clusters."""
    adjacency_matrix = np.zeros((len(all_clusters), len(all_clusters)))
    for i in tqdm(range(len(all_clusters))):
        for j in range(i + 1, len(all_clusters)):
            i_row = tfidf_matrix.getrow(i)
            j_row = tfidf_matrix.getrow(j)
            similarity = i_row.dot(j_row.T).toarray()[0][0]
            adjacency_matrix[i][j] = similarity
            adjacency_matrix[j][i] = similarity
    return adjacency_matrix


def prepare_documents(iterator_cluster_posts: iter) -> pd.Series:
    """
    Generate and return a list of document strings aggregated by cluster and a list of corresponding cluster identifiers.

    Parameters:
    - con (Connection): The database connection object.
    - ids (list[int]): the ids of all the post.
    - clusters (list[int]): the cluster to which each post belongs to. 
    - table_name (str): The name of the table from which to retrieve the post data.

    Returns:
    - pd.Series: A pandas Series where each element is a string consisting of concatenated titles and selftexts 
                 from posts within the same cluster.
    - list[int]: A list of integers representing the unique cluster identifiers for which documents were generated.
    """
    all_text_per_cluster = []
    unique_clusters = []
    for cluster, posts in tqdm(iterator_cluster_posts):
        unique_clusters.append(int(cluster))
        cluster_words = " ".join([title + " " + selftext for title, selftext in posts])
        all_text_per_cluster.append(cluster_words)
    return pd.Series(all_text_per_cluster), unique_clusters

def load_data(REDDIT_DATA_DIR, PROCESSED_REDDIT_DATA, TABLE_NAME, CLUSTER_DB_NAME, IDS_DB_NAME):
    # Create a database connection
    con = create_database_connection(REDDIT_DATA_DIR, TABLE_NAME, ["id", "title", "selftext"])
    ids = load_h5py(PROCESSED_REDDIT_DATA, IDS_DB_NAME)
    clusters = load_h5py(PROCESSED_REDDIT_DATA, CLUSTER_DB_NAME)
    return con, ids, clusters

def TF_IDF_matrix(documents:pd.Series, TFIDF_MAX_FEATURES:str):
    my_stop_words = list(text.ENGLISH_STOP_WORDS.union(["https", "com", "www", "ve", "http", "amp"]))
    tfidf_vectorizer = cuml.feature_extraction.text.TfidfVectorizer(stop_words=my_stop_words, lowercase=True,  max_features=TFIDF_MAX_FEATURES)
    tfidf_matrix = execute_with_gpu_logging(tfidf_vectorizer.fit_transform, documents)
    feature_names = tfidf_vectorizer.get_feature_names()  # Get all feature names from the vectorizer

    return tfidf_matrix, feature_names

def main(REDDIT_DATA_DIR:str, PROCESSED_REDDIT_DATA:str, TABLE_NAME:str, CLUSTER_DB_NAME:str, IDS_DB_NAME:str, TFIDF_MAX_FEATURES:str, TFIDF_FILE:str, ADJACENCY_MATRIX:str, TFIDF_WORDS_PER_CLUSTER:int):
    """Main function to compute the TF-IDF matrix and adjacency matrix."""

    con, ids, clusters = load_data(REDDIT_DATA_DIR, PROCESSED_REDDIT_DATA, TABLE_NAME, CLUSTER_DB_NAME, IDS_DB_NAME)
    iterator_cluster_documents = get_cluster_posts(con, ids, clusters, TABLE_NAME)
    documents, unique_clusters = prepare_documents(iterator_cluster_documents)
    tfidf_matrix, feature_names = TF_IDF_matrix(documents, TFIDF_MAX_FEATURES)

    # only needed for hierarchical clustering
    # adjacency_matrix = compute_adjacency_matrix(tfidf_matrix, unique_clusters)
    # save_h5py(adjacency_matrix, ADJACENCY_MATRIX, "data")   

    top_words_per_document = extract_top_words(tfidf_matrix, feature_names, unique_clusters, TFIDF_WORDS_PER_CLUSTER)
    save_json(top_words_per_document, TFIDF_FILE)


if __name__ == "__main__":

    # config.CLUSTER_FILE = config.SUBCLUSTER_FILE
    # config.TFIDF_FILE = config.SUBCLUSTER_TFIDF_FILE
    print("Total running time:", run_function_with_overrides(main, config))



    