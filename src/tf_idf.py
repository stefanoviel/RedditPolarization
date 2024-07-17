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



def get_cluster_posts(con, ids, clusters, TABLE_NAME):
    """
    Yield the title and selftext for each cluster by executing a database query for each cluster.
    """
    # Map IDs to their clusters, assuming clusters and IDs are in the same order
    cluster_to_ids = {}
    for id, cluster in zip(ids, clusters):
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

def extract_top_words(tfidf_matrix, feature_names, all_clusters, top_n=10):
    """Extract top words for each document."""
    top_words_per_document = {}
    for doc_index in tqdm(range(tfidf_matrix.shape[0])):
        row = tfidf_matrix.getrow(doc_index)
        indices = row.indices
        data = row.data
        top_indices = np.argpartition(data, -top_n)[-top_n:]
        top_indices_sorted = top_indices[np.argsort(data[top_indices])[::-1]]
        original_indices = indices[top_indices_sorted]
        top_features = [feature_names[ind] for ind in original_indices]
        cluster_key = str(all_clusters[doc_index])
        if cluster_key not in top_words_per_document:
            top_words_per_document[cluster_key] = []
        top_words_per_document[cluster_key].append(top_features)
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


def prepare_documents(con, ids, clusters, table_name):
    """Generate a list of document strings from the clusters."""
    all_words = []
    all_clusters = []
    for cluster, posts in tqdm(get_cluster_posts(con, ids, clusters, table_name), total=len(set(clusters))):
        all_clusters.append(int(cluster))
        cluster_words = " ".join([title + " " + selftext for title, selftext in posts])
        all_words.append(cluster_words)
    return pd.Series(all_words), all_clusters

def TF_IDF_matrix(REDDIT_DATA_DIR:str, TABLE_NAME:str, CLUSTER_FILE:str, IDS_FILE:str, TFIDF_MAX_FEATURES:str):

    # Create a database connection
    con = create_database_connection(REDDIT_DATA_DIR, TABLE_NAME, ["id", "title", "selftext"])
    ids = load_json(IDS_FILE)
    clusters = load_h5py(CLUSTER_FILE, 'data')

    documents, all_clusters = prepare_documents(con, ids, clusters, TABLE_NAME)

    my_stop_words = list(text.ENGLISH_STOP_WORDS.union(["https", "com", "www", "ve", "http", "don", "amp", "didn"]))
    tfidf_vectorizer = cuml.feature_extraction.text.TfidfVectorizer(stop_words=my_stop_words, lowercase=True,  max_features=TFIDF_MAX_FEATURES)
    tfidf_matrix = execute_with_gpu_logging(tfidf_vectorizer.fit_transform, documents)
    feature_names = tfidf_vectorizer.get_feature_names()  # Get all feature names from the vectorizer

    return tfidf_matrix, feature_names, all_clusters


def main(REDDIT_DATA_DIR:str, TABLE_NAME:str, CLUSTER_FILE:str, IDS_FILE:str, TFIDF_MAX_FEATURES:str, TFIDF_FILE:str, ADJACENCY_MATRIX:str, CLUSTER_ORDER:str):
    """Main function to compute the TF-IDF matrix and adjacency matrix."""

    tfidf_matrix, feature_names, all_clusters = TF_IDF_matrix(REDDIT_DATA_DIR, TABLE_NAME, CLUSTER_FILE, IDS_FILE, TFIDF_MAX_FEATURES)
    adjacency_matrix = compute_adjacency_matrix(tfidf_matrix, all_clusters)
    save_h5py(adjacency_matrix, ADJACENCY_MATRIX, "data")
    save_json({"cluster_order": list(all_clusters)}, CLUSTER_ORDER)

    top_words_per_document = extract_top_words(tfidf_matrix, feature_names, all_clusters)
    save_json(top_words_per_document, TFIDF_FILE)



if __name__ == "__main__":

    # config.CLUSTER_FILE = config.SUBCLUSTER_FILE
    # config.TFIDF_FILE = config.SUBCLUSTER_TFIDF_FILE
    run_function_with_overrides(main, config)


    