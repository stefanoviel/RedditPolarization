
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
logger = configure_get_logger(config.OUTPUT_DIR, config.EXPERIMENT_NAME, executed_file_name = __file__, log_level='INFO')

import h5py
import duckdb
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text
import numpy as np
import cuml
import pandas as pd

from src.utils.function_runner import run_function_with_overrides, execute_with_gpu_logging

def create_database_connection(parquet_directory:str, table_name:str):
    """Create and return a database connection using the provided configuration."""
    files = [f'{parquet_directory}/{file}' for file in os.listdir(parquet_directory)]
    con = duckdb.connect(database=':memory:')
    query_files = ', '.join(f"'{f}'" for f in files)
    sql_query = f"CREATE TABLE {table_name} AS SELECT id, title, selftext FROM read_parquet([{query_files}], union_by_name=True)"
    con.execute(sql_query)
    return con

def load_cluster_ids(file_path):
    with open(file_path, 'r') as file:
        ids = json.load(file)
    return ids


def get_cluster_posts(con, ids, clusters, TABLE_NAME):
    """
    Yield the title and selftext for each cluster by executing a database query for each cluster.

    Args:
        REDDIT_DATA_DIR (str): Directory containing the database.
        TABLE_NAME (str): The name of the table in the database.
        CLUSTER_FILE (str): HDF5 file path that has the cluster for each element.
        IDS_FILE (str): HDF5 file path that contains all the IDs to consider.

    Yields:
        tuple: cluster identifier, list of tuples (title, selftext) for that clustersr  
    """

    # Map IDs to their clusters
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


def get_important_words(posts, max_features):
    """
    Compute the most important words in posts using TF-IDF.

    Args:
        posts (list of tuples): Each tuple contains (title, selftext) from the database.
        max_features (int): Number of top features (words) to return based on TF-IDF scores.

    Returns:
        list: A list of the most important words in the given posts.
    """
    if not posts:
        return []

    # Combine title and selftext for each post to create a document
    documents = [title + " " + selftext for title, selftext in posts]
    documents = pd.Series(documents)

    # part of urls would get tokenized as words due to punctuation removal
    my_stop_words = list(text.ENGLISH_STOP_WORDS.union(["just", "https", "com", "www", "ve", "http", "don", "amp", "didn"]))

    tfidf_vectorizer = cuml.feature_extraction.text.TfidfVectorizer(stop_words=my_stop_words, lowercase=True,  max_features=max_features)
    tfidf_matrix = execute_with_gpu_logging(tfidf_vectorizer.fit_transform, documents)

    # Get feature names to use as output instead of feature indices
    feature_names = tfidf_vectorizer.get_feature_names()
    sums = tfidf_matrix.sum(axis=0)

    data = []
    for col, term in enumerate(feature_names.to_pandas()):
        data.append((term, sums[0, col]))

    important_words = sorted(data, key=lambda x: x[1], reverse=True)

    return [word for word, _ in important_words]

def find_save_important_words(REDDIT_DATA_DIR:str, TABLE_NAME:str, CLUSTER_FILE:str, IDS_FILE:str, TFIDF_MAX_FEATURES:str, TFIDF_FILE:str):
    """
    Find and save the most important words for each cluster.

    Args:
        REDDIT_DATA_DIR (str): Directory containing the database.
        TABLE_NAME (str): The name of the table in the database.
        CLUSTER_FILE (str): HDF5 file path that has the cluster for each element.
        IDS_FILE (str): HDF5 file path that contains all the IDs to consider.
        OUTPUT_DIR (str): Directory to save the results.
    """

    topic_cluster = {}

    # Create a database connection
    con = create_database_connection(REDDIT_DATA_DIR, TABLE_NAME)

    # Load cluster IDs
    ids = load_cluster_ids(IDS_FILE)

    # Load cluster information
    with h5py.File(CLUSTER_FILE, 'r') as cluster_file:
        clusters = cluster_file['data'][:]

    for cluster, posts in get_cluster_posts(con, ids, clusters, TABLE_NAME):
        important_words = get_important_words(posts, max_features=TFIDF_MAX_FEATURES)
        
        topic_cluster[int(cluster)] = important_words
    
    with open(TFIDF_FILE, 'w') as file:
        json.dump(topic_cluster, file, indent=4)


if __name__ == "__main__":


    run_function_with_overrides(find_save_important_words, config)