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
import time
from tqdm import tqdm
import random
from typing import List, Dict
from src.utils.function_runner import run_function_with_overrides, execute_with_gpu_logging
from src.utils.utils import connect_to_existing_database, load_json, save_h5py, load_h5py, save_json

from cuml.common import logger
logger.set_level(logger.level_error)

if not os.path.exists(config.OUTPUT_DIR):
    os.makedirs(config.OUTPUT_DIR)
logger = configure_get_logger(config.OUTPUT_DIR, config.EXPERIMENT_NAME, executed_file_name = __file__, log_level='INFO')


def map_ids_to_clusters(ids, cluster_assignment):
    """
    Map IDs to their clusters, assuming clusters and IDs are in the same order.
    """
    cluster_to_ids = {}
    for id, cluster in zip(ids, cluster_assignment):
        if cluster not in cluster_to_ids:
            cluster_to_ids[cluster] = []
        cluster_to_ids[cluster].append(id)
    return cluster_to_ids

def yield_post_per_cluster(con: duckdb.DuckDBPyConnection, cluster_to_ids: dict, TABLE_NAME: str, N_POST_PER_CLUSTER: int):
    """
    Yield the title and selftext for a limited number of posts in each cluster by executing a database query for each cluster.
    """
    # Execute a query for each cluster and yield results
    for cluster, ids in cluster_to_ids.items():
        
        # take a N_POST_PER_CLUSTER random sample from each cluster
        if len(ids) > N_POST_PER_CLUSTER:
            ids = np.random.choice(ids, N_POST_PER_CLUSTER, replace=False)
            
        decode_ids = [id.decode('utf-8') for id in ids]

        placeholders = ','.join(['?'] * len(decode_ids))  # Prepare placeholders for SQL query

        s = time.time()
        # not good practice, sorry
        query = f"""
        SELECT title, selftext 
        FROM {TABLE_NAME} 
        WHERE id IN ({placeholders}) 
        """
        cursor = con.execute(query, decode_ids)
        posts = cursor.fetchall()

        all_posts_in_cluster = " ".join([title + " " + selftext for title, selftext in posts])
        yield all_posts_in_cluster
        print(f"Cluster {cluster} has {len(posts)} posts (limited to {N_POST_PER_CLUSTER}) and took {time.time() - s} seconds to process.")



def fetch_posts_per_cluster(
    con: duckdb.DuckDBPyConnection,
    cluster_to_ids: Dict[int, List[bytes]],
    TABLE_NAME: str,
    N_POST_PER_CLUSTER: int
) -> List[str]:

    # Prepare data for the query
    all_ids = []
    cluster_mapping = []
    for cluster, ids in cluster_to_ids.items():
        # Sample N_POST_PER_CLUSTER ids if necessary
        sampled_ids = random.sample(ids, min(len(ids), N_POST_PER_CLUSTER))
        all_ids.extend(id.decode('utf-8') for id in sampled_ids)
        cluster_mapping.extend([cluster] * len(sampled_ids))

    # Process in batches of 50,000
    BATCH_SIZE = 500_000
    cluster_posts = {}
    total_fetched = 0
    start_time = time.time()

    for i in range(0, len(all_ids), BATCH_SIZE):
        batch_ids = all_ids[i:i+BATCH_SIZE]
        batch_clusters = cluster_mapping[i:i+BATCH_SIZE]

        # Prepare the query for this batch
        placeholders = ','.join(['?'] * len(batch_ids))
        query = f"""
        SELECT 
            title,
            selftext
        FROM {TABLE_NAME}
        WHERE id IN ({placeholders})
        """

        # Execute the query for this batch
        print(f"Executing query for batch {i//BATCH_SIZE + 1}, number of posts to fetch: {len(batch_ids)}")
        cursor = con.execute(query, batch_ids)
        results = cursor.fetchall()

        # Process the results for this batch
        for cluster, title, selftext in zip(batch_clusters, *zip(*results)):
            if cluster not in cluster_posts:
                cluster_posts[cluster] = []
            cluster_posts[cluster].append(f"{title} {selftext}")

        total_fetched += len(results)

    # Concatenate posts for each cluster
    concatenated_posts = [" ".join(posts) for posts in cluster_posts.values()]

    print(f"Fetched {total_fetched} posts for {len(cluster_posts)} clusters in {time.time() - start_time:.2f} seconds.")

    return concatenated_posts



def extract_top_words(tfidf_matrix, feature_names, unique_clusters, top_n=10):
    """Extract top words for each document (cluster in our case) from the tfidf matrix."""
    top_words_per_document = {}
    for cluster_index in tqdm(range(tfidf_matrix.shape[0])):
        cluster_key = str(unique_clusters[cluster_index])
        # if cluster_key == "-1":
        #     continue
        row = tfidf_matrix.getrow(cluster_index)
        indices = row.indices
        data = row.data
        if len(data) == 0:
            top_words_per_document[cluster_key] = []
            continue
        
        # Ensure top_n does not exceed the length of data
        actual_top_n = min(top_n, len(data))
        
        # Get the indices of the top `actual_top_n` elements
        top_indices = np.argpartition(data, -actual_top_n)[-actual_top_n:]
        
        # Sort the top indices by their values in descending order
        top_indices_sorted = top_indices[np.argsort(data[top_indices])[::-1]]
        
        # Get the original indices corresponding to the top sorted indices
        original_indices = indices[top_indices_sorted]
        
        # Map indices to feature names
        top_features = [feature_names[ind] for ind in original_indices]
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


def TF_IDF_matrix(documents, TFIDF_MAX_FEATURES:str):
    my_stop_words = list(text.ENGLISH_STOP_WORDS.union(["https", "com", "www", "ve", "http", "amp"]))
    print('stop words')
    tfidf_vectorizer = TfidfVectorizer(stop_words=my_stop_words, lowercase=True,  max_features=TFIDF_MAX_FEATURES)
    tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
    print('done with fit transform')
    feature_names = tfidf_vectorizer.get_feature_names_out()  # Get all feature names from the vectorizer

    return tfidf_matrix, feature_names

def run_tf_idf(DATABASE_PATH:str, PROCESSED_REDDIT_DATA:str, TABLE_NAME:str, CLUSTER_DB_NAME:str, IDS_DB_NAME:str, TFIDF_MAX_FEATURES:str, TFIDF_FILE:str, ADJACENCY_MATRIX:str, TFIDF_WORDS_PER_CLUSTER:int, N_POST_PER_CLUSTER:int):
    """Main function to compute the TF-IDF matrix and adjacency matrix."""

    ids = load_h5py(PROCESSED_REDDIT_DATA, IDS_DB_NAME)
    post_cluster_assignment = load_h5py(PROCESSED_REDDIT_DATA, CLUSTER_DB_NAME)
    con = connect_to_existing_database(DATABASE_PATH)
    cluster_to_ids = map_ids_to_clusters(ids, post_cluster_assignment)
    iterator_posts_in_cluster = yield_post_per_cluster(con, cluster_to_ids, TABLE_NAME, N_POST_PER_CLUSTER)
    tfidf_matrix, feature_names = TF_IDF_matrix(iterator_posts_in_cluster, TFIDF_MAX_FEATURES)

    unique_cluster_order = list(cluster_to_ids.keys())
    top_words_per_document = extract_top_words(tfidf_matrix, feature_names, unique_cluster_order, TFIDF_WORDS_PER_CLUSTER)
    save_json(top_words_per_document, TFIDF_FILE)

def tf_idf_on_subclusters(DATABASE_PATH: str, PROCESSED_REDDIT_DATA: str, TABLE_NAME: str, SUBCLUSTER_DB_NAME: str, CLUSTER_DB_NAME: str, IDS_DB_NAME: str, TFIDF_MAX_FEATURES: str, SUBCLUSTER_TFIDF_FILE: str, ADJACENCY_MATRIX: str, TFIDF_WORDS_PER_CLUSTER: int, N_POST_PER_CLUSTER: int):
    ids = load_h5py(PROCESSED_REDDIT_DATA, IDS_DB_NAME)
    post_subcluster_assignment = load_h5py(PROCESSED_REDDIT_DATA, SUBCLUSTER_DB_NAME)
    post_cluster_assignment = load_h5py(PROCESSED_REDDIT_DATA, CLUSTER_DB_NAME)
    con = connect_to_existing_database(DATABASE_PATH)

    # Create a dictionary to map cluster IDs to subcluster IDs
    cluster_to_subclusters_to_id = {}
    for cluster, subcluster, id in zip(post_cluster_assignment, post_subcluster_assignment, ids):
        if cluster not in cluster_to_subclusters_to_id:
            cluster_to_subclusters_to_id[cluster] = {}
        if subcluster not in cluster_to_subclusters_to_id[cluster]:
            cluster_to_subclusters_to_id[cluster][subcluster] = []
        cluster_to_subclusters_to_id[cluster][subcluster].append(id)

    all_top_words = {}

    for cluster, subclusters in tqdm(cluster_to_subclusters_to_id.items(), desc="Processing clusters"):
        if cluster == -1:
            continue

        subclusters = {subcluster: ids for subcluster, ids in subclusters.items()}

        # Generate posts for each subcluster
        all_posts_per_cluster = fetch_posts_per_cluster(con, subclusters, TABLE_NAME, N_POST_PER_CLUSTER)

        # Compute TF-IDF matrix for the subclusters
        tfidf_matrix, feature_names = TF_IDF_matrix(all_posts_per_cluster, TFIDF_MAX_FEATURES)

        # Get unique subclusters for this cluster
        unique_subcluster_order = list(subclusters.keys())

        # Extract top words for each subcluster
        top_words_per_subcluster = extract_top_words(tfidf_matrix, feature_names, unique_subcluster_order, TFIDF_WORDS_PER_CLUSTER)

        # Store the results
        all_top_words[str(cluster)] = top_words_per_subcluster

        print("top_words_per_subcluster", top_words_per_subcluster)

    # Save the results
    save_json(all_top_words, SUBCLUSTER_TFIDF_FILE)



if __name__ == "__main__":
    # print("Total running time:", run_function_with_overrides(run_tf_idf, config))

    print("Total running time:", run_function_with_overrides(tf_idf_on_subclusters, config))




