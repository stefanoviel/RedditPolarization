# probably not necessary, delete if not used

import os
import sys

# adding root directory to paths
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from logging_config import configure_get_logger
import numpy as np
from collections import defaultdict, Counter
import os
import duckdb

from src.utils.function_runner import run_function_with_overrides, execute_with_gpu_logging
from src.utils.utils import create_database_connection, load_json, save_h5py, load_h5py, save_json

class TFIDF:
    def __init__(self):
        self.df = defaultdict(int)
        self.total_documents = 0
        self.vocab = set()

    def map_ids_to_clusters(self, ids, cluster_assignment):
        """
        Map IDs to their clusters, assuming clusters and IDs are in the same order.
        """
        cluster_to_ids = {}
        for id, cluster in zip(ids, cluster_assignment):
            if cluster not in cluster_to_ids:
                cluster_to_ids[cluster] = []
            cluster_to_ids[cluster].append(id)
        return cluster_to_ids

    def yield_post_per_cluster(self, con: duckdb.DuckDBPyConnection, ids:list, cluster_assignment:list, TABLE_NAME:str):
        """
        Yield the title and selftext for all the posts in each cluster by executing a database query for each cluster.
        """
        # Map IDs to their clusters, assuming clusters and IDs are in the same order
        cluster_to_ids = self.map_ids_to_clusters(ids, cluster_assignment)

        # Execute a query for each cluster and yield results
        for cluster, cluster_ids in cluster_to_ids.items():
            placeholders = ','.join(['?'] * len(cluster_ids))  # Prepare placeholders for SQL query
            query = f"SELECT title, selftext FROM {TABLE_NAME} WHERE id IN ({placeholders})"
            cursor = con.execute(query, cluster_ids)
            posts = cursor.fetchall()
            yield cluster, posts

    def load_connections_ids_clusters(REDDIT_DATA_DIR, PROCESSED_REDDIT_DATA, TABLE_NAME, CLUSTER_DB_NAME, IDS_FILE):
        # Create a database connection
        con = create_database_connection(REDDIT_DATA_DIR, TABLE_NAME, ["id", "title", "selftext"])
        ids = load_h5py(PROCESSED_REDDIT_DATA, CLUSTER_DB_NAME)
        clusters = load_h5py(PROCESSED_REDDIT_DATA, CLUSTER_DB_NAME)
        return con, ids, clusters


    def process_documents(self, iterator_cluster_posts):
        # Step 1: Compute DF for each term

        for cluster, posts in iterator_cluster_posts:
            doc = " ".join([title + " " + selftext for title, selftext in posts])
            terms = doc.split()
            self.vocab.update(terms)
            unique_terms = set(terms)
            for term in unique_terms:
                self.df[term] += 1
            self.total_documents += 1

        # Step 2: Truncate the vocabulary to the top 50,000 terms by document frequency
        # Sort terms by their frequency in descending order and select the top 50,000
        sorted_vocab = sorted(self.df.items(), key=lambda x: x[1], reverse=True)
        truncated_vocab = {term for term, count in sorted_vocab[:50000]}

        # Update self.vocab to only include these 50,000 terms
        self.vocab = truncated_vocab

        # Optionally, adjust self.df to reflect the truncated vocabulary
        self.df = {term: self.df[term] for term in self.vocab}

    def compute_idf(self):
        # Step 2: Compute IDF for each term in the vocabulary
        idf = {}
        for term in self.vocab:
            idf[term] = np.log(self.total_documents / (1 + self.df[term]))
        return idf

    def compute_tfidf(self, doc, idf):
        # Step 3: Compute TF-IDF for a single document, only for terms in the truncated vocabulary
        terms = doc.split()
        term_count = Counter(terms)
        tfidf = {}
        total_terms = sum(term_count[term] for term in self.vocab)  # Total count of terms in the truncated vocabulary

        for term in term_count:
            if term in self.vocab:  # Check if the term is in the truncated vocabulary
                tf = term_count[term] / total_terms  # Use count of valid terms for normalization
                tfidf[term] = tf * idf.get(term, 0)

        return tfidf

    def main(self, REDDIT_DATA_DIR, PROCESSED_REDDIT_DATA, TABLE_NAME, CLUSTER_DB_NAME, IDS_FILE):
        # Load connections, IDs, and cluster assignments
        con, ids, clusters = self.load_connections_ids_clusters(REDDIT_DATA_DIR, PROCESSED_REDDIT_DATA, TABLE_NAME, CLUSTER_DB_NAME, IDS_FILE)

        iterator_cluster_posts = self.yield_post_per_cluster(con, ids, clusters, TABLE_NAME)
        # Compute TF-IDF
        tfidf = TFIDF()
        tfidf.process_documents(iterator_cluster_posts)
        idf = tfidf.compute_idf()

        iterator_cluster_posts = self.yield_post_per_cluster(con, ids, clusters, TABLE_NAME)

        # tfidf_scores = tfidf.compute_tfidf(documents[0], idf)
        print(tfidf_scores)
