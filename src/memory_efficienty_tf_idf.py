# probably not necessary, delete if not used

import os
import sys

# adding root directory to paths
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from logging_config import configure_get_logger
import config
import numpy as np
from collections import defaultdict, Counter
import os
import duckdb
from tqdm import tqdm
import string
from itertools import chain
from sklearn.feature_extraction import text
import re

if not os.path.exists(config.OUTPUT_DIR):
    os.makedirs(config.OUTPUT_DIR)
logger = configure_get_logger(config.OUTPUT_DIR, config.EXPERIMENT_NAME, executed_file_name = __file__, log_level='INFO')

from src.utils.function_runner import run_function_with_overrides, execute_with_gpu_logging
from src.utils.utils import create_database_connection, load_json, save_h5py, load_h5py, save_json

class TFIDF:
    def __init__(self):
        self.df = defaultdict(int)
        self.total_documents = 0
        self.vocab = set()
        self.stop_words = text.ENGLISH_STOP_WORDS.union(["https", "com", "www", "ve", "http", "amp"])
    
    def clean_text(self, text):
        """
        Converts text to lowercase, removes punctuation, numbers, and excludes stop words.
        """
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        # Convert to lowercase
        text = text.lower()
        # Remove numbers (and other non-linguistic characters if needed)
        text = re.sub(r'\d+', '', text)  # Remove digits
        # Split into words
        words = text.split()
        # Remove stop words and filter out any remaining non-alphabetic characters
        return [word for word in words if word not in self.stop_words and word.isalpha()]



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

    def load_connections_ids_clusters(self, REDDIT_DATA_DIR, PROCESSED_REDDIT_DATA, TABLE_NAME, CLUSTER_DB_NAME, IDS_DB_NAME):
        # Create a database connection
        con = create_database_connection(REDDIT_DATA_DIR, TABLE_NAME, ["id", "title", "selftext"])
        ids = load_h5py(PROCESSED_REDDIT_DATA, IDS_DB_NAME)
        clusters = load_h5py(PROCESSED_REDDIT_DATA, CLUSTER_DB_NAME)
        return con, ids, clusters


    def compute_df(self, iterator_cluster_posts):
        for cluster, posts in tqdm(iterator_cluster_posts):
            # Process each post to remove stop words and handle case
            terms = list(chain.from_iterable(self.clean_text(title + " " + selftext) for title, selftext in posts))
            print(f'len(terms) {len(terms):,}')
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


def main(REDDIT_DATA_DIR, PROCESSED_REDDIT_DATA, TABLE_NAME, CLUSTER_DB_NAME, IDS_DB_NAME, TFIDF_FILE):
    
    tfidf = TFIDF()
    # Load connections, IDs, and cluster assignments
    con, ids, clusters = tfidf.load_connections_ids_clusters(REDDIT_DATA_DIR, PROCESSED_REDDIT_DATA, TABLE_NAME, CLUSTER_DB_NAME, IDS_DB_NAME)

    iterator_cluster_posts = tfidf.yield_post_per_cluster(con, ids, clusters, TABLE_NAME)

    logger.info("Computing DF")
    tfidf.compute_df(iterator_cluster_posts)    
    idf = tfidf.compute_idf()

    iterator_cluster_posts = tfidf.yield_post_per_cluster(con, ids, clusters, TABLE_NAME)

    logger.info("Computing TF-IDF")
    top_words_per_cluster = {}
    for cluster, posts in tqdm(iterator_cluster_posts):
        doc = " ".join([title + " " + selftext for title, selftext in posts])
        tfidf_scores = tfidf.compute_tfidf(doc, idf)
        sorted_tfidf = sorted(tfidf_scores.items(), key=lambda x: x[1], reverse=True)

        top_words_per_cluster[str(cluster)] =  [word for word, _ in sorted_tfidf[:50]]

    save_json(top_words_per_cluster, TFIDF_FILE)

if __name__ == "__main__":
    print("Total running time:", run_function_with_overrides(main, config))

