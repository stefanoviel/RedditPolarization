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

from src.tf_idf import prepare_documents, TF_IDF_matrix, extract_top_words
from src.utils.utils import create_database_connection, load_json, load_h5py
import igraph as ig
import leidenalg
from tqdm import tqdm
import cuml

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import fcluster

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

    plt.figure(figsize=(25, 10))
    dn = dendrogram(Z, labels=topic_labels, leaf_rotation=90)
    plt.savefig('dendrogram.png')

    return Z


def compute_cluster_labels_at_each_merge(Z, cluster_order, cluster_per_post):
    """
    This function computes the cluster labels for each threshold at which a merge occurs in the hierarchical clustering.
    """
    thresholds = np.unique(Z[:, 2])

    results = []
    for threshold in thresholds:
        labels = fcluster(Z, t=threshold, criterion='distance')

        label_map = {old_label: new_label for old_label, new_label in zip(cluster_order, labels)}
        mapped_labels = [label_map[cluster_label] for cluster_label in cluster_per_post]
        results.append({'Threshold': threshold, 'Labels': mapped_labels})

    results_df = pd.DataFrame(results)
    return results_df


def compute_coherence(REDDIT_DATA_DIR, TABLE_NAME, TFIDF_MAX_FEATURES, IDS_FILE, merged_clusters):
    
    db_connection = create_database_connection(REDDIT_DATA_DIR, TABLE_NAME,  ["id", "title", "selftext"])
    ids = load_json(IDS_FILE)
    wiki_corpus = api.load('wiki-english-20171001')

    texts = []
    for n, text in tqdm(enumerate(wiki_corpus)): 
        for string in text['section_texts']:
            texts.append(simple_preprocess(string))
        if n > 10: # TODO: remove, I'm considering only the first 10 corpus from wikipedia to speed up the process
            break
    
    dictionary = Dictionary(texts)

    documents, all_clusters = prepare_documents(db_connection, ids, merged_clusters, TABLE_NAME)
    tfidf_matrix, feature_names = TF_IDF_matrix(documents, TFIDF_MAX_FEATURES)
    top_words_per_document = extract_top_words(tfidf_matrix, feature_names, all_clusters)

    # extract the top words for each cluster and create a list of string 
    topics_word = []
    for cluster in top_words_per_document:
        topics_word.append(top_words_per_document[cluster])

    cm = CoherenceModel(topics=topics_word, texts=texts, dictionary=dictionary, coherence='c_v')
    coherence = cm.get_coherence()
    return coherence

        
if __name__ == "__main__":
    # run_function_with_overrides(main, config)

    adjacency_matrix_file = load_h5py(config.ADJACENCY_MATRIX, 'data') 
    cluster_order = load_json(config.CLUSTER_ORDER)['cluster_order']
    cluster_id_per_post = load_h5py(config.CLUSTER_FILE, 'data')
    Z = hierarchical_topics_from_similarity(adjacency_matrix_file, cluster_order)
    results_df = compute_cluster_labels_at_each_merge(Z, cluster_order, cluster_id_per_post)

    # all_coherence = []
    
    for index, row in results_df.iterrows():
        coherence = compute_coherence(config.REDDIT_DATA_DIR, config.TABLE_NAME, config.TFIDF_MAX_FEATURES, config.IDS_FILE, row['Labels'])
        print(f"threshold: { row['Threshold']}, coherence: {coherence}")
        # all_coherence.append(coherence)

    # plt.plot(results_df['Threshold'], all_coherence)
    # plt.savefig('coherence.png')


