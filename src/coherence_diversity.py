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

from src.tf_idf import  TF_IDF_matrix, extract_top_words
from src.utils.utils import connect_to_existing_database, load_json, save_h5py, load_h5py, save_json, append_to_json
import igraph as ig
import leidenalg
from tqdm import tqdm
import cuml
import pickle
from gensim import corpora

import numpy as np
import pandas as pd
import json
from datetime import datetime
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import fcluster


def get_random_posts(DATABASE_PATH, TABLE_NAME, num_posts):
    con = connect_to_existing_database(DATABASE_PATH)
    
    query = f"""
    SELECT title, selftext FROM {TABLE_NAME}
    ORDER BY RANDOM()
    LIMIT {num_posts}
    """
    
    cursor = con.execute(query)
    title_selftext = cursor.fetchall()
    print('retrived n posts:', len(title_selftext))
    posts = [title + " " + selftext for title, selftext in title_selftext]

    con.close()
    
    return posts

def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)



def compute_coherence(TFIDF_FILE, DATABASE_PATH, TABLE_NAME, COHERENCE_FILE, N_POSTS):

    tf_idf_dictionary = load_json(TFIDF_FILE)
    texts = get_random_posts(DATABASE_PATH, TABLE_NAME, N_POSTS)
    dictionary = corpora.Dictionary([simple_preprocess(text) for text in texts])
    print(dictionary)
    corpus = [dictionary.doc2bow(simple_preprocess(text)) for text in texts]

    tf_idf_topics = [topic for topic in tf_idf_dictionary.values()]

    cm = CoherenceModel(topics=tf_idf_topics, corpus=corpus, dictionary=dictionary, coherence='u_mass')
    coherence = cm.get_coherence_per_topic()
    overall_coherence = cm.get_coherence()
    
    print(coherence)
    print(f"Coherence: {overall_coherence}")

    current_date = datetime.now().isoformat()
    coherence_data = {
        "tf_idf_file": TFIDF_FILE,
        "date": current_date,
        "coherence_per_topic": coherence,
        "overall_coherence": overall_coherence
    }

    append_to_json(COHERENCE_FILE, coherence_data)


if __name__ == "__main__":
    run_function_with_overrides(compute_coherence, config)

