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
from src.utils.utils import connect_to_existing_database, load_json, save_h5py, load_h5py, save_json
import igraph as ig
import leidenalg
from tqdm import tqdm
import cuml
import pickle
from gensim import corpora

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import fcluster


def get_random_posts(DATABASE_PATH, TABLE_NAME):
    con =  connect_to_existing_database(DATABASE_PATH)
    
    query = f"""SELECT title, selftext FROM {TABLE_NAME}"""
    
    cursor = con.execute(query)
    title_selftext = cursor.fetchall()
    posts = [title + " " + selftext for title, selftext in title_selftext]

    return posts

# def compute_coherence(TFIDF_FILE, REDDIT_DATA_DIR, TABLE_NAME, MIN_SCORE, MIN_POST_LENGTH, N_POSTS):

#     tf_idf_dictionary = load_json(TFIDF_FILE)

#     wiki_corpus = api.load('wiki-english-20171001')
#     texts = []
#     for n, text in enumerate(wiki_corpus):
#         texts.extend([simple_preprocess(string) for string in text['section_texts']])
#         if n % 1000 == 0:
#             print(f"Processed {n} documents from Wikipedia corpus")
#         if n == 10000:
#             break

#     texts = get_random_posts(REDDIT_DATA_DIR, TABLE_NAME, MIN_SCORE, MIN_POST_LENGTH, N_POSTS)
#     dictionary = Dictionary(texts)

#     tf_idf_topics = [topic for topic in tf_idf_dictionary.values()]

#     cm = CoherenceModel(topics=tf_idf_topics, texts=texts, dictionary=dictionary, coherence='u_mass')
#     coherence = cm.get_coherence_per_topic()

#     print(f"Coherence: {coherence}")

def compute_coherence(TFIDF_FILE, DATABASE_PATH):


    tf_idf_dictionary = load_json(TFIDF_FILE)
    texts = get_random_posts(DATABASE_PATH)
    dictionary = corpora.Dictionary([simple_preprocess(text) for text in texts])
    corpus = [dictionary.doc2bow(simple_preprocess(text)) for text in texts]

    tf_idf_topics = [topic for topic in tf_idf_dictionary.values()]

    cm = CoherenceModel(topics=tf_idf_topics, corpus=corpus, dictionary=dictionary, coherence='u_mass')
    coherence = cm.get_coherence_per_topic()
    print(coherence)
    print(f"Coherence: {cm.get_coherence()}")


if __name__ == "__main__":
    run_function_with_overrides(compute_coherence, config)

