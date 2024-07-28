import os
import time

EXPERIMENT_NAME = "testing_pipeline_small"
OUTPUT_DIR = os.path.join("output" , EXPERIMENT_NAME)
# OUTPUT_DIR = os.path.join("output" , EXPERIMENT_NAME)
if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)

# datasets names
# REDDIT_DATA_DIR = 'data/big_test'
REDDIT_DATA_DIR = 'data/parquet'
PROCESSED_REDDIT_DATA = os.path.join(OUTPUT_DIR, "reddit_data.h5")

# Database configuration and parameters for filtering
TABLE_NAME = 'submissions'
MIN_SCORE = 10
MIN_POST_LENGTH = 40
START_DATE= 1262304000 # 2010-01-01
END_DATE= 1672531200 # 2023-01-01

# Embeddings 
EMBEDDING_DB_NAME = "embeddings"
IDS_DB_NAME = "ids"
MODEL_NAME = 'all-MiniLM-L6-v2' 
# number of samples to load on the gpu at once 
MODEL_BATCH_SIZE = 320_000  

# Dimensionality reduction 
DIMENSIONALITY_REDUCTION_DB_NAME = "dimensional_reduction"
UMAP_COMPONENTS = 5
UMAP_N_Neighbors = 4 # increase this to get more global structure
UMAP_MINDIST = 0
PARTIAL_FIT_DIM_REDUCTION = 0.1
NEGATIVE_SAMPLE_RATE = 5 # understand better what it 
LEARNING_RATE = 1.0
UMAP_N_EPOCHS = 500

# clustering 
CLUSTER_DB_NAME =  "clusters"
SUBCLUSTER_DB_NAME = "subclusters"
HDBS_MIN_CLUSTERSIZE= 300
HDBS_MIN_SAMPLES = 20
HDBS_MIN_CLUSTERSIZE_PERCENTAGE_SEARCH = [0.0005, 0.0025, 0.001, 0.005, 0.01]
HDBS_MIN_SAMPLES_SEARCH = [5, 10, 20]
PARTIAL_FIT_CLUSTER = 0.1  # it might not be used, depending on the employed function

# tfidf
TFIDF_MAX_FEATURES = 50_000  # Size of the vocabulary, None means no limitation
TFIDF_WORDS_PER_CLUSTER = 15
TFIDF_FILE = os.path.join(OUTPUT_DIR, "tfidf.json")
SUBCLUSTER_TFIDF_FILE = os.path.join(OUTPUT_DIR, "subcluster_tfidf.json")

# topic Naming
LLM_NAME = "Qwen/Qwen2-7B-Instruct-GPTQ-Int8"
TOPIC_NAMING_FILE = os.path.join(OUTPUT_DIR, "topic_naming.json")
PROMPT = """ Given the following lists of words, each associated with a cluster number, identify a succinct topic that captures the essence of the words in each list. Below are some examples of how the output should be structured in JSON format.

    Examples:
    - Cluster 4: "game, team, season, like, time, year, player, play, games, 10" -> "Sports Analysis"
    - Cluster -1: "new, like, time, know, game, people, think, make, good, really" -> "General Discussion"
    - Cluster 32: "team, vs, game, twitch, tv, twitter, youtube, 00, logo, mt" -> "Live Streaming and Social Media"
    - Cluster 24: "art, oc, painting, like, drawing, new, paint, pen, imgur, time" -> "Art and Drawing"

    Your task is to find an appropriate topic for the list from cluster 0. Present your output in the following JSON format:

    {{
    "topic": "Your identified topic here"
    }}

    Please perform the same task for the list associated with cluster 0:
    - Cluster 0: {list_of_words} ->

    """

# hierarchical topic discovery
ADJACENCY_MATRIX = os.path.join(OUTPUT_DIR, "adjacency_matrix.h5")
RESOLUTION_PARAMETER = [0.5, 1, 1.5, 2]

# quiz & coherence
NUMBER_OF_OPTIONS = 14
N_POSTS = 1e7