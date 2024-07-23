import os
import time

EXPERIMENT_NAME = "testing_pipeline"
OUTPUT_DIR = os.path.join("output" , EXPERIMENT_NAME)
# OUTPUT_DIR = os.path.join("output" , EXPERIMENT_NAME)
if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)

# where to get the data from
REDDIT_DATA_DIR = 'data/parquet'

# Database configuration and parameters for filtering
TABLE_NAME = 'submissions'
MIN_SCORE = 10
MIN_POST_LENGTH = 40

# Embeddings 
EMBEDDINGS_FILE = os.path.join(OUTPUT_DIR, "embeddings.h5")
IDS_FILE = os.path.join(OUTPUT_DIR, "ids.json")
MODEL_NAME = 'all-MiniLM-L6-v2' 
# number of samples to load on the gpu at once 
MODEL_BATCH_SIZE = 320_000  

# Dimensionality reduction 
DIMENSIONALITY_REDUCTION_FILE = os.path.join(OUTPUT_DIR, "dimensional_reduction.h5")
UMAP_COMPONENTS = 5
UMAP_N_Neighbors = 4 # increase this to get more global structure
UMAP_MINDIST = 0
PARTIAL_FIT_DIM_REDUCTION = 0.1
NEGATIVE_SAMPLE_RATE = 15 # understand better what it 
LEARNING_RATE = 1.0
UMAP_N_EPOCHS = 500

# clustering 
CLUSTER_FILE =  os.path.join(OUTPUT_DIR, "clusters.h5")
SUBCLUSTER_FILE = os.path.join(OUTPUT_DIR, "subclusters.h5")
HDBS_MIN_CLUSTERSIZE= 300
HDBS_MIN_SAMPLES = 20
PARTIAL_FIT_CLUSTER = 0.1

# tfidf
TFIDF_MAX_FEATURES = 50_000  # Size of the vocabulary, None means no limitation
TFIDF_WORDS_PER_CLUSTER = 50
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
CLUSTER_ORDER = os.path.join(OUTPUT_DIR, "cluster_order.json")
RESOLUTION_PARAMETER = [0.5, 1, 1.5, 2]