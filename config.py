import os
import time

EXPERIMENT_NAME = "full_2017"
OUTPUT_DIR = os.path.join("output" , EXPERIMENT_NAME)
# OUTPUT_DIR = os.path.join("output" , EXPERIMENT_NAME)
if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)

# where to get the data from
REDDIT_DATA_DIR = 'data/big_test'

# Database configuration and parameters for filtering
TABLE_NAME = 'submissions'
MIN_SCORE = 10
MIN_POST_LENGTH = 20
SUBSET_FRACTION = 0.1   # randomly select a subset of the data to load in DB
ATTRIBUTE_TO_EXTRACT = {  # changing this will require changes in load_data_to_db.py
    "subreddit",
    "score",
    "author",
    "created_utc",
    "title",
    "id",
    "num_comments",
    "selftext",
}

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
UMAP_MINDIST = 0.01
PARTIAL_FIT_DIM_REDUCTION = 0.1
NEGATIVE_SAMPLE_RATE = 15 # understand better what it 
LEARNING_RATE = 0.25
UMAP_N_EPOCHS = 5000

# clustering 
CLUSTER_FILE =  os.path.join(OUTPUT_DIR, "clusters.h5")
HDBS_MIN_CLUSTERSIZE= 4000
HDBS_MIN_SAMPLES = 20
PARTIAL_FIT_CLUSTER = 0.1

# tfidf
TFIDF_MAX_FEATURES = 10
TFIDF_FILE = os.path.join(OUTPUT_DIR, "tfidf.json")
