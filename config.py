import os
import time

OUTPUT_DIR = "/media/data/stviel/RedditPolarization/output/testing_pipeline"  # where the experiment outputs will be saved

# where to get the data from
REDDIT_DATA_DIR = 'data'

# Database configuration and parameters for filtering
TABLE_NAME = 'food_movies'
MIN_SCORE = 10
MIN_POST_LENGTH = 20
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
EMBEDDINGS_FILE = os.path.join(OUTPUT_DIR, "original_embeddings/embeddings.h5")
if not os.path.exists(os.path.join(OUTPUT_DIR, "embeddings")):
    os.mkdir(os.path.join(OUTPUT_DIR, "embeddings"))
MODEL_NAME = 'all-MiniLM-L6-v2' 
# number of samples to load on the gpu at once 
MODEL_BATCH_SIZE = 320000  # 320k samples will fit in 10GB of GPU memory

# Dimensionality reduction 
DIMENSIONALITY_REDUCTION_FILE = os.path.join(OUTPUT_DIR, "food_movies/dimensional_reduction.h5")
UMAP_COMPONENTS = 5
UMAP_N_Neighbors = 3
UMAP_MINDIST = 0.01
PARTIAL_FIT_SAMPLE_SIZE = 0.1
