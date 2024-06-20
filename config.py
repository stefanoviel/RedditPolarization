import os
import time

# add timestamp to not have same experiments with overlapping names
timestamp = time.strftime("%Y%m%d_%H%M%S") 

OUTPUT_DIR = "/media/data/stviel/RedditPolarization/output"  # where the experiment will be saved
EXPERIMENT_NAME = "testing_pipeline"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, f"{timestamp}_{EXPERIMENT_NAME}")

REDDIT_DATA_DIR = 'data'

# Database configuration and parameters for filtering
REDDIT_DB_FILE = 'data/dbs/duck_db.db'
TABLE_NAME = 'submissions'
MIN_SCORE = 10
MIN_POST_LENGTH = 20

EMBEDDINGS_FILE = 'data/embeddings/embeddings.h5'

MODEL_NAME = 'all-MiniLM-L6-v2' 

UMAP_COMPONENTS = 5
UMAP_N_Neighbors = 50
UMAP_MINDIST = 0.01
PARTIAL_FIT_SAMPLE_SIZE = 0.1