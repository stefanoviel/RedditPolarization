import os
import time

# add timestamp to not have same experiments with overlapping names
timestamp = time.strftime("%Y%m%d_%H%M%S") 


OUTPUT_DIR = "/media/data/stviel/RedditPolarization/output"  # where the experiment will be saved

EXPERIMENT_NAME = "testing_duckdb"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, f"{timestamp}_{EXPERIMENT_NAME}")

REDDIT_DATA_DIR = 'data'
REDDIT_DB_FILE = 'data/dbs/duck_db_test.db'

