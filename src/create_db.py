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
logger = configure_get_logger(config.OUTPUT_DIR, config.EXPERIMENT_NAME, executed_file_name = __file__, log_level='INFO')

import pandas as pd
from src.utils.utils import connect_to_existing_database, load_json, load_h5py
from src.utils.function_runner import run_function_with_overrides


def create_db(DATABASE_PATH, PROCESSED_REDDIT_DATA, IDS_DB_NAME, CLUSTER_DB_NAME, TFIDF_FILE, TABLE_NAME, FINAL_DATAFRAME):
    con = connect_to_existing_database(DATABASE_PATH)
    topic_description = load_json(TFIDF_FILE)
    ids = load_h5py(PROCESSED_REDDIT_DATA, IDS_DB_NAME)
    post_cluster_assignment = load_h5py(PROCESSED_REDDIT_DATA, CLUSTER_DB_NAME)

    decoded_ids = [id.decode('utf-8') for id in ids]

    query = f"SELECT id, subreddit, created_utc, author FROM {TABLE_NAME} WHERE id IN ({','.join(['?']*len(ids))})"
    cursor = con.execute(query, decoded_ids)
    
    # Fetch all rows from the executed query
    rows = cursor.fetchall()
    
    # Define the column names based on the SELECT statement
    columns = ['id', 'subreddit', 'created_utc', 'author']
    
    # Create a pandas DataFrame from the fetched rows
    df = pd.DataFrame(rows, columns=columns)
    
    # Add the post_cluster_assignment as a new column
    df['cluster'] = post_cluster_assignment

    df.to_csv(FINAL_DATAFRAME, index=False)


if __name__ == "__main__":
    run_function_with_overrides(create_db, config)