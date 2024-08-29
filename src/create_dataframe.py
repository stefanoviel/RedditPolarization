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

    query = f"SELECT id, subreddit, created_utc, author, title FROM {TABLE_NAME} WHERE id IN ({','.join(['?']*len(ids))})"
    cursor = con.execute(query, decoded_ids)
    print('query executed')
    
    # Fetch all rows from the executed query
    rows = cursor.fetchall()
    print('everything fetched')
    
    # Define the column names based on the SELECT statement
    columns = ['id', 'subreddit', 'created_utc', 'author', 'title']
    print('first rows fetched', rows[:5])
    
    # Create a pandas DataFrame from the fetched rows
    df = pd.DataFrame(rows, columns=columns)
    
    # Add the post_cluster_assignment as a new column

    cluster_topic_df = pd.DataFrame([[cluster] for cluster in post_cluster_assignment], columns=['cluster'])
    cluster_topic_df['id'] = decoded_ids
    df = df.merge(cluster_topic_df, on='id')
    print('merged')

    df.to_csv(FINAL_DATAFRAME, index=False)


def create_db_chunked(DATABASE_PATH, PROCESSED_REDDIT_DATA, IDS_DB_NAME, CLUSTER_DB_NAME, SUBCLUSTER_DB_NAME, TABLE_NAME, FINAL_DATAFRAME, CHUNK_SIZE):
    con = connect_to_existing_database(DATABASE_PATH)
    ids = load_h5py(PROCESSED_REDDIT_DATA, IDS_DB_NAME)
    post_cluster_assignment = load_h5py(PROCESSED_REDDIT_DATA, CLUSTER_DB_NAME)
    post_subcluster_assignment = load_h5py(PROCESSED_REDDIT_DATA, SUBCLUSTER_DB_NAME)

    decoded_ids = [id.decode('utf-8') for id in ids]
    CHUNK_SIZE = int(CHUNK_SIZE)

    # remove db if it exists
    if os.path.exists(FINAL_DATAFRAME):
        os.remove(FINAL_DATAFRAME)

    # Open the CSV file in write mode for the first chunk
    is_first_chunk = True
 
    print('Total chunks:', len(decoded_ids)//CHUNK_SIZE)
    for i in range(0, len(decoded_ids), CHUNK_SIZE):
        # Create a chunk of IDs
        chunk_ids = decoded_ids[i:i + CHUNK_SIZE]
        chunk_clusters = post_cluster_assignment[i:i + CHUNK_SIZE]
        chunk_subclusters = post_subcluster_assignment[i:i + CHUNK_SIZE]

        # Formulate the query for this chunk
        query = f"SELECT DISTINCT id, subreddit, created_utc, author FROM {TABLE_NAME} WHERE id IN ({','.join(['?']*len(chunk_ids))})"
        cursor = con.execute(query, chunk_ids)
        print(f'Query executed for chunk {i//CHUNK_SIZE + 1}')
        
        # Fetch the rows and create a DataFrame
        rows = cursor.fetchall()
        print(f'Rows fetched for chunk {i//CHUNK_SIZE + 1}')
        
        # Define the column names based on the SELECT statement
        columns = ['id', 'subreddit', 'created_utc', 'author']
        chunk_df = pd.DataFrame(rows, columns=columns)
        urls = [f'https://www.reddit.com/r/{subreddit}/comments/{id}/' for subreddit, id in zip(chunk_df.subreddit, chunk_df.id)]

        # Get the corresponding cluster for each chunk_df.id (some ids might be duplicates)
        chunk_df['cluster'] = [chunk_clusters[chunk_ids.index(id)] for id in chunk_df.id]
        chunk_df['subcluster'] = [chunk_subclusters[chunk_ids.index(id)] for id in chunk_df.id]
        chunk_df['url'] = urls

        chunk_df = chunk_df[chunk_df.cluster != -1]  # Remove the unclustered posts

        # Save the chunk DataFrame to the CSV file
        # If it's the first chunk, write the header; otherwise, append without header   
        chunk_df.to_csv(FINAL_DATAFRAME, mode='a', index=False, header=is_first_chunk)
        is_first_chunk = False  # After the first chunk, no need to write the header again

        print(f'Chunk {i//CHUNK_SIZE + 1} saved to {FINAL_DATAFRAME}')

    # Close the database connection
    con.close()
    print('Database connection closed.')


if __name__ == "__main__":
    run_function_with_overrides(create_db_chunked, config)