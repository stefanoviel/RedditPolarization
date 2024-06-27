import os
import sys

# adding root directory to paths
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from logging_config import configure_get_logger
import config

if not os.path.exists(config.OUTPUT_DIR):
    os.makedirs(config.OUTPUT_DIR)
logger = configure_get_logger(config.OUTPUT_DIR, config.EXPERIMENT_NAME, executed_file_name = __file__)
os.environ["NUMEXPR_MAX_THREADS"] = "32"
import numexpr

import random
from typing import Dict, Any, Optional, List
import json
from tqdm import tqdm
import zstandard as zstd
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import time
from langdetect import detect
from src.utils_run_single_step import run_function_with_overrides
import psycopg2
import yaml
from psycopg2.extras import execute_values


def load_config():
    with open('db_config.yaml', 'r') as file:
        return yaml.safe_load(file)

def create_database_connection() -> psycopg2.extensions.connection:
    """Create and return a database connection using the provided configuration."""

    config = load_config()
    db_config = config['db']

    return psycopg2.connect(**db_config)

def filter_submissions_by_content_length(
    line_json: Dict[str, Any], attributes_to_extract: set, min_num_characters: int = 20, min_score: int = 10
) -> Optional[Dict[str, Any]]:
    """Filter out submissions with less than min_num_characters in the title and selftext,
    ensuring they're in English, and have a minimum number of interactions."""

    # if there is media don't include self text in the embedding as it will be an URL
    if line_json.get("media", None) is not None:
        selftext = ""
    else:
        selftext = line_json.get("selftext", "")

    # TODO: check detect english    

    combined_text = line_json.get("title", "") + selftext
    score = line_json.get("score", 0)
    if len(combined_text) <= min_num_characters or  (score < -min_score)  and (score > min_score): 
        return None

    filtered_post = {
        key: line_json[key] for key in attributes_to_extract if key in line_json
    }
    filtered_post["selftext"] = selftext  # Update selftext based on media presence

    return filtered_post


def setup_database_create_table(table_name) -> None:
    """Set up the database schema for storing Reddit posts."""

    conn = create_database_connection()
    with conn.cursor() as cursor:
        cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
        cursor.execute(
            f"""CREATE TABLE IF NOT EXISTS {table_name} (
            subreddit TEXT,
            score INTEGER,
            author TEXT,
            created_utc INTEGER,
            title TEXT,
            id TEXT PRIMARY KEY,
            num_comments INTEGER,
            selftext TEXT
        
        )"""
        ) # TODO: add media
    conn.commit()
    conn.close()


def add_compressed_file_to_db(
    file_path: str,
    index: int,
    batch_size: int,
    min_post_length: int,
    min_score: int,
    table_name: str,
    attributes_to_extract: set,
    subset_fraction: float
) -> None:
    """Read a zstd-compressed file and insert relevant data into a  database."""

    logger.info(f"Process {index} started")
    local_conn = create_database_connection() # Create a  cursor for the thread
    discarded_rows = 0
    row_count = 0
    rows_buffer = []
    with open(file_path, "rb") as file_handle:
        # max_window_size is set to 2 GB otherwise the default 128 MB is too small for some files
        reader = zstd.ZstdDecompressor(max_window_size=2147483648).stream_reader(
            file_handle
        )
        while True:
            chunk = reader.read(2**27)  # Read 128 MB at a time
            if not chunk:
                break

            try: 
                data = chunk.decode("utf-8").split("\n")
            except UnicodeDecodeError:
                continue
            
            for line in data:
                if random.random() > subset_fraction:
                    continue

                try:
                    line_json = json.loads(line)
                    # print(json.dumps(line_json, indent=4))
                    filtered_rows = filter_submissions_by_content_length(
                        line_json, attributes_to_extract, min_post_length, min_score
                    )
                    if filtered_rows:
                        rows_buffer.append(filtered_rows)
                        if len(rows_buffer) >= batch_size:
                            inserted_rows = insert_into_db(local_conn, rows_buffer, table_name, attributes_to_extract)
                            row_count += len(inserted_rows)
                            rows_buffer.clear()
                    else:
                        discarded_rows += 1

                except json.JSONDecodeError:
                    continue

    # Insert any remaining rows in the buffer after loop ends
    if rows_buffer:
        inserted_rows = insert_into_db(local_conn, rows_buffer, table_name, attributes_to_extract)
        row_count += len(inserted_rows)


    logger.info(
        f"Process {index} finished. File {file_path} processed. Added {row_count:,} rows to the database. Discarded {discarded_rows:,} rows. Percentage discarded: {discarded_rows / (row_count + discarded_rows) * 100:.2f}%"
    )




def insert_into_db(conn, data_batch: List[Dict[str, Any]], table_name:str, attributes_to_extract:set) -> None:
    """
    Inserts data into a PostgreSQL database from a list of dictionaries.
    Each dictionary represents a row to be inserted, with the keys as column names.
    
    :param conn: psycopg2 connection object to the database
    :param data_batch: list of dictionaries, where each dictionary represents data for one row
    """
    # Check if data_batch is empty
    if not data_batch:
        print("No data to insert.")
        return

    # Extracting all column names from the first dictionary assuming all dictionaries have the same keys
    columns = data_batch[0].keys()
    column_names = ", ".join(columns)
    value_placeholders = ", ".join(["%s"] * len(columns))

    # Preparing the INSERT INTO statement

    sql = f"INSERT INTO {table_name} ({column_names}) VALUES ({value_placeholders})"

    # Prepare the list of tuples for the executemany() function
    values_to_insert = []
    for data in data_batch:
        if set(data.keys()) == attributes_to_extract:
            values_to_insert.append(tuple(data[col] for col in columns))

    # Create a cursor and execute the insertion
    cursor = conn.cursor()
    try:
        cursor.executemany(sql, values_to_insert)
        conn.commit()

        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        row_count = cursor.fetchone()[0]

    except Exception as e:
        conn.rollback()
        print(f"An error occurred: {e}")
    finally:
        cursor.close()

    return values_to_insert


def main_load_files_in_db(
    REDDIT_DATA_DIR: str, TABLE_NAME: str, ATTRIBUTE_TO_EXTRACT: set, MIN_POST_LENGTH: int, MIN_SCORE: int, SUBSET_FRACTION: float
) -> None:
    """Main function to load Reddit data into a database."""
    global stop_monitor

    setup_database_create_table(TABLE_NAME)

    processes = []
    count = 0
    for file_name in os.listdir(REDDIT_DATA_DIR):
        if file_name.endswith(".zst"):
            file_path = os.path.join(REDDIT_DATA_DIR, file_name)

            # batches of 20k rows seems a good trade-off between writing speed and and overhead due to the commit operation
            process = multiprocessing.Process(
                target=add_compressed_file_to_db,
                args=(file_path, count, 20000, MIN_POST_LENGTH, MIN_SCORE, TABLE_NAME, ATTRIBUTE_TO_EXTRACT, SUBSET_FRACTION),
            )
            processes.append(process)
            process.start()
            count += 1

    for process in processes:
        process.join()


if __name__ == "__main__":
    # just for testing    
    run_function_with_overrides(main_load_files_in_db, config)