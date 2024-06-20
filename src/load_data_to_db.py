import os
import sys 
# adding root directory to paths
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from logging_config import configure_get_logger
import config
if not os.path.exists(config.OUTPUT_PATH):
    os.makedirs(config.OUTPUT_PATH)
logger = configure_get_logger(config.OUTPUT_PATH)
os.environ['NUMEXPR_MAX_THREADS'] = '32'  
import numexpr

from typing import Dict, Any, Optional, List
import json
import duckdb
from tqdm import tqdm
import zstandard as zstd
from concurrent.futures import ProcessPoolExecutor, as_completed
import threading
import time
import duckdb
from langdetect import detect

# Global counter and lock for thread-safe operations
row_count = 0
discarded_rows = 0
row_count_lock = threading.Lock()
stop_monitor = False

def filter_submissions_by_content_length(line_json: Dict[str, Any], min_num_characters: int = 20, min_score: int = 10) -> Optional[Dict[str, Any]]:
    """Filter out submissions with less than min_num_characters in the title and selftext, 
    ensuring they're in English, and have a minimum number of interactions."""

    # if there is media don't include self text in the embedding as it will be an URL
    if line_json.get('media', None) is not None:
        selftext = ''
    else:
        selftext = line_json.get('selftext', '')

    combined_text = line_json.get('title', '') + selftext
    if len(combined_text) <= min_num_characters:
        return None
    
    score = line_json.get('score', 0)
    if - min_score < score  < min_score: # a comment is relevant if it obtained a very positive or very negative response
        return None
    
    # Language detection to filter only English posts PROBLEM: it is too slow
    # try:
    #     if detect(combined_text) != 'en':
    #         return None
    # except:
    #     return None  

    # Select desired attributes
    desired_attributes = {'subreddit', 'score', 'author', 'created_utc', 'title', 'id', 'num_comments', 'upvote_ratio', 'selftext'}
    filtered_post = {key: line_json[key] for key in desired_attributes if key in line_json}
    filtered_post['selftext'] = selftext  # Update selftext based on media presence
    
    return filtered_post
 


def setup_database_create_table(db_file: str) -> None:
    """Set up the database schema for storing Reddit posts."""

    conn = duckdb.connect(db_file)
    conn.execute(f"DROP TABLE IF EXISTS {config.TABLE_NAME}")
    conn.execute(f"""CREATE TABLE IF NOT EXISTS {config.TABLE_NAME} (
        subreddit TEXT,
        score INTEGER,
        author TEXT,
        created_utc INTEGER,
        title TEXT,
        id TEXT PRIMARY KEY,
        num_comments INTEGER,
        upvote_ratio REAL,
        selftext TEXT
    )""")
    conn.close()


def add_compressed_file_to_db(file_path: str, db_file: str, index: int, batch_size: int) -> None:
    """Read a zstd-compressed file and insert relevant data into a DuckDB database."""

    logger.info(f"Thread {index} started")
    local_conn = duckdb.connect(db_file).cursor()  # Create a DuckDB cursor for the thread
    discarded_rows_local = 0
    rows_buffer = [] 
    with open(file_path, 'rb') as file_handle:
        # max_window_size is set to 2 GB otherwise the default 128 MB is too small for some files
        reader = zstd.ZstdDecompressor(max_window_size=2147483648).stream_reader(file_handle)
        while True:
            chunk = reader.read(2**27)  # Read 128 MB at a time
            if not chunk:
                break
            data = chunk.decode('utf-8').split('\n')
            for line in data:
                try:
                    line_json = json.loads(line)
                    # print(json.dumps(line_json, indent=4))
                    filtered_rows = filter_submissions_by_content_length(line_json, config.MIN_POST_LENGTH, config.MIN_SCORE)
                    if filtered_rows:
                        rows_buffer.append(filtered_rows)
                        if len(rows_buffer) >= batch_size:
                            insert_into_db(local_conn, rows_buffer)
                            rows_buffer.clear()
                            with row_count_lock: # Keep track of the total number of rows written
                                global row_count
                                row_count += batch_size 
                    else:
                        discarded_rows_local += 1


                except json.JSONDecodeError:
                    continue

    # Insert any remaining rows in the buffer after loop ends
    if rows_buffer:
        insert_into_db(local_conn, rows_buffer)
    
    with row_count_lock:
        global discarded_rows
        discarded_rows += discarded_rows_local

    logger.info(f"Thread {index} finished. File {file_path} processed. Added {row_count:,} rows to the database. Discarded {discarded_rows:,} rows. Percentage discarded: {discarded_rows / (row_count + discarded_rows) * 100:.2f}%")


def monitor_rows() -> None:
    """Monitor the total number of rows written to the database."""
    global stop_monitor
    while not stop_monitor:
        with row_count_lock:
            print(f"Total rows written: {row_count:,}", end='\r')
        time.sleep(1)  # Update every second

def insert_into_db(connection: duckdb.DuckDBPyConnection, data_batch: List[Dict[str, Any]]) -> None:
    """Insert a batch of data into the database."""

    if not data_batch:
        return

    expected_keys = {'subreddit', 'score', 'author', 'created_utc', 'title', 'id', 'num_comments', 'selftext'}
    columns = ', '.join(data_batch[0].keys())
    placeholders = ', '.join(['?'] * len(data_batch[0]))
    query = f"INSERT INTO {config.TABLE_NAME} ({columns}) VALUES ({placeholders})"

    valid_data_batch = []

    # Filter out rows with missing or None values 
    for row in data_batch:
        if set(row.keys()) == expected_keys and None not in row.values():
            valid_data_batch.append(row)

    # Prepare values for insertion
    values = [tuple(row[key] for key in expected_keys) for row in valid_data_batch]

    try:
        if values:
            connection.begin()
            connection.executemany(query, values)
            connection.commit()
        else:
            logger.warning("No valid data to insert.")
    except Exception as e:
        logger.warning(f"Error during database insert operation: {e}")
        connection.rollback()


def main_load_files_in_db() -> None:
    """Main function to load Reddit data into a DuckDB database."""
    global stop_monitor

    setup_database_create_table(config.REDDIT_DB_FILE)

    threads = []
    count = 0 
    for file_name in os.listdir(config.REDDIT_DATA_DIR):
        if file_name.endswith('.zst'):
            file_path = os.path.join(config.REDDIT_DATA_DIR, file_name)

            # batches of 20k rows seems a good trade-off between writing speed and and overhead due to the commit operation
            thread = threading.Thread(target=add_compressed_file_to_db, args=(file_path, config.REDDIT_DB_FILE, count, 20000)) 
            threads.append(thread)
            thread.start()
            count += 1

    monitor_thread = threading.Thread(target=monitor_rows)
    monitor_thread.start()

    for thread in threads:
        thread.join()

    stop_monitor = True
    monitor_thread.join()
    logger.info(f"Final total rows written: {row_count:,} rows. Discarded {discarded_rows:,} rows. Percentage discarded: {discarded_rows / (row_count + discarded_rows) * 100:.2f}%")



if __name__ == '__main__':
    # just for testing
    main_load_files_in_db()