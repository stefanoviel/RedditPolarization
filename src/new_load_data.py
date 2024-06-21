import os
import sys

# adding root directory to paths
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from logging_config import configure_get_logger
import config

if not os.path.exists(config.OUTPUT_DIR):
    os.makedirs(config.OUTPUT_DIR)
logger = configure_get_logger(config.OUTPUT_DIR, executed_file_name = __file__)
os.environ["NUMEXPR_MAX_THREADS"] = "32"
import numexpr

from typing import Dict, Any, Optional, List, Tuple
import json
from tqdm import tqdm
import zstandard as zstd
from concurrent.futures import ProcessPoolExecutor, as_completed
import threading
import time
from langdetect import detect
from src.run_single_step import run_function_with_overrides
import psycopg2
from psycopg2.extras import execute_values

# Global counter and lock for thread-safe operations
row_count = 0
discarded_rows = 0
row_count_lock = threading.Lock()
stop_monitor = False



def create_database_connection(db_config: Dict[str, str]) -> psycopg2.extensions.connection:
    """Create and return a database connection using the provided configuration."""
    return psycopg2.connect(**db_config)

def setup_database_schema(cursor: psycopg2.extensions.cursor, table_name: str) -> None:
    """Set up the database schema for storing Reddit posts using the provided cursor and table name."""
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
        upvote_ratio REAL,
        selftext TEXT
    )""")

def filter_submission_by_criteria(submission: Dict[str, Any], min_length: int, min_score: int) -> Optional[Dict[str, Any]]:
    """Filter out submissions based on content length and score criteria."""
    if submission.get("media"):
        selftext = ""
    else:
        selftext = submission.get("selftext", "")

    combined_text = submission.get("title", "") + selftext
    if len(combined_text) <= min_length or (-min_score < submission.get("score", 0) < min_score):
        return None

    # Create a dictionary of desired attributes if they exist in the submission
    desired_attributes = {"subreddit", "score", "author", "created_utc", "title", "id", "num_comments", "upvote_ratio", "selftext"}
    filtered_post = {key: submission[key] for key in desired_attributes if key in submission}
    filtered_post["selftext"] = selftext
    return filtered_post

def insert_data_batch(cursor: psycopg2.extensions.cursor, data_batch: List[Dict[str, Any]], table_name: str) -> None:
    """Insert a batch of data into the database using the provided cursor."""
    if not data_batch:
        return

    columns = data_batch[0].keys()
    placeholders = ", ".join(["%s"] * len(columns))
    sql = f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES ({placeholders})"

    values_to_insert = [tuple(data[col] for col in columns) for data in data_batch]
    cursor.executemany(sql, values_to_insert)
    cursor.connection.commit()


def process_compressed_file(file_path: str, db_cursor: psycopg2.extensions.cursor, min_post_length: int, min_score: int, table_name: str,  batch_size: int,) -> Tuple[int, int]:
    """Process a compressed file, filter data, and insert it into the database, returning counts of inserted and discarded rows."""
    inserted_rows = 0
    discarded_rows = 0
    discarded_rows_local = 0
    local_row_count = 0
    rows_buffer = []
    with open(file_path, 'rb') as file_handle:
        reader = zstd.ZstdDecompressor().stream_reader(file_handle)
        while chunk := reader.read(2**27):  # Read 128 MB at a time
            data = chunk.decode('utf-8').split('\n')
            data_batch = []
            for line in data:
                try:
                    submission = json.loads(line)
                    filtered_submission = filter_submission_by_criteria(submission, min_post_length, min_score)

                    if filtered_submission:
                        rows_buffer.append(filtered_submission)
                        if len(rows_buffer) >= batch_size:
                            insert_data_batch(db_cursor, rows_buffer, table_name)
                            rows_buffer.clear()
                            with (row_count_lock):  # Keep track of the total number of rows written
                                global row_count
                                row_count += batch_size
                                local_row_count += batch_size           

                except json.JSONDecodeError:
                    continue

    if data_batch:
        insert_data_batch(db_cursor, data_batch, table_name)
        inserted_rows += len(data_batch)

    with row_count_lock:
        global discarded_rows
        discarded_rows += discarded_rows_local

    return inserted_rows, discarded_rows

def monitor_rows() -> None:
    """Monitor the total number of rows written to the database."""
    global stop_monitor
    while not stop_monitor:
        with row_count_lock:
            print(f"Total rows written: {row_count:,}", end="\r")
        time.sleep(1)  # Update every second


def start_file_processing_thread(file_path: str, db_cursor: psycopg2.extensions.cursor, min_post_length: int, min_score: int, table_name: str, thread_index: int) -> threading.Thread:
    """Start a new thread to process a compressed file and insert data into the database."""
    thread = threading.Thread(
        target=process_compressed_file, 
        args=(file_path, db_cursor, min_post_length, min_score, table_name)
    )
    thread.start()
    return thread
