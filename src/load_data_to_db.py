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


from src.unpack_zst import read_lines_zst
from typing import Dict, Any, Optional
import json
import duckdb
from tqdm import tqdm
import zstandard as zstd
from concurrent.futures import ProcessPoolExecutor, as_completed
import threading
import time
import duckdb

# Global counter and lock for thread-safe operations
row_count = 0
row_count_lock = threading.Lock()
stop_monitor = False
 
def keep_relevant_submissions(line_json, min_num_characters=20):
    if len(line_json.get('title', '') + line_json.get('selftext', '')) > min_num_characters:
        desired_attributes = {'subreddit', 'score', 'author', 'created_utc', 'title', 'id', 'num_comments', 'upvote_ratio', 'selftext'}
        return {key: line_json[key] for key in desired_attributes if key in line_json}
    return None


def setup_database(db_file):
    # Establish a connection to handle schema setup.
    conn = duckdb.connect(db_file)
    conn.execute("DROP TABLE IF EXISTS submissions")
    conn.execute("""CREATE TABLE IF NOT EXISTS submissions (
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


def process_file(file_path, db_file, index, batch_size=100):
    print(f"Thread {index} started")
    local_conn = duckdb.connect(db_file).cursor()  # Create a DuckDB cursor for the thread

    buffer = []  # Buffer to hold data rows before batch insert

    with open(file_path, 'rb') as file_handle:
        reader = zstd.ZstdDecompressor(max_window_size=2147483648).stream_reader(file_handle)
        while True:
            chunk = reader.read(2**27)  # Read 128 MB at a time
            if not chunk:
                break
            data = chunk.decode('utf-8').split('\n')
            for line in data[:-1]:
                try:
                    line_json = json.loads(line)
                    filtered_data = keep_relevant_submissions(line_json)
                    if filtered_data:
                        buffer.append(filtered_data)
                        if len(buffer) >= batch_size:
                            insert_into_db(local_conn, buffer)
                            buffer.clear()
                            with row_count_lock:
                                global row_count
                                row_count += batch_size 
                except json.JSONDecodeError:
                    continue

    # Insert any remaining rows in the buffer after loop ends
    if buffer:
        insert_into_db(local_conn, buffer)

    print(f"Thread {index} finished")

def insert_into_db(connection, data_batch):
    if not data_batch:
        return

    # Assuming all data in batch have the same keys
    expected_keys = {'subreddit', 'score', 'author', 'created_utc', 'title', 'id', 'num_comments', 'selftext'}
    columns = ', '.join(data_batch[0].keys())
    placeholders = ', '.join(['?'] * len(data_batch[0]))
    query = f"INSERT INTO submissions ({columns}) VALUES ({placeholders})"

   # Filter and validate data entries
    valid_data_batch = []
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
            print("No valid data to insert.")
    except Exception as e:
        print(f"Error during database insert operation: {e}")
        connection.rollback()

    # if len(valid_data_batch) < len(data_batch):
    #     print(f"Removed {len(data_batch) - len(valid_data_batch)} invalid entries out of {len(data_batch)} total.")


def monitor_rows():
    global stop_monitor
    while not stop_monitor:
        with row_count_lock:
            print(f"Total rows written: {row_count:,}", end='\r')
        time.sleep(1)  # Update every second

def main(directory, db_file):
    global stop_monitor

    setup_database(db_file)

    threads = []
    count = 0 
    for file_name in os.listdir(directory):
        if file_name.endswith('.zst'):
            file_path = os.path.join(directory, file_name)
            thread = threading.Thread(target=process_file, args=(file_path, db_file, count, 20000))
            threads.append(thread)
            thread.start()
            count += 1

    monitor_thread = threading.Thread(target=monitor_rows)
    monitor_thread.start()

    for thread in threads:
        thread.join()

    stop_monitor = True
    monitor_thread.join()
    print(f"Final total rows written: {row_count}")



if __name__ == '__main__':

    logger.info("Loading data to DuckDB")
    main(config.REDDIT_DATA_DIR, config.REDDIT_DB_FILE)
    logger.info("Data loaded successfully")