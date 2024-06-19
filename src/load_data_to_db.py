
import os
import sys 
# adding root directory to paths
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from logging_config import configure_get_logger
import config
logger = configure_get_logger(__name__)


from src.unpack_zst import read_lines_zst
from typing import Dict, Any, Optional
import json

    
def keep_relevant_submissions(line_json: Dict[str, Any], min_num_characters: int = 20) -> Optional[Dict[str, Any]]:
    """Filters out submission with a lenght lower than `min_num_characters"""
    if len(line_json.get('title', '') + line_json.get('selftext', '') ) > min_num_characters:
        desired_attributes = {'subreddit', 'score', 'author', 'created_utc', 'title', 'id', 'num_comments', 'upvote_ratio', 'selftext'}
        return {key: line_json[key] for key in desired_attributes if key in line_json}
    return None


def create_submissions_table(table_name, sql_db):
    """Creates a table in the database if it doesn't exist."""
    sql_db.sql(f"DROP TABLE IF EXISTS {table_name}")
    sql_db.sql(
        f"CREATE TABLE {table_name} (NUM INT PRIMARY KEY , score INT, subreddit VARCHAR, author VARCHAR, created_utc INTEGER, "
        "title VARCHAR , id VARCHAR, parent_id VARCHAR, body VARCHAR)")
    

def _add_to_db_json(good_lines, table_name, sql_db):
    with open(temp_file_path_j, 'w') as f:
        json.dump(good_lines, f)
    sql_db.execute(
        f"COPY {table_name} FROM '{temp_file_path_j}' (FORMAT JSON, AUTO_DETECT true) ;")
    #print("added to collection", len(good_lines))
    

def extract_submissions(input_file_name: str, sql_db, table_name: str):
    reader_chunk_size: int = 2 ** 27
    reader_window_size: int = 2 ** 31
    good_lines = []

    if ((table_name,) in sql_db.execute("SHOW TABLES;").fetchall()):
        max_id = sql_db.sql(f"SELECT MAX(num) FROM {table_name}").fetchone()
        if max_id[0] is not None:
            good_lines_count = max_id[0] + 1
        else: 
            good_lines_count = 0
    else: 
        create_submissions_table(table_name, sql_db)
        good_lines_count = 0
    #initialize tracking variables
    bad_lines, file_lines, file_bytes_processed = 0, 0, 0
    file_size = os.stat(input_file_name).st_size

    # Loop through every line in the file
    for line, file_bytes_processed in read_lines_zst(file_name=input_file_name, reader_window_size=reader_window_size,
                                                     reader_chunk_size=reader_chunk_size):
        try:
            line_json = json.loads(line)
            filtered_line = _filter_submission(line_json)  #do basic preselection of lines
            if filtered_line is not None:
                filtered_line['NUM'] = good_lines_count
                good_lines_count += 1
                good_lines.append(filtered_line)  #add to list of valid lines

        except (KeyError, json.JSONDecodeError) as err:
            bad_lines += 1
        file_lines += 1

        # Log progress
        if file_lines % 200_000 == 0:
            logger.info(
                f": {good_lines_count:,} {file_lines:,} : {bad_lines:,} : {file_bytes_processed:,}:{(file_bytes_processed / file_size) * 100:.0f}%")

        # Write the lines to the db file
        if len(good_lines) > 10_000:  #for now only opening json file once is fastest..
            _add_to_db_json(good_lines, table_name, sql_db)
            good_lines = []

    #add last bit of lines
    _add_to_db_json(good_lines, table_name, sql_db)

    print('total added lines: ', sql_db.sql(f"SELECT COUNT(*) FROM {table_name}"))
    logger.info(f"Complete : {file_lines:,} : {bad_lines:,} : {good_lines_count:,}")