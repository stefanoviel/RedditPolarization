
import os
import sys 
# adding root directory to paths
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from logging_config import configure_get_logger
import config

from typing import Dict, Any, Optional

def keep_relevant_comments(line_json: Dict[str, Any], min_num_characters:int = 20):
    """Filters out the non relevant rows, only keep some columns"""
    if (len(line_json['body']) > min_num_characters):
        desired_attributes = {'score', 'subreddit', 'author', 'created_utc', 'id',
                              'parent_id'}  # Specify the attributes you want to keep
        filtered = {key: value for key, value in line_json.items() if key in desired_attributes}
        filtered['title'] = line_json['body'] # rename to create consistency with submissions
        return filtered
    else:
        return None

def keep_relevant_submissions(line_json: Dict[str, Any], min_num_characters:int = 20):
    """Filters out the non relevant rows, only keep some columns"""
    if (len(line_json['title']) > min_num_characters) and (line_json['media'] is None):
        desired_attributes = {'subreddit', 'score', 'author', 'created_utc', 'title', 'id', 'num_comments',
                              'upvote_ratio'}  # Specify the attributes you want to keep
        return {key: value for key, value in line_json.items() if key in desired_attributes}
    else:
        return None


def create_comment_submissions_table(table_name, sql_db):
    sql_db.sql(f"DROP TABLE IF EXISTS {table_name}")
    sql_db.sql(
        f"CREATE TABLE {table_name} (NUM INT PRIMARY KEY , score INT, subreddit VARCHAR, author VARCHAR, created_utc INTEGER, "
        "title VARCHAR , id VARCHAR, parent_id VARCHAR )")