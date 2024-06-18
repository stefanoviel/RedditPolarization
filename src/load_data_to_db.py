
import os
import sys 
# adding root directory to paths
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from logging_config import configure_get_logger
import config



def _filter_comment(line_json):
    
    if (len(line_json['body']) > 20):
        desired_attributes = {'score', 'subreddit', 'author', 'created_utc', 'id',
                              'parent_id'}  # Specify the attributes you want to keep
        filtered = {key: value for key, value in line_json.items() if key in desired_attributes}
        filtered['title'] = line_json['body'] # rename to create consistency with submissions
        return filtered
    else:
        return None