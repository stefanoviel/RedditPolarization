
import os
import sys 
# adding root directory to paths
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from logging_config import configure_get_logger
import config

from src.embed_dataset import main_embed_data
from src.load_data_to_db import main_load_files_in_db


def main():
    if not os.path.exists(config.OUTPUT_PATH):
        os.makedirs(config.OUTPUT_PATH)
    logger = configure_get_logger(config.OUTPUT_PATH)

    main_load_files_in_db()
    main_embed_data()


if __name__ == '__main__':
    main()
