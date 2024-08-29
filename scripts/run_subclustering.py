import os
import sys
import argparse

# adding root directory to paths
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from logging_config import configure_get_logger
import config
import time

from src.clustering import apply_clustering_existing_clusters
from src.utils.function_runner import run_function_with_overrides
from src.tf_idf import tf_idf_on_subclusters
from src.naming_topics import naming_topics_in_tfidf_file


def main():
    logger = configure_get_logger(config.OUTPUT_DIR, config.EXPERIMENT_NAME, executed_file_name=__file__)

    time_hdbscan = run_function_with_overrides(apply_clustering_existing_clusters, config)
    time_tfidf = run_function_with_overrides(tf_idf_on_subclusters, config)
    # time_naming_topics = run_function_with_overrides(naming_topics_in_tfidf_file, config)

    logger.info("-------------------------------")
    logger.info(f"Time for HDBSCAN: {time_hdbscan} s")
    logger.info(f"Time for TF-IDF: {time_tfidf:,} s")
    # logger.info(f"Time for naming topics: {time_naming_topics} s")


if __name__ == "__main__":
    main()
