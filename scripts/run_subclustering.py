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
from src.tf_idf import run_tf_idf
from src.quiz_llm import run_quiz_multiple_times


def main():
    logger = configure_get_logger(config.OUTPUT_DIR, config.EXPERIMENT_NAME, executed_file_name=__file__)

    time_hdbscan = run_function_with_overrides(apply_clustering_existing_clusters, config)

    # use tf idf just with different parameters
    config.CLUSTER_DB_NAME = config.SUBCLUSTER_DB_NAME
    config.TFIDF_FILE = config.SUBCLUSTER_TFIDF_FILE
    time_tfidf = run_function_with_overrides(run_tf_idf, config)

    logger.info("-------------------------------")
    logger.info(f"Time for HDBSCAN: {time_hdbscan} s")
    logger.info(f"Time for TF-IDF: {time_tfidf:,} s")


if __name__ == "__main__":
    main()
