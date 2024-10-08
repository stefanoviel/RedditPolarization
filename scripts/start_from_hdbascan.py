import os
import sys
import argparse

# adding root directory to paths
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from logging_config import configure_get_logger
import config
import time

from src.clustering import hdbscan_cluster_data
from src.utils.function_runner import run_function_with_overrides
from src.tf_idf import run_tf_idf
from src.quiz_llm import run_quiz_multiple_times

# this script assume that the embedding have already been created
def main(partial_fit_hdbscan, tf_idf_file):
    config.PARTIAL_FIT_CLUSTER = partial_fit_hdbscan
    config.PARTIAL_FIT_DIM_REDUCTION = 0.1
    folder = os.path.join(config.OUTPUT_DIR, 'tf_idfs')  # create a folder to save all tf-idf file 
    
    if not os.path.exists(folder):
        os.makedirs(folder)
    config.TFIDF_FILE = os.path.join(folder, tf_idf_file)

    if not os.path.exists(config.OUTPUT_DIR):
        os.makedirs(config.OUTPUT_DIR)
    
    logger = configure_get_logger(config.OUTPUT_DIR, config.EXPERIMENT_NAME, executed_file_name=__file__)

    time_hdbscan = run_function_with_overrides(hdbscan_cluster_data, config)
    time_tfidf = run_function_with_overrides(run_tf_idf, config)
    time_quiz = run_function_with_overrides(run_quiz_multiple_times, config)

    logger.info("-------------------------------")
    logger.info(f"Time for HDBSCAN: {time_hdbscan} s")
    logger.info(f"Time for TF-IDF: {time_tfidf:,} s")
    logger.info(f"Time for quiz: {time_quiz} s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run main script with arguments for PARTIAL_FIT_DIM_REDUCTION and TFIDF_FILE")
    parser.add_argument("--partial_fit_hdbscan", type=float, required=True, help="Float value for PARTIAL_FIT_DIM_REDUCTION")
    parser.add_argument("--tfidf_file", type=str, required=True, help="Path to the TFIDF file")

    args = parser.parse_args()

    main(args.partial_fit_hdbscan, args.tfidf_file)
