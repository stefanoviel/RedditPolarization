import os
import sys

# adding root directory to paths
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from logging_config import configure_get_logger
import config
import time

from src.embed_dataset import create_and_save_embeddings
from src.dimensionality_reduction import UMAP_partial_fit_partial_transform
from src.clustering import hdbscan_cluster_data
from src.utils.function_runner import run_function_with_overrides
from src.tf_idf import run_tf_idf


def main():
    if not os.path.exists(config.OUTPUT_DIR):
        os.makedirs(config.OUTPUT_DIR)
        
    logger = configure_get_logger(config.OUTPUT_DIR, config.EXPERIMENT_NAME, executed_file_name = __file__)

    # time_embeddings = run_function_with_overrides(create_and_save_embeddings, config)
    time_umap = run_function_with_overrides(UMAP_partial_fit_partial_transform, config)
    # time_hdbscan = run_function_with_overrides(hdbscan_cluster_data, config)
    # time_tfidf = run_function_with_overrides(run_tf_idf, config)


    logger.info("-------------------------------")
    # logger.info(f"Time for embeddings: {time_embeddings/60:,} min")
    logger.info(f"Time for UMAP: {time_umap} s")
    # logger.info(f"Time for HDBSCAN: {time_hdbscan} s")
    # logger.info(f"Time for TF-IDF: {time_tfidf:,} s")


if __name__ == "__main__":
    main()

