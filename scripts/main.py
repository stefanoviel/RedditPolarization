import os
import sys

# adding root directory to paths
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from logging_config import configure_get_logger
import config
import time

from src.embed_dataset import process_and_save_embeddings
from src.dimensionality_reduction import UMAP_transform_partial_fit
from src.hdbscan import run_dbscan
from src.utils.function_runner import run_function_with_overrides


def main():
    if not os.path.exists(config.OUTPUT_DIR):
        os.makedirs(config.OUTPUT_DIR)
        
    logger = configure_get_logger(config.OUTPUT_DIR, config.EXPERIMENT_NAME, executed_file_name = __file__)

    memory_embeddings, time_embeddings = run_function_with_overrides(process_and_save_embeddings, config)
    time.sleep(5)  # wait for the GPU to free up memory
    memory_umap, time_umap = run_function_with_overrides(UMAP_transform_partial_fit, config)
    time.sleep(5)  # wait for the GPU to free up memory
    memory_hdbscan, time_hdbscan = run_function_with_overrides(run_dbscan, config)

    logger.info("-------------------------------")
    logger.info("Memory and time usage:")
    logger.info(f"Memory usage for embeddings: {memory_embeddings} MB")
    logger.info(f"Time for embeddings: {time_embeddings} s")
    logger.info(f"Memory usage for UMAP: {memory_umap} MB")
    logger.info(f"Time for UMAP: {time_umap} s")
    logger.info(f"Memory usage for HDBSCAN: {memory_hdbscan} MB")
    logger.info(f"Time for HDBSCAN: {time_hdbscan} s")


if __name__ == "__main__":
    main()
