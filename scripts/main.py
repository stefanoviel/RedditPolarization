import os
import sys

# adding root directory to paths
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from logging_config import configure_get_logger
import config

from src.embed_dataset import process_and_save_embeddings
from src.load_data_to_db import main_load_files_in_db
from src.dimensionality_reduction import UMAP_transform_partial_fit
from src.hdbscan import run_dbscan


def main():
    if not os.path.exists(config.OUTPUT_DIR):
        os.makedirs(config.OUTPUT_DIR)
        
    logger = configure_get_logger(config.OUTPUT_DIR, config.EXPERIMENT_NAME, executed_file_name = __file__)

    # main_load_files_in_db(
    #     config.REDDIT_DATA_DIR,
    #     config.TABLE_NAME,
    #     config.ATTRIBUTE_TO_EXTRACT,
    #     config.MIN_POST_LENGTH,
    #     config.MIN_SCORE,
    #     config.SUBSET_FRACTION,
    # )

    process_and_save_embeddings(
        config.DB_FILEPATH,
        config.MODEL_NAME,
        config.TABLE_NAME,
        config.MODEL_BATCH_SIZE,
        config.EMBEDDINGS_FILE
    )

    UMAP_transform_partial_fit(
        config.EMBEDDINGS_FILE,
        config.UMAP_N_Neighbors,
        config.UMAP_COMPONENTS,
        config.UMAP_MINDIST,
        config.PARTIAL_FIT_SAMPLE_SIZE,
        config.DIMENSIONALITY_REDUCTION_FILE,
    )

    run_dbscan(
        HDBS_MIN_CLUSTERSIZE,
        HDBS_MIN_SAMPLES,
        DIMENSIONALITY_REDUCTION_FILE,
        CLUSTER_FILE
    )


if __name__ == "__main__":
    main()
