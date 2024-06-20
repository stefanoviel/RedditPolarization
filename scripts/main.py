import os
import sys

# adding root directory to paths
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from logging_config import configure_get_logger
import config

from src.embed_dataset import main_embed_data
from src.load_data_to_db import main_load_files_in_db
from src.dimensionality_reduction import UMAP_transform_partial_fit


def main():
    if not os.path.exists(config.OUTPUT_PATH):
        os.makedirs(config.OUTPUT_PATH)
    logger = configure_get_logger(config.OUTPUT_PATH)

    main_load_files_in_db(
        config.REDDIT_DATA_DIR,
        config.REDDIT_DB_FILE,
        config.MIN_POST_LENGTH,
        config.MIN_SCORE,
    )

    main_embed_data(
        config.MODEL_NAME,
        config.REDDIT_DB_FILE,
        config.TABLE_NAME,
        config.EMBEDDINGS_FILE,
    )

    UMAP_transform_partial_fit(
        config.EMBEDDINGS_FILE,
        config.UMAP_N_Neighbors,
        config.UMAP_COMPONENTS,
        config.UMAP_MINDIST,
        config.PARTIAL_FIT_SAMPLE_SIZE,
        config.DIMENSIONALITY_REDUCTION_FILE,
    )


if __name__ == "__main__":
    main()
