import os
import sys

# adding root directory to paths
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from logging_config import configure_get_logger
import config

if not os.path.exists(config.OUTPUT_DIR):
    os.makedirs(config.OUTPUT_DIR)
logger = configure_get_logger(config.OUTPUT_DIR, config.EXPERIMENT_NAME, executed_file_name = __file__)

import torch
from sentence_transformers import SentenceTransformer
import duckdb
import h5py
import json
import yaml
from tqdm import tqdm
import langid

from src.utils.utils import create_database_connection
from src.utils.function_runner import run_function_with_overrides
    

def initialize_model(model_name:str) -> SentenceTransformer:
    """
    Initialize the Sentence Transformer model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SentenceTransformer(model_name)
    model.to(device)
    model.eval()
    return model


def prepare_texts_and_ids(data: list[tuple[int, str, str]]) -> tuple[list[str], list[int]]:
    """
    Prepare texts by concatenating title and selftext, and extract IDs, including only posts likely in English using langid.
    
    Args:
        data (list of tuple): Each tuple contains (id, title, selftext).
    
    Returns:
        tuple: A tuple containing a list of concatenated texts and a list of IDs, filtered by English language detection.
    """
    texts = []
    ids = []

    for id, title, text in tqdm(data):
        full_text = title + " " + text
        # Detect the language of the concatenated text
        lang, _ = langid.classify(full_text)
        if lang == 'en':
            texts.append(full_text)
            ids.append(id)

    return texts, ids


def generate_embeddings(model:SentenceTransformer, texts:list[str]) -> torch.Tensor:
    """
    Generate embeddings for the given texts using the provided model.
    """
    with torch.no_grad():
        # I tried different batch sizes, there is no significant difference in performance
        embeddings = model.encode(
            texts, show_progress_bar=True, convert_to_tensor=True, batch_size=1024, precision='float32'
        )
        embeddings = embeddings.cpu().numpy()  # Move embeddings back to CPU for storage
    return embeddings


def count_rows_to_embed(con, table_name:str,  min_score: int, min_post_length: int) -> int:
    """
    Count the number of rows in the specified table.
    """
    query = f"""SELECT COUNT(*)
            FROM {table_name}
            WHERE LENGTH(title) + LENGTH(selftext) > {min_post_length}
            AND score > {min_score};
            """

    con.execute(f"SELECT COUNT(*) FROM {table_name};")
    tot_rows = con.fetchone()[0]

    con.execute(query)
    rows_to_embed = con.fetchall()

    return tot_rows, rows_to_embed

def fetch_data_in_batches(con, table_name:str, batch_size: int, min_score: int, min_post_length: int):
    """Fetch data from the specified table in batches."""

    query = f"""SELECT id, title, selftext
            FROM {table_name}
            WHERE LENGTH(title) > {min_post_length}
            AND score > {min_score}
            AND selftext NOT LIKE '%[deleted]%'
            AND selftext NOT LIKE '%[removed]%'
            AND media = FALSE 
            """
    
    con.execute(query)

    while True:
        batch = con.fetchmany(batch_size)
        if not batch:
            break
        yield batch

def append_to_h5(file_path: str, embeddings, ids):
    """Append embeddings and IDs to the H5 file."""
    with h5py.File(file_path, "a") as f:
        embed_dset = f["embeddings"]
        ids_dset = f["ids"]
        
        current_size = embed_dset.shape[0]
        new_size = current_size + embeddings.shape[0]
        
        embed_dset.resize((new_size, embed_dset.shape[1]))
        embed_dset[current_size:new_size] = embeddings
        
        ids_dset.resize((new_size,))
        ids_dset[current_size:new_size] = ids


def initialize_h5_file(file_path: str, embedding_dim: int):
    """Initialize the H5 file with datasets for embeddings and IDs."""
    with h5py.File(file_path, "w") as f:
        f.create_dataset("embeddings", shape=(0, embedding_dim), maxshape=(None, embedding_dim), dtype='float32', chunks=True)
        f.create_dataset("ids", shape=(0,), maxshape=(None,), dtype=h5py.string_dtype(encoding='utf-8'), chunks=True)
    return file_path


def create_and_save_embeddings(REDDIT_DATA_DIR: str, MODEL_NAME: str, TABLE_NAME: str, MODEL_BATCH_SIZE: int, H5_FILE: str, MIN_SCORE: int, MIN_POST_LENGTH: int):
    """Fetch data in batches from db, generate embeddings, and save them incrementally along with their corresponding IDs."""
    model = initialize_model(MODEL_NAME)
    con = create_database_connection(REDDIT_DATA_DIR, TABLE_NAME, ["author", "id", "title", "selftext", "score", "num_comments", "subreddit", 'created_utc', "media"])
    
    h5_file = None
    total_processed = 0

    for batch in fetch_data_in_batches(con, TABLE_NAME, MODEL_BATCH_SIZE, MIN_SCORE, MIN_POST_LENGTH):
        texts, batch_ids = prepare_texts_and_ids(batch)
        
        if texts is not None:
            embeddings = generate_embeddings(model, texts)

            if h5_file is None:
                h5_file = initialize_h5_file(H5_FILE, embeddings.shape[1])
            
            append_to_h5(h5_file, embeddings, batch_ids)
            total_processed += len(batch_ids)
            
            logger.info(f"Processed {total_processed} items")

    logger.info(f"Embeddings and IDs saved to {H5_FILE}")
    logger.info(f"Total items processed: {total_processed}")

    with h5py.File(H5_FILE, "r") as f:
        logger.info(f"Final shape of the embeddings dataset: {f['embeddings'].shape}")
        logger.info(f"Final shape of the IDs dataset: {f['ids'].shape}")

# def create_and_save_embeddings(REDDIT_DATA_DIR: str, MODEL_NAME: str, TABLE_NAME: str, MODEL_BATCH_SIZE: int, EMBEDDINGS_FILE: str, IDS_FILE: str, MIN_SCORE: int, MIN_POST_LENGTH: int):
#     """Fetch data in batches from db, generate embeddings, and save them incrementally along with their corresponding IDs."""
#     model = initialize_model(MODEL_NAME)
#     con = create_database_connection(REDDIT_DATA_DIR, TABLE_NAME, ["author", "id", "title", "selftext", "score", "num_comments", "subreddit", 'created_utc', "media"])

#     tot_rows, rows_to_embed = count_rows_to_embed(con, TABLE_NAME, MIN_SCORE, MIN_POST_LENGTH)
#     logger.info(f"Total rows to embed: {tot_rows:,}, filtered rows: {rows_to_embed[0][0]:,}")

#     # Open the HDF5 file for writing embeddings and prepare a list for IDs
#     with h5py.File(EMBEDDINGS_FILE, "w") as f:
#         data_initialized = False
#         embeddings_dataset = None
#         ids = []
        
#         for batch in fetch_data_in_batches(con, TABLE_NAME, MODEL_BATCH_SIZE, MIN_SCORE, MIN_POST_LENGTH):
#             texts, batch_ids = prepare_texts_and_ids(batch)

#             if texts:
#                 embeddings = generate_embeddings(model, texts)

#                 if not data_initialized:
#                     num_embeddings, embedding_dim = embeddings.shape
#                     embeddings_maxshape = (None, embedding_dim)
#                     embeddings_dataset = f.create_dataset("data", shape=(0, embedding_dim), maxshape=embeddings_maxshape, dtype='float32', chunks=True)
#                     data_initialized = True

#                 current_shape = embeddings_dataset.shape
#                 new_shape = (current_shape[0] + embeddings.shape[0], embedding_dim)
#                 embeddings_dataset.resize(new_shape)
#                 embeddings_dataset[current_shape[0]:new_shape[0], :] = embeddings

#                 ids.extend(batch_ids)  # Append new IDs to the list

#         # Once all data is processed, save IDs to a JSON file
#         with open(IDS_FILE, 'w') as id_file:
#             json.dump(ids, id_file)

#         logger.info(f"shape of the embeddings dataset: {embeddings_dataset.shape}")
#         logger.info(f"Embeddings saved to {EMBEDDINGS_FILE}")
#         logger.info(f"IDs saved to {IDS_FILE}")




if __name__ == "__main__":
    # just for testing
    run_function_with_overrides(create_and_save_embeddings, config)