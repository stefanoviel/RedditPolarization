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
import psycopg2
import h5py
import yaml
from tqdm import tqdm
from src.utils.function_runner import run_function_with_overrides


def create_database_connection(parquet_directory:str, table_name:str):
    """Create and return a database connection using the provided configuration."""
    files = [f'{parquet_directory}/{file}' for file in os.listdir(parquet_directory)]

    con = duckdb.connect(database=':memory:')

    # # Construct a SQL statement to read all files
    query_files = ', '.join(f"'{f}'" for f in files)
    sql_query = f"CREATE TABLE {table_name} AS SELECT author, id, title, selftext, score, num_comments, subreddit, created_utc  FROM read_parquet([{query_files}], union_by_name=True)"
    con.execute(sql_query)
    return con
    

def initialize_model(model_name:str) -> SentenceTransformer:
    """
    Initialize the Sentence Transformer model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SentenceTransformer(model_name)
    model.to(device)
    model.eval()
    return model

def prepare_texts(data:list[tuple[str, str]]) -> list[str]:
    """
    Prepare texts by concatenating title and selftext.
    """
    return [title + " " + text for title, text in data]


def generate_embeddings(model:SentenceTransformer, texts:list[str]) -> torch.Tensor:
    """
    Generate embeddings for the given texts using the provided model.
    """
    with torch.no_grad():
        # I tried different batch sizes, there is no significant difference in performance
        embeddings = model.encode(
            texts, show_progress_bar=False, convert_to_tensor=True, batch_size=1024, precision='float32'
        )
        embeddings = embeddings.cpu().numpy()  # Move embeddings back to CPU for storage
    return embeddings


# def count_rows(db_file_path:str, table_name:str) -> int:
#     """
#     Count the number of rows in the specified table.
#     """
#     cursor = create_database_connection(db_file_path)
#     cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
#     count = cursor.fetchone()[0]
#     cursor.close()

#     return count

def fetch_data_in_batches(con, table_name:str, batch_size: int):
    """Fetch data from the specified table in batches."""

    query = f"SELECT title, selftext FROM {table_name};"
    con.execute(query)

    while True:
        batch = con.fetchmany(batch_size)
        if not batch:
            break
        yield batch



def process_and_save_embeddings(REDDIT_DATA_DIR:str, MODEL_NAME: str, TABLE_NAME: str, MODEL_BATCH_SIZE: int, EMBEDDINGS_FILE: str):
    """Fetch data in batches, generate embeddings, and save them incrementally."""
    model = initialize_model(MODEL_NAME)

    con = create_database_connection(REDDIT_DATA_DIR, TABLE_NAME)

    # Open the HDF5 file for writing
    with h5py.File(EMBEDDINGS_FILE, "w") as f:
        # Initialize variables to keep track of dataset dimensions
        data_initialized = False
        dataset = None
        
        for batch in fetch_data_in_batches(con, TABLE_NAME, MODEL_BATCH_SIZE):
            texts = prepare_texts(batch)
            print(len(texts))
            if texts:
                
                embeddings = generate_embeddings(model, texts)

                if not data_initialized:
                    # Create dataset with initial dimensions and enable resizing
                    num_embeddings, embedding_dim = embeddings.shape
                    maxshape = (None, embedding_dim)  # Allow unlimited rows
                    dataset = f.create_dataset("embeddings", shape=(0, embedding_dim), maxshape=maxshape, dtype='float32', chunks=True)
                    data_initialized = True

                # Resize the dataset to accommodate new embeddings
                current_shape = dataset.shape
                new_shape = (current_shape[0] + embeddings.shape[0], embedding_dim)
                dataset.resize(new_shape)
                
                # Write new embeddings to the dataset
                print("inserting from ", current_shape[0], " to ", new_shape[0], " shape: ", embeddings.shape)
                dataset[current_shape[0]:new_shape[0], :] = embeddings
        

        logger.info(f"shape of the dataset: {dataset.shape}")   
        logger.info(f"Embeddings saved to {EMBEDDINGS_FILE}")





if __name__ == "__main__":
    # just for testing
    run_function_with_overrides(process_and_save_embeddings, config)
