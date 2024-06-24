import os
import sys

# adding root directory to paths
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from logging_config import configure_get_logger
import config

if not os.path.exists(config.OUTPUT_DIR):
    os.makedirs(config.OUTPUT_DIR)
logger = configure_get_logger(config.OUTPUT_DIR, executed_file_name = __file__)


import torch
from sentence_transformers import SentenceTransformer
import duckdb
import psycopg2
import h5py
import yaml
from tqdm import tqdm
from utils_run_single_step import run_function_with_overrides


def load_config():
    with open('db_config.yaml', 'r') as file:
        return yaml.safe_load(file)

def create_database_connection() -> psycopg2.extensions.connection:
    """Create and return a database connection using the provided configuration."""

    config = load_config()
    db_config = config['db']

    return psycopg2.connect(**db_config)

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
    return [title + " " + text for title, text in data if title and text]


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


def count_rows(table_name:str) -> int:
    """
    Count the number of rows in the specified table.
    """
    conn = create_database_connection()
    cursor = conn.cursor()
    cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
    count = cursor.fetchone()[0]
    cursor.close()
    conn.close()
    return count

def fetch_data_in_batches(table_name: str, batch_size: int):
    """Fetch data from the specified table in batches."""
    conn = create_database_connection()
    cursor = conn.cursor(name='fetch_cursor')  # Server-side cursor
    query = f"SELECT title, selftext FROM {table_name};"
    cursor.execute(query)

    while True:
        batch = cursor.fetchmany(batch_size)
        if not batch:
            break
        yield batch

    cursor.close()
    conn.close()

def process_and_save_embeddings(MODEL_NAME: str, TABLE_NAME: str, MODEL_BATCH_SIZE: int, EMBEDDINGS_FILE: str):
    """Fetch data in batches, generate embeddings, and save them incrementally."""
    model = initialize_model(MODEL_NAME)
    rows_number = count_rows(TABLE_NAME)
    
    # Open the HDF5 file for writing
    with h5py.File(EMBEDDINGS_FILE, "w") as f:
        # Initialize variables to keep track of dataset dimensions
        data_initialized = False
        dataset = None
        i = 0

        for batch in tqdm(fetch_data_in_batches(TABLE_NAME, MODEL_BATCH_SIZE), total=rows_number // MODEL_BATCH_SIZE):
            texts = prepare_texts(batch)
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
                dataset[current_shape[0]:new_shape[0], :] = embeddings
            

        print("Embeddings saved incrementally.")




if __name__ == "__main__":
    # just for testing
    run_function_with_overrides(process_and_save_embeddings, config)
