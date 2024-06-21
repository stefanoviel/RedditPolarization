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
from src.run_single_step import run_function_with_overrides


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


def fetch_data(table_name:str) -> list[tuple[str, str]]:
    """
    Connect to the db and fetch data from the specified table.
    """
    conn = create_database_connection()
    cursor = conn.cursor()  # Create a cursor object
    query = f"SELECT title, selftext FROM {table_name};"
    cursor.execute(query)
    result = cursor.fetchall()

    cursor.close()
    conn.close()

    return result


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
        embeddings = model.encode(
            texts, show_progress_bar=True, convert_to_tensor=True, batch_size=1024
        )
        embeddings = embeddings.cpu().numpy()  # Move embeddings back to CPU for storage
    return embeddings


def save_embeddings(embeddings:torch.Tensor, file_path:str) -> None:
    """
    Save embeddings to an HDF5 file.
    """
    with h5py.File(file_path, "w") as f:
        f.create_dataset("embeddings", data=embeddings)
    print("Embeddings saved.")


# Usage example
def main_embed_data(MODEL_NAME:str, TABLE_NAME:str, EMBEDDINGS_FILE:str) -> None:
    model = initialize_model(MODEL_NAME)
    data = fetch_data(TABLE_NAME)
    texts = prepare_texts(data)
    embeddings = generate_embeddings(model, texts)
    save_embeddings(embeddings, EMBEDDINGS_FILE)


if __name__ == "__main__":
    # just for testing
    run_function_with_overrides(main_embed_data, config)
