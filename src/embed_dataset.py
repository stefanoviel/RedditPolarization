import os
import sys

# adding root directory to paths
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from logging_config import configure_get_logger
import config

if not os.path.exists(config.OUTPUT_PATH):
    os.makedirs(config.OUTPUT_PATH)
logger = configure_get_logger(config.OUTPUT_PATH)
os.environ["NUMEXPR_MAX_THREADS"] = "32"
import numexpr

import torch
from sentence_transformers import SentenceTransformer
import duckdb
import h5py


def initialize_model(model_name):
    """
    Initialize the Sentence Transformer model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SentenceTransformer(model_name)
    model.to(device)
    model.eval()
    return model


def fetch_data(database_file, table_name):
    """
    Connect to DuckDB and fetch data from the specified table.
    """
    conn = duckdb.connect(database=database_file, read_only=True)
    query = f"SELECT title, selftext FROM {table_name};"
    result = conn.execute(query).fetchall()
    conn.close()
    print("Database connection closed.")
    return result


def prepare_texts(data):
    """
    Prepare texts by concatenating title and selftext.
    """
    return [title + " " + text for title, text in data if title and text]


def generate_embeddings(model, texts):
    """
    Generate embeddings for the given texts using the provided model.
    """
    with torch.no_grad():
        embeddings = model.encode(
            texts, show_progress_bar=True, convert_to_tensor=True, batch_size=1024
        )
        embeddings = embeddings.cpu().numpy()  # Move embeddings back to CPU for storage
    return embeddings


def save_embeddings(embeddings, file_path):
    """
    Save embeddings to an HDF5 file.
    """
    with h5py.File(file_path, "w") as f:
        f.create_dataset("embeddings", data=embeddings)
    print("Embeddings saved.")


# Usage example
def main_embed_data(model_name, db_file, table_name, embeddings_file):
    model = initialize_model(model_name)
    data = fetch_data(db_file, table_name)
    texts = prepare_texts(data)
    embeddings = generate_embeddings(model, texts)
    save_embeddings(embeddings, embeddings_file)


if __name__ == "__main__":
    # just for testing
    main_embed_data(
        config.MODEL_NAME,
        config.REDDIT_DB_FILE,
        config.TABLE_NAME,
        config.EMBEDDINGS_FILE,
    )
