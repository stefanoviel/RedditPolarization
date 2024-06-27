
import h5py
import numpy as np


def load_embeddings(file_path: str, db_name:str) -> np.ndarray:
    """
    Load embeddings from an HDF5 file into a NumPy array.
    """
    with h5py.File(file_path, "r") as file:
        # Assume the dataset name in HDF5 file is 'embeddings'
        embeddings = file[db_name][:]
    return embeddings