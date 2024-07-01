
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

def sample_hdf5(input_filename, output_filename, sample_fraction=0.1):
    # Open the input HDF5 file
    with h5py.File(input_filename, 'r') as file:
        # Create the output HDF5 file
        with h5py.File(output_filename, 'w') as outfile:
            # Iterate over each dataset in the input file
            for dataset_name in file:
                data = file[dataset_name][...]
                
                # Calculate the number of samples to take
                num_samples = int(data.shape[0] * sample_fraction)
                
                # Sample the data
                if num_samples > 0:
                    indices = np.random.choice(data.shape[0], num_samples, replace=False)
                    sampled_data = data[indices]
                    
                    # Write the sampled data to the new file
                    outfile.create_dataset(dataset_name, data=sampled_data)
                else:
                    print(f"Not enough data to sample in dataset {dataset_name}")