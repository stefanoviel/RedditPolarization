import os
import random
import subprocess
import h5py

def copy_random_sample_files(source_folder, destination_folder, sample_size):
    # Ensure the source folder exists
    if not os.path.exists(source_folder):
        raise ValueError(f"Source folder '{source_folder}' does not exist.")

    # Ensure the destination folder exists, if not, create it
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Get a list of all files in the source folder
    all_files = [f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))]

    # Check if sample size is greater than the number of available files
    if sample_size > len(all_files):
        raise ValueError(f"Sample size '{sample_size}' is greater than the number of files in the source folder.")

    # Randomly sample the files
    sampled_files = random.sample(all_files, sample_size)

    # Copy the sampled files to the destination folder
    for file_name in sampled_files:
        full_file_name = os.path.join(source_folder, file_name)
        destination_file_name = os.path.join(destination_folder, file_name)
        subprocess.run(['cp', full_file_name, destination_file_name])
        print(f"Copied '{file_name}' to '{destination_folder}'")


def extract_and_save_specific_db_h5py(input_file, output_file):
    # Open the input HDF5 file in read mode
    with h5py.File(input_file, 'r') as infile:
        # Check if the required datasets exist in the input file
        if 'clusters' not in infile or 'ids' not in infile:
            print("Error: Required datasets 'clusters' or 'ids' not found in the input file.")
            return
        
        # Extract the datasets
        clusters_data = infile['clusters'][:]
        ids_data = infile['ids'][:]
        dim_reduction = infile['dimensional_reduction'][:]

    # Open the output HDF5 file in write mode (this will create a new file)
    with h5py.File(output_file, 'w') as outfile:
        # Save the extracted datasets to the new file
        outfile.create_dataset('clusters', data=clusters_data)
        outfile.create_dataset('ids', data=ids_data)
        outfile.create_dataset('dimensional_reduction', data=dim_reduction)

    print(f"Data successfully extracted and saved to {output_file}")


if __name__ == "__main__":
    # Example usage
    input_file = 'reddit_data.h5'
    output_file = 'clusters_ids_dim_red.h5'
    extract_and_save_specific_db_h5py(input_file, output_file)


    # # Example usage
    # source_folder = "data/big_test"
    # destination_folder = "data/random_sample"
    # sample_size = 10

    # copy_random_sample_files(source_folder, destination_folder, sample_size)
