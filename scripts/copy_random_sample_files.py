import os
import random
import subprocess

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

# Example usage
source_folder = "data/big_test"
destination_folder = "data/random_sample"
sample_size = 10

copy_random_sample_files(source_folder, destination_folder, sample_size)
