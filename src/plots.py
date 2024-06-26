
import matplotlib.pyplot as plt
import numpy as np
import h5py
import os
import cuml
import json
import random
from tqdm import tqdm
import pandas as pd
import zstandard as zstd


def load_vectors(file_path, dataset_name):
    with h5py.File(file_path, 'r') as file:
        vectors = np.array(file[dataset_name])
    return vectors


def compute_plot_trutworthiness(original_file, original_dataset_name, reduced_files_directory, reduced_dataset_name, label=None, plot=True):
    """Compute trustworthiness between the original dataset and all the reduced dataset (run with different parameters) in the reduced_files_directory."""

    original_vectors = load_vectors(original_file, original_dataset_name)

    trustworthiness_scores_original = {}

    for file_path in os.listdir(reduced_files_directory):
        file_path = os.path.join(reduced_files_directory, file_path)
        reduced_vectors = load_vectors(file_path, reduced_dataset_name)
        tw = cuml.metrics.trustworthiness(original_vectors, reduced_vectors)
        trustworthiness_scores_original[file_path] = tw
        print(f'Trustworthiness for {file_path}: {tw}')
    
    if plot:
        plt.plot(list(trustworthiness_scores_original.keys()), list(trustworthiness_scores_original.values()), label=label)
        plt.legend()

    return trustworthiness_scores_original


def extract_data_from_compressed_file_limited(file_path: str, subset_fraction:float) -> list:
    """Read a zstd-compressed file and extract length and score of each JSON entry."""

    results = []
    with open(file_path, "rb") as file_handle:
        # Setup decompressor with a large max_window_size for large files
        decompressor = zstd.ZstdDecompressor(max_window_size=2147483648)
        reader = decompressor.stream_reader(file_handle)

        while True:
            chunk = reader.read(2**27)  # Read 128 MB at a time
            if not chunk:
                break

            try:
                data = chunk.decode("utf-8").split("\n")
            except UnicodeDecodeError:
                continue
            
            for line in data:
                if random.random() > subset_fraction:
                    continue

                try:
                    line_json = json.loads(line)
                    
                    # if there is media don't include self text in the embedding as it will be an URL
                    if line_json.get("media", None) is not None:
                        selftext = ""
                    else:
                        selftext = line_json.get("selftext", "")

                    combined_text = line_json.get("title", "") + selftext
                    score = line_json.get("score", -1)
                    num_comments = line_json.get("num_comments", -1)
                    up_votes = line_json.get("ups", -1)
                    down_votes = line_json.get("downs", -1)
                    has_media = "media" in line_json and line_json["media"] is not None
                    subreddit = line_json.get("subreddit", "")

                    results.append({
                                    'length': len(combined_text), 
                                    'score': score, 
                                    'num_comments': num_comments, 
                                    'upvotes': up_votes, 
                                    'downvotes': down_votes,
                                    "has_media": has_media,
                                    "subreddit": subreddit
                                })
                    

                    
                except json.JSONDecodeError:
                    continue

    return results


def extract_year_month(filename):
    """Extract the year and month from a filename formatted as 'RS_YYYY-MM.zst'."""
    base = os.path.basename(filename)
    parts = base.split('-')
    year = parts[0][-4:]
    month = parts[1][:2]
    return year, month

def extract_statistics_from_folder(directory, num_files_to_process, subset_fraction=0.1):
    """Process all .zst files in the directory, extract data, and store in a pandas DataFrame."""
    all_data = []
    
    counter = 0
    for file in os.listdir(directory):
        if file.endswith('.zst'):
            
            year, month = extract_year_month(file)
            file_path = os.path.join(directory, file)
            data = extract_data_from_compressed_file_limited(file_path, subset_fraction)
            
            for d in (data):
                d['year'] = year
                d['month'] = month
                all_data.append(d)

            print("Processed file:", file, "number of entries:", len(data))

            # Break after the first file for demonstration; remove this in actual use to process all files
            counter += 1
            if counter >= num_files_to_process:
                break

    df = pd.DataFrame(all_data)
    return df