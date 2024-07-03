
import matplotlib.pyplot as plt
import numpy as np
import h5py
import os
import cuml
import json
import sklearn
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
        tw = cuml.metrics.trustworthiness(original_vectors, reduced_vectors, n_neighbors= 30, batch_size=4000)
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


def match_clusters(labels1, labels2):

    # Determine unique clusters and index elements in each cluster
    unique_clusters1 = np.unique(labels1)
    unique_clusters2 = np.unique(labels2)

    # Create mappings from labels to list indices
    cluster_map1 = {label: np.where(labels1 == label)[0] for label in unique_clusters1}
    cluster_map2 = {label: np.where(labels2 == label)[0] for label in unique_clusters2}

    # Determine intersections and track the best matches
    best_matches = {}
    for label1, elements1 in cluster_map1.items():
        max_intersection = -1
        best_label2 = None
        for label2, elements2 in cluster_map2.items():
            intersection_size = len(np.intersect1d(elements1, elements2))
            # If current intersection is greater than max recorded, update the match
            if intersection_size > max_intersection:
                max_intersection = intersection_size
                best_label2 = label2
        best_matches[label1] = best_label2

    # Generate new labels based on matches
    new_label_counter = 0
    new_label_map = {}
    new_labels1 = np.zeros_like(labels1)
    new_labels2 = np.zeros_like(labels2)

    for label1, label2 in best_matches.items():
        if label1 not in new_label_map or label2 not in new_label_map:
            new_label_map[label1] = new_label_counter
            new_label_map[label2] = new_label_counter
            new_label_counter += 1
        new_labels1[np.where(labels1 == label1)] = new_label_map[label1]
        new_labels2[np.where(labels2 == label2)] = new_label_map[label2]

    return new_labels1.tolist(), new_labels2.tolist()


def match_and_compute_cluster_metrics(function, ground_truth_file, ground_truth_dataset_name, directory_with_predictions, directory_with_predictions_dataset_name):
    """Compute the Adjusted Rand Index between ground truth and predicted labels."""
    ground_truth_labels = load_vectors(ground_truth_file, ground_truth_dataset_name)        
    print("ground truth", len(ground_truth_labels))

    ari_scores = {}

    for file in os.listdir(directory_with_predictions):
        print(file, ground_truth_file)
        if file.endswith('.h5'):
            file_path = os.path.join(directory_with_predictions, file)
            predicted_labels = load_vectors(file_path, directory_with_predictions_dataset_name)
            print(file, "number of unique predictedlabels:", len(np.unique(predicted_labels)))

            # Match clusters between ground truth and predicted labels
            ground_truth_labels_matched, predicted_labels = match_clusters(ground_truth_labels, predicted_labels)

            # using sklearn as the cuml one is broken
            ari = function(ground_truth_labels_matched, predicted_labels)
            print(f'ADJ for {file_path}: {ari}')
            ari_scores[file_path] = ari
    
    return ari_scores


def plot_ARI(ari_scores, label=None):
    """Plot the ARI scores."""
    plt.plot(list(ari_scores.keys()), list(ari_scores.values()), label=label)
    plt.legend()