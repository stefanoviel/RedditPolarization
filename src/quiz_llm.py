import os
import sys

# adding root directory to paths
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from logging_config import configure_get_logger
import config

from cuml.common import logger
logger.set_level(logger.level_error)

if not os.path.exists(config.OUTPUT_DIR):
    os.makedirs(config.OUTPUT_DIR)
logger = configure_get_logger(config.OUTPUT_DIR, config.EXPERIMENT_NAME, executed_file_name = __file__, log_level='INFO')

import h5py
import duckdb
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text
import numpy as np
import cuml
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.utils.function_runner import run_function_with_overrides, execute_with_gpu_logging
from src.utils.utils import connect_to_existing_database, load_json, load_h5py, load_model_and_tokenizer, append_to_json
from src.utils.LLM_utils import create_tokenized_prompt, generate_response_local_model, generate_response_lama_server, generate_response_gpt


def get_random_posts_with_clusters(con, ids, clusters, TABLE_NAME, n_quiz):

    # Filter out IDs corresponding to clusters with labels -1, we don't want to have noise as an option in the quiz 
    filtered_ids_clusters = [(id, cluster) for id, cluster in zip(ids, clusters) if cluster != -1]
    filtered_ids, filtered_clusters = zip(*filtered_ids_clusters) if filtered_ids_clusters else ([], [])
    id_to_cluster = {id.decode('utf-8'): cluster for id, cluster in zip(filtered_ids, filtered_clusters)}
    decoded_ids = [id.decode('utf-8') for id in filtered_ids]

    random_decoded_ids = np.random.choice(decoded_ids, n_quiz, replace=False)
    placeholders = ','.join(['?'] * len(random_decoded_ids))
    
    # Execute a single query to select n random posts from all clusters
    query = f"SELECT id, title, selftext FROM {TABLE_NAME} WHERE id IN ({placeholders})"
    
    # decoded_ids + [n_quiz] combines the list of IDs and the limit value for the query parameters
    cursor = con.execute(query, list(random_decoded_ids))
    posts = cursor.fetchall()
    
    # Map each post back to its cluster
    results = [(id_to_cluster[post[0]], post[1], post[2]) for post in posts]
    
    return results


def get_nearest_clusters(PROCESSED_REDDIT_DATA: str, CENTROIDS_DB_NAME: str, target_cluster: int, n_options: int = 5):
    # Load the centroids
    centroids = load_h5py(PROCESSED_REDDIT_DATA, CENTROIDS_DB_NAME)

    print("centroids", centroids.shape)

    # Get the centroid of the target cluster
    target_centroid = centroids[target_cluster]

    print("target_centroid", target_centroid.shape)

    # Compute the Euclidean distances between the target centroid and all other centroids
    distances = np.linalg.norm(centroids - target_centroid, axis=1)
    print("distances", distances)

    del centroids

    # Exclude the target cluster from the list of distances by setting its distance to infinity
    distances[target_cluster] = np.inf
    nearest_clusters = np.argsort(distances)[:n_options-1]

    return nearest_clusters


def generate_quiz(con, ids, clusters, topic_description, TABLE_NAME, N_QUIZ, NUMBER_OF_OPTIONS, PROCESSED_REDDIT_DATA, CENTROIDS_DB_NAME,):

    questions = []
    for cluster, title, body in get_random_posts_with_clusters(con, ids, clusters, TABLE_NAME, N_QUIZ):
        if len(title) == 0:
            continue

        corect_topic_for_post = topic_description[str(cluster)]
        
        # check that there are enough topics to sample from 
        if NUMBER_OF_OPTIONS > len(topic_description):
            NUMBER_OF_OPTIONS = len(topic_description)

        closest_clusters = get_nearest_clusters(PROCESSED_REDDIT_DATA, CENTROIDS_DB_NAME, cluster, NUMBER_OF_OPTIONS)
        all_topics = [topic_description[str(c)] for c in closest_clusters]

        correct_topic_position = np.random.randint(0, NUMBER_OF_OPTIONS)
        
        all_topics.insert(correct_topic_position, str(corect_topic_for_post))
        all_topic_string = "\n".join([f"{chr(65+i)}) {t}" for i, t in enumerate(all_topics)])

        question =  f"""You will receive five topics, each represented by a list of words ordered by importance (from the most to the less important), along with a post composed of a title and body. 
        \nEach topic is also assigned a letter, which you will use to identify it.
        \nYour task is to identify the topic that best represents the post. Please return the letter corresponding to the correct topic in the following JSON format: {{"answer": "insert the letter here"}}.
        \nPost title: {title}\nPost body: {body}\n\nTopics:\n{all_topic_string}"""

        questions.append({
            'question': question,
            'answer': chr(65+correct_topic_position)
        })
        print(question)

    return questions

def solve_quiz(PROCESSED_REDDIT_DATA, CLUSTER_DB_NAME, IDS_DB_NAME, TABLE_NAME, TFIDF_FILE, DATABASE_PATH, NUMBER_OF_OPTIONS, N_QUIZ, CENTROIDS_DB_NAME): 
    topic_description = load_json(TFIDF_FILE)
    ids = load_h5py(PROCESSED_REDDIT_DATA, IDS_DB_NAME)
    post_cluster_assignment = load_h5py(PROCESSED_REDDIT_DATA, CLUSTER_DB_NAME)
    con = connect_to_existing_database(DATABASE_PATH)
    print("Generating quiz...")
    questions = generate_quiz(con, ids, post_cluster_assignment, topic_description, TABLE_NAME, N_QUIZ, NUMBER_OF_OPTIONS, PROCESSED_REDDIT_DATA, CENTROIDS_DB_NAME)

    # not using local model for now
    # model, tokenizer = load_model_and_tokenizer(LLM_NAME)
    correct_answers_count = 0

    for question in questions:
        prompt = question['question']
        # model_inputs = create_tokenized_prompt(prompt, tokenizer, model.device)
        # generated_ids = generate_response_local_model(model, model_inputs)
        # response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        response = generate_response_gpt(prompt)
        # response = generate_response_lama_server(prompt)
        try:
            response = json.loads(response)
        except Exception as e:
            continue

        if response['answer'] == question['answer']:
            correct_answers_count += 1
        print(prompt)
        print("Correct answer: ", question['answer'])
        print(f"Response: {response['answer']}")
        print("=======================================================================")

    accuracy = correct_answers_count / len(questions)
    print(f"Number of correct answers: {correct_answers_count}/{len(questions)}, accuracy: {accuracy}")

    return accuracy


def run_quiz_multiple_times(PROCESSED_REDDIT_DATA, CLUSTER_DB_NAME, IDS_DB_NAME, TABLE_NAME, TFIDF_FILE, DATABASE_PATH, LLM_NAME, NUMBER_OF_OPTIONS, TEST_LLM_ACCURACY_FILE, N_QUIZ, NUM_RUNS, CENTROIDS_DB_NAME):
    all_accuracies = []

    for _ in range(NUM_RUNS):
        accuracy = solve_quiz(
            PROCESSED_REDDIT_DATA, CLUSTER_DB_NAME, IDS_DB_NAME, TABLE_NAME, TFIDF_FILE, DATABASE_PATH, NUMBER_OF_OPTIONS, N_QUIZ, CENTROIDS_DB_NAME
        )
        all_accuracies.append(accuracy)

    accuracy_data = {
        "tf_idf_file": TFIDF_FILE,
        "accuracies": all_accuracies,
        "num_runs": NUM_RUNS
    }

    append_to_json(TEST_LLM_ACCURACY_FILE, accuracy_data)

    return all_accuracies

if __name__ == "__main__": 
    run_function_with_overrides(run_quiz_multiple_times, config)