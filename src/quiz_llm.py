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
from src.utils.LLM_utils import create_tokenized_prompt, generate_response_local_model, generate_response


def get_random_posts_with_clusters(con, ids, clusters, TABLE_NAME, n_quiz):
    # Filter out IDs corresponding to clusters with labels -1
    filtered_ids_clusters = [(id, cluster) for id, cluster in zip(ids, clusters) if cluster != -1]

    # Separate the filtered IDs and clusters
    filtered_ids, filtered_clusters = zip(*filtered_ids_clusters) if filtered_ids_clusters else ([], [])

    # Map filtered IDs to their clusters
    id_to_cluster = {id.decode('utf-8'): cluster for id, cluster in zip(filtered_ids, filtered_clusters)}
    
    # Decode byte string IDs to regular strings
    decoded_ids = [id.decode('utf-8') for id in filtered_ids]
    
    # Prepare placeholders for SQL query
    placeholders = ','.join(['?'] * len(decoded_ids))
    
    # Execute a single query to select n random posts from all clusters
    query = f"SELECT id, title, selftext FROM {TABLE_NAME} WHERE id IN ({placeholders}) ORDER BY RANDOM() LIMIT ?"
    
    # decoded_ids + [n_quiz] combines the list of IDs and the limit value for the query parameters
    cursor = con.execute(query, decoded_ids + [n_quiz])
    posts = cursor.fetchall()
    
    # Map each post back to its cluster
    results = [(id_to_cluster[post[0]], post[1], post[2]) for post in posts]
    
    return results


def generate_quiz(con, ids, clusters, topic_description, TABLE_NAME, N_QUIZ, n_options):

    questions = []
    for cluster, title, body in get_random_posts_with_clusters(con, ids, clusters, TABLE_NAME, N_QUIZ):
        if len(title) == 0:
            continue

        corect_topic_for_post = topic_description[str(cluster)]
        
        # check that there are enough topics to sample from 
        if n_options > len(topic_description):
            n_options = len(topic_description)

        all_topics = [topic_description[str(c)] for c in np.random.choice(list(topic_description.keys()), n_options-1, replace=False) if int(c) != int(cluster)]

        correct_topic_position = np.random.randint(0, n_options)
        
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

    return questions

def solve_quiz(PROCESSED_REDDIT_DATA, CLUSTER_DB_NAME, IDS_DB_NAME, TABLE_NAME, TFIDF_FILE, DATABASE_PATH, LLM_NAME, NUMBER_OF_OPTIONS, N_QUIZ): 
    topic_description = load_json(TFIDF_FILE)
    ids = load_h5py(PROCESSED_REDDIT_DATA, IDS_DB_NAME)
    post_cluster_assignment = load_h5py(PROCESSED_REDDIT_DATA, CLUSTER_DB_NAME)
    con = connect_to_existing_database(DATABASE_PATH)
    questions = generate_quiz(con, ids, post_cluster_assignment, topic_description, TABLE_NAME, N_QUIZ, NUMBER_OF_OPTIONS)

    # not using local model for now
    # model, tokenizer = load_model_and_tokenizer(LLM_NAME)
    correct_answers_count = 0

    for question in questions:
        prompt = question['question']
        # model_inputs = create_tokenized_prompt(prompt, tokenizer, model.device)
        # generated_ids = generate_response_local_model(model, model_inputs)
        # response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        response = generate_response(prompt)
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


def run_quiz_multiple_times(PROCESSED_REDDIT_DATA, CLUSTER_DB_NAME, IDS_DB_NAME, TABLE_NAME, TFIDF_FILE, DATABASE_PATH, LLM_NAME, NUMBER_OF_OPTIONS, TEST_LLM_ACCURACY_FILE, N_QUIZ, NUM_RUNS):
    all_accuracies = []

    for _ in range(NUM_RUNS):
        accuracy = solve_quiz(
            PROCESSED_REDDIT_DATA, CLUSTER_DB_NAME, IDS_DB_NAME, TABLE_NAME, TFIDF_FILE, DATABASE_PATH, LLM_NAME, NUMBER_OF_OPTIONS, N_QUIZ
        )
        all_accuracies.append(accuracy)

    accuracy_data = {
        "tf_idf_file": TFIDF_FILE,
        "accuracies": all_accuracies,
        "num_runs": NUM_RUNS
    }

    append_to_json(TEST_LLM_ACCURACY_FILE, accuracy_data)

    return all_accuracies


def solve_multiple_quiz_save_accuracy(PROCESSED_REDDIT_DATA, CLUSTER_DB_NAME, IDS_DB_NAME,TABLE_NAME, DATABASE_PATH, LLM_NAME, NUMBER_OF_OPTIONS, TF_IDF_FOLDER, TEST_LLM_ACCURACY_FILE): 
    accuracies = []
    for tf_idf_file in os.listdir(TF_IDF_FOLDER):
        tf_idf_path = os.path.join(TF_IDF_FOLDER, tf_idf_file)
        print(tf_idf_path)
        accuracy = solve_quiz(PROCESSED_REDDIT_DATA, CLUSTER_DB_NAME, IDS_DB_NAME,TABLE_NAME, tf_idf_path, DATABASE_PATH, LLM_NAME, NUMBER_OF_OPTIONS)
        accuracies.append(accuracy)

    with open(TEST_LLM_ACCURACY_FILE, 'w') as file:
        json.dump(accuracies, file)


if __name__ == "__main__": 
    run_function_with_overrides(run_quiz_multiple_times, config)