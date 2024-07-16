
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
import json
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.utils.function_runner import run_function_with_overrides, execute_with_gpu_logging
from src.utils.utils import create_database_connection, load_json, load_h5py, load_model_and_tokenizer, create_tokenized_prompt, generate_response



def get_random_cluster_post(con, ids, clusters, TABLE_NAME):

    # Map IDs to their clusters, assuming clusters and IDs are in the same order
    cluster_to_ids = {}
    for id, cluster in zip(ids, clusters):
        if cluster not in cluster_to_ids:
            cluster_to_ids[cluster] = []
        cluster_to_ids[cluster].append(id)

    # Execute a query for each cluster and yield results
    for cluster, cluster_ids in cluster_to_ids.items():
        placeholders = ','.join(['?'] * len(cluster_ids))  # Prepare placeholders for SQL query
        query = f"SELECT title, selftext FROM {TABLE_NAME} WHERE id IN ({placeholders}) ORDER BY RANDOM() LIMIT 1"
        cursor = con.execute(query, cluster_ids)
        posts = cursor.fetchall()
        yield cluster, posts


def generate_quiz(con, ids, clusters, topic_description, TABLE_NAME):

    questions = []
    for cluster, posts in get_random_cluster_post(con, ids, clusters, TABLE_NAME):
        if len(posts) == 0:
            continue

        post = posts[0]
        corect_topic = topic_description[str(cluster)]
        all_topics = [topic_description[str(c)] for c in np.random.choice(list(topic_description.keys()), 4, replace=False) if c != cluster]

        correct_topic_position = np.random.randint(0, 5)
        
        all_topics.insert(correct_topic_position, str(corect_topic))
        all_topic_string = "\n".join([f"{chr(65+i)}) {t}" for i, t in enumerate(all_topics)])

        question =  f"""You will receive five topics, each represented by a list of words ordered by importance (from the most to the less important), along with a post composed of a title and body. 
        \nEach topic is also assigned a letter, which you will use to identify it.
        \nYour task is to identify the topic that best represents the post. Please return the letter corresponding to the correct topic in the following JSON format: {{"answer": "insert the letter here"}}.
        \nPost title: {post[0]}\nPost body: {post[1]}\n\nTopics:\n{all_topic_string}"""

        questions.append({
            'question': question,
            'answer': chr(65+correct_topic_position)
        })

    return questions

def solve_quiz(IDS_FILE, CLUSTER_FILE, TABLE_NAME, TFIDF_FILE, REDDIT_DATA_DIR, LLM_NAME): 

    topic_description = load_json(TFIDF_FILE)
    ids = load_json(IDS_FILE)
    clusters = load_h5py(CLUSTER_FILE, 'data')
    con = create_database_connection(REDDIT_DATA_DIR, TABLE_NAME, ['id', 'title', 'selftext'])
    questions = generate_quiz(con, ids, clusters, topic_description, TABLE_NAME)

    model, tokenizer = load_model_and_tokenizer(LLM_NAME)
    correct_answers_count = 0

    for question in questions:
        prompt = question['question']
        model_inputs = create_tokenized_prompt(prompt, tokenizer, model.device)
        generated_ids = generate_response(model, model_inputs)
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        response = json.loads(response)
        if response['answer'] == question['answer']:
            correct_answers_count += 1
        print(prompt)
        print("Correct answer: ", question['answer'])
        print(f"Response: {response['answer']}")
        print("=======================================================================")

    print(f"Number of correct answers: {correct_answers_count}/{len(questions)}, accuracy: {correct_answers_count/len(questions)}")


if __name__ == "__main__": 
    
    run_function_with_overrides(solve_quiz, config)