import os
import sys

# adding root directory to paths
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from logging_config import configure_get_logger
import config

if not os.path.exists(config.OUTPUT_DIR):
    os.makedirs(config.OUTPUT_DIR)
logger = configure_get_logger(config.OUTPUT_DIR, config.EXPERIMENT_NAME, executed_file_name = __file__, log_level='INFO')

from src.utils.function_runner import run_function_with_overrides, execute_with_gpu_logging
from src.utils.utils import load_json,  save_json_file, load_model_and_tokenizer
from src.utils.LLM_utils import  generate_response
import pandas as pd
import torch
import json

def save_to_csv(data, file_path, columns):
    """Save a list of lists to a CSV file with specified columns."""
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(file_path, index=False)

def generate_prompt(list_of_words):
    """Generate a prompt based on a list of words."""
    PROMPT_TEMPLATE = """Given the following lists of words, each associated with a cluster number, identify a succinct topic that captures the essence of the words in each list. Below are examples of the expected JSON output format.

    Examples:
    - "game, team, season, like, time, year, player, play, games, 10" -> {{"topic": "Sports Analysis"}}
    - "new, like, time, know, game, people, think, make, good, really" -> {{"topic": "General Discussion"}}
    - "team, vs, game, twitch, tv, twitter, youtube, 00, logo, mt" -> {{"topic": "Live Streaming and Social Media"}}
    - "art, oc, painting, like, drawing, new, paint, pen, imgur, time" -> {{"topic": "Art and Drawing"}}

    Your task:

    Given a list of words, output only the identified topic in the following JSON format, ensuring there are no additional characters or formatting around the JSON:

    {{"topic": "Your identified topic here"}}

    Please perform the same task for the following list of words:
    - {list_of_words}

    Output Format:
    {{"topic": "Identified topic"}}

    Ensure that the output is in valid JSON format and is not surrounded by any extra formatting like '''json '''.
    """
    return PROMPT_TEMPLATE.format(list_of_words=", ".join(list_of_words))


def parse_response(response, cluster):
    """Parse the model's response and return the topic. If parsing fails, return an error topic."""
    try:
        response_json = json.loads(response)
        if "topic" in response_json:
            return response_json["topic"]
        else:
            print(f"Error in cluster {cluster}: Missing 'topic' key in response")
            return "Error"
    except json.JSONDecodeError:
        print(f"Error in cluster {cluster}: Invalid JSON response")
        print(f"Error response: {response}")
        return "Error"


def process_tfidf_file(tfidf_data,  llm_name):
    """Process each topic and generate names using the model."""
    topic_naming = []
    
    for cluster, words in list(tfidf_data.items()):
        if cluster == '-1': 
            continue
        prompt_text = generate_prompt(words)
        response = generate_response(prompt_text, llm_name)
        
        topic = parse_response(response, cluster)
        topic_naming.append([int(cluster), words, topic])
        
        print(f"Words {words} -> {topic}")
    
    return topic_naming

def process_subtopics_tfidf_file(tfidf_data,  llm_name):
    """Process each subtopic and generate names using the model."""
    topic_naming = []
    
    for cluster, subtopics in list(tfidf_data.items()):
        for subcluster, words in list(subtopics.items()):
            if subcluster == '-1': 
                continue

            prompt_text = generate_prompt(words)
            response = generate_response(prompt_text, llm_name)
            
            topic = parse_response(response, cluster)
            topic_naming.append([int(cluster), int(subcluster), words, topic])
            
            print(f"Words {words} -> {topic}")
    
    return topic_naming


def naming_topics_tfidf_file(TFIDF_FILE, CLUSTER_AND_TOPIC_NAMES, LLM_NAME):    
    tfidf_data = load_json(TFIDF_FILE)
    topic_naming = process_tfidf_file(tfidf_data,  LLM_NAME)
    save_to_csv(topic_naming, CLUSTER_AND_TOPIC_NAMES, columns=["cluster", "words", "topic"])


def naming_subtopics_subtfidf_file(SUBCLUSTER_TFIDF_FILE, SUBCLUSTER_AND_TOPIC_NAMES, LLM_NAME):    
    tfidf_data = load_json(SUBCLUSTER_TFIDF_FILE)
    topic_naming = process_subtopics_tfidf_file(tfidf_data, LLM_NAME)
    save_to_csv(topic_naming, SUBCLUSTER_AND_TOPIC_NAMES, columns=["cluster", "subcluster", "words_subcluster", "topic_subcluster"])


if __name__ == "__main__": 
    run_function_with_overrides(naming_subtopics_subtfidf_file, config)