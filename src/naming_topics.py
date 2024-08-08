
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

def process_topics(tfidf_data, prompt, LLM_NAME):
    """Process each topic and generate names using the model."""
    topic_naming = []
    for cluster, words in list(tfidf_data.items()):
        prompt_text = prompt.format(list_of_words=words)
        response = generate_response(prompt_text, LLM_NAME)
        try: 
            response = json.loads(response)
        except:
            print(f"Error in cluster {cluster}")
            print(f"Error response {response}")
            response = {"topic": "Error"}
        if "topic" not in response:
            continue
        topic_naming.append([int(cluster), response["topic"]])
        print(f"Words {words} -> {response['topic']}")
    return topic_naming

def main(TFIDF_FILE, FINAL_DATAFRAME, LLM_NAME):

    # TODO: fix prompt, sometime it outputs '''json '''

    PROMPT = """ Given the following lists of words, each associated with a cluster number, identify a succinct topic that captures the essence of the words in each list. Below are some examples of how the output should be structured in JSON format.

    Examples:
    - "game, team, season, like, time, year, player, play, games, 10" -> {{"topic" : "Sports Analysis"}}
    - "new, like, time, know, game, people, think, make, good, really" -> {{"topic" : "General Discussion"}}
    - "team, vs, game, twitch, tv, twitter, youtube, 00, logo, mt" -> {{"topic" : "Live Streaming and Social Media"}}
    - "art, oc, painting, like, drawing, new, paint, pen, imgur, time" -> {{"topic" : "Art and Drawing"}}

    Your task is to find an appropriate topic for the list of words reported below. Present your output in the following JSON format:

    {{
    "topic": "Your identified topic here"
    }}

    Please perform the same task for the following lists of words:
    - {list_of_words} ->

    ONLY output the topic name in the JSON format above. Do not include any other information in the JSON output.
    """

    tfidf_data = load_json(TFIDF_FILE)
    topic_naming = process_topics(tfidf_data, PROMPT, LLM_NAME)
    df = pd.DataFrame(topic_naming, columns=["cluster", "topic"])
    final_df = pd.read_csv(FINAL_DATAFRAME)
    final_df = final_df.merge(df, on="cluster", how="left")
    final_df.to_csv(FINAL_DATAFRAME, index=False)





if __name__ == "__main__": 
    run_function_with_overrides(main, config)