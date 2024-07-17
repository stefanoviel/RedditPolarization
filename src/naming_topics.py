
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
from src.utils.utils import load_json, format_prompt, create_tokenized_prompt, generate_response, save_json_file, load_model_and_tokenizer

import torch
import json


def process_topics(tfidf_data, tokenizer, model, device, prompt):
    """Process each topic and generate names using the model."""
    topic_naming = {}
    for cluster, words in tfidf_data.items():
        words = [word[0] for word in words]
        prompt_text = format_prompt(prompt, words)
        model_inputs = create_tokenized_prompt(prompt_text, tokenizer, device)
        generated_ids = generate_response(model, model_inputs)
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        response = json.loads(response)
        if "topic" not in response:
            print(f"Error in cluster {cluster}")
            continue
        topic_naming[cluster] = {"words": words, "topic": response['topic']}
    return topic_naming

def main(TFIDF_FILE, LLM_NAME, TOPIC_NAMING_FILE, PROMPT):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_model_and_tokenizer(LLM_NAME)
    tfidf_data = load_json(TFIDF_FILE)
    topic_naming = process_topics(tfidf_data, tokenizer, model, device, PROMPT)
    save_json_file(topic_naming, TOPIC_NAMING_FILE)


if __name__ == "__main__": 
    run_function_with_overrides(main, config)