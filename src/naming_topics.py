
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
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json


def load_model_and_tokenizer(model_name):
    """Initialize the model and tokenizer."""
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def format_prompt(prompt, list_of_words):
    """Format the prompt text with specified words."""
    return prompt.format(list_of_words=list_of_words)

def load_json_file(file_path):
    """Load data from a JSON file."""
    with open(file_path, "r") as f:
        return json.load(f)

def save_json_file(data, file_path):
    """Save data to a JSON file."""
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)

def create_tokenized_prompt(prompt_text, tokenizer, device):
    """Tokenize and prepare the prompt for the model."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt_text}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    return tokenizer([text], return_tensors="pt").to(device)

def generate_response(model, model_inputs):
    """Generate a response using the model."""
    generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=512)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
    return generated_ids

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
    tfidf_data = load_json_file(TFIDF_FILE)
    topic_naming = process_topics(tfidf_data, tokenizer, model, device, PROMPT)
    save_json_file(topic_naming, TOPIC_NAMING_FILE)


if __name__ == "__main__": 
    run_function_with_overrides(main, config)