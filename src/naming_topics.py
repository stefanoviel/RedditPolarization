
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


def init_model_tokenizer(model_name): 
    model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    return model, tokenizer


def create_tokenized_prompt(prompt, list_of_words, tokenizer, device):

    prompt = prompt.format(list_of_words= list_of_words)

    print(prompt)
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    return tokenizer([text], return_tensors="pt").to(device)

def generate_topics_names(TFIDF_FILE:str, LLM_NAME:str, TOPIC_NAMING_FILE:str, PROMPT:str):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = init_model_tokenizer(LLM_NAME)

    with open(TFIDF_FILE, "r") as f:
        tfidf = json.load(f)

    topic_naming = {}

    for cluster, words in tfidf.items():
        model_inputs = create_tokenized_prompt(PROMPT, words,  tokenizer, device)

        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=512
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        response = json.loads(response)
        if "topic" not in response:
            logger.info(f"Error in cluster {cluster}")
            continue

        topic_naming[cluster] = {"words": words, "topic": response['topic']}

    with open(TOPIC_NAMING_FILE, "w") as f:
        json.dump(topic_naming, f, indent=4)


if __name__ == "__main__": 
    run_function_with_overrides(generate_topics_names, config)