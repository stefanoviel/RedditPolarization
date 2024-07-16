import os
import h5py
import duckdb
import json
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_h5py(file_path: str, db_name:str) -> np.ndarray:
    """
    Load from an HDF5 file into a NumPy array.
    """
    with h5py.File(file_path, "r") as file:
        # Assume the dataset name in HDF5 file is 'data'
        embeddings = file[db_name][:]
    return embeddings

def save_h5py(data: np.ndarray, file_path: str, db_name:str):
    """
    Save a NumPy array to an HDF5 file.
    """
    with h5py.File(file_path, "w") as file:
        file.create_dataset(db_name, data=data)

def load_json(file_path):
    with open(file_path, 'r') as file:
        ids = json.load(file)
    return ids

def sample_hdf5(input_filename, output_filename, sample_fraction=0.1):
    # Open the input HDF5 file
    with h5py.File(input_filename, 'r') as file:
        # Create the output HDF5 file
        with h5py.File(output_filename, 'w') as outfile:
            # Iterate over each dataset in the input file
            for dataset_name in file:
                data = file[dataset_name][...]
                
                # Calculate the number of samples to take
                num_samples = int(data.shape[0] * sample_fraction)
                
                # Sample the data
                if num_samples > 0:
                    indices = np.random.choice(data.shape[0], num_samples, replace=False)
                    sampled_data = data[indices]
                    
                    # Write the sampled data to the new file
                    outfile.create_dataset(dataset_name, data=sampled_data)
                else:
                    print(f"Not enough data to sample in dataset {dataset_name}")


def create_database_connection(parquet_directory:str, table_name:str, columns: list) -> duckdb.DuckDBPyConnection:
    """Create and return a database connection using the provided configuration."""
    files = [f'{parquet_directory}/{file}' for file in os.listdir(parquet_directory)]
    con = duckdb.connect(database=':memory:')
    query_files = ', '.join(f"'{f}'" for f in files)
    columns_str = ', '.join(columns)
    sql_query = f"CREATE TABLE {table_name} AS SELECT {columns_str} FROM read_parquet([{query_files}], union_by_name=True)"
    con.execute(sql_query)
    return con


def load_model_and_tokenizer(model_name):
    """Initialize the model and tokenizer."""
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def format_prompt(prompt, list_of_words):
    """Format the prompt text with specified words."""
    return prompt.format(list_of_words=list_of_words)


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