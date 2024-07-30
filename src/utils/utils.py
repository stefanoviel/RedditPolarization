import os
import h5py
import duckdb
import json
import numpy as np
import pyarrow.parquet as pq
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_h5py(file_path: str, db_name:str) -> np.ndarray:
    """
    Load from an HDF5 file into a NumPy array.
    """
    with h5py.File(file_path, "r") as file:
        # Assume the dataset name in HDF5 file is 'data'
        embeddings = file[db_name][:]
    return embeddings


def get_indices_for_random_h5py_subset(filename: str, dataset_name, subset_fraction: float):
    """Extract data points corresponding to indices"""

    with h5py.File(filename, "r") as file:
        dataset = file[dataset_name]
        total_samples = dataset.shape[0]
        num_samples = int(total_samples * subset_fraction)

        partial_fit_indices = np.random.choice(total_samples, num_samples, replace=False)
        partial_fit_indices.sort()

    return partial_fit_indices, total_samples, num_samples

def load_with_indices_h5py_efficient(file_path: str, db_name: str, indices: np.ndarray) -> np.ndarray:
    """
    Load specific indices from an HDF5 file into a NumPy array using an efficient block loading strategy.
    It's necessary to to load everything into memory, otherwise it gets very slow.
    """
    indices = np.array(indices)
    min_idx, max_idx = indices.min(), indices.max()

    with h5py.File(file_path, "r") as file:
        dataset = file[db_name]
        data_block = dataset[min_idx:max_idx+1]  # +1 because max_idx is inclusive

    selected_data = data_block[indices - min_idx]

    return selected_data

def save_h5py(data: np.ndarray, file_path: str, db_name: str):
    """
    Save a NumPy array to an HDF5 file without deleting existing datasets.
    """
    with h5py.File(file_path, "a") as file:  # Changed "w" to "a" to prevent deletion of existing data
        # Check if dataset exists and delete if it does to prevent error on creation
        if db_name in file:
            del file[db_name]
        file.create_dataset(db_name, data=data)

def load_json(file_path):
    with open(file_path, 'r') as file:
        ids = json.load(file)
    return ids

def save_json(data, file_path):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

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



def create_filtered_database(parquet_directory: str, table_name: str, columns: list, min_score: int, min_post_length: int, start_date: int, end_date: int, database_path: str, max_expression_depth: int = 2500) -> duckdb.DuckDBPyConnection:
    """Create a filtered database connection and load only the necessary data, saving it to disk."""
    # List all files in the directory
    all_files = os.listdir(parquet_directory)
    
    valid_files = []
    for file in all_files:
        file_path = f'{parquet_directory}/{file}'
        try:
            # Read the schema of the Parquet file
            parquet_file = pq.ParquetFile(file_path)
            schema = parquet_file.schema.to_arrow_schema()
            # Check if 'media' field exists and is of boolean type
            if schema.get_field_index('media') != -1 and schema.field('media').type == 'bool':
                valid_files.append(file)
            del parquet_file
        except Exception as e:
            print(f"Error reading schema of file {file}: {e}")
    
    # print(valid_files)
    files = [f'{parquet_directory}/{file}' for file in valid_files]
    con = duckdb.connect(database=database_path)
    con.execute(f"SET max_expression_depth TO {max_expression_depth}")
    query_files = ', '.join(f"'{f}'" for f in files)
    
    # Define the schema with the specified columns
    columns_str = ', '.join(columns)
    
    # only load in the database what is necessary
    sql_query = f"""
    CREATE TABLE {table_name} AS 
    SELECT {columns_str} 
    FROM read_parquet([{query_files}], union_by_name=True)
    WHERE LENGTH(title) > {min_post_length}
      AND score > {min_score}
      AND selftext NOT LIKE '%[deleted]%'
      AND selftext NOT LIKE '%[removed]%'
      AND media = FALSE 
      AND {start_date} < created_utc 
      AND created_utc < {end_date}
    """
    
    try:
        con.execute(sql_query)
    except duckdb.ConversionException as e:
        print(f"Error creating table {table_name}: {e}")
        raise
    
    return con


def connect_to_existing_database(database_path: str) -> duckdb.DuckDBPyConnection:
    """
    Connect to an existing DuckDB database given the path where it is saved.
    """
    try:
        # Connect to the existing database
        con = duckdb.connect(database=database_path, read_only=False)
        print(f"Successfully connected to the database at {database_path}")
        return con
    except Exception as e:
        print(f"Error connecting to the database at {database_path}: {e}")
        raise


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