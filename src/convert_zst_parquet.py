import os
import zstandard as zstd
import pandas as pd
from io import StringIO

import os
import zstandard as zstd
import pandas as pd
from io import StringIO


def process_chunk(chunk, min_num_characters, min_score, required_columns):
    """ Process a single chunk of data, filtering based on given criteria. """
    chunk['combined_text'] = chunk['title'] + chunk['selftext']
    return chunk[(chunk['combined_text'].str.len() > min_num_characters) & (chunk['score'] >= min_score)][required_columns]


def convert_to_parquet(input_dir, output_dir, min_num_characters, min_score):
    total_rows = 0
    total_filtered_rows = 0
    required_columns = ['id', 'author', 'title', 'selftext', 'subreddit', 'created_utc', 'score', 'num_comments', 'media']

    for filename in os.listdir(input_dir):
        if filename.endswith('.zst'):
            file_path = os.path.join(input_dir, filename)
            print(f"Processing {file_path}...")
            buffer = ""

            # try:
            with open(file_path, 'rb') as compressed:
                decompressor = zstd.ZstdDecompressor(max_window_size=2147483648)
                reader = decompressor.stream_reader(compressed)

                while True:
                    chunk = reader.read(2**30) # Read 1 GB at a time  
                    if not chunk:
                        break
                    raw_data = buffer + chunk.decode('utf-8')

                    try:
                        data = pd.read_json(StringIO(raw_data), lines=True)
                        buffer = ""  # Clear buffer as data is correctly parsed
                    except ValueError:
                        # Likely got cut in the middle of a JSON object; find the last newline
                        last_newline = raw_data.rfind('\n')
                        data = pd.read_json(StringIO(raw_data[:last_newline]), lines=True)
                        buffer = raw_data[last_newline+1:]  # Save remainder to buffer

                    if all(col in data.columns for col in required_columns):
                        filtered_chunk = process_chunk(data, min_num_characters, min_score, required_columns)
                        print(filtered_chunk.head())
                        total_rows += len(data)
                        total_filtered_rows += len(filtered_chunk)

                        # Construct the output path for the Parquet file
                        parquet_path = os.path.join(output_dir, os.path.splitext(filename)[0] + '.parquet')

                        # Save the DataFrame as a Parquet file, appending to existing file
                        filtered_chunk.to_parquet(parquet_path, compression='snappy', index=False)
                        print(f"Processed a chunk. Saved to {parquet_path}")
                    else:
                        print("Some required columns are missing in the chunk.")

            # except Exception as e:
            #     print(f"Failed to process {file_path}: {e}")

    print(f"Total rows processed: {total_rows}")
    print(f"Total filtered rows: {total_filtered_rows}")




if __name__ == '__main__':
    input_dir = 'data/compressed'
    output_dir = 'data/parquet_1'
    min_num_characters = 25  
    min_score = 10  # TODO: computed dynamically based on month/year
    convert_to_parquet(input_dir, output_dir, min_num_characters, min_score)