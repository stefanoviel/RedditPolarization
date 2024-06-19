import io
import zstandard as zstd

def count_lines_in_zst_file(file_path):
    """Counts the number of lines in a Zstandard compressed file."""
    line_count = 0
    with open(file_path, 'rb') as file_handle:
        # Create a decompressor object
        decompressor = zstd.ZstdDecompressor(max_window_size=2147483648)
        # Stream reader provides a file-like interface for reading the decompressed data
        with decompressor.stream_reader(file_handle) as reader:
            # TextIOWrapper allows us to read lines from binary stream
            text_stream = io.TextIOWrapper(reader, encoding='utf-8')
            for line in text_stream:
                line_count += 1
                print(f'line count: {line_count:,}', end='\r')

    return line_count

# Usage
file_path = "data/RS_2014-08.zst"
total_lines = count_lines_in_zst_file(file_path)
print(f"The file {file_path} contains {total_lines:,} lines.")
