import pandas as pd
import zstandard
from typing import List


def read_and_decode(reader: zstandard.ZstdDecompressionReader, chunk_size: int, max_window_size: int, previous_chunk=None, bytes_read: int = 0) -> str:
    chunk = reader.read(chunk_size)
    bytes_read += chunk_size
    if previous_chunk is not None:
        chunk = previous_chunk + chunk
    try:
        return chunk.decode()
    # If we get a UnicodeDecodeError, it's likely because we have a character that's split between two chunks
    # The simplest way to handle this is to read another chunk and append it to the previous one
    except UnicodeDecodeError:
        if bytes_read > max_window_size:
            raise UnicodeError(f"Unable to decode frame after reading {bytes_read:,} bytes")
        print(f"Decoding error with {bytes_read:,} bytes, reading another chunk")
        return read_and_decode(reader, chunk_size, max_window_size, chunk, bytes_read)


def read_lines_zst(file_name: str, reader_window_size: int = 2 ** 31, reader_chunk_size: int = 2 ** 27):
    with open(file_name, 'rb') as file_handle:
        buffer = ''
        reader = zstandard.ZstdDecompressor(max_window_size=reader_window_size).stream_reader(file_handle)
        # Loop through the entire file and read it in chunks
        while True:
            chunk = read_and_decode(reader=reader, chunk_size=reader_chunk_size, max_window_size=reader_window_size // 4)

            if not chunk:
                break
            lines = (buffer + chunk).split("\n")

            for line in lines[:-1]:
                yield line, file_handle.tell()

            buffer = lines[-1]

        reader.close()



