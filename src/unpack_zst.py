import zstandard
from typing import List


def read_and_decode(reader: zstandard.ZstdDecompressionReader, chunk_size: int, max_window_size: int, previous_chunk=None, bytes_read: int = 0) -> str:
    """
    Reads and decodes chunks from a Zstandard compressed stream reader.
    
    Args:
        reader (zstandard.ZstdDecompressionReader): The Zstandard decompression reader object.
        chunk_size (int): Size of each chunk to read.
        max_window_size (int): Maximum allowed size of the window for decoding.
        previous_chunk (Optional[bytes]): Previously read chunk to concatenate with the current chunk.
        bytes_read (int, optional): Total bytes read so far. Defaults to 0.
        
    Returns:
        str: Decoded string from the concatenated chunks.
        
    Raises:
        UnicodeError: If unable to decode due to incomplete character frames after reaching `max_window_size`.
    """
    chunk = reader.read(chunk_size)
    bytes_read += chunk_size
    
    if previous_chunk is not None:
        chunk = previous_chunk + chunk
        
    try:
        return chunk.decode()
    except UnicodeDecodeError:
        if bytes_read > max_window_size:
            raise UnicodeError(f"Unable to decode frame after reading {bytes_read:,} bytes")
        print(f"Decoding error with {bytes_read:,} bytes, reading another chunk")
        return read_and_decode(reader, chunk_size, max_window_size, chunk, bytes_read)


def read_lines_zst(file_name: str, reader_window_size: int = 2 ** 31, reader_chunk_size: int = 2 ** 27):
    """
    Generator function to read lines from a Zstandard compressed file.

    Args:
        file_name (str): Path to the Zstandard compressed file.
        reader_window_size (int, optional): Maximum window size for the Zstandard decompressor. Defaults to 2 ** 31.
        reader_chunk_size (int, optional): Chunk size for reading from the Zstandard decompressor. Defaults to 2 ** 27.
        
    Yields:
        Tuple[str, int]: Each line from the file and its corresponding file position.
    """
    with open(file_name, 'rb') as file_handle:
        buffer = ''
        reader = zstandard.ZstdDecompressor(max_window_size=reader_window_size).stream_reader(file_handle)
        
        while True:
            chunk = read_and_decode(reader=reader, chunk_size=reader_chunk_size, max_window_size=reader_window_size // 4)

            if not chunk:
                break
            
            lines = (buffer + chunk).split("\n")

            for line in lines[:-1]:
                yield line, file_handle.tell()

            buffer = lines[-1]

        reader.close()


def write_lines_zst(writer: zstandard.ZstdCompressionWriter, lines: List[str]):
    """
    Writes lines to a Zstandard compressed stream writer.

    Args:
        writer (zstandard.ZstdCompressionWriter): Zstandard compression writer object.
        lines (List[str]): List of lines to write to the stream.
    """
    for line in lines:
        writer.write(line.encode('utf-8'))
        writer.write("\n".encode('utf-8'))


if __name__ == '__main__':

    # test that everything is working
    for n, line in enumerate(read_lines_zst(file_name='/cluster/work/coss/anmusso/reddit/submissions/RS_2023-09.zst')): 
        print(line)
        if n > 5: 
            break