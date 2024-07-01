import sys
from transform import zst_to_parquet

if __name__ == '__main__':
    file_name = sys.argv[1]
    zst_folder_path = sys.argv[2]
    parquet_folder_path = sys.argv[3]
    zst_to_parquet(zst_file_path=f'{zst_folder_path}/{file_name}.zst',
                   parquet_folder_path=parquet_folder_path,
                   parquet_file_name=file_name)

