from typing import List
import os


def extract_zst_file_names(folder_path: str) -> List[str]:
    file_names = os.listdir(folder_path)
    file_names = [f.split('.')[0] for f in file_names if f.startswith('RS')]
    return file_names


def launch_jobs(cluster: bool = True):
    if cluster:
        zst_folder_path = '/cluster/work/coss/anmusso/reddit/submissions'
        parquet_folder_path = '/cluster/work/coss/anmusso/reddit_parquet/submissions'
    else:
        zst_folder_path = '/Users/andrea/Desktop/PhD/Projects/Current/Reddit/data/comments'
        parquet_folder_path = '/Users/andrea/Desktop/PhD/Projects/Current/Reddit/data/parquet_test'

    file_names = extract_zst_file_names(folder_path=zst_folder_path)
    file_names = sorted(file_names)
    file_names = ['RS_2022-01', 'RS_2022-02', 'RS_2022-03', 'RS_2022-04', 'RS_2022-05', 'RS_2022-06', 'RS_2022-07', 'RS_2022-08', 'RS_2022-09', 'RS_2022-10', 'RS_2022-11', 'RS_2022-12', 'RS_2023-01', 'RS_2023-02', 'RS_2023-03', 'RS_2023-04']
    print(file_names)

    os.system('chmod +x run_script.sh')
    for file_name in file_names:
        if cluster:
            os.system(f'sbatch run_script.sh {file_name} {zst_folder_path} {parquet_folder_path}')
        else:
            os.system(f'./run_script.sh {file_name} {zst_folder_path} {parquet_folder_path}')


if __name__ == '__main__':
    is_cluster = 'cluster' in os.getcwd()
    launch_jobs(cluster=is_cluster)