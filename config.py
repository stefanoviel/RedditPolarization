import os
import time

# add timestamp to not have same experiments with overlapping names
timestamp = time.strftime("%Y%m%d_%H%M%S") 

EXPERIMENT_NAME = "first"
OUTPUT_PATH = os.path.join("/cluster/work/coss/stviel", f"{timestamp}_{EXPERIMENT_NAME}")








# OLD parameters

MONTHS_COMMENTS = []
MONTHS_SUBMISSIONS = ['05', '06']

EMBEDD_SET = ['S']
ANALYSE_SET = 'S'

MODEL_PATH = f'/cluster/work/coss/anmusso/victoria/model/all-mpnet-base-v2'
INPUT_FILE_BASE_PATH = '/cluster/work/coss/anmusso/reddit'
EMBEDINGS_BASE_PATH = '/cluster/work/coss/anmusso/victoria/embeddings' #where embeddingfiles will be saved
DATA_BASE_PATH = '/cluster/work/coss/anmusso/victoria/loaded_data/loaded_data.db'

MODEL_NAME = 'all-mpnet-base-v2' # available models: 'all-mpnet-base-v2' , 'all-MiniLM-L6-v2', 'multi-qa-mpnet-base-dot-v1'

FILTER_UMAP= False
UMAP_COMPONENTS = 5
UMAP_N_Neighbors = 50
UMAP_MINDIST = 0.01

HDBS_MIN_CLUSTERSIZE= 50
#HDBS_ALG = ''

TERMS = ['blm', 'Black', 'Lives', 'Matter', 'BLM', 'racist', 'racism', 'defund', 'police', 'George', 'Floyd']
SUBREDDIT = 'BlackLivesMatter'

