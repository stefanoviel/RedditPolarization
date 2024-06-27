import os
import sys

# adding root directory to paths
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from logging_config import configure_get_logger
import config

if not os.path.exists(config.OUTPUT_DIR):
    os.makedirs(config.OUTPUT_DIR)
logger = configure_get_logger(config.OUTPUT_DIR, executed_file_name = __file__)

from src.utils_run_single_step import run_function_with_overrides
import h5py
from tqdm import tqdm
import numpy as np
import cuml
from cuml.manifold import UMAP
from dask_cuda import LocalCUDACluster
from dask.distributed import Client
import dask.array as da
# from cuml.dask.manifold import UMAP as MNMG_UMAP
os.environ["NUMEXPR_MAX_THREADS"] = "32"
import numexpr


logger.info("WORKING")