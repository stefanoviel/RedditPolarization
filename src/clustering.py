import os
import sys

# adding root directory to paths
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from logging_config import configure_get_logger
import config

import cuml


def hdbscan_coordinates(coordinates, old_table, info_table, sql_db):
    scanner = cuml.cluster.hdbscan.HDBSCAN(min_cluster_size=config.HDBS_MIN_CLUSTERSIZE, min_samples=config.HDBS_MIN_SAMPLES)
    clusters = scanner.fit_predict(coordinates.iloc[:, :-1]) 