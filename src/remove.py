from dask_cuda import LocalCUDACluster
from dask.distributed import Client
import dask.array as da
from cuml.datasets import make_blobs
from cuml.manifold import UMAP
from cuml.dask.manifold import UMAP as MNMG_UMAP
import numpy as np

def main():
    # Start the Dask cluster
    cluster = LocalCUDACluster(threads_per_worker=1)
    client = Client(cluster)

    # Generate synthetic data
    features, y = make_blobs(1000000, 10, centers=42, cluster_std=0.1, dtype=np.float32, random_state=10)

    # Initialize and fit the local UMAP model on a subset of the data
    local_model = UMAP(random_state=10)

    selection = np.random.RandomState(10).choice(1000, 100)
    X_train = features[selection]
    y_train = y[selection]
    local_model.fit(X_train)

    # Initialize the distributed UMAP model
    distributed_model = MNMG_UMAP(model=local_model)

    # Convert data to Dask array and perform transformation
    scattered_features = client.scatter(features, broadcast=True)

    distributed_X = da.from_delayed(scattered_features, shape=features.shape, dtype=features.dtype)
    embedding = distributed_model.transform(distributed_X)
    result = embedding.compute()

    print(result)

    # Close the Dask client and cluster
    client.close()
    cluster.close()

if __name__ == "__main__":
    main()
