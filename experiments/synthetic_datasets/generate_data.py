"""
Use make_blobs to generate synthetic datasets and save in datasets/ dir.
"""

from sklearn.datasets import make_blobs
from collections import namedtuple
from sklearn.metrics import pairwise_distances
from pathlib import Path
import numpy as np
import json


Parameters = namedtuple(
    "Parameters", ["n_samples", "n_features", "centers", "cluster_std"]
)


if __name__ == "__main__":

    datasets_dir = Path(__file__).parent / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)

    make_blobs_parameters = [
        Parameters(n_samples=400, n_features=64, centers=5, cluster_std=6),
        Parameters(n_samples=400, n_features=64, centers=2, cluster_std=2),
        Parameters(n_samples=400, n_features=128, centers=7, cluster_std=3),
        Parameters(n_samples=1000, n_features=300, centers=5, cluster_std=2),
        Parameters(n_samples=10000, n_features=32, centers=20, cluster_std=2),
        Parameters(n_samples=10000, n_features=1024, centers=20, cluster_std=4),
    ]

    random_state = 0

    for parameters in make_blobs_parameters:

        dataset_tag = f"{parameters.n_samples}-{parameters.n_features}-{parameters.centers}-{parameters.cluster_std}"

        dataset_dir = Path(__file__).parent / "datasets" / dataset_tag
        dataset_dir.mkdir(parents=True, exist_ok=True)

        X, y = make_blobs(
            n_samples=parameters.n_samples,
            n_features=parameters.n_features,
            centers=parameters.centers,
            cluster_std=parameters.cluster_std,
            random_state=random_state,
        )

        D = pairwise_distances(X=X, metric="euclidean")

        np.save(f"{dataset_dir}/feature_vectors", X)
        np.save(f"{dataset_dir}/labels", y)
        np.save(f"{dataset_dir}/distance_matrix", D)

        with open(f"{dataset_dir}/parameters.json", "w") as f:
            json.dump(parameters._asdict(), f, indent=4)
