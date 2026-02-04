"""
Generate table comparing kmeodids ASW with upper bound.
"""

from pathlib import Path
import json
import numpy as np
import kmedoids
from silhouette_upper_bound import upper_bound
from sklearn.metrics import (
    silhouette_score,
    adjusted_rand_score,
    adjusted_mutual_info_score,
)


def table_row(
    labels: np.ndarray,
    distance_matrix: np.ndarray,
    n_clusters: int,
    random_state: int = 42,
    fast: bool = True,
):

    # Upper bound

    ub = upper_bound(distance_matrix)

    # Kmedoids clustering

    if fast:
        cluster_labels = kmedoids.fastmsc(
            diss=distance_matrix, medoids=n_clusters, random_state=random_state
        ).labels
    else:
        cluster_labels = kmedoids.pamsil(
            diss=distance_matrix, medoids=n_clusters, random_state=random_state
        ).labels

    # Evaluation

    asw = silhouette_score(distance_matrix, cluster_labels, metric="precomputed")

    wcre = 1 - asw / ub

    ari = adjusted_rand_score(labels, cluster_labels)

    ami = adjusted_mutual_info_score(labels, cluster_labels)

    measurements = [n_clusters, ari, ami, asw, ub, wcre]

    headers = ["Clusters", "ARI", "AMI", "ASW", "UB", "WCRE"]

    w = 10
    print("".join(f"{h:<{w}}" for h in headers))
    print("-" * (w * len(headers)))
    print("".join(f"{m:<{w}.3f}" for m in measurements))


if __name__ == "__main__":

    # Loop through each synthetic dataset and perform cluster analysis

    base_path = Path(__file__).parent / "datasets"

    for folder in base_path.iterdir():

        print(f"Dataset: {folder.name}")

        _, labels, distance_matrix = (
            np.load(f"{folder}/feature_vectors.npy"),
            np.load(f"{folder}/labels.npy"),
            np.load(f"{folder}/distance_matrix.npy"),
        )

        # Get parameters

        params_file = folder / "parameters.json"

        with open(params_file, "r", encoding="utf-8") as f:

            dataset_parameters = json.load(f)

        # Print table row
        table_row(labels, distance_matrix, dataset_parameters["centers"])
        print("\n")
