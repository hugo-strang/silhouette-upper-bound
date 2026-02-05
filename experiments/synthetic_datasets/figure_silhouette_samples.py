"""
This file generates a figure that shows individual silhouette widths compared to their corresponding upper bounds for a synthetic dataset.
"""

from silhouette_upper_bound import upper_bound_samples
import numpy as np
import matplotlib.pyplot as plt
import json
from sklearn.cluster import KMeans
from pathlib import Path
from sklearn.metrics import silhouette_samples
import sys
from pathlib import Path

parent_dir = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, parent_dir)
import utils


if __name__ == "__main__":

    # Figure for repo

    base_path = Path(__file__).parent

    # 1. load data

    datasets_path = base_path / "datasets"

    folder = f"{datasets_path}/400-64-5-6"

    feature_vectors, labels, distance_matrix = (
        np.load(f"{folder}/feature_vectors.npy"),
        np.load(f"{folder}/labels.npy"),
        np.load(f"{folder}/distance_matrix.npy"),
    )

    params_file = f"{folder}/parameters.json"

    with open(params_file, "r", encoding="utf-8") as f:
        dataset_parameters = json.load(f)

    # 2. Kmeans clustering
    cluster_labels = (
        KMeans(n_clusters=dataset_parameters["centers"], random_state=0).fit_predict(
            feature_vectors
        )
        + 1
    )

    scores = silhouette_samples(distance_matrix, cluster_labels, metric="precomputed")

    # 3. upper bound
    ub_samples = upper_bound_samples(D=distance_matrix)

    # 4. aggregate silhouette and upper bound
    score, ub = np.mean(scores), np.mean(ub_samples)
    print(f"ASW = {score} | ub = {ub}")

    # 5. data for figure
    data = utils.get_silhouette_plot_data(
        cluster_labels, scores, dataset_parameters["centers"], ub_samples
    )

    # 6. generate figure

    fig, ax = plt.subplots(figsize=(10, 6))

    for x in data.keys():

        # Cluster Silhouette scores
        ax.fill_betweenx(
            np.arange(data[x]["y_lower"], data[x]["y_upper"]),
            0,
            data[x]["sorted_silhouettes"],
            facecolor=data[x]["color"],
            edgecolor="black",
            alpha=0.8,
        )

        # Cluster Silhouette bounds
        ax.fill_betweenx(
            np.arange(data[x]["y_lower"], data[x]["y_upper"]),
            0,
            data[x]["sorted_ub_values"],
            facecolor=data[x]["color"],
            edgecolor=data[x]["color"],
            alpha=0.5,
        )

        # Label cluster number
        ax.text(-0.05, data[x]["y_lower"] + 0.5 * data[x]["size_cluster_i"], str(x))

    ax.axvline(x=ub, color="black", linestyle="--", label=rf"upper bound")
    ax.axvline(x=score, color="black", linestyle="-", label="ASW")
    ax.set_xlim([-0.1, 0.5])
    ax.set_yticks([])
    ax.legend(fontsize=12, title_fontsize=13, loc="upper right")
    ax.set_ylabel("Cluster label", fontsize=14)
    ax.set_xlabel(
        "Silhouette width (opaque)\nand corresponding upper bound (transparent)",
        fontsize=14,
    )

    plt.savefig(f"{base_path}/figures/silhouette_samples.png", bbox_inches="tight")
    plt.close()
