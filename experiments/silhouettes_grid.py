from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MultipleLocator
import utils
from scipy.spatial.distance import squareform, pdist
from silhouette_upper_bound import (
    upper_bound,
    upper_bound_macro_silhouette,
    upper_bound_samples,
)


rows, cols = 2, 4
fig, axes = plt.subplots(rows, cols, figsize=(16, 10))
axes = axes.flatten()

datasets = [
    ("ceramic", 2, "cityblock"),
    ("customers", 2, "cosine"),
    ("dermatology", 6, "cosine"),
    ("heart_statlog", 2, "euclidean"),
    ("optdigits", 10, "euclidean"),
    ("rna", 5, "euclidean"),
    ("wdbc", 2, "euclidean"),
    ("wine", 3, "cityblock"),
]

for i, (dataset, n_clusters, metric) in enumerate(datasets):

    results = pd.read_pickle(f"results/{dataset}.pkl")
    results = results[results["n_clusters"] == n_clusters]
    ax = axes[i]

    labels, scores, min_cluster_size, cluster_sizes = (
        results["cluster_labels"].values[0],
        results["silh_samples"].values[0],
        results["min_cluster_size"].values[0],
        results["cluster_sizes"].values[0],
    )

    # load array of feature vectors and distance matrix
    X = np.load(f"arrays/{dataset}.npy")
    D = squareform(pdist(X, metric=metric))

    ub_samples = upper_bound_samples(D)

    ub_samples_adjusted = upper_bound_samples(D, m=min_cluster_size)

    data = utils.get_silhouette_plot_data(labels, scores, n_clusters, ub_samples)

    data_adjusted = utils.get_silhouette_plot_data(
        labels, scores, n_clusters, ub_samples_adjusted
    )

    score, ub, ub_adjusted = (
        np.mean(scores),
        np.mean(ub_samples),
        np.mean(ub_samples_adjusted),
    )

    for x in data.keys():

        # Cluster Silhouette bounds
        ax.fill_betweenx(
            np.arange(data[x]["y_lower"], data[x]["y_upper"]),
            0,
            data[x]["sorted_ub_values"],
            facecolor=data[x]["color"],
            edgecolor=data[x]["color"],
            alpha=0.6,
        )

        # Cluster Silhouette bounds (m = min cluster size)
        ax.fill_betweenx(
            np.arange(data_adjusted[x]["y_lower"], data_adjusted[x]["y_upper"]),
            0,
            data_adjusted[x]["sorted_ub_values"],
            facecolor=data_adjusted[x]["color"],
            edgecolor=data_adjusted[x]["color"],
            alpha=1.0,
        )

        # Cluster Silhouette scores
        ax.fill_betweenx(
            np.arange(data[x]["y_lower"], data[x]["y_upper"]),
            0,
            data[x]["sorted_silhouettes"],
            facecolor=data[x]["color"],
            edgecolor="black",
            alpha=1.0,
        )

        # Label cluster number
        ax.text(-0.05, data[x]["y_lower"] + 0.5 * data[x]["size_cluster_i"], str(x))

    ax.axvline(x=ub, color="black", linestyle="--", label=rf"ASW upper bound global")
    ax.axvline(
        x=ub_adjusted,
        color="black",
        linestyle="dotted",
        label=rf"ASW upper bound adjusted",
    )
    ax.axvline(x=score, color="orange", linestyle="-", label="ASW")
    ax.set_title(dataset.replace("_", " ").title(), fontsize=15)
    ax.set_xlim([-0.1, 1.1])
    ax.set_yticks([])
    ax.legend(fontsize=8, loc="upper right")

plt.tight_layout()
plt.savefig("silhouettes_grid.pdf", bbox_inches="tight")
print("Silhouette grid plot generated!")
plt.close()
