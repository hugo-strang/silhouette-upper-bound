"""
This file generates a figure that shows individual silhouette widths compared to their corresponding upper bounds for a synthetic dataset.
"""

from silhouette_upper_bound import upper_bound_samples
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import utils


# Figure for repo

n_samples, n_features, centers, cluster_std = 400, 64, 5, 6

X, _ = make_blobs(
    n_samples=n_samples,
    n_features=n_features,
    centers=centers,
    cluster_std=cluster_std,
    random_state=0,
)
D = utils.data_to_distance_matrix(data=X, metric="euclidean")

# clustering
kmeans_dict = utils.asw_optimization(
    algorithm=utils.algorithm_kmedoids,
    data=D,
    k_range=range(centers, centers + 1),
    asw_metric="precomputed",
)

scores = kmeans_dict["best_scores"]

# upper bound
ub_samples = upper_bound_samples(D=D)

data = utils.get_silhouette_plot_data(
    kmeans_dict["best_labels"], scores, centers, ub_samples
)

score, ub = np.mean(scores), np.mean(ub_samples)

print(score)

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

plt.savefig("../figures/silhouette_samples.png", bbox_inches="tight")
plt.close()
