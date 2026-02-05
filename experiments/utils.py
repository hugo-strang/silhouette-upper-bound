from sklearn.metrics import silhouette_samples
import numpy as np
import kmedoids
import matplotlib.pyplot as plt


# ==========
# Algorithm
# ==========


def algorithm_kmedoids(
    data: np.ndarray, k: int, random_state: int = 42, fast=False
) -> np.ndarray:

    if fast:
        cluster_labels = (
            kmedoids.fastmsc(diss=data, medoids=k, random_state=random_state).labels + 1
        )
    else:
        cluster_labels = (
            kmedoids.pamsil(diss=data, medoids=k, random_state=random_state).labels + 1
        )

    return cluster_labels


# ========
# Plotting
# ========


def get_silhouette_plot_data(labels, scores, n_clusters, ub_samples):
    """
    Cluster labels should be 1-indexed, i.e. 1, 2, 3, ... n_clusters
    """

    data = {i: {} for i in range(1, n_clusters + 1)}

    y_lower = 10
    for i in sorted(list(data.keys())):

        indices = np.where(labels == i)[0]
        cluster_silhouettes = scores[indices]
        cluster_ub_values = ub_samples[indices]

        # Get sorted order of silhouette values
        sorted_order = np.argsort(cluster_silhouettes)

        sorted_silhouettes = cluster_silhouettes[sorted_order]
        sorted_ub_values = cluster_ub_values[sorted_order]

        size_cluster_i = sorted_silhouettes.shape[0]
        y_upper = y_lower + size_cluster_i

        color = plt.cm.viridis(float(i) / n_clusters)

        data[i]["y_lower"] = y_lower
        data[i]["y_upper"] = y_upper
        data[i]["sorted_silhouettes"] = sorted_silhouettes
        data[i]["color"] = color
        data[i]["sorted_ub_values"] = sorted_ub_values
        data[i]["size_cluster_i"] = size_cluster_i

        # update y_lower
        y_lower = y_upper + 10

    return data


# ========
# Macro silhouette
# ========


def macro_averaged_silhouette(dissimilarity_matrix, labels):

    silhouette_scores = silhouette_samples(
        X=dissimilarity_matrix, labels=labels, metric="precomputed"
    )

    mac_silh = []

    for cluster_id in np.unique(labels):
        scores = silhouette_scores[labels == cluster_id]

        mac_silh.append(np.mean(scores))

    return np.mean(mac_silh)
