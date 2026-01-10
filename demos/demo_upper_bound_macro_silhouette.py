"""
In this example, we show how to use the macro-averaged silhouette upper bound on a synthetic dataset.
"""

import numpy as np
from silhouette_upper_bound import upper_bound_macro_silhouette
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_samples, pairwise_distances
from collections import Counter


def _macro_averaged_silhouette(dissimilarity_matrix, labels):

    silhouette_scores = silhouette_samples(
        X=dissimilarity_matrix, labels=labels, metric="precomputed"
    )

    mac_silh = []

    for cluster_id in np.unique(labels):
        scores = silhouette_scores[labels == cluster_id]

        mac_silh.append(np.mean(scores))

    return np.mean(mac_silh)


def main():
    # Generate synthetic data
    X, _ = make_blobs(n_samples=3000, n_features=32, centers=7, random_state=42)
    D = pairwise_distances(X)

    # Cluster with KMeans
    kmeans = KMeans(n_clusters=7, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(X)

    # Fixed cluster sizes that match our empirical results (this defines our solution space)
    cluster_sizes = Counter(labels).values()

    # Macro-averaged silhouette score
    score = _macro_averaged_silhouette(D, labels)
    print(f"KMeans macro silhouette: {score:.3f} | Cluster sizes: {cluster_sizes}")

    print(
        f"Macro silhouette upper bound: {upper_bound_macro_silhouette(D, cluster_sizes):.3f}"
    )


if __name__ == "__main__":
    main()
