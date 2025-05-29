"""
In this example, we show how to lower the upper bound assuming the restriction
that no cluster is smaller than kappa, where kappa > 2. 
"""

import numpy as np
from silhouette_upper_bound import upper_bound
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score, pairwise_distances

def main():
    # Generate synthetic data
    X, _ = make_blobs(n_samples=3000, n_features=32, centers=7, random_state=42)
    D = pairwise_distances(X)
    
    # Cluster with KMeans
    kmeans = KMeans(n_clusters=7, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(X)
    
    # Overall silhouette score
    score = silhouette_score(X, labels)
    print(f"KMeans Silhouette score: {score:.3f}")

    print(f"Silhouette upper bound: {upper_bound(D):.3f}")

    # Upper bound with kappa > 2
    min_cluster_size = np.bincount(labels).min() 
    print(f"Silhouette upper bound (given that we don't allow cluster sizes smaller than {min_cluster_size}): {upper_bound(D, kappa=min_cluster_size):.3f}")

if __name__ == "__main__":
    main()
