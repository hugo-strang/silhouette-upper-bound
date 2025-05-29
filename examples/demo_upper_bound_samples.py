"""
In this example, we cluster a synthetic dataset using KMeans. 
We then plot the Silhouette score of each sample and compare it to its upper bound. 
"""

import matplotlib.pyplot as plt
import numpy as np
from silhouette_upper_bound import upper_bound, upper_bound_samples
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score, silhouette_samples, pairwise_distances

def main():
    # Generate synthetic data
    X, _ = make_blobs(n_samples=500, n_features=16, centers=3, cluster_std=2, random_state=42)
    D = pairwise_distances(X)
    
    # Cluster with KMeans
    kmeans = KMeans(n_clusters=3, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(X)
    
    # Overall silhouette score
    score = silhouette_score(X, labels)

    # Upper bound
    ub = upper_bound(D)

    # Upper bound for each sample 
    ub_samples = upper_bound_samples(D)
    
    # Compute silhouette values for each sample
    sample_silhouette_values = silhouette_samples(X, labels)
    
    fig, ax = plt.subplots(figsize=(8, 5))

    # Silhouette plot setup
    ax.set_xlim([-0.1, 1])
    ax.set_ylim([0, len(X) + (3 + 1) * 10])  # 3 clusters + padding

    y_lower = 10
    for i in range(3):

        indices = np.where(labels == i)[0]
        cluster_silhouettes = sample_silhouette_values[indices]
        cluster_ub_values = ub_samples[indices]

        # Get sorted order of silhouette values
        sorted_order = np.argsort(cluster_silhouettes)
        
        sorted_silhouettes = cluster_silhouettes[sorted_order]
        sorted_ub_values = cluster_ub_values[sorted_order]    

        size_cluster_i = sorted_silhouettes.shape[0]
        y_upper = y_lower + size_cluster_i

        color = plt.cm.viridis(float(i) / 3)

        # Cluster Silhouette scores
        ax.fill_betweenx(np.arange(y_lower, y_upper),
                        0, sorted_silhouettes,
                        facecolor=color, edgecolor=color, alpha=0.7)
        
        # Cluster Silhouette bounds
        ax.fill_betweenx(np.arange(y_lower, y_upper),
                        0, sorted_ub_values,
                        facecolor=color, edgecolor=color, alpha=0.2)

        # Label cluster number
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        y_lower = y_upper + 10  # 10 for spacing between clusters

    ax.set_title("Silhouette plot for clusters")
    ax.set_xlabel("Silhouette coefficient values and upper bounds (shaded)")
    ax.set_ylabel("Cluster label")
    ax.axvline(x=score, color="red", linestyle="--", label="Average silhouette score")
    ax.axvline(x=ub, color="black", linestyle="--", label="Upper bound")
    ax.set_yticks([])  # Clear y-axis labels for clarity
    ax.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
