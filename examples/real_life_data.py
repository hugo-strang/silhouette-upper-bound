"""
In this script, we generate a table containing results on syntehtic datasets.
"""

from silhouette_upper_bound import upper_bound
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score, pairwise_distances
from collections import Counter
import pandas as pd


def _run_experiment(dataset_name: str, krange=range(2, 20)):

    path = f"data/{dataset_name}/data.csv"

    print(f"==== Running dataset: {dataset_name} ====\n")

    data = pd.read_csv(path).select_dtypes(include="number")
    print(f"Data shape: {data.shape}")

    D = pairwise_distances(X=data.to_numpy())

    best_k, best_score, best_min_cluster_size = 2, 0, 0
    for k in krange:

        kmeans = KMeans(n_clusters=k, random_state=42)

        cluster_labels = kmeans.fit_predict(data)

        min_cluster_size = min(Counter(cluster_labels).values())

        silh_score = silhouette_score(data, cluster_labels)

        if min_cluster_size == 1:
            continue  # We don't allow singletons

        if silh_score > best_score:
            best_k = k
            best_score = silh_score
            best_min_cluster_size = min_cluster_size

    print(f"Best Silhouette = {round(best_score,3)} at K = {best_k}")

    print(f"Upper Bound: {round(upper_bound(D),3)}")

    print(f"Min cluster size at K = {best_k}: {best_min_cluster_size}")

    print(
        f"Upper Bound (kappa = min cluster size): {round(upper_bound(D, kappa=best_min_cluster_size),3)}\n"
    )


if __name__ == "__main__":

    _run_experiment(dataset_name="rna")

    _run_experiment(dataset_name="biblical_texts")

    _run_experiment(dataset_name="frogs")

    _run_experiment(dataset_name="ceramic")

    _run_experiment(dataset_name="conference_papers")
