from silhouette_upper_bound import upper_bound_samples
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_samples, pairwise_distances
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter


def get_data(dataset: str, transpose=False) -> np.ndarray:
    path = f"data/{dataset}/data.csv"

    print(f"==== Running dataset: {dataset} ====\n")

    df = pd.read_csv(path)

    data = df.select_dtypes(include="number")

    if transpose:
        data = data.transpose()

    print(f"Data shape: {data.shape}")

    data = data.to_numpy()

    # Removing zero-vectors
    non_zero_rows = ~np.all(data == 0, axis=1)

    data = data[non_zero_rows]

    print(f"Data shape (zeros removed): {data.shape}")

    return data


def _optim_iteration(data, cluster_labels, metric, best_solution):

    try:
        silh_samples = silhouette_samples(data, cluster_labels, metric=metric)
    except:
        silh_samples = np.zeros(data.shape[0])

    silh_score = np.mean(silh_samples)

    if silh_score > best_solution["best_score"]:

        best_solution["best_score"] = silh_score
        best_solution["best_scores"] = silh_samples
        best_solution["best_labels"] = cluster_labels

    return best_solution


def kmeans_optimized(data, k_range=range(2, 31), random_state=42, n_init="auto"):
    """

    Parameters
    ----------
        data: np.ndarray
            array of shape (n_samples, n_features)
        k_range: range
            specifies which values for K to use
    """

    best_solution = {
        "best_score": 0,  # ASW
        "best_scores": None,  # Silhouette samples
        "best_labels": None,  # Cluster labels
    }

    print(f"Kmeans optimization")

    for k in tqdm(k_range):

        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)

        cluster_labels = kmeans.fit_predict(data) + 1  # 1:indexed

        best_solution = _optim_iteration(
            data=data,
            cluster_labels=cluster_labels,
            metric="euclidean",
            best_solution=best_solution,
        )

    n_clusters = len(Counter(best_solution["best_labels"]))
    if n_clusters <= 6:
        clusters = {int(k): v for k, v in Counter(best_solution["best_labels"]).items()}
        print(
            f"best score: {best_solution['best_score']} | n clusters: {n_clusters} | clusters: {sorted(clusters.items())}"
        )
    else:
        print(f"best score: {best_solution['best_score']} | n clusters: {n_clusters}")

    return best_solution


def hierarchical_optimized(
    data: np.ndarray, metric: str, method="single", t_range=range(2, 31), TOL=1e-10
):
    """

    Parameters
    ----------
        D: np.ndarray
            distance matrix of shape (n_samples, n_features)
        metric: str
            distance metric, e.g. "euclidean" or "cosine"
        method: str
            specifies the linkage method
        t_range: range
            specifies which values for K to use

    """

    D = pairwise_distances(data, metric=metric)  # convert data to dissimilarity matrix

    assert np.linalg.norm(D - D.T, ord="fro") < TOL, f"Matrix X is not symmetric!"
    assert (
        np.abs(np.diag(D)).max() < TOL
    ), f"Diagonal entries of X are not close to zero!"

    vector_D = squareform(
        D, checks=False
    )  # convert dissimilarity matrix to vector-form distance vector

    best_solution = {
        "best_score": 0,  # ASW
        "best_scores": None,  # Silhouette samples
        "best_labels": None,  # Cluster labels
    }

    print(f"{method} optimization")
    for t in tqdm(t_range):

        cluster_labels = fcluster(
            linkage(vector_D, method=method), t=t, criterion="maxclust"
        )

        best_solution = _optim_iteration(
            data=D,
            cluster_labels=cluster_labels,
            metric="precomputed",
            best_solution=best_solution,
        )

    print(
        f"best score: {best_solution['best_score']} | n clusters: {len(Counter(best_solution['best_labels']))}"
    )

    return best_solution


def get_upper_bound(data: np.ndarray, metric: str) -> dict:

    D = pairwise_distances(data, metric=metric)  # convert data to dissimilarity matrix

    print(f"Computing upper bound")

    ubs = upper_bound_samples(D)

    ub = np.mean(ubs)
    ubs_min = np.min(ubs)
    ubs_max = np.max(ubs)

    print(f"UB: {ub}")

    return {"ub": ub, "min": ubs_min, "max": ubs_max, "samples": ubs}
