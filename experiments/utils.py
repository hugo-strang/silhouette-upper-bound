from silhouette_upper_bound import upper_bound_samples
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_samples, pairwise_distances
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter
import kmedoids
from sklearn.preprocessing import StandardScaler
from scipy.io import arff
from scipy.spatial.distance import pdist
import os
from typing import Callable
import logging
from logging import Logger

# =======
# Logging
# =======


def get_logger(name: str) -> Logger:
    """
    Get a configured logger with the given name.
    Ensures no duplicate handlers are added.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:  # Avoid adding multiple handlers
        handler = logging.StreamHandler()  # console output
        formatter = logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


logger = get_logger(__name__)

# ==========
# Algorithms
# ==========


def algorithm_kmeans(data, k, random_state=42, n_init="auto") -> np.ndarray:
    """
    Apply kmeans to data.

    Parameters
    ----------
        data: np.ndarray
            Shape n_samples x n_features.

        k: int
            Number of clusters.

    Returns
    -------
        np.ndarray
            Cluster labels
    """

    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)

    return kmeans.fit_predict(data) + 1


def algorithm_hierarchical(data: np.ndarray, k: int, method: str) -> np.ndarray:
    """
    ...
    """

    vector_form_data = squareform(
        data, checks=False
    )  # convert dissimilarity matrix to vector-form distance vector

    return fcluster(linkage(vector_form_data, method=method), t=k, criterion="maxclust")


def algorithm_kmedoids(data: np.ndarray, k: int, random_state: int = 42) -> np.ndarray:

    if data.shape[0] < 1000:
        cluster_labels = (
            kmedoids.pamsil(diss=data, medoids=k, random_state=random_state).labels + 1
        )
    else:
        cluster_labels = (
            kmedoids.fastmsc(diss=data, medoids=k, random_state=random_state).labels + 1
        )

    return cluster_labels


# ============
# Optimization
# ============


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


def asw_optimization(
    algorithm: Callable,
    data: np.ndarray,
    k_range: range,
    asw_metric: str,
    ub_reference: float | None = None,
    epsilon: float = 0.15,
    **kwargs,
):
    """

    Parameters
    ----------
        algorithm: Callable
            function that returns cluster labels corresponding to dataset

        data: np.ndarray
            if algorithm is kmeans, then shape should be n_samples x n_features,
            otherwise shape should be n_samples x n_samples (distance_matrix).

        k_range: range
            k candidates

        asw_metric: str
            e.g. 'euclidean' or 'precomputed'.

        ub_reference: float | None
            used for early stopping (default is None, which means no early stopping is applied).

        epsilon: float
            early stopping tolerance (default is 0.15).

    """

    # Inititalize best solution
    best_solution = {
        "best_score": 0,  # ASW
        "best_scores": None,  # Silhouette samples
        "best_labels": None,  # Cluster labels
        "stopped_early": False,  # yes/no early stopping applied
    }

    logger.info(f"Optimizing ASW")
    for k in tqdm(k_range):

        cluster_labels = algorithm(data, k, **kwargs)

        best_solution = _optim_iteration(
            data=data,
            cluster_labels=cluster_labels,
            metric=asw_metric,
            best_solution=best_solution,
        )

        if ub_reference is not None:

            if (ub_reference - best_solution["best_score"]) / ub_reference < epsilon:
                logger.info("Stopping early!")
                best_solution["stopped_early"] = True
                return best_solution

    return best_solution


# ===========
# Upper bound
# ===========


def get_upper_bound(data: np.ndarray, metric: str) -> dict:

    D = pairwise_distances(data, metric=metric)  # convert data to dissimilarity matrix

    logger.info(f"Computing upper bound")

    ubs = upper_bound_samples(D)

    ub = np.mean(ubs)
    ubs_min = np.min(ubs)
    ubs_max = np.max(ubs)

    logger.info(f"UB: {ub}")

    return {"ub": ub, "min": ubs_min, "max": ubs_max, "samples": ubs}


# ===========
# Load data
# ===========


def data_to_distance_matrix(data, metric, TOL=1e-10):
    """
    Parameters
    ----------
        data: np.ndarray
            shape n_samples x n_features.

        metric: str
            distance metric.

        TOL: float
            tolerance for matrix symmetry.

    Returns
    -------
        np.ndarray
            distance matrix of shape n_samples x n_samples.
    """

    D = pairwise_distances(data, metric=metric)  # convert data to dissimilarity matrix

    assert np.linalg.norm(D - D.T, ord="fro") < TOL, f"Matrix X is not symmetric!"
    assert (
        np.abs(np.diag(D)).max() < TOL
    ), f"Diagonal entries of X are not close to zero!"

    return D


def load_unlabeled_data(dataset: str, transpose: bool = False) -> np.ndarray:
    """
    Load unlabeled dataset.

    Parameters
    ----------
        dataset: str
            Should be 'ceramic', 'conference_papers', 'religious_papers' or 'rna'

        transpose: bool
            Should be True if data has shape n_features x n_samples (default is False).
    """

    path = f"data/unlabeled/{dataset}/data.csv"

    logger.info(f"==== Running dataset: {dataset} ====\n")

    df = pd.read_csv(path)

    data = df.select_dtypes(include="number")

    if transpose:
        data = data.transpose()

    logger.info(f"Data shape: {data.shape}")

    data = data.to_numpy()

    # Removing zero-vectors
    non_zero_rows = ~np.all(data == 0, axis=1)

    data = data[non_zero_rows]

    logger.info(f"Data shape (zeros removed): {data.shape}")

    return data


def load_arff_as_distance_matrix(path, metric="euclidean", scale=False):
    # Load
    data, meta = arff.loadarff(path)
    df = pd.DataFrame(data)
    fname = os.path.basename(path).lower()

    if "wdbc" in fname:
        # First column is ID, second is label
        y = df.iloc[:, 1].astype(str).to_numpy()
        X = df.iloc[:, 2:].to_numpy()

    elif "wine" in fname:
        # Frst column is label
        y = df.iloc[:, 0].astype(str).to_numpy()
        X = df.iloc[:, 1:].to_numpy()

    elif "yeast" in fname:
        # Frst column is ID
        y = df.iloc[:, -1].astype(str).to_numpy()
        X = df.iloc[:, 1:].to_numpy()

    elif "mopsi-joensuu" in fname:
        # Frst column is ID
        y = np.zeros(df.shape[0])
        X = df.iloc[:, :].to_numpy()

    else:
        # Last column is label
        y = df.iloc[:, -1].astype(str).to_numpy()
        X = df.iloc[:, :-1].to_numpy()

    # Replace missing values ("?")
    X = np.where(X == b"?", np.nan, X).astype(float)
    col_means = np.nanmean(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_means, inds[1])

    if scale:
        X = StandardScaler().fit_transform(X)

    # Remove potential zero vecs
    X = X[~np.all(np.isclose(X, 0, atol=1e-12), axis=1)]

    # Distance matrix
    D = squareform(pdist(X, metric=metric))

    return D, X, y
