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
import matplotlib.pyplot as plt

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

    if len(k_range) > 1:
        logger.info(f"Optimizing ASW")

    for k in tqdm(k_range, disable=bool(len(k_range) == 1)):

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


# ========
# Plotting
# ========


def get_silhouette_plot_data(labels, scores, n_clusters, ub_samples):

    data = {i: {} for i in range(1, n_clusters + 1)}

    y_lower = 10
    for i in data.keys():

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


def _macro_averaged_silhouette(dissimilarity_matrix, labels):

    silhouette_scores = silhouette_samples(
        X=dissimilarity_matrix, labels=labels, metric="precomputed"
    )

    mac_silh = []

    for cluster_id in np.unique(labels):
        scores = silhouette_scores[labels == cluster_id]

        mac_silh.append(np.mean(scores))

    return np.mean(mac_silh)
