import numpy as np
from collections.abc import Iterable
from .utils import _row_f, _row_f_given_list, _check_dissimilarity_matrix
from numba.typed import List


def upper_bound_samples(D: np.ndarray, m: int | Iterable = 1) -> np.ndarray:
    """
    Compute a sharp upper bound of the Silhouette coefficient for each data point.

    Parameters
    ----------
    D: np.ndarray
        Square matrix of pairwise distances (or dissimilarities) (shape: [n_samples, n_samples]).
    m: int
        Lower limit of cluster size (default is 1).

    Returns
    -------
    np.ndarray
        A 1D array where the i:th element is a sharp upper bound of the Silhouette coefficient s(i).

    References
    ----------
    .. [1] Silhouette (clustering). Wikipedia. https://en.wikipedia.org/wiki/Silhouette_(clustering)
    .. [2] An upper bound of the silhouette validation metric for clustering. arXiv. https://arxiv.org/abs/2509.08625
    """

    _check_dissimilarity_matrix(D=D)

    # Remove diagonal from distance matrix and then sort
    D_hat = np.sort(D[~np.eye(D.shape[0], dtype=bool)].reshape(D.shape[0], -1))

    n = D_hat.shape[0]
    if n < 4:
        raise ValueError("Matrix must be at least of size 4x4.")

    if isinstance(m, (int, np.integer)):  # m integer
        if m < 1 or m > n // 2:
            raise ValueError("The parameter m is out of range.")
        per_row_computation = _row_f
    elif isinstance(m, Iterable):  # m iterable
        if sum(m) != n:
            raise ValueError("There is a mismatch in input and number of datapoints.")
        elif len(m) < 2:
            raise ValueError("Number of clusters is smaller than two.")
        per_row_computation = _row_f_given_list
        typed_m = List()
        [typed_m.append(k) for k in m]

        m = typed_m
    else:
        raise ValueError("Wrong input type.")

    # Compute bounds
    bounds = np.apply_along_axis(
        lambda row: per_row_computation(row, m=m, n=n), axis=1, arr=D_hat
    )

    return bounds


def upper_bound(D: np.ndarray, m: int | Iterable = 1) -> float:
    """
    Compute an upper bound of the Average Silhouette Width (ASW). The upper bound ranges from 0 to 1.

    Parameters
    ----------
    D: np.ndarray
        Square matrix of pairwise distances (or dissimilarities) (shape: [n_samples, n_samples]).
    m: int
        Lower limit of cluster size (default is 1).

    Returns
    -------
    float
        An upper bound of the ASW.

    Notes
    -----
    We emphasize that the upper bound is not guaranteed to be close to the true global ASW-maximum.
    Comparison with outputs from suitable clustering algorithms is advised.

    References
    ----------
    .. [1] Silhouette (clustering). Wikipedia. https://en.wikipedia.org/wiki/Silhouette_(clustering)
    .. [2] An upper bound of the silhouette validation metric for clustering. arXiv. https://arxiv.org/abs/2509.08625
    """

    point_bounds = upper_bound_samples(D=D, m=m)

    return np.mean(point_bounds)


def upper_bound_macro_silhouette(D: np.ndarray, cluster_sizes: Iterable) -> float:
    """
    Compute an upper bound of the macro-averaged silhouette. The upper bound ranges from 0 to 1.

    Parameters
    ----------
    D: np.ndarray
        Square matrix of pairwise distances (or dissimilarities) (shape: [n_samples, n_samples]).
    cluster_sizes: Iterable
        Fixed cluster sizes that define our constrained solution space.

    Returns
    -------
    float
        An upper bound of the macro-averaged silhouette.

    Notes
    -----
    We emphasize that the upper bound is not guaranteed to be close to the true macro-silhouette-maximum.
    Comparison with outputs from suitable clustering algorithms is advised.
    Moreover, the upper bound covers only clusterings with cluster sizes matching the given input.

    References
    ----------
    .. [1] Silhouette (clustering). Wikipedia. https://en.wikipedia.org/wiki/Silhouette_(clustering)
    .. [2] An upper bound of the silhouette validation metric for clustering. arXiv. https://arxiv.org/abs/2509.08625
    .. [3] Revisiting Silhouette Aggregation. arXiv. https://arxiv.org/abs/2401.05831
    """

    if not isinstance(cluster_sizes, Iterable):
        raise ValueError("Wrong input type.")

    point_bounds_sorted = np.sort(upper_bound_samples(D=D, m=cluster_sizes))

    n_clusters = len(cluster_sizes)

    _sum = 0

    cluster_sizes = sorted(cluster_sizes, reverse=True)

    id_counter = 0
    for cluster_size in cluster_sizes:

        _sum += np.sum(
            point_bounds_sorted[id_counter : id_counter + cluster_size] / cluster_size
        )

        id_counter += cluster_size

    return _sum / n_clusters
