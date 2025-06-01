import numpy as np
from .utils import _row_f, _check_dissimilarity_matrix


def upper_bound_samples(D: np.ndarray, kappa: int = 2) -> np.ndarray:
    """
    Compute an upper bound of the Silhouette coefficient for each sample.

    Notation:
        - D: n x n dissimilarity matrix corresponding to dataset.
        - C: Clustering of data points from D.
        - s(i): Silhouette coefficient of the i:th data point.

    Let C* denote a globally Silhouette-optimal clustering. To construct an upper bound of each s(i),
    we use the fact that, in C*, each data point i belongs to a cluster of size Delta.
    The average distance between i and every other data point in the same cluster
    is not smaller than the average distance between i and the Delta - 1 points closest to i -- call this average x(Delta).
    Furthermore, the average distance between i and the n - Delta points farthest from i -- call this average y(Delta) -- is not smaller than
    the average distance between i and every point in the neighboring cluster closest to i.
    By combining these two observations, we construct a value guaranteed to be greater than or equal to s(i),
    namely UB(i,D) := 1 - min(x(Delta) / y(Delta)), where the minimum is taken over Delta = kappa, ..., n - kappa.

    Parameters
    ----------
    D: np.ndarray
        Square matrix of pairwise distances (or dissimilarities) (shape: [n_samples, n_samples]).
    kappa: int
        Lower limit for cluster size.

    Returns
    -------
    np.ndarray
        An array where the i:th element is an upper bound of the Silhouette coefficient s(i).

    References
    ----------
    .. [1] Silhouette (clustering). Wikipedia. https://en.wikipedia.org/wiki/Silhouette_(clustering)
    """

    _check_dissimilarity_matrix(D=D)

    # Remove diagonal from distance matrix and then sort
    D_hat = np.sort(D[~np.eye(D.shape[0], dtype=bool)].reshape(D.shape[0], -1))

    n = D_hat.shape[0]
    if n < 4:
        raise ValueError("Matrix must be at least of size 4x4.")

    if kappa < 1 or kappa > n - 2:
        raise ValueError("The parameter kappa is out of range.")

    # Compute bounds
    bounds = np.apply_along_axis(
        lambda row: _row_f(row, kappa=kappa, n=n), axis=1, arr=D_hat
    )

    return bounds


def upper_bound(D: np.ndarray, kappa: int = 2) -> float:
    """
    Compute an upper bound of the Average Silhouette Width (ASW). The upper bound ranges from 0 to 1.

    Notation:
        - D: n x n dissimilarity matrix corresponding to dataset.
        - C: Clustering of data points from D.
        - s(i): Silhouette coefficient of the i:th data point.
        - ASW(C,D): Mean s(i) over all data points, given the clustering C.

    Let C* denote a globally Silhouette-optimal clustering. To construct an upper bound of ASW(C*,D),
    we use the fact that, in C*, each data point i belongs to a cluster of size Delta.
    The average distance between i and every other data point in the same cluster
    is not smaller than the average distance between i and the Delta - 1 points closest to i -- call this average x(Delta).
    Furthermore, the average distance between i and the n - Delta points farthest from i -- call this average y(Delta) -- is not smaller than
    the average distance between i and every point in the neighboring cluster closest to i.
    By combining these two observations, we construct a value guaranteed to be greater than or equal to s(i),
    namely UB(i,D) := 1 - min(x(Delta) / y(Delta)), where the minimum is taken over Delta = kappa, ..., n - kappa.
    The overall upper bound is obtained by taking the mean of these values. To summarize, we have ASW(C*,D) <= mean(UB(i,D)).

    To obtain the upper bound for each sample, use `upper_bound_samples()`.

    We emphasize that the upper bound is not guaranteed to be close to the true global ASW-maximum.
    In other words, its usefulness is expected to vary across different datasets. However, exploring the individual bounds UB(i,D)
    as well as the effects of increasing the kappa-parameter should provide valuable insights in most clustering contexts.

    Parameters
    ----------
    D: np.ndarray
        Square matrix of pairwise distances (or dissimilarities) (shape: [n_samples, n_samples]).
    kappa: int
        Lower limit for cluster size.

    Returns
    -------
    float
        An upper bound of the ASW.

    References
    ----------
    .. [1] Silhouette (clustering). Wikipedia. https://en.wikipedia.org/wiki/Silhouette_(clustering)
    """

    point_bounds = upper_bound_samples(D=D, kappa=kappa)

    return np.mean(point_bounds)
