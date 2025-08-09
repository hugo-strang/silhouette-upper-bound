import numpy as np
from .utils import _row_f, _check_dissimilarity_matrix


def upper_bound_samples(D: np.ndarray, kappa: int = 1) -> np.ndarray:
    """
    Compute a sharp upper bound of the Silhouette coefficient for each data point.

    Parameters
    ----------
    D: np.ndarray
        Square matrix of pairwise distances (or dissimilarities) (shape: [n_samples, n_samples]).
    kappa: int
        Lower limit of cluster size (default is 1).

    Returns
    -------
    np.ndarray
        A 1D array where the i:th element is a sharp upper bound of the Silhouette coefficient s(i).

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

    if kappa < 1 or kappa > n - 1:
        raise ValueError("The parameter kappa is out of range.")

    # Compute bounds
    bounds = np.apply_along_axis(
        lambda row: _row_f(row, kappa=kappa, n=n), axis=1, arr=D_hat
    )

    return bounds


def upper_bound(D: np.ndarray, kappa: int = 1) -> float:
    """
    Compute an upper bound of the Average Silhouette Width (ASW). The upper bound ranges from 0 to 1.

    Parameters
    ----------
    D: np.ndarray
        Square matrix of pairwise distances (or dissimilarities) (shape: [n_samples, n_samples]).
    kappa: int
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
    """

    point_bounds = upper_bound_samples(D=D, kappa=kappa)

    return np.mean(point_bounds)
