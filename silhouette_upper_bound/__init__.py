
import numpy as np
from .utils import _row_f, _check_dissimilarity_matrix

def upper_bound_samples(D: np.ndarray, kappa: int = 2) -> np.ndarray:
    """
    Compute an upper bound of the Silhouette coefficient for each sample. 

    References
    ----------
    .. [1] Silhouette (clustering). Wikipedia. https://en.wikipedia.org/wiki/Silhouette_(clustering)
    """

    _check_dissimilarity_matrix(D=D)
    
    # Remove diagonal from distance matrix and then sort
    D_hat = np.sort(D[~np.eye(D.shape[0],dtype=bool)].reshape(D.shape[0],-1))

    n = D_hat.shape[0]
    if n < 4:
        raise ValueError("Matrix must be at least of size 4x4.")
    
    if kappa < 1 or kappa > n - 2:
        raise ValueError("The parameter kappa is out of range.")
    
    # Compute bounds
    bounds = np.apply_along_axis(lambda row: _row_f(row, kappa=kappa, n=n), axis=1, arr=D_hat)

    return bounds

def upper_bound(D: np.ndarray, kappa: int = 2) -> float:
    """
    Compute an upper bound of the Average Silhouette Width (ASW). The upper bound ranges from 0 to 1.
    
    When C is a clustering corresponding to the dissimilarity matrix D, we denote its Silhouette score by ASW(C,D).
    Let C* denote a globally Silhouette-optimal clustering. To construct an upper bound of ASW(C*,D), 
    we use the fact that, in C*, each i belongs to a cluster of size 
    2 <= delta <= n-2. The average distance between $i$ and every other data point in the same cluster 
    is not smaller than the average distance between $i$ and the $\Delta-1$ points closest to $i$. Furthermore,
    the average distance between $i$ and the $n-\Delta$ points farthest from $i$ is not smaller than 
    the average distance between $i$ and every point in the neighboring cluster closest to $i$.
    By combining these two observations, we construct a value guaranteed to be greater than or equal to $s(i,\mcC_K,D)$.
    
    Then ASW(C*,D) <= 1 - 

    Parameters
    ----------
    D: np.ndarray
        Square matrix of pairwise distances (or dissimilarities) (shape: [n_samples, n_samples]).
    kappa: int
        Lower limit of cluster size. 

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


