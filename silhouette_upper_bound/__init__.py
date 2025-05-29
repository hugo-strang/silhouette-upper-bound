
import numpy as np
from .utils import _row_f, _check_dissimilarity_matrix

def upper_bound_samples(D: np.ndarray, kappa: int = 2) -> np.ndarray:
    """
    ...
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
    ...
    """

    point_bounds = upper_bound_samples(D=D, kappa=kappa)

    return np.mean(point_bounds)


