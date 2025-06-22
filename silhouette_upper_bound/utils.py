import numpy as np
from numba import njit


def _check_dissimilarity_matrix(D: np.ndarray, tol: float = 1e-15):

    # Check that D is a valid dissimilarity matrix
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError("Matrix must be square.")

    if not np.all(D >= -tol):
        raise ValueError("Matrix must be non-negative.")

    if not np.allclose(np.diag(D), 0, atol=tol):
        raise ValueError("Matrix must have zero diagonal.")

    if not np.allclose(D, D.T, atol=tol):
        raise ValueError("Matrix must be symmetric.")


@njit
def _row_f(row: np.ndarray, kappa: int, n: int) -> float:

    y = np.sum(row[kappa - 1 :])
    # Initialize q
    if kappa == 1:
        x = 0
        q = 1
    else:
        x = np.sum(row[: kappa - 1])
        q = (x / (kappa - 1)) / (y / (n - kappa))

    for delta in range(kappa + 1, n - kappa + 1):
        d_to_move = row[delta - 2]

        x += d_to_move
        y -= d_to_move

        q_candidate = (x / (delta - 1)) / (y / (n - delta))

        if q_candidate < q:
            q = q_candidate

    return 1 - q
