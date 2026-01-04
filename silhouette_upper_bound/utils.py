import numpy as np
from collections.abc import Iterable
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
def _row_f(row: np.ndarray, m: int, n: int) -> float:

    y = np.sum(row[m - 1 :])
    # Initialize q
    if m == 1:
        x = 0
        q = 1
    else:
        x = np.sum(row[: m - 1])
        q = (x / (m - 1)) / (y / (n - m))

    for k in range(m + 1, n - m + 1):
        d_to_move = row[k - 2]

        x += d_to_move
        y -= d_to_move

        q_candidate = (x / (k - 1)) / (y / (n - k))

        if q_candidate < q:
            q = q_candidate

    return 1 - q


@njit
def _row_f_given_list(row: np.ndarray, m: Iterable, n: int) -> float:

    m = list(set(m))

    m.sort()

    m_init = m[0]

    y = np.sum(row[m_init - 1 :])
    # Initialize q
    if m_init == 1:
        x = 0
        q = 1
    else:
        x = np.sum(row[: m_init - 1])
        q = (x / (m_init - 1)) / (y / (n - m_init))

    for k in m[1:]:

        x = np.sum(row[: k - 1])
        y = np.sum(row[k - 1 :])

        q_candidate = (x / (k - 1)) / (y / (n - k))

        if q_candidate < q:
            q = q_candidate

    return 1 - q
