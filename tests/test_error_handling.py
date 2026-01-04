import pytest
import numpy as np
from silhouette_upper_bound import upper_bound, upper_bound_macro_silhouette


def test_upper_bound_input():

    np.random.seed(42)

    # ---- Wrong diagonal ----
    A = np.random.rand(100, 100)
    D = (A + A.T) / 2
    np.fill_diagonal(D, 1)

    # check that ValueError is raised
    with pytest.raises(ValueError):
        upper_bound(D=D)

    with pytest.raises(ValueError):
        upper_bound(D=D, m=5)

    # ---- Wrong dimensions ----
    A = np.random.rand(100, 101)

    # check that ValueError is raised
    with pytest.raises(ValueError):
        upper_bound(D=A)

    with pytest.raises(ValueError):
        upper_bound(D=A, m=5)

    # ---- Wrong m parameter ----
    A = np.random.rand(100, 100)
    D = (A + A.T) / 2
    np.fill_diagonal(D, 0)

    wrong_ms = [0, -10, 1.5, 51, 98, 101, np.float16(3.0)]

    for m in wrong_ms:
        # check that ValueError is raised
        with pytest.raises(ValueError):
            upper_bound(D=D, m=m)

    # ---- Ok m parameter ----
    ok_ms = [
        1,
        25,
        50,
        [30, 50, 20],
        [1, 99],
        np.int64(3),
        np.int32(2),
        np.int16(10),
    ]
    for m in ok_ms:
        # Shouldn't raise error
        try:
            upper_bound(D=D, m=m)
        except Exception as e:
            pytest.fail(f"upper_bound raised {type(e)} unexpectedly!")


def test_upper_bound_macro_silhouette_input():

    np.random.seed(42)

    # ---- Missing cluster sizes ----
    A = np.random.rand(100, 100)
    D = (A + A.T) / 2
    np.fill_diagonal(D, 0)

    # check that TypeError is raised
    with pytest.raises(TypeError):
        upper_bound_macro_silhouette(D=D)

    # ---- Wrong cluster sizes inputs ----

    wrong_inputs = [0, -10, 1.5, 51, 98, 101, [49, 52], [33, 33, 33]]

    for cluster_sizes in wrong_inputs:
        # check that ValueError is raised
        with pytest.raises(ValueError):
            upper_bound_macro_silhouette(D=D, cluster_sizes=cluster_sizes)

    # ---- Ok cluster sizes inputs ----

    ok_inputs = [
        [33, 33, 34],
        [50, 50],
        np.array([10, 20, 70]),
        np.array([10 for _ in range(10)]),
    ]
    for cluster_sizes in ok_inputs:
        # Shouldn't raise error
        try:
            upper_bound_macro_silhouette(D=D, cluster_sizes=cluster_sizes)
        except Exception as e:
            pytest.fail(f"upper_bound_macro_silhouette raised {type(e)} unexpectedly!")
