import numpy as np
import pytest
from silhouette_upper_bound import upper_bound, upper_bound_samples
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score, silhouette_samples, pairwise_distances
from collections import Counter
import kmedoids
import os
import csv


# Only write CSV if explicitly enabled
CSV_REPORT = os.getenv("CSV_REPORT", "0") == "1"
REPORT_FILE = "results/test_report_standard_silhouette.csv"

# If enabled, prepare the file with a header
if CSV_REPORT:
    with open(REPORT_FILE, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "method",
                "metric",
                "n_samples",
                "n_features",
                "centers",
                "cluster_std",
                "ASW",
                "UB",
                "UB-Score",
                "Wors-case relative error",
            ]
        )


def test_basic():

    d1 = np.array(
        [
            [0, 1, 1, 1, 1],
            [1, 0, 1, 1, 1],
            [1, 1, 0, 1, 1],
            [1, 1, 1, 0, 1],
            [1, 1, 1, 1, 0],
        ]
    )

    assert upper_bound(d1) == 0.0

    d2 = np.array(
        [
            [0, 1, 5, 5, 5],
            [1, 0, 5, 5, 5],
            [5, 5, 0, 1, 1],
            [5, 5, 1, 0, 1],
            [5, 5, 1, 1, 0],
        ]
    )

    assert upper_bound(d2) == 1 - 1 / 5

    d3 = np.array(
        [
            [0, 1, 6, 6.5, 7, 8],
            [1, 0, 5, 7.5, 6.88, 8],
            [6, 5, 0, 0.5, 0.78, 1.1],
            [6.5, 7.5, 0.5, 0, 0.5, 1],
            [7, 6.88, 0.78, 0.5, 0, 1],
            [8, 8, 1.1, 1, 1, 0],
        ]
    )

    d3f = np.array(
        [
            4 * 1 / (6 + 6.5 + 7 + 8),
            4 * 1 / (5 + 7.5 + 6.88 + 8),
            (2 / 3) * (0.5 + 0.78 + 1.1) / (6 + 5),
            (2 / 3) * (0.5 + 0.5 + 1) / (6.5 + 7.5),
            (2 / 3) * (0.78 + 0.5 + 1) / (7 + 6.88),
            (2 / 3) * (1.1 + 1 + 1) / (8 + 8),
        ]
    )

    assert np.abs(upper_bound(d3) - np.mean(1 - d3f)) < 1e-15


def _test_helper(
    D: np.ndarray, score: float, score_samples: np.ndarray, labels: np.ndarray, *args
):

    # 1. Test standard upper bound
    ub = upper_bound(D)

    assert 0 <= ub and ub <= 1
    assert ub - score >= -1e-15

    # 2. Test each sample
    ub_samples = upper_bound_samples(D)
    assert np.all(0 <= ub_samples) and np.all(ub_samples <= 1)
    assert np.all(ub_samples - score_samples >= -1e-15)

    # Upper bound with m > 1 (constraint on smallest cluster size)
    min_cluster_size = min(Counter(labels).values())

    assert min_cluster_size > 0

    if min_cluster_size > 1:

        # 3. Test restricted upper bound
        ub_m = upper_bound(D, m=min_cluster_size)

        assert 0 <= ub_m and ub_m <= 1
        assert ub_m - score >= -1e-15

        # 4. Test each sample for restricted upper bound
        ub_m_samples = upper_bound_samples(D, m=min_cluster_size)
        assert np.all(0 <= ub_m_samples) and np.all(ub_m_samples <= 1)
        assert np.all(ub_m_samples - score_samples >= -1e-15)

        # 5. Check that restricted bound is not greater than standard bounds
        assert ub - ub_m >= -1e-15

    # --- CSV reporting (only if CSV_REPORT=1) ---
    if CSV_REPORT:
        with open(REPORT_FILE, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                list(args)
                + [
                    "{:.5f}".format(score),
                    "{:.5f}".format(ub),
                    "{:.5f}".format(ub - score),
                    format((ub - score) / ub, ".0%"),
                ]
            )


@pytest.mark.parametrize("n_samples", [100, 200, 300, 400, 500])
@pytest.mark.parametrize("n_features", [10, 15, 20])
@pytest.mark.parametrize("centers", [3, 6, 9])
@pytest.mark.parametrize("cluster_std", [1.0, 2.0, 3.0, 10.0])
def test_blobs_kmeans(n_samples, n_features, centers, cluster_std):
    X, _ = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=centers,
        cluster_std=cluster_std,
        random_state=42,
    )

    D = pairwise_distances(X)

    # KMeans clustering
    model = KMeans(n_clusters=centers, random_state=42, n_init="auto")
    labels = model.fit_predict(X)
    score = silhouette_score(X, labels)
    score_samples = silhouette_samples(X, labels)

    _test_helper(
        D,
        score,
        score_samples,
        labels,
        "kmeans",
        "euclidean",
        n_samples,
        n_features,
        centers,
        cluster_std,
    )


@pytest.mark.parametrize("metric", ["euclidean", "manhattan", "chebyshev"])
@pytest.mark.parametrize("n_samples", [100, 200, 300, 400, 500])
@pytest.mark.parametrize("n_features", [10, 15, 20])
@pytest.mark.parametrize("centers", [3, 6, 9])
@pytest.mark.parametrize("cluster_std", [1.0, 2.0, 3.0, 10.0])
def test_blobs_kmedoids(metric, n_samples, n_features, centers, cluster_std):
    X, _ = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=centers,
        cluster_std=cluster_std,
        random_state=42,
    )

    D = pairwise_distances(X, metric=metric)

    # KMedoids clustering
    labels = kmedoids.fastmsc(diss=D, medoids=centers, random_state=42).labels + 1
    score = silhouette_score(X=D, labels=labels, metric="precomputed")
    score_samples = silhouette_samples(X=D, labels=labels, metric="precomputed")

    _test_helper(
        D,
        score,
        score_samples,
        labels,
        "kmedoids",
        metric,
        n_samples,
        n_features,
        centers,
        cluster_std,
    )


@pytest.mark.parametrize("n_samples", [2000])
@pytest.mark.parametrize("n_features", [164])
@pytest.mark.parametrize("centers", [3, 7])
@pytest.mark.parametrize("cluster_std", [2.0, 3.0])
def test_dense_blobs_kmeans(n_samples, n_features, centers, cluster_std):
    X, _ = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=centers,
        cluster_std=cluster_std,
        random_state=42,
    )

    D = pairwise_distances(X)

    # KMeans clustering
    model = KMeans(n_clusters=centers, random_state=42, n_init="auto")
    labels = model.fit_predict(X)
    score = silhouette_score(X, labels)
    score_samples = silhouette_samples(X, labels)

    _test_helper(
        D,
        score,
        score_samples,
        labels,
        "kmeans",
        "euclidean",
        n_samples,
        n_features,
        centers,
        cluster_std,
    )


@pytest.mark.parametrize("metric", ["euclidean", "manhattan", "chebyshev"])
@pytest.mark.parametrize("n_samples", [2000])
@pytest.mark.parametrize("n_features", [164])
@pytest.mark.parametrize("centers", [3, 7])
@pytest.mark.parametrize("cluster_std", [2.0, 3.0])
def test_dense_blobs_kmedoids(metric, n_samples, n_features, centers, cluster_std):
    X, _ = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=centers,
        cluster_std=cluster_std,
        random_state=42,
    )

    D = pairwise_distances(X, metric=metric)

    # KMedoids clustering
    labels = kmedoids.fastmsc(diss=D, medoids=centers, random_state=42).labels + 1
    score = silhouette_score(X=D, labels=labels, metric="precomputed")
    score_samples = silhouette_samples(X=D, labels=labels, metric="precomputed")

    _test_helper(
        D,
        score,
        score_samples,
        labels,
        "kmedoids",
        metric,
        n_samples,
        n_features,
        centers,
        cluster_std,
    )
