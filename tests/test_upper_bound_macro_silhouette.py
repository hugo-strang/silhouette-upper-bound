import numpy as np
import pytest
from silhouette_upper_bound import upper_bound_macro_silhouette
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score, silhouette_samples, pairwise_distances
from collections import Counter
import kmedoids
import os
import csv


# Only write CSV if explicitly enabled
CSV_REPORT = os.getenv("CSV_REPORT", "0") == "1"
REPORT_FILE = "results/test_report_macro_silhouette.csv"

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
                "Macro Silhouette",
                "UB",
                "UB-Score",
                "Wors-case relative error",
                "cluster sizes",
            ]
        )


def _macro_averaged_silhouette(dissimilarity_matrix, labels):

    silhouette_scores = silhouette_samples(
        X=dissimilarity_matrix, labels=labels, metric="precomputed"
    )

    mac_silh = []

    for cluster_id in np.unique(labels):
        scores = silhouette_scores[labels == cluster_id]

        mac_silh.append(np.mean(scores))

    return np.mean(mac_silh)


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
    score = _macro_averaged_silhouette(dissimilarity_matrix=D, labels=labels)

    cluster_sizes = list(Counter(labels).values())

    ub = upper_bound_macro_silhouette(D=D, cluster_sizes=cluster_sizes)

    assert 0 <= ub and ub <= 1

    assert ub - score >= -1e-15

    # --- CSV reporting (only if CSV_REPORT=1) ---
    if CSV_REPORT:
        with open(REPORT_FILE, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "kmeans",
                    "euclidean",
                    n_samples,
                    n_features,
                    centers,
                    cluster_std,
                    "{:.5f}".format(score),
                    "{:.5f}".format(ub),
                    "{:.5f}".format(ub - score),
                    format((ub - score) / ub, ".0%"),
                    cluster_sizes,
                ]
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
    score = _macro_averaged_silhouette(dissimilarity_matrix=D, labels=labels)

    cluster_sizes = list(Counter(labels).values())

    ub = upper_bound_macro_silhouette(D=D, cluster_sizes=cluster_sizes)

    assert 0 <= ub and ub <= 1

    assert ub - score >= -1e-15

    # --- CSV reporting (only if CSV_REPORT=1) ---
    if CSV_REPORT:
        with open(REPORT_FILE, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "kmedoids",
                    metric,
                    n_samples,
                    n_features,
                    centers,
                    cluster_std,
                    "{:.5f}".format(score),
                    "{:.5f}".format(ub),
                    "{:.5f}".format(ub - score),
                    format((ub - score) / ub, ".0%"),
                    cluster_sizes,
                ]
            )
