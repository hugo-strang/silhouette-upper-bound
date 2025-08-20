from silhouette_upper_bound import upper_bound_samples
from sklearn.datasets import make_blobs
from sklearn.metrics import pairwise_distances
from utils import kmeans_optimized, hierarchical_optimized, Counter, kmedoids_optimized
import numpy as np


def table(rows):

    headers = [
        "Dataset",
        "Metric",
        "Hierarchical weighted",
        "Hierarchical single",
        "KMeans",
        "Upper bound",
        "Min",
        "Max",
    ]

    # Format header
    header_line = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join(["---"] * len(headers)) + " |"

    print(header_line)
    print(separator)

    # Format rows
    for row in rows:

        print(" & ".join(f"${str(cell)}$" for cell in row) + " \\\ ")


def table_row(params):

    n_samples, n_features, centers, cluster_std = params
    # Generate synthetic data
    X, _ = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=centers,
        cluster_std=cluster_std,
        random_state=0,
    )
    D = pairwise_distances(X)

    # Compute upper bound
    ubs = upper_bound_samples(D)
    ub = np.mean(ubs)
    ubs_min = np.min(ubs)
    ubs_max = np.max(ubs)

    # Kmeans
    #kmeans_dict = kmeans_optimized(data=X, n_init=10)
    kmeans_dict = kmedoids_optimized(data=X, metric="euclidean", k_range=range(2, 51))

    # Single
    single_dict = hierarchical_optimized(data=X, metric="euclidean", method="single", t_range=range(2, 3))

    # Weigthed
    weighted_dict = hierarchical_optimized(
        data=X, metric="euclidean", method="weighted"
    )

    kmeans_str = f"${kmeans_dict['best_score']:.3f}$ ({len(Counter(kmeans_dict['best_labels']))})"
    weighted_str = f"${weighted_dict['best_score']:.3f}$ ({len(Counter(weighted_dict['best_labels']))})"
    single_str = f"${single_dict['best_score']:.3f}$ ({len(Counter(single_dict['best_labels']))})"

    return (
        "-".join(str(x) for x in params),
        "Euclidean",
        weighted_str,
        single_str,
        kmeans_str,
        ub,
        ubs_min,
        ubs_max,
    )


def table(caseparams: list):
    """
    Print table in terminal.
    """

    headers = [
        "Dataset",
        "Metric",
        "Hierarchical weighted",
        "Hierarchical single",
        "KMeans",
        "UB(D)",
        "minUB(D)",
        "maxUB(D)",
    ]

    lines = []

    # Format header
    header_line = "| " + " | ".join(headers) + " |"
    lines.append(header_line)
    separator = "| " + " | ".join(["---"] * len(headers)) + " |"
    lines.append(separator)

    for params in caseparams:
        row = table_row(params=params)

        lines.append(
            " & ".join(
                f"${cell:.3f}$" if type(cell) is not str else f"{cell}" for cell in row
            )
            + " \\\ "
        )

    # Print table to terminal
    print("\nTABLE\n")
    for line in lines:
        print(line)


if __name__ == "__main__":

    # n_samples, n_features, n_centers, cluster_std
    case1params = (400, 64, 5, 6) # best score: 0.24871862135056447 | n clusters: 5
    case2params = (400, 64, 2, 2) # best score: 0.6730066307708127 | n clusters: 2
    case3params = (400, 128, 7, 3) # best score: 0.5215261021315551 | n clusters: 7
    case4params = (1000, 161, 2, 13) # best score: 0.04598680544028653 | n clusters: 3
    case5params = (1000, 300, 5, 2) # best score: 0.6677330849537492 | n clusters: 5
    case6params = (10000, 32, 20, 2) # best score: 0.6258718209220459 | n clusters: 20
    case7params = (10000, 1024, 20, 4) # best score: 0.4172529997913183 | n clusters: 20

    table(
        [
            #case1params,
            #case2params,
            #case3params,
            case4params,
            case5params,
            case6params,
            case7params,
        ]
    )
