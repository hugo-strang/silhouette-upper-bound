"""
This file generates a table comparing empirical ASW values with the upper bound for synthetic datasets.
"""

from sklearn.datasets import make_blobs
from scipy.spatial.distance import pdist
import utils


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
    D = utils.data_to_distance_matrix(data=X, metric="euclidean")

    # Compute upper bound
    ub_dict = utils.get_upper_bound(data=X, metric="euclidean")

    # Kmedoids
    kmedoids_dict = utils.asw_optimization(
        algorithm=utils.algorithm_kmedoids,
        data=D,
        k_range=range(centers, centers + 1),
        asw_metric="precomputed",
        fast=True,
    )

    kmedoids_str = f"{kmedoids_dict['best_score']:.3f}"

    return (
        "-".join(str(x) for x in params),
        str(int(centers)),
        kmedoids_str,
        ub_dict["ub"],
        (ub_dict["ub"] - kmedoids_dict["best_score"]) / ub_dict["ub"],
    )


def table(caseparams: list):
    """
    Print table in terminal.
    """

    headers = [
        "Dataset",
        "K",
        "ASW",
        "UB",
        "wcre",
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
                f"{cell:.3f}" if type(cell) is not str else f"{cell}" for cell in row
            )
            + " \\\ "
        )

    # Print table to terminal
    print("\nTABLE\n")
    for line in lines:
        print(line)


def cluster_centers(params):
    n_samples, n_features, centers, cluster_std = params

    # Generate synthetic data
    X, y, cluster_centers = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=centers,
        cluster_std=cluster_std,
        random_state=0,
        return_centers=True,
    )

    # Compute all pairwise distances
    dists = pdist(cluster_centers, metric="euclidean")

    # Average distance between centers
    avg_dist = dists.mean()

    print(f"params: {params}")
    print("cluster_centers shape:", cluster_centers.shape)
    print(f"min inter-center distance: {dists.min()}")
    print(f"max inter-center distance: {dists.max()}")
    print(f"max - min inter-center distance: {dists.max() - dists.min()}")

    print("\naverage inter-center distance:", f"{avg_dist:.1f}")

    return avg_dist


if __name__ == "__main__":

    # n_samples, n_features, n_centers, cluster_std
    case1params = (400, 64, 5, 6)
    case2params = (400, 64, 2, 2)
    case3params = (400, 128, 7, 3)
    case4params = (1000, 300, 5, 2)
    case5params = (10000, 32, 20, 2)
    case6params = (10000, 1024, 20, 4)

    table(
        [
            case1params,
            case2params,
            case3params,
            case5params,
            case6params,
        ]
    )

    cluster_centers(case6params)
