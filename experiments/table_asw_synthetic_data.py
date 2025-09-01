"""
This file generates a table comparing empirical ASW values with the upper bound for synthetic datasets.
"""

from sklearn.datasets import make_blobs
import utils


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


def table_row(params, k_range: range = range(2, 31)):

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

    # Weigthed
    weighted_dict = utils.asw_optimization(algorithm=utils.algorithm_hierarchical,
                                               data=D,
                                               k_range=k_range,
                                               asw_metric="precomputed",
                                               method="weighted"
                                               )

    # Single
    single_dict = single_dict = utils.asw_optimization(algorithm=utils.algorithm_hierarchical,
                                               data=D,
                                               k_range=k_range,
                                               asw_metric="precomputed",
                                               method="single"
                                               )

    # Kmeans
    kmeans_dict = utils.asw_optimization(algorithm=utils.algorithm_kmeans,
                                             data=X,
                                             k_range=k_range,
                                             asw_metric="euclidean"
                                             )

    # Kmedoids
    kmedoids_dict = utils.asw_optimization(algorithm=utils.algorithm_kmedoids,
                                             data=D,
                                             k_range=k_range,
                                             asw_metric="precomputed"
                                             )


    weighted_str = f"${weighted_dict['best_score']:.3f}$ ({len(utils.Counter(weighted_dict['best_labels']))})"
    single_str = f"${single_dict['best_score']:.3f}$ ({len(utils.Counter(single_dict['best_labels']))})"
    kmeans_str = f"${kmeans_dict['best_score']:.3f}$ ({len(utils.Counter(kmeans_dict['best_labels']))})"
    kmedoids_str = f"${kmeans_dict['best_score']:.3f}$ ({len(utils.Counter(kmedoids_dict['best_labels']))})"

    return (
        "-".join(str(x) for x in params),
        "Euclidean",
        weighted_str,
        single_str,
        kmeans_str,
        kmedoids_str,
        ub_dict["ub"],
        ub_dict["min"],
        ub_dict["max"],
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
        "KMedoids",
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
    case1params = (400, 64, 5, 6)
    case2params = (400, 64, 2, 2)
    case3params = (400, 128, 7, 3)
    case4params = (1000, 161, 2, 13)
    case5params = (1000, 300, 5, 2)
    case6params = (10000, 32, 20, 2)
    case7params = (10000, 1024, 20, 4)

    table(
        [
            case1params,
            case2params,
            case3params,
            case4params,
            case5params,
            case6params,
            case7params,
        ]
    )
