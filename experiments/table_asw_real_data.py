"""
This file generates a table comparing empirical ASW values with the upper bound for real datasets.

Note:
    - The real datasets are available at https://archive.ics.uci.edu/
"""

import utils

logger = utils.get_logger(__name__)


def table_row(dataset: str, metric: str, k_range: range = range(2, 3)):

    logger.info(f"\nDistance metric: {metric}")

    # Prepare data

    if dataset == "conference_papers":
        data = utils.load_unlabeled_data(dataset=dataset, transpose=True)
    else:
        data = utils.load_unlabeled_data(dataset=dataset)

    n = data.shape[0]

    ub_dict = utils.get_upper_bound(data=data, metric=metric)

    dissimilarity_matrix = utils.data_to_distance_matrix(data=data, metric=metric)

    # Weighted
    if n > 1000:
        weighted_dict = utils.asw_optimization(
            algorithm=utils.algorithm_hierarchical,
            data=dissimilarity_matrix,
            k_range=k_range,
            asw_metric="precomputed",
            method="weighted",
        )
    else:
        weighted_dict = utils.asw_optimization(
            algorithm=utils.algorithm_hierarchical,
            data=dissimilarity_matrix,
            k_range=range(2, n + 1),
            asw_metric="precomputed",
            method="weighted",
        )

    # Single
    if n > 1000:
        single_dict = utils.asw_optimization(
            algorithm=utils.algorithm_hierarchical,
            data=dissimilarity_matrix,
            k_range=k_range,
            asw_metric="precomputed",
            method="single",
        )
    else:
        single_dict = utils.asw_optimization(
            algorithm=utils.algorithm_hierarchical,
            data=dissimilarity_matrix,
            k_range=range(2, n + 1),
            asw_metric="precomputed",
            method="single",
        )

    # Kmeans
    if metric == "euclidean":
        kmeans_dict = utils.asw_optimization(
            algorithm=utils.algorithm_kmeans,
            data=data,
            k_range=k_range,
            asw_metric=metric,
        )
        kmeans_str = f"${kmeans_dict['best_score']:.3f}$ ({len(utils.Counter(kmeans_dict['best_labels']))})"
    else:
        kmeans_dict = {"best_score": "N/A"}
        kmeans_str = "N/A"

    # Kmedoids
    kmedoids_dict = utils.asw_optimization(
        algorithm=utils.algorithm_kmedoids,
        data=dissimilarity_matrix,
        k_range=k_range,
        asw_metric="precomputed",
    )

    kmedoids_str = f"${kmedoids_dict['best_score']:.3f}$ ({len(utils.Counter(kmedoids_dict['best_labels']))})"
    weighted_str = f"${weighted_dict['best_score']:.3f}$ ({len(utils.Counter(weighted_dict['best_labels']))})"
    single_str = f"${single_dict['best_score']:.3f}$ ({len(utils.Counter(single_dict['best_labels']))})"

    return [
        dataset,
        metric,
        weighted_str,
        single_str,
        kmeans_str,
        kmedoids_str,
        ub_dict["ub"],
        ub_dict["min"],
        ub_dict["max"],
    ]


def table(dataset_metric: list):
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

    for dataset, metric in dataset_metric:
        row = table_row(dataset=dataset, metric=metric)

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

    dataset_metric = [
        ("rna", "correlation"),
        ("religious_texts", "cosine"),
        ("conference_papers", "cosine"),
        ("religious_texts", "euclidean"),
        ("ceramic", "euclidean"),
        ("conference_papers", "euclidean"),
        ("rna", "euclidean"),
        ("religious_texts", "jaccard"),
        ("conference_papers", "jaccard"),
    ]

    table(dataset_metric=dataset_metric)
