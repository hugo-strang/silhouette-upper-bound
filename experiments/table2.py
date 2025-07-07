import utils


def table_row(dataset: str, metric: str):

    print(f"\nDistance metric: {metric}")

    if dataset == "conference_papers":
        data = utils.get_data(dataset=dataset, transpose=True)
    else:
        data = utils.get_data(dataset=dataset)

    n = data.shape[0]

    ub_dict = utils.get_upper_bound(data=data, metric=metric)

    # Weighted
    if n > 1000:
        weighted_dict = utils.hierarchical_optimized(
            data=data, metric=metric, method="weighted", t_range=range(2, 51)
        )
    else:
        weighted_dict = utils.hierarchical_optimized(
            data=data, metric=metric, method="weighted", t_range=range(2, n)
        )

    # Single
    if n > 1000:
        single_dict = utils.hierarchical_optimized(
            data=data, metric=metric, method="single", t_range=range(2, 51)
        )
    else:
        single_dict = utils.hierarchical_optimized(
            data=data, metric=metric, method="single", t_range=range(2, n)
        )

    # Kmeans
    if metric == "euclidean":
        kmeans_dict = utils.kmeans_optimized(data=data, k_range=range(2, 51))
        kmeans_str = f"${kmeans_dict['best_score']:.3f}$ ({len(utils.Counter(kmeans_dict['best_labels']))})"
    else:
        kmeans_dict = {"best_score": "N/A"}
        kmeans_str = "N/A"

    weighted_str = f"${weighted_dict['best_score']:.3f}$ ({len(utils.Counter(weighted_dict['best_labels']))})"
    single_str = f"${single_dict['best_score']:.3f}$ ({len(utils.Counter(single_dict['best_labels']))})"

    return [
        dataset,
        metric,
        weighted_str,
        single_str,
        kmeans_str,
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
