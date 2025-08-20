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
            data=data, metric=metric, method="weighted", t_range=range(2, 3)
        )
    else:
        weighted_dict = utils.hierarchical_optimized(
            data=data, metric=metric, method="weighted", t_range=range(2, 3)
        )

    # Single
    if n > 1000:
        single_dict = utils.hierarchical_optimized(
            data=data, metric=metric, method="single", t_range=range(2, 3)
        )
    else:
        single_dict = utils.hierarchical_optimized(
            data=data, metric=metric, method="single", t_range=range(2, 3)
        )

    # Kmeans
    # if metric == "euclidean":
    #     kmeans_dict = utils.kmeans_optimized(data=data, k_range=range(2, 51))
    #     kmeans_str = f"${kmeans_dict['best_score']:.3f}$ ({len(utils.Counter(kmeans_dict['best_labels']))})"
    # else:
    #     kmeans_dict = {"best_score": "N/A"}
    #     kmeans_str = "N/A"
    kmeans_dict = utils.kmedoids_optimized(data=data, metric=metric, k_range=range(2, 51))
    kmeans_str = f"${kmeans_dict['best_score']:.3f}$ ({len(utils.Counter(kmeans_dict['best_labels']))})"

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
        ("rna", "correlation"), # best score: 0.39478298909442816 | n clusters: 7
        ("religious_texts", "cosine"), # best score: 0.08652382921487772 | n clusters: 23
        ("conference_papers", "cosine"), # best score: 0.1262421716484819 | n clusters: 2
        ("religious_texts", "euclidean"), # best score: 0.8461517181892778 | n clusters: 2
        ("ceramic", "euclidean"), # best score: 0.5840130686182088 | n clusters: 2
        ("conference_papers", "euclidean"), # best score: 0.3837160375878645 | n clusters: 2
        ("rna", "euclidean"), # best score: 0.22994911168195492 | n clusters: 8
        ("religious_texts", "jaccard"), # best score: 0.024181961360913204 | n clusters: 24
        ("conference_papers", "jaccard"), # best score: 0.15994838028119676 | n clusters: 2
    ]

    table(dataset_metric=dataset_metric)
