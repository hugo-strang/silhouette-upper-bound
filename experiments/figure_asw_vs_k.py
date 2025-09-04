"""
This file generates a figure showing empirical ASW values for different number of clusters K for a synthetic dataset.
"""

import numpy as np
from silhouette_upper_bound import upper_bound_samples
from sklearn.datasets import make_blobs
import utils
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd
import seaborn as sns

logger = utils.get_logger(__name__)


def graph(params):

    # Parameters
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
    ref = np.mean(upper_bound_samples(D=D))

    ref_80 = np.mean(upper_bound_samples(D=D, kappa=80))

    # Storage
    k_list = []
    silh_list = []
    single_list = []
    weighted_list = []
    kmedoids_list = []

    for k in range(2, 21):

        # Kmeans
        kmeans_score = utils.asw_optimization(
            algorithm=utils.algorithm_kmeans,
            data=X,
            k_range=range(k, k + 1),
            asw_metric="euclidean",
            n_init=10,
        )["best_score"]

        # Single
        single_score = utils.asw_optimization(
            algorithm=utils.algorithm_hierarchical,
            data=D,
            k_range=range(k, k + 1),
            asw_metric="precomputed",
            method="single",
        )["best_score"]

        # Weighted
        weighted_score = utils.asw_optimization(
            algorithm=utils.algorithm_hierarchical,
            data=D,
            k_range=range(k, k + 1),
            asw_metric="precomputed",
            method="weighted",
        )["best_score"]

        # Kmedoids
        kmedoids_score = utils.asw_optimization(
            algorithm=utils.algorithm_kmedoids,
            data=D,
            k_range=range(k, k + 1),
            asw_metric="precomputed",
        )["best_score"]

        k_list.append(k)
        silh_list.append(kmeans_score)
        single_list.append(single_score)
        weighted_list.append(weighted_score)
        kmedoids_list.append(kmedoids_score)

    # Put data into a tidy DataFrame for seaborn
    df = pd.DataFrame(
        {
            "K": k_list,
            "KMeans Silhouette": silh_list,
            "PAMSIL Silhouette": kmedoids_list,
            "Weighted-Linkage Silhouette": weighted_list,
            "Single-Linkage Silhouette": single_list,
        }
    )

    # Melt the DataFrame for seaborn
    df_melted = df.melt(
        id_vars="K",
        value_vars=[
            "KMeans Silhouette",
            "PAMSIL Silhouette",
            "Weighted-Linkage Silhouette",
            "Single-Linkage Silhouette",
        ],
        var_name="Method",
        value_name=" ",
    )

    # Plot
    sns.set(style="whitegrid", context="talk")

    plt.figure(figsize=(10, 6))
    ax = sns.lineplot(
        data=df_melted,
        x="K",
        y=" ",
        hue="Method",
        style="Method",
        markers=True,
        dashes=True,
        linewidth=2.5,
    )

    # Reference lines
    logger.info(f"Upper bound = {ref}")
    logger.info(f"Upper bound (kappa 80) = {ref_80}")
    plt.axhline(
        y=ref, color="black", linestyle="--", linewidth=1.5, label=f"Upper bound"
    )
    plt.axhline(
        y=ref_80,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=rf"Upper bound ($\kappa$={80})",
    )

    # Adjust axes
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.ylim(0, 0.4)

    # Titles and labels
    plt.xlabel("Number of clusters (K)", fontsize=14)
    plt.legend(fontsize=12, title_fontsize=13, loc="best")

    plt.tight_layout()
    plt.savefig("koptim.pdf")


if __name__ == "__main__":

    caseparams = (400, 64, 5, 6)
    graph(params=caseparams)
