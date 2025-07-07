import numpy as np
from silhouette_upper_bound import upper_bound_samples
from sklearn.datasets import make_blobs
from sklearn.metrics import pairwise_distances
from utils import kmeans_optimized, hierarchical_optimized
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd
import seaborn as sns


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
    D = pairwise_distances(X)
    ref = np.mean(upper_bound_samples(D=D))

    ref_80 = np.mean(upper_bound_samples(D=D, kappa=80))

    # Storage
    k_list = []
    silh_list = []
    single_list = []
    weighted_list = []

    for k in range(2, 21):

        # Kmeans
        kmeans_scores = kmeans_optimized(data=X, k_range=range(k, k + 1), n_init=10)[
            "best_scores"
        ]

        # Single
        single_scores = hierarchical_optimized(
            data=X, metric="euclidean", method="single", t_range=range(k, k + 1)
        )["best_scores"]

        # Weighted
        weighted_scores = hierarchical_optimized(
            data=X, metric="euclidean", method="weighted", t_range=range(k, k + 1)
        )["best_scores"]

        k_list.append(k)
        silh_list.append(np.mean(kmeans_scores))
        single_list.append(np.mean(single_scores))
        weighted_list.append(np.mean(weighted_scores))

    # Put data into a tidy DataFrame for seaborn
    df = pd.DataFrame(
        {
            "K": k_list,
            "KMeans Silhouette": silh_list,
            "Weighted-Linkage Silhouette": weighted_list,
            "Single-Linkage Silhouette": single_list,
        }
    )

    # Melt the DataFrame for seaborn
    df_melted = df.melt(
        id_vars="K",
        value_vars=[
            "KMeans Silhouette",
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
        dashes=False,
        linewidth=2.5,
    )

    # Reference lines
    print(f"Upper bound = {ref}")
    print(f"Upper bound (kappa 80) = {ref_80}")
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
