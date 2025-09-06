"""
This file generates figures that show individual silhouette widths compared to their corresponding upper bounds for labeled datasets.

Notes:
    - The labeled datasets are available at https://github.com/deric/clustering-benchmark/tree/master/src/main/resources/datasets/real-world
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from utils import (
    load_arff_as_distance_matrix,
    asw_optimization,
    algorithm_kmedoids,
    get_silhouette_plot_data,
)
from silhouette_upper_bound import upper_bound, upper_bound_samples
from collections import Counter


# -------------------------------------------------
# 1. Collect datasets
# -------------------------------------------------
dataset_dir = "data/labeled/real_world"

datasets = []
for fname in os.listdir(dataset_dir):
    if fname.endswith(".arff"):
        try:
            path = os.path.join(dataset_dir, fname)
            D, X, y = load_arff_as_distance_matrix(path, scale=True)
            n_samples, n_features = X.shape
            class_counts = Counter(y)
            ub_asw = upper_bound(D=D)

            ub_samples_dict = {}

            for kappa_s in range(
                1, int(n_samples * 0.10) + 1, int(n_samples * 0.10) // 10
            ):
                ub_samples_dict[kappa_s] = upper_bound_samples(D=D, kappa=kappa_s)

            datasets.append(
                {
                    "name": fname.replace(".arff", "") + f" {n_samples, n_features}",
                    "X_plot": X,
                    "D": D,
                    "y": y,
                    "ub_asw": ub_asw,
                    "ub_samples_dict": ub_samples_dict,
                }
            )
        except:
            continue


datasets = datasets[:20]  # pick a subset

print("Datasets processed!")

# -------------------------------------------------
# 2. Plot grid with silhouette plots
# -------------------------------------------------

k_ground_truth = True  # change to False to optimise over a wider range of K

n = len(datasets)
rows, cols = 3, 4
fig, axes = plt.subplots(rows, cols, figsize=(20, 15))
axes = axes.flatten()

for i, dataset in enumerate(datasets):
    X_plot, D, y, ub_asw = (
        dataset["X_plot"],
        dataset["D"],
        dataset["y"],
        dataset["ub_asw"],
    )
    ax = axes[i]

    # Compute silhouette values per sample
    ub_samples = upper_bound_samples(D)

    # Ground truth stats
    n_clusters = len(Counter(y))

    # Decide which K to consider
    if k_ground_truth:
        k_range = range(n_clusters, n_clusters + 1)
    else:
        k_range = range(2, 13)

    # Generate clustering through kmedoids
    _kmedoids_optimized = asw_optimization(
        algorithm=algorithm_kmedoids, data=D, k_range=k_range, asw_metric="precomputed"
    )
    kmedoids_scores = _kmedoids_optimized["best_scores"]
    kmedoids_labels = _kmedoids_optimized["best_labels"]

    print(f"\n-----------------------------------")
    print(f"{i} Optim ASW: {np.mean(kmedoids_scores)}")
    print(f"{i} UB: {np.mean(ub_samples)}")
    y_lower = 10

    y = kmedoids_labels

    data = get_silhouette_plot_data(
        kmedoids_labels, kmedoids_scores, n_clusters, ub_samples
    )

    for x in data.keys():

        # Cluster Silhouette scores
        ax.fill_betweenx(
            np.arange(data[x]["y_lower"], data[x]["y_upper"]),
            0,
            data[x]["sorted_silhouettes"],
            facecolor=data[x]["color"],
            edgecolor="black",
            alpha=0.8,
        )

        # Cluster Silhouette bounds
        ax.fill_betweenx(
            np.arange(data[x]["y_lower"], data[x]["y_upper"]),
            0,
            data[x]["sorted_ub_values"],
            facecolor=data[x]["color"],
            edgecolor=data[x]["color"],
            alpha=0.5,
        )

        # Label cluster number
        ax.text(-0.05, data[x]["y_lower"] + 0.5 * data[x]["size_cluster_i"], str(x))

    ax.axvline(
        ub_asw, color="black", linestyle="--", label=rf"upper bound ($\kappa$={1})"
    )  # show UB-ASW reference
    ax.axvline(
        np.mean(kmedoids_scores), color="black", linestyle="-", label=rf"ASW"
    )  # show UB-ASW reference
    ax.set_title(f"{dataset['name']}")
    ax.set_xlim([-0.1, 1.1])
    ax.set_yticks([])

    ax.legend(fontsize=8, loc="upper right")

# Save to PDF
if k_ground_truth:
    plt.savefig("silhouette_grid_gt.pdf", bbox_inches="tight")
else:
    plt.savefig("silhouette_grid.pdf", bbox_inches="tight")
print("Silhouette grid plot generated!")
plt.close()

# -------------------------------------------------
# 3. Plot grid with violin plots
# -------------------------------------------------
n = len(datasets)
rows, cols = 3, 4
fig, axes = plt.subplots(
    rows, cols, figsize=(20, 15), sharey=True
)  # share y-axis for comparability
axes = axes.flatten()

for i, dataset in enumerate(datasets):
    ub_samples_dict = dataset["ub_samples_dict"]
    ax = axes[i]

    # Sort kappas for consistency
    kappas = sorted(ub_samples_dict.keys())
    data = [ub_samples_dict[kappa] for kappa in kappas]

    vp = ax.violinplot(data, positions=range(len(kappas)), showmeans=True)

    # --- Prettify violins ---
    for body in vp["bodies"]:
        body.set_facecolor("#87CEFA")  # soft blue
        body.set_edgecolor("black")
        body.set_alpha(0.7)

    # --- Style the mean lines (default black → red) ---
    if "cmeans" in vp:
        vp["cmeans"].set_color("red")
        vp["cmeans"].set_linewidth(2)

    ax.set_xticks(range(len(kappas)))
    ax.set_xticklabels(kappas, rotation=30, fontsize=12)

    ax.set_ylim(0, 1.05)
    ax.set_title(f"{dataset['name']}", fontsize=12, pad=8)

    if i % cols == 0:
        ax.set_ylabel("upper bounds", fontsize=12)
    ax.set_xlabel(r"$\kappa$", fontsize=12)

    ax.grid(axis="y", linestyle="--", alpha=0.6)


plt.tight_layout()
plt.savefig("violin_grid.pdf", bbox_inches="tight")
print("Violin grid plot generated!")
