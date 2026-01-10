from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MultipleLocator


rows, cols = 2, 4
fig, axes = plt.subplots(rows, cols, figsize=(16, 10))
axes = axes.flatten()

datasets = [
    "ceramic",
    "customers",
    "dermatology",
    "heart_statlog",
    "optdigits",
    "rna",
    "wdbc",
    "wine",
]

for i, dataset in enumerate(datasets):
    results = pd.read_pickle(f"results/{dataset}.pkl")
    ax = axes[i]

    df = pd.DataFrame(
        {
            "K": results["n_clusters"],
            "ASW": results["asw"],
            "Macro Silhouette": results["macro_silhouette"],
            "ASW upper bound adjusted": results["ub_asw_min_cluster_size"],
            "Macro Silhouette upper bound ": results["ub_macro"],
        }
    )

    # Melt for seaborn
    df_melted = df.melt(
        id_vars="K",
        value_vars=[
            "ASW",
            "Macro Silhouette",
            "ASW upper bound adjusted",
            "Macro Silhouette upper bound ",
        ],
        var_name="Method",
        value_name="Value",
    )

    # Plot
    sns.lineplot(
        data=df_melted,
        x="K",
        y="Value",
        hue="Method",
        style="Method",
        markers=["o", "o", "X", "X"],
        dashes=["", "", (2, 2), (2, 2)],
        linewidth=1.5,
        ax=ax,
    )

    # Reference line
    ax.axhline(
        y=results["ub_asw"].values[0],
        color="black",
        linestyle="--",
        linewidth=1.5,
        label="ASW upper bound global",
    )

    # Axes formatting
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Number of clusters (K)", fontsize=13)
    ax.set_ylabel("")
    ax.set_title(dataset.replace("_", " ").title(), fontsize=15)
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, which="major", linestyle="--", linewidth=0.7, alpha=0.6)

plt.tight_layout()
plt.savefig("asw_vs_k_grid.pdf", bbox_inches="tight")
print("✅ Silhouette grid plot generated and saved as asw_vs_k_grid.pdf!")
plt.close()
