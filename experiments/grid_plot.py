import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_samples, silhouette_score
from scipy.io import arff
from scipy.spatial.distance import pdist, squareform
from silhouette_upper_bound import upper_bound
from collections import Counter


# -------------------------------------------------
# 2. Load datasets
# -------------------------------------------------
def load_arff_as_distance_matrix(path, metric="euclidean", scale=False):
    # Load
    data, meta = arff.loadarff(path)
    df = pd.DataFrame(data)
    fname = os.path.basename(path).lower()


    if "wdbc" in fname:
        # First column is ID, second is label
        y = df.iloc[:, 1].astype(str).to_numpy()
        X = df.iloc[:, 2:].to_numpy()  

    elif "wine" in fname:
        # Frst column is label
        y = df.iloc[:, 0].astype(str).to_numpy()
        X = df.iloc[:, 1:].to_numpy() 

    elif "yeast" in fname:
        # Frst column is ID 
        y = df.iloc[:, -1].astype(str).to_numpy()
        X = df.iloc[:, 1:].to_numpy() 

    elif "mopsi-joensuu" in fname:
        # Frst column is ID 
        y = np.zeros(df.shape[0])
        X = df.iloc[:, :].to_numpy() 

    else:
        # Last column is label
        # arrhythmia, balance-scale, cpu, dermatology, ecoli, glass, haberman, heart-statlog, iono, iris, letter, segment, sonar, tae, thy, vehicle, wisc, zoo
        y = df.iloc[:, -1].astype(str).to_numpy()
        X = df.iloc[:, :-1].to_numpy()
    
    # Replace missing values ("?")
    X = np.where(X == b'?', np.nan, X).astype(float)
    col_means = np.nanmean(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_means, inds[1])

    if scale:
        X = StandardScaler().fit_transform(X)

    # Distance matrix
    D = squareform(pdist(X, metric=metric))

    return D, X, y


# -------------------------------------------------
# 3. Collect datasets
# -------------------------------------------------
# Example: assuming you’ve cloned the GitHub repo locally
# e.g. `git clone https://github.com/deric/clustering-benchmark.git`
dataset_dir = "data/clustering_benchmarks/real_world"

datasets = []
for fname in os.listdir(dataset_dir):
    if fname.endswith(".arff"):
        try:
            path = os.path.join(dataset_dir, fname)
            D, X, y = load_arff_as_distance_matrix(path, scale=True)
            n_samples, n_features = X.shape
            class_counts = Counter(y)
            kappa_r = int(min(class_counts.values()))
            ub_asw_r = upper_bound(D=D, kappa=kappa_r)
            kappa = 1
            ub_asw = upper_bound(D=D, kappa=kappa)
            datasets.append({
                "name": fname.replace(".arff", "")+f" {n_samples, n_features}",
                "X_plot": X,
                "y": y,
                "kappa": kappa,
                "ub_asw": ub_asw,
                "kappa_r": kappa_r,
                "ub_asw_r": ub_asw_r
            })
        except:
            continue


datasets = datasets[:20]  # pick a subset

print("Datasets processed!")

# -------------------------------------------------
# 4. Plot grid with silhouette plots
# -------------------------------------------------
n = len(datasets)
rows, cols = 3, 4
fig, axes = plt.subplots(rows, cols, figsize=(20, 20))
axes = axes.flatten()

for i, dataset in enumerate(datasets):
    X_plot, y, ub_asw, kappa, ub_asw_r, kappa_r = dataset["X_plot"], dataset["y"], dataset["ub_asw"], dataset["kappa"], dataset["ub_asw_r"], dataset["kappa_r"]
    ax = axes[i]

    # Compute silhouette values per sample
    sil_values = silhouette_samples(X_plot, y, metric="euclidean")
    cluster_labels = np.unique(y)
    y_lower = 10

    for j, cluster in enumerate(cluster_labels):
        cluster_sil = sil_values[y == cluster]
        cluster_sil.sort()
        size = len(cluster_sil)
        y_upper = y_lower + size

        # use index j to pick color, not the raw cluster label
        color = plt.cm.tab10(j % 10)
        ax.fill_betweenx(
            np.arange(y_lower, y_upper),
            0, cluster_sil,
            facecolor=color, edgecolor=color, alpha=0.7
        )
        ax.text(-0.05, y_lower + 0.5*size, str(cluster))
        y_lower = y_upper + 10  # gap between clusters

    ax.axvline(np.mean(sil_values), color="red", linestyle="--", label = "ASW")
    ax.axvline(ub_asw, color="blue", linestyle=":", label = rf"upper bound ($\kappa$={kappa})")  # show UB-ASW reference
    ax.axvline(ub_asw_r, color="black", linestyle="--", label = rf"upper bound ($\kappa$={kappa_r})")  # show UB-ASW reference
    ax.set_title(f"{dataset['name']}")
    ax.set_xlim([-0.1, ub_asw + 0.05])
    ax.set_yticks([])

    ax.legend(fontsize=8, loc="upper right")

# Adjust spacing
plt.subplots_adjust(wspace=0.4, hspace=0.4)

# Save to PDF
plt.savefig("silhouette_grid.pdf", bbox_inches="tight")
