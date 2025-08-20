import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.io import arff
from scipy.spatial.distance import pdist, squareform
from silhouette_upper_bound import upper_bound
from collections import Counter


# -------------------------------------------------
# 2. Load datasets
# -------------------------------------------------
def load_arff_as_distance_matrix(path, metric="euclidean"):
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

    # Distance matrix
    D = squareform(pdist(X, metric=metric))

    return D, X, y


def preprocess_for_plot(X):
    """Standardize and reduce to 2D (for visualization only)"""
    X_scaled = StandardScaler().fit_transform(X)
    if X_scaled.shape[1] > 2:
        X_plot = PCA(n_components=2).fit_transform(X_scaled)
    else:
        X_plot = X_scaled
    return X_plot


# -------------------------------------------------
# 3. Collect datasets
# -------------------------------------------------
# Example: assuming you’ve cloned the GitHub repo locally
# e.g. `git clone https://github.com/deric/clustering-benchmark.git`
dataset_dir = "data/clustering_benchmarks/use"

datasets = []
for fname in os.listdir(dataset_dir):
    if fname.endswith(".arff"):
        try:
            path = os.path.join(dataset_dir, fname)
            D, X, y = load_arff_as_distance_matrix(path)
            n_samples, n_features = X.shape
            class_counts = Counter(y)
            kappa = int(min(class_counts.values()))
            #kappa = 1
            ub_asw = upper_bound(D=D, kappa=kappa)
            X_plot = preprocess_for_plot(X)
            datasets.append({
                "name": fname.replace(".arff", "")+f" ({n_samples, n_features})\n kappa = {kappa}",
                "X_plot": X_plot,
                "y": y,
                "ub_asw": ub_asw
            })
        except:
            continue



datasets = datasets[:20]  # pick a subset

print("Datasets processed!")

# -------------------------------------------------
# 4. Plot grid
# -------------------------------------------------
n = len(datasets)
rows, cols = 5, 4
fig, axes = plt.subplots(rows, cols, figsize=(20, 20))
axes = axes.flatten()

for i, dataset in enumerate(datasets):
    X_plot, y, ub_asw = dataset["X_plot"], dataset["y"], dataset["ub_asw"]
    unique_labels, numeric_labels = np.unique(y, return_inverse=True)
    ax = axes[i]
    sc = ax.scatter(X_plot[:,0], X_plot[:,1], c=numeric_labels, s=10, cmap="tab10")
    ax.set_title(f"{dataset['name']} (UB-ASW={ub_asw:.2f})")
    ax.set_xticks([])
    ax.set_yticks([])

# Hide unused axes if < 20 datasets
for j in range(i+1, len(axes)):
    axes[j].axis("off")

plt.tight_layout()
plt.show()
