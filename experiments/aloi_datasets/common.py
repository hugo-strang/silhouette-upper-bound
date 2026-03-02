from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import (
    adjusted_rand_score,
    silhouette_samples,
    adjusted_mutual_info_score,
)
from silhouette_upper_bound import upper_bound
from collections import Counter


def get_df_sampled(dataset_name):
    """
    Help function to load dataframe.
    """

    base_path = Path.cwd()
    path = base_path / f"datasets/{dataset_name}.csv"

    # 1. Load whitespace-separated data
    df = pd.read_csv(
        path, sep=r"\s+", header=None, engine="python"  # whitespace  # no header row
    )  # needed because of regex separator

    # 2. add class label and drop instances with missing class
    df["class"] = df.iloc[:, -1].str.extract(r"(\d+)", expand=False)
    df = df[df["class"].notna()]
    df["class"] = df["class"].astype(int)

    # 3. sample n instances from each class
    df_sampled = (
        df.groupby("class", group_keys=False)
        .apply(lambda g: g.sample(n=100, random_state=42))
        .reset_index(drop=True)
    )

    return df_sampled


def eval(D, y, cluster_labels, ub):
    """
    Help function to print evaluation data.
    """

    # summary
    cluster_sizes = list(Counter(cluster_labels).values())
    min_cluster_size = min(cluster_sizes)
    print(f"Min cluster size = {min_cluster_size}")
    print(f"K = {len(cluster_sizes)}")

    # silhouette samples
    silh_samples = silhouette_samples(X=D, labels=cluster_labels, metric="precomputed")

    # ASW
    asw = np.mean(silh_samples)
    print(f"ASW = {asw}")
    print(f"ub = {ub}")
    print(f"WCRE = {(ub - asw)/ub}")

    # constrained
    uba = upper_bound(D, min_cluster_size)
    print(f"uba = {uba}\nwcre = {(uba - asw)/uba}")

    # external validation
    # AMI and ARI
    ari = adjusted_rand_score(cluster_labels, y)
    ami = adjusted_mutual_info_score(cluster_labels, y)

    print(f"Adjusted Rand Index vs. true labels: {ari:.3f}")
    print(f"Adjusted Mutual Info vs. true labels: {ami:.3f}")

    print(
        f"\n{len(cluster_sizes)}&{min_cluster_size}&{ari:.3f}&{ami:.3f}&{asw:.3f}&{ub:.3f}&{((ub - asw)/ub):.3f}&{uba:.3f}&{((uba - asw)/uba):.3f}"
    )
