"""
In this script, we generate a table containing results on syntehtic datasets.
"""

from silhouette_upper_bound import upper_bound
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score, pairwise_distances

def print_markdown_table(rows, headers= ["Dataset", "KMeans ASW", "ASW upper bound", "Diff."]):
    # Format header
    header_line = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join(["---"] * len(headers)) + " |"

    print(header_line)
    print(separator)

    # Format rows
    for row in rows:
        print("| " + " | ".join(str(cell) for cell in row) + " |")

def silhoutte_and_ub(params):

    n_samples, n_features, centers, cluster_std = params
    # Generate synthetic data
    X, _ = make_blobs(n_samples=n_samples, n_features=n_features, centers=centers, cluster_std=cluster_std, random_state=0)
    D = pairwise_distances(X)

    # Cluster with KMeans
    kmeans = KMeans(n_clusters=centers, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)

    # Compute score and upper bound
    score = silhouette_score(X, labels)
    ub = upper_bound(D)
    diff = ub - score 

    return round(score, 3), round(ub, 3), round(diff, 3)

def run_cases(cases):

    rows = []
    for caseparams in cases:
        row = []
        row.append("-".join(str(x) for x in caseparams))

        score, ub, diff = silhoutte_and_ub(caseparams)

        row += [f"{score}", f"{ub}", f"{diff}"]
        
        rows.append(row)
    
    print_markdown_table(rows=rows)



if __name__ == "__main__":
    
    case1params = (400, 64, 5, 6)
    case2params = (400, 64, 2, 2)
    case3params = (400, 128, 7, 3)
    case4params = (1000, 161, 2, 13)

    run_cases([case1params, case2params, case3params, case4params])