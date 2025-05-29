import numpy as np
import pytest
from silhouette_upper_bound import upper_bound, upper_bound_samples
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score, silhouette_samples, pairwise_distances

def test_basic():

    d1 = np.array([[0, 1, 1, 1, 1], 
                   [1, 0, 1, 1, 1], 
                   [1, 1, 0, 1, 1], 
                   [1, 1, 1, 0, 1], 
                   [1, 1, 1, 1, 0]])

    assert upper_bound(d1) == 0.0 

    d2 = np.array([[0, 1, 5, 5, 5],  
                   [1, 0, 5, 5, 5],  
                   [5, 5, 0, 1, 1],  
                   [5, 5, 1, 0, 1],  
                   [5, 5, 1, 1, 0]]) 

    assert upper_bound(d2) == 1 - 1 / 5

    d3 = np.array([[0, 1, 6, 6.5, 7, 8],  
                   [1, 0, 5, 7.5, 6.88, 8],  
                   [6, 5, 0, 0.5, 0.78, 1.1],  
                   [6.5, 7.5, 0.5, 0, 0.5, 1],  
                   [7, 6.88, 0.78, 0.5, 0, 1],
                   [8, 8, 1.1, 1, 1, 0]]) 
    
    d3f = np.array([4 * 1 / (6 + 6.5 + 7 + 8), 
                    4 * 1 / (5 + 7.5 + 6.88 + 8), 
                    (2 / 3) * (0.5 + 0.78 + 1.1) / (6 + 5), 
                    (2 / 3) * (0.5 + 0.5 + 1) / (6.5 + 7.5), 
                    (2 / 3) * (0.78 + 0.5 + 1) / (7 + 6.88),
                    (2 / 3) * (1.1 + 1 + 1) / (8 + 8)])
    
    assert np.abs(upper_bound(d3) - np.mean(1 - d3f)) < 1e-15


@pytest.mark.parametrize("n_samples", [100, 200, 300, 400, 500])
@pytest.mark.parametrize("n_features", [10, 15, 20])
@pytest.mark.parametrize("centers", [3, 6, 9])
@pytest.mark.parametrize("cluster_std", [1.0, 2.0, 3.0])
def test_blobs(n_samples, n_features, centers, cluster_std):
    X, _ = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=centers,
        cluster_std=cluster_std,
        random_state=42,
    )

    D = pairwise_distances(X)

    # KMeans clustering
    model = KMeans(n_clusters=centers, random_state=42, n_init="auto")
    labels = model.fit_predict(X)
    score = silhouette_score(X, labels)

    # Test standard upper bound
    ub = upper_bound(D)

    assert 0 <= ub and ub <= 1
    assert ub - score >= - 1e-15

    # Test each sample 
    score_samples = silhouette_samples(X, labels)
    ub_samples = upper_bound_samples(D)
    assert np.all(0 <= ub_samples) and np.all(ub_samples <= 1)
    assert np.all(ub_samples - score_samples >= - 1e-15)

    # Upper bound with kappa > 2
    min_cluster_size = np.bincount(labels).min() 

    if min_cluster_size > 2:

        ub_kappa = upper_bound(D, kappa=min_cluster_size)

        assert 0 <= ub_kappa and ub_kappa <= 1
        assert ub_kappa - score >= - 1e-15
    
        # Test each sample 
        ub_kappa_samples = upper_bound_samples(D, kappa=min_cluster_size)
        assert np.all(0 <= ub_kappa_samples) and np.all(ub_kappa_samples <= 1)
        assert np.all(ub_kappa_samples - score_samples >= - 1e-15)

