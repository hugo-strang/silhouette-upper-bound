# Silhouette Upper Bound
An upper bound of the [Average Silhouette Width](https://en.wikipedia.org/wiki/Silhouette_(clustering)).

## Overview
Evaluating clustering quality is a fundamental task in cluster analysis, and the
[Average Silhouette Width](https://en.wikipedia.org/wiki/Silhouette_(clustering)) (ASW) is one of the most widely used metrics for this purpose. ASW scores range from $-1$ to $1$, where:

* Values near 1 indicate well-separated, compact clusters

* Values around 0 suggest overlapping or ambiguous cluster assignments

* Values near -1 imply that many points may have been misassigned

Optimizing the Silhouette score is a common objective in clustering workflows. However, since we rarely know the true global ASW-maximum achievable for a dataset, it's difficult to assess how close a given clustering result is to being globally optimal. Simply comparing to the theoretical maximum of 1 is often misleading, as the structure of the dataset imposes inherent limits on what is achievable.

This project introduces a data-dependent upper bound on the ASW that hopefully can provide a more meaningful reference point than the fixed value of 1. The upper bound helps answer a key question: How close is my clustering result to the best possible outcome on this specific data?

To compute the upper bound, the method requires a dissimilarity matrix as input.

## Installation
```
pip install silhouette-upper-bound
```

## Examples

To help you get started, we provide example scripts demonstrating common use cases.
You can find these in the [`demos/`](./demos) folder.

## Quickstart
```python
import numpy as np
from silhouette_upper_bound import upper_bound

if __name__ == '__main__':

    np.random.seed(42)

    # dummy data
    A = np.random.rand(100, 100)
    D = (A + A.T) / 2
    np.fill_diagonal(D, 0)

    # ASW upper bound
    ub = upper_bound(D)

    print(f"There is no clustering of the data points of D that has a higher Silhouette score than {ub}.")
```

## Experimental results

We evaluate the performance of the upper bound using synthetic datasets generated with `scikit-learn`’s `make_blobs()` [function](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_blobs.html). Each dataset is identified by a label of the form `n_samples`-`n_features`-`centers`-`cluster_std`, which corresponds to the parameters used in the data generation.

The code that generates the results below can be found in 
[`experiments/table1.py`](./experiments/table1.py).

| Dataset | KMeans ASW | ASW upper bound | Worst-case relative error |
| --- | --- | --- | --- |
| 400-64-5-6 | 0.249 | 0.376 | 0.38 |
| 400-64-2-2 | 0.673 | 0.673 | .00 |
| 400-128-7-3 | 0.522 | 0.566 | 0.08 |
| 1000-161-2-13 | 0.084 | 0.182 | 0.54 |

Note that the upper bound confirms global optimality for KMeans on dataset 400-64-2-2.

## Contribution

Contributions are welcome! If you have suggestions for improvements, bug reports, or new features, feel free to open an issue or submit a pull request.

To contribute:

1. Fork the repository.
2. Create a new branch for your feature or fix.
3. Submit a pull request.

Thank you for helping improve this project!