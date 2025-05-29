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
You can find these in the [`examples/`](./examples) folder.