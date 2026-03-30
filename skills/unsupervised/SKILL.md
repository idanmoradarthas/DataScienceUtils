---
name: ds-utils-unsupervised
description: >
  Provides evaluation and visualization for unsupervised learning and clustering. Use when the user asks to plot cluster cardinality, wants to visualize cluster magnitude, compare magnitude versus cardinality to find anomalies, or needs to determine the optimal number of clusters in a Python data science project using sklearn-compatible models.
metadata:
  author: Idan Morad
  documentation: https://datascienceutils.readthedocs.io/en/stable/
  package: data-science-utils
  repository: https://github.com/idanmoradarthas/DataScienceUtils
---
# Unsupervised — ds_utils.unsupervised

Tools for analyzing and visualizing unsupervised clustering models.

## Installation

```bash
pip install data-science-utils
# or
conda install -c idanmorad data-science-utils
```

## Import

```python
from ds_utils.unsupervised import plot_cluster_cardinality
from ds_utils.unsupervised import plot_cluster_magnitude
from ds_utils.unsupervised import plot_magnitude_vs_cardinality
from ds_utils.unsupervised import plot_loss_vs_cluster_number
```

---

## plot_cluster_cardinality

Plots the number of points (cardinality) in each cluster as a bar chart.

```python
from ds_utils.unsupervised import plot_cluster_cardinality
import matplotlib.pyplot as plt

# complete usage example
plot_cluster_cardinality(kmeans.labels_)
plt.show()
```

**Parameters:**
- `labels` — array-like, Cluster labels assigned by the clustering algorithm.

**Returns:** matplotlib Axes.

**Common mistakes:**
- Requires the fitted `labels_` attribute from the estimator, not the dataset itself.

---

## plot_cluster_magnitude

Plots the Total Point-to-Centroid Distance per cluster.

```python
from ds_utils.unsupervised import plot_cluster_magnitude
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt

# complete usage example
plot_cluster_magnitude(X, kmeans.labels_, kmeans.cluster_centers_, euclidean)
plt.show()
```

**Parameters:**
- `X` — array-like, The dataset.
- `labels` — array-like, Assigned cluster labels.
- `cluster_centers` — array-like, Cluster center coordinates.
- `distance_function` — callable, The distance function.
- `ax` — Axes, optional matplotlib axes.

**Returns:** matplotlib Axes.

**Common mistakes:**
- The distance function argument MUST be a **callable** (e.g. `euclidean` from `scipy.spatial.distance`), never a string `"euclidean"`.

---

## plot_magnitude_vs_cardinality

Detects anomalies by plotting cluster magnitude against its cardinality.

```python
from ds_utils.unsupervised import plot_magnitude_vs_cardinality
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt

# complete usage example
plot_magnitude_vs_cardinality(X, kmeans.labels_, kmeans.cluster_centers_, euclidean)
plt.show()
```

**Parameters:**
- `X` — array-like, The dataset.
- `labels` — array-like, Assigned cluster labels.
- `cluster_centers` — array-like, Cluster center coordinates.
- `distance_function` — callable, The distance function.
- `ax` — Axes, optional matplotlib axes.

**Returns:** matplotlib Axes.

**Common mistakes:**
- The distance function must be a callable.
- `plot_magnitude_vs_cardinality` detects anomalous clusters where cardinality does not correlate with magnitude.

---

## plot_loss_vs_cluster_number

Helps find the optimum number of clusters by iterating K-Means and plotting the sum of distances loss.

```python
from ds_utils.unsupervised import plot_loss_vs_cluster_number
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt

# complete usage example
plot_loss_vs_cluster_number(X, 3, 20, euclidean)
plt.show()
```

**Parameters:**
- `X` — array-like, The data.
- `min_k` — int, The minimum number of clusters. (Default: 3)
- `max_k` — int, The maximum number of clusters. (Default: 20)
- `distance_function` — callable, The distance function to use.
- `ax` — Axes, optional matplotlib axes.
- `random_state` — int, random state for K-Means.

**Returns:** matplotlib Axes.

**Common mistakes:**
- The distance function must be a callable.
- `plot_loss_vs_cluster_number` only works with `sklearn.cluster.KMeans`. Do NOT pass a pre-fitted estimator or expect it to work natively with DBSCAN/Hierarchy.

---

## Typical Workflow

```python
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial.distance import euclidean
from ds_utils.unsupervised import plot_loss_vs_cluster_number, plot_magnitude_vs_cardinality

# Find optimal k
plot_loss_vs_cluster_number(X, 3, 20, euclidean)
plt.show()

# Run clustering
kmeans = KMeans(n_clusters=8, random_state=42)
kmeans.fit(X)

# Analyze clusters
plot_magnitude_vs_cardinality(X, kmeans.labels_, kmeans.cluster_centers_, euclidean)
plt.show()
```
