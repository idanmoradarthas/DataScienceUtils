---
name: ds-utils-unsupervised
description: >
  Provides evaluation and visualization for unsupervised learning and clustering. Use when the user asks to plot cluster cardinality, wants to visualize cluster magnitude, compare magnitude versus cardinality to find anomalies, needs to determine the optimal number of clusters, or wants a 2D scatter plot of clustering results (PCA/t-SNE/UMAP, matplotlib or Plotly) in a Python data science project using sklearn-compatible models.
license: MIT
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
from ds_utils.unsupervised import plot_clusters
from ds_utils.unsupervised import plot_clusters_plotly
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
plot_loss_vs_cluster_number(
    X,
    k_min=3,
    k_max=20,
    distance_function=euclidean,
    algorithm_parameters={"random_state": 42},
)
plt.show()
```

**Parameters:**
- `X` — array-like, The data.
- `k_min` — int, **Required.** The minimum number of clusters to evaluate.
- `k_max` — int, **Required.** The maximum number of clusters to evaluate.
- `distance_function` — callable, **Required.** The distance function to use.
- `algorithm_parameters` — dict, optional. Additional keyword arguments
  passed to `KMeans()`. Use this to set `random_state`, `n_init`, etc.
  Example: `algorithm_parameters={"random_state": 42}`.
- `ax` — matplotlib Axes, optional. Target axes for the plot.

**Returns:** matplotlib Axes.

**Common mistakes:**
- The distance function must be a callable.
- `k_min` and `k_max` are **required** positional arguments — there are no
  defaults. Omitting them raises a `TypeError`.
- Only works with `sklearn.cluster.KMeans`. Do NOT pass a pre-fitted
  estimator or expect it to work with DBSCAN or hierarchical clustering.
- Pass `random_state` via `algorithm_parameters={"random_state": 42}`,
  not as a direct keyword argument.

---

## plot_clusters

Creates a 2D scatter plot of clustering results. For data with more than 2 features, PCA (default), t-SNE, or UMAP is applied automatically. Cluster centroids can optionally be overlaid.

```python
from ds_utils.unsupervised import plot_clusters
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt

estimator = KMeans(n_clusters=8, random_state=42)
estimator.fit(X)

plot_clusters(X, estimator.labels_, estimator.cluster_centers_, reduction_method="pca", random_state=42)
plt.show()
```

**Parameters:**
- `X` — array-like of shape (n_samples, n_features). Numeric 2D array with at least 2 features.
- `cluster_labels` — array-like of shape (n_samples,). Cluster labels.
- `cluster_centers` — array-like of shape (n_clusters, n_features), optional. Centroids to overlay.
- `reduction_method` — `'pca'` (default), `'tsne'`, or `'umap'`. Used only when `n_features > 2`.
- `random_state` — int, optional. Passed to the dimensionality reduction algorithm.
- `ax` — matplotlib Axes, optional.

**Returns:** matplotlib Axes.

**Common mistakes:**
- `X` must be numeric. Use `df.select_dtypes(include="number").values` if needed.
- `reduction_method="umap"` requires `umap-learn` (`pip install data-science-utils[umap]`).
- t-SNE cannot transform new points, so centroids are skipped (with a warning) when `reduction_method="tsne"`.

---

## plot_clusters_plotly

Creates an interactive Plotly scatter plot of clustering results. Same reduction options as `plot_clusters`.

```python
from ds_utils.unsupervised import plot_clusters_plotly

fig = plot_clusters_plotly(X, estimator.labels_, estimator.cluster_centers_)
fig.show()
```

**Parameters:**
- Same as `plot_clusters`, except `fig` (plotly Figure, optional) replaces `ax`.
- `show_legend` — bool, default=True.

**Returns:** `plotly.graph_objects.Figure` (not a matplotlib Axes).

**Common mistakes:**
- Call `.show()` on the returned figure.
- Do not treat the return value as a matplotlib Axes.

---

## Typical Workflow

```python
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial.distance import euclidean
from ds_utils.unsupervised import plot_loss_vs_cluster_number, plot_magnitude_vs_cardinality, plot_clusters

# Find optimal k
plot_loss_vs_cluster_number(
    X,
    k_min=3,
    k_max=20,
    distance_function=euclidean,
    algorithm_parameters={"random_state": 42},
)
plt.show()

# Run clustering
kmeans = KMeans(n_clusters=8, random_state=42)
kmeans.fit(X)

# Analyze clusters
plot_magnitude_vs_cardinality(X, kmeans.labels_, kmeans.cluster_centers_, euclidean)
plt.show()

# Visualize clusters
plot_clusters(X, kmeans.labels_, kmeans.cluster_centers_, random_state=42)
plt.show()
```
