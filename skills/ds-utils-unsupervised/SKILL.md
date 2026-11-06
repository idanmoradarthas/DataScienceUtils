---
name: ds-utils-unsupervised
description: >
  Unsupervised clustering analysis, optimal cluster count selection, and 2D scatter plots (PCA, t-SNE, UMAP). Use for elbow loss curves, cluster cardinality, centroid magnitude, and cluster assignments. Do NOT use for supervised classification metrics or text binarization.
version: 1.0.0
license: MIT
metadata:
  author: "Idan Morad <idanmorad.arthas@gmail.com>"
  tags:
    - clustering
    - unsupervised-learning
    - kmeans
    - cluster-cardinality
    - elbow-method
    - cluster-visualization
    - pca
    - tsne
    - umap
  documentation: "https://datascienceutils.readthedocs.io/en/stable/"
  package: "data-science-utils"
  repository: "https://github.com/idanmoradarthas/DataScienceUtils"
---

# ds-utils-unsupervised

Tools for choosing cluster count, diagnosing clustering quality using cardinality and magnitude analysis, and visualizing cluster assignments in 2D scatter plots.

## Purpose

ds_utils.unsupervised provides diagnostics for centroid-based clustering (specifically scikit-learn's `KMeans`) and 2D scatter visualization of clustering results. It helps data scientists determine the optimal number of clusters $k$ via custom distance loss sweeps, evaluate cluster compactness and stability using cardinality (number of points) and magnitude (total distance to centroid), and visualize cluster assignments via PCA, t-SNE, or UMAP reduction in matplotlib or Plotly.

- **When to use**: Selecting $k$ with the elbow method, verifying that clusters are balanced and compact, spotting anomalous or diffuse clusters, and creating 2D scatter plots of clustering results.
- **When NOT to use**: For supervised classification/regression metrics (use `ds-utils-metrics`), PCA explained variance (use `ds-utils-preprocess`), or using the 2D reduction coordinates as clustering input (they are for display only).

## Prerequisites

- Python 3.11 or higher.
- `data-science-utils` package installed (`pip install data-science-utils`).
- Core dependencies: `numpy`, `pandas`, `scikit-learn`, `scipy`, `matplotlib`, `plotly`.
- Optional: `umap-learn` for UMAP reduction (`pip install data-science-utils[umap]`).

## Instructions

### Task-to-Approach Map

| Clustering Diagnostic Task | Function | Required Inputs | Output |
|---|---|---|---|
| Determine optimal cluster count ($k$) | `plot_loss_vs_cluster_number` | `X`, `k_min`, `k_max`, `distance_function` | Matplotlib Axes |
| Check balance / points per cluster | `plot_cluster_cardinality` | `labels_` | Matplotlib Axes |
| Measure compactness per cluster | `plot_cluster_magnitude` | `X`, `labels_`, `cluster_centers_`, `distance_func` | Matplotlib Axes |
| Identify anomalous or diffuse clusters | `plot_magnitude_vs_cardinality` | `X`, `labels_`, `cluster_centers_`, `distance_func` | Matplotlib Axes |
| Visualize clusters in 2D (matplotlib) | `plot_clusters` | `X`, `cluster_labels`, optional `cluster_centers` | Matplotlib Axes |
| Visualize clusters in 2D (Plotly) | `plot_clusters_plotly` | `X`, `cluster_labels`, optional `cluster_centers` | Plotly Figure |

Full signatures, parameter options, and common mistakes are documented in [references/api_reference.md](references/api_reference.md).

---

### End-to-End Clustering Workflow

Follow this sequence to go from unscaled features to a validated clustering model:

```text
               Unlabeled Data Matrix
                         │
           Scale Features (e.g. StandardScaler)
                         │
                         ▼
        Sweep k and Identify Elbow Point
          (plot_loss_vs_cluster_number)
                         │
                         ▼
             Fit Final KMeans(n_clusters=k)
                         │
         ┌───────────────┴───────────────┐
         ▼                               ▼
  Check Balance                   Check Compactness
(plot_cluster_cardinality)      (plot_cluster_magnitude)
         │                               │
         └───────────────┬───────────────┘
                         ▼
           Diagnose Cluster Consistency
          (plot_magnitude_vs_cardinality)
                         |
                         v
           Visualize Cluster Assignments
      (plot_clusters / plot_clusters_plotly)
```

#### Step 1: Sweep $k$ and Select Elbow Point (`plot_loss_vs_cluster_number`)
Test a range of cluster counts to observe the diminishing return in distance loss.
- **Input requirement**: `k_min` and `k_max` are required integers without defaults.
- **Distance function**: Must be a **callable** (such as `euclidean` from scipy.spatial.distance), never a string. Pass algorithm parameters (e.g. `random_state`) via the `algorithm_parameters` dict.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import StandardScaler
from ds_utils.unsupervised import plot_loss_vs_cluster_number

# 1. Generate synthetic clusters and scale data
from sklearn.datasets import make_blobs

X_raw, _ = make_blobs(n_samples=150, n_features=4, centers=3, random_state=42)
X_scaled = StandardScaler().fit_transform(X_raw)

# 2. Sweep k from 2 to 8
ax_loss = plot_loss_vs_cluster_number(
    X_scaled,
    k_min=2,
    k_max=8,
    distance_function=euclidean,
    algorithm_parameters={"random_state": 42},
)
plt.close()
```
- **What to look for**: The "elbow" point where adding more clusters no longer yields sharp decreases in loss (loss flattens noticeably after $k=3$).
- **Action**: Pick the elbow $k$ (e.g., $k=3$) and fit your final model. Because the elbow method can be subjective, consider calculating `sklearn.metrics.silhouette_score` as a complementary validation check.

#### Step 2: Fit Final Model and Inspect Cardinality (`plot_cluster_cardinality`)
Fit `KMeans` with the chosen $k$ and inspect cluster member counts.

```python
from sklearn.cluster import KMeans
from ds_utils.unsupervised import plot_cluster_cardinality

kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(X_scaled)

ax_card = plot_cluster_cardinality(labels)
plt.close()
```
- **What to look for**: Tiny outlier clusters (cardinality near 1 or 2) or one massive cluster that swallowed the entire dataset.
- **Action**: If one cluster dominates heavily, inspect whether feature scaling was missed or if $k$ should be increased.

#### Step 3: Measure Compactness (`plot_cluster_magnitude`)
Magnitude measures the sum of point-to-centroid distances within each cluster.

```python
from ds_utils.unsupervised import plot_cluster_magnitude

ax_mag = plot_cluster_magnitude(X_scaled, labels, kmeans.cluster_centers_, distance_function=euclidean)
plt.close()
```
- **What to look for**: Disproportionately high magnitude in clusters indicates high internal dispersion / loose clusters.
- **Action**: If a cluster has excessive magnitude relative to others, evaluate whether it contains multiple sub-populations.

#### Step 4: Diagnose Anomalies (`plot_magnitude_vs_cardinality`)
Under uniform density, cluster magnitude should scale roughly linearly with cardinality.

```python
from ds_utils.unsupervised import plot_magnitude_vs_cardinality

ax_joint = plot_magnitude_vs_cardinality(X_scaled, labels, kmeans.cluster_centers_, distance_function=euclidean)
plt.close()
```
- **Interpretation**: Points far above the trendline represent clusters with low cardinality but high magnitude (i.e. dispersed outliers). Points far below are unusually dense cores.
- **Action**: Inspect high-magnitude/low-cardinality clusters for data quality issues or anomalous instances.

#### Step 5: Visualize Cluster Assignments (`plot_clusters` / `plot_clusters_plotly`)
Create a 2D scatter plot of the clustering results. For data with more than 2 features, PCA (default), t-SNE, or UMAP reduces to 2D automatically.

**Matplotlib variant:**

```python
from ds_utils.unsupervised import plot_clusters

ax_viz = plot_clusters(
    X_scaled,
    labels,
    kmeans.cluster_centers_,
    reduction_method="pca",
    random_state=42,
)
plt.show()
```

**Plotly variant (interactive):**

```python
from ds_utils.unsupervised import plot_clusters_plotly

fig = plot_clusters_plotly(
    X_scaled,
    labels,
    kmeans.cluster_centers_,
    reduction_method="pca",
    random_state=42,
)
fig.show()
```

- **Axis labels**: With >2 features, axes are labeled PC1/PC2, t-SNE 1/2, or UMAP 1/2 depending on the method. With exactly 2 features, labels are "Feature 1"/"Feature 2" and no reduction is performed.
- **Legend**: Entries are "Cluster <label>" plus "Centroids" when centers are drawn.
- **t-SNE caveat**: t-SNE cannot transform new data, so centroids are **not** plotted (a `UserWarning` is emitted).
- **Return types**: plot_clusters returns a matplotlib Axes (call plt.show()); plot_clusters_plotly returns a plotly Figure (call fig.show()).

---

## Examples

### Complete Clustering Diagnostics Script

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from ds_utils.unsupervised import (
    plot_loss_vs_cluster_number,
    plot_cluster_cardinality,
    plot_cluster_magnitude,
    plot_magnitude_vs_cardinality,
    plot_clusters,
    plot_clusters_plotly,
)

# 1. Generate synthetic data
X, _ = make_blobs(n_samples=200, centers=4, n_features=3, random_state=42)
X_norm = StandardScaler().fit_transform(X)

# 2. Find optimal k
plot_loss_vs_cluster_number(X_norm, k_min=2, k_max=6, distance_function=euclidean)
plt.close()

# 3. Fit optimal KMeans
model = KMeans(n_clusters=4, random_state=42)
model.fit(X_norm)

# 4. Evaluate cardinality & magnitude
plot_cluster_cardinality(model.labels_)
plt.close()

plot_cluster_magnitude(X_norm, model.labels_, model.cluster_centers_, euclidean)
plt.close()

plot_magnitude_vs_cardinality(X_norm, model.labels_, model.cluster_centers_, euclidean)
plt.close()

# 5. Visualize clusters (matplotlib)
plot_clusters(X_norm, model.labels_, model.cluster_centers_, reduction_method="pca", random_state=42)
plt.close()

# 6. Visualize clusters (Plotly)
fig = plot_clusters_plotly(X_norm, model.labels_, model.cluster_centers_, reduction_method="pca", random_state=42)
```

## Limitations

- **Centroid-based algorithms only**: Diagnostic functions (`plot_cluster_magnitude`, `plot_magnitude_vs_cardinality`) expect centroid coordinates (`cluster_centers_`) and discrete partition labels. They do not support soft clustering or agglomerative clustering without centroids.
- **Distance function callable**: Distance parameters must be callable functions, not strings.
- **t-SNE draws no centroids**: t-SNE cannot transform new data points, so `cluster_centers` are silently skipped (with a warning) when `reduction_method='tsne'`.
- **DBSCAN noise label counts as a cluster**: The DBSCAN noise label `-1` counts as a unique label in the centers check. Passing KMeans centers (computed for real clusters only) will raise `ValueError` because the number of centers does not match the number of unique labels including `-1`. Omit `cluster_centers` when using DBSCAN.
- **2D reduction is for display only**: The reduced coordinates from PCA/t-SNE/UMAP are used solely for visualization. Never feed them back as clustering input.

## Troubleshooting

| Error / Symptom | Root Cause | Solution |
|---|---|---|
| `ValueError: No valid results were obtained. Check your input data and parameters.` (preceded by `Error occurred for k=...: 'str' object is not callable`) | Passed string metric (e.g. `"euclidean"`) instead of callable | Pass callable metric function, e.g., `euclidean` from scipy.spatial.distance |
| `TypeError: missing 2 required positional arguments: 'k_min' and 'k_max'` | Omitted `k_min` or `k_max` | Specify both integers explicitly, e.g. `k_min=2, k_max=10` |
| `AttributeError: 'DBSCAN' object has no attribute 'cluster_centers_'` when preparing arguments for `plot_cluster_magnitude` / `plot_magnitude_vs_cardinality` | These functions need centroid coordinates, which DBSCAN does not provide | Compute centroids per label yourself (for example the mean of `X` per label) and pass them as `cluster_centers`, or use KMeans. `plot_cluster_cardinality` only needs labels and works with DBSCAN. |
| Plot axes blank or not appearing | Using non-interactive backend without saving | Call `plt.show()` or `plt.savefig(output_path)` |
| `ValueError: X must have at least 2 features for 2D scatter visualization.` | `X` has only 1 column/feature | Ensure `X` has at least 2 features before calling `plot_clusters` / `plot_clusters_plotly` |
| `ValueError: X and cluster_labels must have the same length.` | Array length mismatch between data and labels | Verify `len(X) == len(cluster_labels)` |
| `ValueError: Number of cluster centers must match the number of unique labels.` | Centers count does not equal unique labels count (common with DBSCAN noise label `-1`) | Omit `cluster_centers` for DBSCAN, or compute one center per unique label including noise |
| `ValueError: Unknown reduction_method: '<name>'. Choose from 'pca', 'tsne', 'umap'.` | Unsupported reduction method string | Use `'pca'`, `'tsne'`, or `'umap'` |
| `ImportError: UMAP is not installed...` | `umap-learn` package not installed | `pip install data-science-utils[umap]` or `pip install umap-learn` |
| `UserWarning: t-SNE cannot transform new data; centroids will not be plotted.` | `reduction_method='tsne'` with `cluster_centers` provided | Use `'pca'` or `'umap'` to display centroids, or accept the warning |
| `ValueError: Invalid property specified for object of type plotly.graph_objs.Scatter: 'ax'` | Passed matplotlib `ax=` argument to `plot_clusters_plotly` | Use `fig=` for Plotly; `ax=` is for `plot_clusters` only |
| `AttributeError: PathCollection.set() got an unexpected keyword argument 'fig'` | Passed Plotly `fig=` argument to `plot_clusters` | Use `ax=` for matplotlib; `fig=` is for `plot_clusters_plotly` only |
