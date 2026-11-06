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

ds_utils.unsupervised provides diagnostics for centroid-based clustering (specifically scikit-learn KMeans) and 2D scatter visualization of clustering results. It helps data scientists determine the optimal number of clusters $k$ via custom distance loss sweeps, evaluate cluster compactness and stability using cardinality (number of points) and magnitude (total distance to centroid), and visualize cluster assignments via PCA, t-SNE, or UMAP reduction in matplotlib or Plotly.

- **When to use**: Selecting $k$ with the elbow method, verifying that clusters are balanced and compact, spotting anomalous or diffuse clusters, and creating 2D scatter plots of clustering results.
- **When NOT to use**: For supervised classification/regression metrics (use ds-utils-metrics), PCA explained variance (use ds-utils-preprocess), or using the 2D reduction coordinates as clustering input (they are for display only).

## Prerequisites

- Python 3.11 or higher.
- data-science-utils package installed (pip install data-science-utils).
- Core dependencies: numpy, pandas, scikit-learn, scipy, matplotlib, plotly.
- Optional: umap-learn for UMAP reduction (pip install data-science-utils[umap]).

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
                 Prepare Features
           (missing values & StandardScaler)
                         │
                         ▼
        Sweep k and Evaluate Diagnostics
          (plot_loss_vs_cluster_number & silhouette)
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
                         │
                         ▼
           Visualize Cluster Assignments
      (plot_clusters / plot_clusters_plotly)
                         │
                         ▼
             Interpret Cluster Profiles
         (groupby mean on original features)
```

#### Inputs you need

The steps below assume:
- An input DataFrame `X` containing numeric feature columns without target labels.
- A decision on feature selection: which columns belong in the clustering feature space.

```python
import pandas as pd

# Load your unlabeled feature matrix (replace with your data)
df = pd.read_csv("your_dataset")
X = df.select_dtypes("number")  # select numeric clustering features
```

#### Step 1: Prepare features
Question: what is the scale spread across features, are there missing values, and how should features be standardized?

```python
from sklearn.preprocessing import StandardScaler

# Check missingness and scale disparities
print("Missingness ratio per feature:\n", X.isna().mean())
print("Scale spread (mean and std):\n", X.describe().loc[["mean", "std"]])

# Impute or drop missing values, then standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```
- **What to look for**: Large standard deviation discrepancies across features (e.g. income in thousands vs age in tens).
- **Action**: Standardize features with StandardScaler before distance-based clustering.
- **Caution**: Distance-based algorithms (like KMeans and Euclidean distance) are dominated by features with large scales if unscaled.

#### Step 2: Sweep $k$ and Select Cluster Count (`plot_loss_vs_cluster_number`)
Question: which cluster count $k$ balances compactness against model complexity?

```python
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from ds_utils.unsupervised import plot_loss_vs_cluster_number

# 1. Sweep loss across k range
k_range = range(2, 7)
ax_loss = plot_loss_vs_cluster_number(
    X_scaled,
    k_min=2,
    k_max=6,
    distance_function=euclidean,
    algorithm_parameters=dict(random_state=42, n_init=10),
)
plt.close()

# 2. Complementary silhouette score sweep
sil_scores = {}
for k in k_range:
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X_scaled)
    sil_scores[k] = silhouette_score(X_scaled, km.labels_)
print("Silhouette scores by k:", sil_scores)
```
- **What to look for**: An elbow bend in the loss curve accompanied by a high silhouette score.
- **Action**: Pick a $k$ that shows good separation and yields interpretable business segments.
- **Caution**: The elbow is often ambiguous on real-world data without clear separation; combine loss curves with silhouette scores and domain interpretability.

#### Step 3: Fit Final Model and Inspect Cardinality (`plot_cluster_cardinality`)
Question: are cluster point counts balanced, or do massive or tiny outlier clusters exist?

```python
from ds_utils.unsupervised import plot_cluster_cardinality

k_optimal = 3
kmeans = KMeans(n_clusters=k_optimal, n_init=10, random_state=42)
labels = kmeans.fit_predict(X_scaled)

ax_card = plot_cluster_cardinality(labels)
plt.close()
```
- **What to look for**: Tiny clusters containing only 1 or 2 points, or one dominant cluster containing almost all instances.
- **Action**: If a cluster is disproportionately small, investigate whether those points are data entry errors or anomalous records.
- **Caution**: Extreme cardinality imbalance can indicate unhandled extreme outliers pulling centroids away.

#### Step 4: Measure Compactness (`plot_cluster_magnitude`)
Question: which clusters are tightly packed around their centroids, and which are loose or diffuse?

```python
from ds_utils.unsupervised import plot_cluster_magnitude

ax_mag = plot_cluster_magnitude(X_scaled, labels, kmeans.cluster_centers_, distance_function=euclidean)
plt.close()
```
- **What to look for**: Disproportionately high magnitude in clusters indicating high internal dispersion.
- **Action**: Check if high-magnitude clusters contain distinct sub-populations that warrant splitting.
- **Caution**: Total magnitude naturally scales with the number of points; evaluate magnitude in tandem with cardinality.

#### Step 5: Diagnose Anomalies (`plot_magnitude_vs_cardinality`)
Question: do all clusters exhibit consistent density, or do certain clusters have unusually high dispersion for their size?

```python
from ds_utils.unsupervised import plot_magnitude_vs_cardinality

ax_joint = plot_magnitude_vs_cardinality(X_scaled, labels, kmeans.cluster_centers_, distance_function=euclidean)
plt.close()
```
- **What to look for**: Clusters lying far above the expected trend line (low cardinality but high magnitude).
- **Action**: Flag points belonging to high-dispersion clusters for outlier inspection.
- **Caution**: Clusters on opposite sides of the linear fit reflect different densities, not necessarily bad clustering.

#### Step 6: Visualize Cluster Assignments (`plot_clusters` / `plot_clusters_plotly`)
Question: how do the clusters separate when projected into a 2D space?

```python
from ds_utils.unsupervised import plot_clusters

ax_viz = plot_clusters(
    X_scaled,
    labels,
    kmeans.cluster_centers_,
    reduction_method="pca",
    random_state=42,
)
plt.close()
```
- **What to look for**: Visual separation, overlap, or boundary definition across clusters in 2D projection.
- **Action**: Try t-SNE or UMAP if PCA does not resolve overlapping clusters (note: t-SNE cannot plot centroids).
- **Caution**: 2D projection coordinates are for visual diagnosis only; never use reduced 2D coordinates as clustering features.

#### Step 7: Interpret the clusters
Question: what are the concrete feature characteristics and profiles of each discovered cluster?

```python
# Profile clusters on original, unscaled feature values
cluster_summary = X.assign(cluster=labels).groupby("cluster").mean()
cluster_counts = pd.Series(labels, name="size").value_counts()
print("Cluster sizes:\n", cluster_counts)
print("Cluster mean profiles (unscaled):\n", cluster_summary)
```
- **What to look for**: Distinct feature averages differentiating each cluster (e.g. high-spend vs low-frequency).
- **Action**: Assign business labels to each cluster based on its feature profile.
- **Caution**: Profiling on standardized data obscures real-world physical units; always compute profile means on the unscaled features.

---

## Examples

### Runnable demo on a bundled dataset (replace the loading lines with your own data)

```python
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from sklearn.datasets import load_wine
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

# 1. Load bundled dataset and standardize
wine = load_wine(as_frame=True)
X_raw = wine.data
X_scaled = StandardScaler().fit_transform(X_raw)

# 2. Diagnostic sweep for k
ax_loss = plot_loss_vs_cluster_number(
    X_scaled, k_min=2, k_max=6, distance_function=euclidean, algorithm_parameters=dict(random_state=42, n_init=10)
)
plt.close()

# 3. Fit KMeans with k=3
km = KMeans(n_clusters=3, n_init=10, random_state=42)
labels = km.fit_predict(X_scaled)

# 4. Evaluate cardinality & magnitude diagnostics
ax_card = plot_cluster_cardinality(labels)
plt.close()

ax_mag = plot_cluster_magnitude(X_scaled, labels, km.cluster_centers_, euclidean)
plt.close()

ax_joint = plot_magnitude_vs_cardinality(X_scaled, labels, km.cluster_centers_, euclidean)
plt.close()

# 5. Visualize clusters in 2D (matplotlib and plotly variants)
ax_clusters = plot_clusters(X_scaled, labels, km.cluster_centers_, reduction_method="pca", random_state=42)
plt.close()

fig_clusters = plot_clusters_plotly(X_scaled, labels, km.cluster_centers_, reduction_method="pca", random_state=42)

# 6. Profile cluster means on unscaled features
profiles = X_raw.assign(cluster=labels).groupby("cluster").mean()
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
