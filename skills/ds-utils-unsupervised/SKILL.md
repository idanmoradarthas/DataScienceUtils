---
name: ds-utils-unsupervised
description: >
  Unsupervised clustering analysis, optimal cluster count selection, and 2D scatter plots (PCA, t-SNE, UMAP). Use for elbow loss curves, silhouette score sweeps over k, silhouette plots, cluster cardinality, centroid magnitude, and cluster assignments. Do NOT use for supervised classification metrics or text binarization.
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
    - silhouette-analysis
    - cluster-visualization
    - pca
    - tsne
    - umap
  documentation: "https://datascienceutils.readthedocs.io/en/stable/"
  package: "data-science-utils"
  repository: "https://github.com/idanmoradarthas/DataScienceUtils"
---

# ds-utils-unsupervised

Tools for choosing cluster count, diagnosing clustering quality using silhouette, cardinality and magnitude analysis, and visualizing cluster assignments in 2D scatter plots.

## Purpose

ds_utils.unsupervised provides diagnostics for centroid-based clustering (specifically scikit-learn KMeans) and 2D scatter visualization of clustering results. It helps data scientists determine the optimal number of clusters $k$ via custom distance loss sweeps and silhouette score sweeps, validate cluster separation with per-sample silhouette coefficients, evaluate cluster compactness and stability using cardinality (number of points) and magnitude (total distance to centroid), and visualize cluster assignments via PCA, t-SNE, or UMAP reduction in matplotlib or Plotly.

- **When to use**: Selecting $k$ with the elbow method, validating separation with a silhouette plot, verifying that clusters are balanced and compact, spotting anomalous or diffuse clusters, and creating 2D scatter plots of clustering results.
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
| Compare average silhouette scores across $k$ | `plot_silhouette_score_vs_cluster_number` | `X`, optional `k_range` | Matplotlib Axes |
| Validate separation and spot misassigned samples | `plot_silhouette_analysis` | `X`, `labels` | Matplotlib Axes |
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
          (plot_loss_vs_cluster_number,
           plot_silhouette_score_vs_cluster_number,
            plot_silhouette_analysis)
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
X = X.dropna()  # labels then align with the remaining rows of X
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```
- **What to look for**: Large standard deviation discrepancies across features (e.g. income in thousands vs age in tens).
- **Action**: Standardize features with StandardScaler before distance-based clustering; confirm missing rows are removed or imputed so labels align with X.
- **Caution**: Distance-based algorithms (like KMeans and Euclidean distance) are dominated by features with large scales if unscaled.

#### Step 2: Sweep $k$ and Select Cluster Count (`plot_loss_vs_cluster_number` / `plot_silhouette_score_vs_cluster_number` / `plot_silhouette_analysis`)
Question: which cluster count $k$ balances compactness against model complexity?

```python
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import euclidean
from sklearn.cluster import KMeans
from ds_utils.unsupervised import (
    plot_silhouette_score_vs_cluster_number,
    plot_loss_vs_cluster_number,
    plot_silhouette_analysis,
)

# 1. Sweep loss across k range
ax_loss = plot_loss_vs_cluster_number(
    X_scaled,
    k_min=2,
    k_max=6,
    distance_function=euclidean,
    algorithm_parameters=dict(random_state=42, n_init=10),
)
plt.close()

# 2. Complementary silhouette score sweep over the same k range; the best-scoring k is marked on the plot
ax_scores = plot_silhouette_score_vs_cluster_number(X_scaled, range(2, 7))
k_values, scores = ax_scores.lines[0].get_data()  # plotted values; NaN where a score was undefined
k_candidate = int(k_values[np.nanargmax(scores)])
print("Best k by average silhouette score:", k_candidate)
plt.close()

# 3. Per-cluster silhouette plot for the best-scoring k (repeat for the other candidates you are weighing)
candidate_labels = KMeans(n_clusters=k_candidate, n_init=10, random_state=42).fit_predict(X_scaled)
ax_sil = plot_silhouette_analysis(X_scaled, candidate_labels)
plt.close()
```
- **What to look for**: An elbow bend in the loss curve accompanied by a peak in the silhouette score sweep. In the per-cluster silhouette plot, every cluster band should extend past the dashed average line, have similar thickness, and contain few or no negative scores.
- **Action**: Pick a $k$ that shows good separation and yields interpretable business segments; investigate clusters whose band lies mostly left of the average line, since they are weakly separated or contain misassigned points.
- **Caution**: The elbow is often ambiguous on real-world data without clear separation; combine loss curves with silhouette plots and domain interpretability. The highest average silhouette score is a shortlist, not a verdict: it often favors the smallest k, so confirm it with the per-cluster plot. Silhouette needs pairwise distances and the sweep refits the algorithm for every k, so on large datasets compute it on a random subsample.

#### Step 3: Fit Final Model and Inspect Cardinality (`plot_cluster_cardinality`)
Question: are cluster point counts balanced, or do massive or tiny outlier clusters exist?

```python
from ds_utils.unsupervised import plot_cluster_cardinality

k_optimal = 3  # replace with the k you chose in Step 2
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
import matplotlib.pyplot as plt
from ds_utils.unsupervised import plot_clusters, plot_clusters_plotly

# Matplotlib static 2D projection
ax_viz = plot_clusters(
    X_scaled,
    labels,
    kmeans.cluster_centers_,
    reduction_method="pca",
    random_state=42,
)
plt.close()

# Plotly interactive 2D projection
fig_viz = plot_clusters_plotly(
    X_scaled,
    labels,
    kmeans.cluster_centers_,
    reduction_method="pca",
    random_state=42,
)
# fig_viz.show()  # display interactive Plotly figure in browser or notebook
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
    plot_silhouette_score_vs_cluster_number,
    plot_loss_vs_cluster_number,
    plot_silhouette_analysis,
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

# 2. Diagnostic sweeps for k: loss curve and average silhouette scores
ax_loss = plot_loss_vs_cluster_number(
    X_scaled, k_min=2, k_max=6, distance_function=euclidean, algorithm_parameters=dict(random_state=42, n_init=10)
)
plt.close()

ax_scores = plot_silhouette_score_vs_cluster_number(X_scaled, range(2, 7))
plt.close()

# 3. Fit KMeans with k=3
km = KMeans(n_clusters=3, n_init=10, random_state=42)
labels = km.fit_predict(X_scaled)

# 4. Validate cluster separation with a silhouette plot
ax_sil = plot_silhouette_analysis(X_scaled, labels)
plt.close()

# 5. Evaluate cardinality & magnitude diagnostics
ax_card = plot_cluster_cardinality(labels)
plt.close()

ax_mag = plot_cluster_magnitude(X_scaled, labels, km.cluster_centers_, euclidean)
plt.close()

ax_joint = plot_magnitude_vs_cardinality(X_scaled, labels, km.cluster_centers_, euclidean)
plt.close()

# 6. Visualize clusters in 2D (matplotlib and plotly variants)
ax_clusters = plot_clusters(X_scaled, labels, km.cluster_centers_, reduction_method="pca", random_state=42)
plt.close()

fig_clusters = plot_clusters_plotly(X_scaled, labels, km.cluster_centers_, reduction_method="pca", random_state=42)

# 7. Profile cluster means on unscaled features
profiles = X_raw.assign(cluster=labels).groupby("cluster").mean()
```

## Limitations

- **Centroid-based algorithms only**: Diagnostic functions (`plot_cluster_magnitude`, `plot_magnitude_vs_cardinality`) expect centroid coordinates (`cluster_centers_`) and discrete partition labels. They do not support soft clustering or agglomerative clustering without centroids.
- **Silhouette cost and labels only**: `plot_silhouette_analysis` needs only `X` and the labels, so it also works for DBSCAN or agglomerative clustering, but it computes pairwise distances: time and memory grow quadratically with the number of samples, so use a random subsample for large datasets. Every distinct label, including the DBSCAN noise label `-1`, is drawn as a cluster, and at least 2 clusters (and fewer clusters than samples) are required.
- **Silhouette sweep needs `n_clusters`**: `plot_silhouette_score_vs_cluster_number` creates the algorithm as `algorithm(n_clusters=k, **algorithm_parameters)` and calls `fit_predict()`, so it works with K-Means (the default) and agglomerative clustering but not with DBSCAN, which has no `n_clusters`. The default `random_state=42, n_init=10` applies to K-Means only; any other algorithm receives just the `algorithm_parameters` you pass. Values of k whose clustering yields fewer than 2 clusters, or one cluster per sample, have no score and are left out with a warning.
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
| `ValueError: k_range must contain only integers of at least 2.` | `k_range` holds a value below 2, a float (for example 2.0), or a string | Build it from integers, for example `range(2, 11)` |
| `ValueError: k_range values must be smaller than the number of samples.` | The largest k in `k_range` reaches the number of rows of `X` | Lower the upper end of `k_range`, or pass more samples |
| `UserWarning: The silhouette score is undefined for k=[...]` | For those k the algorithm returned a single cluster or one cluster per sample | Those k are skipped in the plot; change `algorithm_parameters` or narrow `k_range` if you need them |
| `TypeError: ... got an unexpected keyword argument 'n_clusters'` from `plot_silhouette_score_vs_cluster_number` | `algorithm` does not take `n_clusters`, as with DBSCAN | Pass `KMeans` or `AgglomerativeClustering`, or evaluate the labels of other algorithms with `plot_silhouette_analysis` |
| `ValueError: X and labels must have the same length.` | `plot_silhouette_analysis` received a feature matrix and labels of different lengths, for example after rows were dropped from `X` | Compute the labels from the same `X` you pass to the plot |
| `ValueError: Silhouette analysis requires at least 2 clusters.` | `labels` holds a single distinct value (one cluster, or only DBSCAN noise) | Re-fit with at least 2 clusters or adjust the algorithm parameters |
| `ValueError: Silhouette analysis requires fewer clusters than samples.` | Every sample has its own label | Reduce the cluster count so that clusters group several samples |
| `UserWarning: colors has 2 entries for 3 clusters; falling back to the default palette.` | `colors` has fewer entries than there are clusters | Pass one color per cluster, or omit `colors` to use the default palette |
| `ValueError: X and cluster_labels must have the same length.` | Array length mismatch between data and labels | Verify `len(X) == len(cluster_labels)` |
| `ValueError: Number of cluster centers must match the number of unique labels.` | Centers count does not equal unique labels count (common with DBSCAN noise label `-1`) | Omit `cluster_centers` for DBSCAN, or compute one center per unique label including noise |
| `ValueError: Unknown reduction_method: '<name>'. Choose from 'pca', 'tsne', 'umap'.` | Unsupported reduction method string | Use `'pca'`, `'tsne'`, or `'umap'` |
| `ImportError: UMAP is not installed...` | `umap-learn` package not installed | `pip install data-science-utils[umap]` or `pip install umap-learn` |
| `UserWarning: t-SNE cannot transform new data; centroids will not be plotted.` | `reduction_method='tsne'` with `cluster_centers` provided | Use `'pca'` or `'umap'` to display centroids, or accept the warning |
| `ValueError: Invalid property specified for object of type plotly.graph_objs.Scatter: 'ax'` | Passed matplotlib `ax=` argument to `plot_clusters_plotly` | Use `fig=` for Plotly; `ax=` is for `plot_clusters` only |
| `AttributeError: PathCollection.set() got an unexpected keyword argument 'fig'` | Passed Plotly `fig=` argument to `plot_clusters` | Use `ax=` for matplotlib; `fig=` is for `plot_clusters_plotly` only |
