# Unsupervised API Reference

Exhaustive parameter lists, function signatures, return specifications, and common mistakes for `ds_utils.unsupervised`.

## Import Paths

```python
from ds_utils.unsupervised import (
    plot_cluster_cardinality,
    plot_cluster_magnitude,
    plot_magnitude_vs_cardinality,
    plot_loss_vs_cluster_number,
    plot_silhouette_analysis,
    plot_clusters,
    plot_clusters_plotly,
)
```

---

## plot_loss_vs_cluster_number

Evaluates K-Means clustering loss across a range of cluster counts to locate the optimal $k$ (elbow method).

### Signature
```python
plot_loss_vs_cluster_number(
    X: numpy.ndarray,
    k_min: int,
    k_max: int,
    distance_function: Callable[[numpy.ndarray, numpy.ndarray], float],
    *,
    algorithm_parameters: Dict[str, Any] = None,
    ax: Optional[matplotlib.axes._axes.Axes] = None,
    **kwargs,
) -> matplotlib.axes._axes.Axes
```

### Parameters
- `X`: Dataset features matrix (`numpy.ndarray` or DataFrame).
- `k_min`: Minimum number of clusters (required `int`).
- `k_max`: Maximum number of clusters (required `int`).
- `distance_function`: Callable distance metric, e.g., `scipy.spatial.distance.euclidean`. Must be a callable, not a string.
- `algorithm_parameters` (*keyword-only*, *dict*, default `None`): Additional keyword arguments passed to `KMeans()` (e.g. `{"random_state": 42}`).
- `ax` (*keyword-only*, optional, default `None`): Target matplotlib axes.
- `**kwargs`: Forwarded to underlying plot function (`Axes.plot`).

### Returns
- `matplotlib.axes._axes.Axes`

---

## plot_silhouette_analysis

Plots the silhouette coefficient of every sample, grouped and sorted per cluster, with an optional dashed line at the average score. Use it to validate a candidate $k$ or to spot weak, uneven or misassigned clusters for any algorithm that produces discrete labels.

### Signature
```python
plot_silhouette_analysis(
    X: numpy.ndarray,
    labels: numpy.ndarray,
    *,
    metric: str = 'euclidean',
    colors: Optional[Sequence[Any]] = None,
    show_avg_line: bool = True,
    ax: Optional[matplotlib.axes._axes.Axes] = None,
    **kwargs,
) -> matplotlib.axes._axes.Axes
```

### Parameters
- `X`: Feature matrix of shape `(n_samples, n_features)` (`numpy.ndarray` or DataFrame), the same data the labels were computed on.
- `labels`: 1D array of shape `(n_samples,)` with the cluster of each sample. Every distinct value, including the DBSCAN noise label `-1`, is drawn as one cluster.
- `metric` (*keyword-only*, default `'euclidean'`): Distance metric name passed to `sklearn.metrics.silhouette_samples`, e.g. `'manhattan'` or `'cosine'`.
- `colors` (*keyword-only*, optional, default `None`): One color per cluster, in the order of the sorted labels. When `None`, or when fewer colors than clusters are given (a `UserWarning` is emitted), a `tab10` (up to 10 clusters) or `tab20` palette is used.
- `show_avg_line` (*keyword-only*, default `True`): Draw the average silhouette score as a dashed red vertical line with a legend entry.
- `ax` (*keyword-only*, optional, default `None`): Target matplotlib axes.
- `**kwargs`: Forwarded to underlying plot function (`Axes.fill_betweenx`), default `alpha=0.7`.

### Returns
- `matplotlib.axes._axes.Axes`

### Raises
- `ValueError`: `"X and labels must have the same length."` — when lengths differ.
- `ValueError`: `"Silhouette analysis requires at least 2 clusters."` — when `labels` holds a single distinct value.
- `ValueError`: `"Silhouette analysis requires fewer clusters than samples."` — when every sample is its own cluster.
- `UserWarning`: `"colors has <n> entries for <k> clusters; falling back to the default palette."` — when `colors` has fewer entries than there are clusters.

### Example
```python
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from ds_utils.unsupervised import plot_silhouette_analysis

labels = AgglomerativeClustering(n_clusters=4).fit_predict(X)
fig, ax = plt.subplots(figsize=(8, 5))
plot_silhouette_analysis(X, labels, metric="manhattan", ax=ax)
ax.set_title("Silhouette analysis")
plt.show()
```

---

## plot_cluster_cardinality

Plots the distribution of sample counts (cardinality) assigned to each cluster.

### Signature
```python
plot_cluster_cardinality(
    labels: numpy.ndarray,
    *,
    ax: Optional[matplotlib.axes._axes.Axes] = None,
    **kwargs,
) -> matplotlib.axes._axes.Axes
```

### Parameters
- `labels`: 1D array of assigned cluster integer IDs (`kmeans.labels_`).
- `ax` (*keyword-only*, optional, default `None`): Target matplotlib axes.
- `**kwargs`: Forwarded to underlying plot function (`Axes.bar`).

### Returns
- `matplotlib.axes._axes.Axes`

---

## plot_cluster_magnitude

Plots the total sum of point-to-centroid distances (magnitude) per cluster.

### Signature
```python
plot_cluster_magnitude(
    X: numpy.ndarray,
    labels: numpy.ndarray,
    cluster_centers: numpy.ndarray,
    distance_function: Callable[[numpy.ndarray, numpy.ndarray], float],
    *,
    ax: Optional[matplotlib.axes._axes.Axes] = None,
    **kwargs,
) -> matplotlib.axes._axes.Axes
```

### Parameters
- `X`: Input feature dataset (`numpy.ndarray`).
- `labels`: Cluster assignment labels (`numpy.ndarray`).
- `cluster_centers`: Centroid coordinates (`kmeans.cluster_centers_`).
- `distance_function`: Callable metric function (e.g. `scipy.spatial.distance.euclidean`).
- `ax` (*keyword-only*, optional, default `None`): Target matplotlib axes.
- `**kwargs`: Forwarded to underlying plot function (`Axes.bar`).

### Returns
- `matplotlib.axes._axes.Axes`

---

## plot_magnitude_vs_cardinality

Plots cluster magnitude against cardinality to diagnose cluster quality and locate anomalous or dispersed clusters.

### Signature
```python
plot_magnitude_vs_cardinality(
    X: numpy.ndarray,
    labels: numpy.ndarray,
    cluster_centers: numpy.ndarray,
    distance_function: Callable[[numpy.ndarray, numpy.ndarray], float],
    *,
    ax: Optional[matplotlib.axes._axes.Axes] = None,
    **kwargs,
) -> matplotlib.axes._axes.Axes
```

### Parameters
Same parameters as `plot_cluster_magnitude` (`X`, `labels`, `cluster_centers`, `distance_function`, keyword-only `ax`), except that `**kwargs` is forwarded to `Axes.scatter`.

### Returns
- `matplotlib.axes._axes.Axes`


---

## plot_clusters

Creates a 2D scatter plot of clustering results. When the input has more than 2 features, dimensionality reduction (PCA, t-SNE, or UMAP) is applied automatically.

### Signature
```python
plot_clusters(
    X: numpy.ndarray,
    cluster_labels: numpy.ndarray,
    cluster_centers: Optional[numpy.ndarray] = None,
    *,
    reduction_method: str = 'pca',
    random_state: Optional[int] = None,
    ax: Optional[matplotlib.axes._axes.Axes] = None,
    **kwargs,
) -> matplotlib.axes._axes.Axes
```

### Parameters
- `X`: Numeric 2D array of shape `(n_samples, n_features)` with at least 2 features. DataFrames are accepted.
- `cluster_labels`: 1D array of shape `(n_samples,)` with cluster labels.
- `cluster_centers` (optional, default `None`): Array of shape `(n_clusters, n_features)`. Centroids to overlay on the scatter plot.
- `reduction_method` (*keyword-only*, default `'pca'`): `'pca'`, `'tsne'`, or `'umap'`. Used only when `n_features > 2`.
- `random_state` (*keyword-only*, optional, default `None`): Random state for the dimensionality reduction algorithm.
- `ax` (*keyword-only*, optional, default `None`): Target matplotlib Axes.
- `**kwargs`: Forwarded to `matplotlib.axes.Axes.scatter()` for cluster points.

### Returns
- `matplotlib.axes._axes.Axes`

### Raises
- `ValueError`: `"X must be a 2D array with at least 1 feature."` — when `X` is not 2-D or has no feature columns.
- `ValueError`: `"X must have at least 2 features for 2D scatter visualization."` — when `X` has only 1 feature.
- `ValueError`: `"X and cluster_labels must have the same length."` — when lengths differ.
- `ValueError`: `"Number of cluster centers must match the number of unique labels."` — when `len(cluster_centers)` does not equal the number of unique labels.
- `ValueError`: `"Unknown reduction_method: '<name>'. Choose from 'pca', 'tsne', 'umap'."` — when an unsupported method is given.
- `ImportError`: `"UMAP is not installed. To use reduction_method='umap', install it via: \`pip install data-science-utils[umap]\` or \`pip install umap-learn\`."` — when `reduction_method='umap'` and `umap-learn` is absent.
- `UserWarning`: `"t-SNE cannot transform new data; centroids will not be plotted."` — when `reduction_method='tsne'` and `cluster_centers` is provided.

### Example
```python
from sklearn.cluster import KMeans
from ds_utils.unsupervised import plot_clusters
import matplotlib.pyplot as plt

kmeans = KMeans(n_clusters=4, random_state=42).fit(X)
plot_clusters(X, kmeans.labels_, kmeans.cluster_centers_, reduction_method="pca", random_state=42)
plt.show()
```

---

## plot_clusters_plotly

Creates an interactive Plotly 2D scatter plot of clustering results. Same dimensionality reduction options as `plot_clusters`.

### Signature
```python
plot_clusters_plotly(
    X: numpy.ndarray,
    cluster_labels: numpy.ndarray,
    cluster_centers: Optional[numpy.ndarray] = None,
    *,
    reduction_method: str = 'pca',
    random_state: Optional[int] = None,
    fig: Optional[plotly.graph_objs._figure.Figure] = None,
    show_legend: bool = True,
    **kwargs,
) -> plotly.graph_objs._figure.Figure
```

### Parameters
- `X`: Numeric 2D array of shape `(n_samples, n_features)` with at least 2 features. DataFrames are accepted.
- `cluster_labels`: 1D array of shape `(n_samples,)` with cluster labels.
- `cluster_centers` (optional, default `None`): Array of shape `(n_clusters, n_features)`. Centroids to overlay.
- `reduction_method` (*keyword-only*, default `'pca'`): `'pca'`, `'tsne'`, or `'umap'`. Used only when `n_features > 2`.
- `random_state` (*keyword-only*, optional, default `None`): Random state for the dimensionality reduction algorithm.
- `fig` (*keyword-only*, optional, default `None`): Target Plotly `Figure`.
- `show_legend` (*keyword-only*, default `True`): Whether to display the legend.
- `**kwargs`: Forwarded to the `plotly.graph_objects.Scatter` trace.

### Returns
- `plotly.graph_objs._figure.Figure`

### Raises
Same exceptions and warning as `plot_clusters` (both share the input validation and the dimensionality reduction); see its **Raises** section above.

### Example
```python
from sklearn.cluster import KMeans
from ds_utils.unsupervised import plot_clusters_plotly

kmeans = KMeans(n_clusters=4, random_state=42).fit(X)
fig = plot_clusters_plotly(X, kmeans.labels_, kmeans.cluster_centers_)
fig.show()
```
