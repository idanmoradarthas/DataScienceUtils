# Unsupervised API Reference

Exhaustive parameter lists, function signatures, return specifications, and common mistakes for `ds_utils.unsupervised`.

## Import Paths

```python
from ds_utils.unsupervised import (
    plot_cluster_cardinality,
    plot_cluster_magnitude,
    plot_magnitude_vs_cardinality,
    plot_loss_vs_cluster_number,
)
```

---

## plot_loss_vs_cluster_number

Evaluates K-Means clustering loss across a range of cluster counts to locate the optimal $k$ (elbow method).

### Signature
```python
plot_loss_vs_cluster_number(
    X: Union[np.ndarray, pd.DataFrame],
    k_min: int,
    k_max: int,
    distance_function: Callable,
    algorithm_parameters: Optional[Dict[str, Any]] = None,
    ax: Optional[matplotlib.axes.Axes] = None,
) -> matplotlib.axes.Axes
```

### Parameters
- `X`: Dataset (features only, without labels).
- `k_min`: Minimum number of clusters (required int).
- `k_max`: Maximum number of clusters (required int).
- `distance_function`: Callable distance metric, e.g., `scipy.spatial.distance.euclidean`. Must be a callable, not a string.
- `algorithm_parameters`: Optional dictionary forwarded to `sklearn.cluster.KMeans` (e.g. `{"random_state": 42}`).
- `ax`: Optional target matplotlib axes.

### Returns
- `matplotlib.axes.Axes`

---

## plot_cluster_cardinality

Plots the distribution of sample counts (cardinality) assigned to each cluster.

### Signature
```python
plot_cluster_cardinality(
    labels: Union[np.ndarray, List[int]],
    ax: Optional[matplotlib.axes.Axes] = None,
) -> matplotlib.axes.Axes
```

### Parameters
- `labels`: 1D array of assigned cluster integer IDs (`kmeans.labels_`).
- `ax`: Optional target matplotlib axes.

### Returns
- `matplotlib.axes.Axes`

---

## plot_cluster_magnitude

Plots the total sum of point-to-centroid distances (magnitude) per cluster.

### Signature
```python
plot_cluster_magnitude(
    X: Union[np.ndarray, pd.DataFrame],
    labels: Union[np.ndarray, List[int]],
    cluster_centers: Union[np.ndarray, List],
    distance_function: Callable,
    ax: Optional[matplotlib.axes.Axes] = None,
) -> matplotlib.axes.Axes
```

### Parameters
- `X`: Input feature dataset.
- `labels`: Cluster assignment labels.
- `cluster_centers`: Centroid coordinates (`kmeans.cluster_centers_`).
- `distance_function`: Callable metric function (e.g. `scipy.spatial.distance.euclidean`).
- `ax`: Optional target matplotlib axes.

### Returns
- `matplotlib.axes.Axes`

---

## plot_magnitude_vs_cardinality

Plots cluster magnitude against cardinality to diagnose cluster quality and locate anomalous or dispersed clusters.

### Signature
```python
plot_magnitude_vs_cardinality(
    X: Union[np.ndarray, pd.DataFrame],
    labels: Union[np.ndarray, List[int]],
    cluster_centers: Union[np.ndarray, List],
    distance_function: Callable,
    ax: Optional[matplotlib.axes.Axes] = None,
) -> matplotlib.axes.Axes
```

### Parameters
- Same as `plot_cluster_magnitude`.

### Returns
- `matplotlib.axes.Axes`
