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
- `X`: Input feature dataset (`numpy.ndarray`).
- `labels`: Assigned cluster labels (`numpy.ndarray`).
- `cluster_centers`: Cluster center coordinates (`kmeans.cluster_centers_`).
- `distance_function`: Callable metric function (`Callable[[numpy.ndarray, numpy.ndarray], float]`).
- `ax` (*keyword-only*, optional, default `None`): Target matplotlib axes.
- `**kwargs`: Forwarded to underlying plot function (`Axes.scatter`).

### Returns
- `matplotlib.axes._axes.Axes`
