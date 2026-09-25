"""Unsupervised learning utilities for clustering visualization and analysis."""

from typing import Optional, Callable, Dict, Any, Tuple
import warnings

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, axes, lines
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from plotly import graph_objects as go

try:
    from umap import UMAP
except ImportError:
    UMAP = None


def _extract_cardinality(labels):
    labels_df = pd.DataFrame(np.transpose(labels), columns=["labels"])
    cardinality = labels_df["labels"].value_counts().sort_index().reset_index()
    return cardinality


def plot_cluster_cardinality(labels: np.ndarray, *, ax: Optional[axes.Axes] = None, **kwargs) -> axes.Axes:
    """Plot the number of points per cluster as a bar chart.

    Cluster cardinality is the number of examples per cluster.

    :param labels: Labels of each point.
    :param ax: Axes object to draw the plot onto; if None, uses the current Axes.
    :param kwargs: Additional keyword arguments passed to `matplotlib.axes.Axes.bar()`.
    :return: The Axes object with the plot drawn onto it.
    :raises ValueError: If labels are empty.
    """
    if ax is None:
        _, ax = plt.subplots()

    if len(labels) == 0:
        raise ValueError("Labels array is empty.")

    cardinality = _extract_cardinality(labels)
    cardinality["count"].plot(kind="bar", ax=ax, **kwargs)
    plt.xticks(rotation=0)
    ax.set_xlabel("Cluster Label")
    ax.set_ylabel("Points in Cluster")

    return ax


def _extract_magnitude(X, labels, cluster_centers, distance_function):
    data = pd.DataFrame({"point": list(X), "label": labels})
    data["center"] = data["label"].apply(lambda label: cluster_centers[label])
    data["distance"] = data.apply(lambda row: distance_function(row["point"], row["center"]), axis=1)
    magnitude = data.groupby("label")["distance"].sum()
    return magnitude


def plot_cluster_magnitude(
    X: np.ndarray,
    labels: np.ndarray,
    cluster_centers: np.ndarray,
    distance_function: Callable[[np.ndarray, np.ndarray], float],
    *,
    ax: Optional[axes.Axes] = None,
    **kwargs,
) -> axes.Axes:
    """Plot the Total Point-to-Centroid Distance per cluster as a bar chart.

    Cluster magnitude is the sum of distances from all examples to the centroid of the cluster.

    :param X: Training instances.
    :param labels: Labels of each point.
    :param cluster_centers: Coordinates of cluster centers.
    :param distance_function: Function to calculate the distance between an instance and its cluster center.
           It should take two ndarrays (instance and center) and return a float.
    :param ax: Axes object to draw the plot onto; if None, uses the current Axes.
    :param kwargs: Additional keyword arguments passed to `matplotlib.axes.Axes.bar()`.
    :return: The Axes object with the plot drawn onto it.
    :raises ValueError: If input arrays have inconsistent shapes or if distance_function is invalid.
    """
    if ax is None:
        _, ax = plt.subplots()

    if len(X) != len(labels):
        raise ValueError("X and labels must have the same length.")

    if len(cluster_centers) != len(np.unique(labels)):
        raise ValueError("Number of cluster centers must match the number of unique labels.")

    try:
        magnitude = _extract_magnitude(X, labels, cluster_centers, distance_function)
    except TypeError:
        raise ValueError("Invalid distance_function provided.")

    magnitude.sort_index().plot(kind="bar", ax=ax, **kwargs)
    plt.xticks(rotation=0)
    ax.set_xlabel("Cluster Label")
    ax.set_ylabel("Total Point-to-Centroid Distance")

    return ax


def plot_magnitude_vs_cardinality(
    X: np.ndarray,
    labels: np.ndarray,
    cluster_centers: np.ndarray,
    distance_function: Callable[[np.ndarray, np.ndarray], float],
    *,
    ax: Optional[axes.Axes] = None,
    **kwargs,
) -> axes.Axes:
    """Plot magnitude against cardinality as a scatter plot to find anomalous clusters.

    Higher cluster cardinality tends to result in a higher cluster magnitude. Clusters are considered
    anomalous when cardinality doesn't correlate with magnitude relative to the other clusters.

    :param X: Training instances.
    :param labels: Labels of each point.
    :param cluster_centers: Coordinates of cluster centers.
    :param distance_function: Function to calculate the distance between an instance and its cluster center.
           It should take two ndarrays (instance and center) and return a float.
    :param ax: Axes object to draw the plot onto; if None, uses the current Axes.
    :param kwargs: Additional keyword arguments passed to `matplotlib.axes.Axes.scatter()`.
    :return: The Axes object with the plot drawn onto it.
    :raises ValueError: If input arrays have inconsistent shapes or if distance_function is invalid.
    """
    if ax is None:
        _, ax = plt.subplots()

    if len(X) != len(labels):
        raise ValueError("X and labels must have the same length.")

    if len(cluster_centers) != len(np.unique(labels)):
        raise ValueError("Number of cluster centers must match the number of unique labels.")

    try:
        cardinality = pd.DataFrame(_extract_cardinality(labels))
        cardinality = cardinality.rename(columns={"labels": "Cardinality"})
        magnitude = pd.DataFrame(_extract_magnitude(X, labels, cluster_centers, distance_function))
        magnitude = magnitude.rename(columns={"distance": "Magnitude"})
    except TypeError:
        raise ValueError("Invalid distance_function provided.")

    merged = cardinality.merge(magnitude, left_index=True, right_index=True)

    merged.plot("Cardinality", "Magnitude", kind="scatter", ax=ax, **kwargs)
    for index, point in merged.iterrows():
        ax.annotate(str(index), (point["Cardinality"], point["Magnitude"]))

    line = lines.Line2D([0, 1], [0, 1], transform=ax.transAxes, color="r", linestyle="--")
    ax.add_line(line)

    ax.set_xlabel("Cardinality")
    ax.set_ylabel("Magnitude")

    return ax


def plot_loss_vs_cluster_number(
    X: np.ndarray,
    k_min: int,
    k_max: int,
    distance_function: Callable[[np.ndarray, np.ndarray], float],
    *,
    algorithm_parameters: Dict[str, Any] = None,
    ax: Optional[axes.Axes] = None,
    **kwargs,
) -> axes.Axes:
    """Plot the Total magnitude (sum of distances) as loss against the number of clusters.

    This method runs the KMeans algorithm with increasing cluster numbers and plots the resulting loss.

    :param X: Training instances.
    :param k_min: The minimum cluster number.
    :param k_max: The maximum cluster number.
    :param distance_function: Function to calculate the distance between an instance and its cluster center.
           It should take two ndarrays (instance and center) and return a float.
    :param algorithm_parameters: Parameters to use for the KMeans algorithm. If None, default parameters will be used.
    :param ax: Axes object to draw the plot onto; if None, uses the current Axes.
    :param kwargs: Additional keyword arguments passed to `matplotlib.axes.Axes.scatter()`.
    :return: The Axes object with the plot drawn onto it.
    :raises ValueError: If k_min > k_max or if invalid parameters are provided.
    """
    if ax is None:
        _, ax = plt.subplots()

    if k_min > k_max:
        raise ValueError("k_min must be less than or equal to k_max.")

    if algorithm_parameters is None:
        algorithm_parameters = {"random_state": 42}
    else:
        algorithm_parameters = algorithm_parameters.copy()

    if "n_clusters" in algorithm_parameters:
        del algorithm_parameters["n_clusters"]

    result = []

    for k in range(k_min, k_max + 1):
        try:
            estimator = KMeans(n_clusters=k, **algorithm_parameters)
            estimator.fit(X)
            magnitude = pd.DataFrame(
                _extract_magnitude(X, estimator.labels_, estimator.cluster_centers_, distance_function)
            )
            result.append({"k": k, "magnitude": magnitude["distance"].sum()})
        except Exception as e:
            print(f"Error occurred for k={k}: {str(e)}")

    if not result:
        raise ValueError("No valid results were obtained. Check your input data and parameters.")

    pd.DataFrame(result).plot("k", "magnitude", kind="scatter", ax=ax, **kwargs)
    plt.xticks(range(max(0, k_min - 1), k_max + 2), rotation=0)
    ax.set_xlabel("Number of clusters")
    ax.set_ylabel("Total Point-to-Centroid Distance")
    ax.set_title("Loss vs Number of Clusters")

    return ax


def _validate_cluster_plot_inputs(
    X: np.ndarray,
    cluster_labels: np.ndarray,
    cluster_centers: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], np.ndarray]:
    """Validate clustering plot inputs and return array-converted copies."""
    X = np.asarray(X)
    cluster_labels = np.asarray(cluster_labels)
    if cluster_centers is not None:
        cluster_centers = np.asarray(cluster_centers)

    if X.ndim != 2 or X.shape[1] < 1:
        raise ValueError("X must be a 2D array with at least 1 feature.")

    if X.shape[1] == 1:
        raise ValueError("X must have at least 2 features for 2D scatter visualization.")

    if len(X) != len(cluster_labels):
        raise ValueError("X and cluster_labels must have the same length.")

    unique_labels = np.unique(cluster_labels)

    if cluster_centers is not None and len(cluster_centers) != len(unique_labels):
        raise ValueError("Number of cluster centers must match the number of unique labels.")

    return X, cluster_labels, cluster_centers, unique_labels


def _plot_cluster_scatter(
    ax: axes.Axes,
    X_plot: np.ndarray,
    cluster_labels: np.ndarray,
    cluster_centers_plot: Optional[np.ndarray],
    colors,
    unique_labels: np.ndarray,
    **kwargs,
) -> None:
    """Create scatter plot of clusters on the given Axes (helper function)."""
    # Plot data points
    for i, cluster_label in enumerate(unique_labels):
        cluster_data = X_plot[cluster_labels == cluster_label]
        color = colors(i) if callable(colors) else colors[i]
        scatter_kwargs = {"alpha": 0.6, "s": 50, **kwargs}
        ax.scatter(
            cluster_data[:, 0],
            cluster_data[:, 1],
            c=[color],
            label=f"Cluster {cluster_label}",
            **scatter_kwargs,
        )

    # Plot cluster centers if available
    if cluster_centers_plot is not None:
        ax.scatter(
            cluster_centers_plot[:, 0],
            cluster_centers_plot[:, 1],
            c="red",
            marker="x",
            s=200,
            linewidths=3,
            label="Centroids",
        )


def _reduce_dimensions(
    X: np.ndarray,
    cluster_centers: Optional[np.ndarray],
    reduction_method: str,
    random_state: Optional[int],
) -> Tuple[np.ndarray, Optional[np.ndarray], str, str, Optional[str]]:
    """Reduce X (and optionally cluster_centers) to 2D."""
    if X.shape[1] <= 2:
        return X, cluster_centers, "Feature 1", "Feature 2", None

    if reduction_method == "pca":
        pca = PCA(n_components=2, random_state=random_state)
        X_2d = pca.fit_transform(X)
        centers_2d = pca.transform(cluster_centers) if cluster_centers is not None else None
        return X_2d, centers_2d, "PC1", "PC2", f"PCA Explained Variance: {pca.explained_variance_ratio_.sum():.1%}"

    if reduction_method == "tsne":
        tsne = TSNE(n_components=2, random_state=random_state)
        X_2d = tsne.fit_transform(X)
        centers_2d = None
        if cluster_centers is not None:
            warnings.warn("t-SNE cannot transform new data; centroids will not be plotted.")
        return X_2d, centers_2d, "t-SNE 1", "t-SNE 2", "t-SNE projection"

    if reduction_method == "umap":
        if UMAP is None:
            raise ImportError(
                "UMAP is not installed. To use reduction_method='umap', install it via: "
                "`pip install data-science-utils[umap]` or `pip install umap-learn`."
            )
        umap_model = UMAP(n_components=2, random_state=random_state)
        X_2d = umap_model.fit_transform(X)
        centers_2d = umap_model.transform(cluster_centers) if cluster_centers is not None else None
        return X_2d, centers_2d, "UMAP 1", "UMAP 2", "UMAP projection"

    raise ValueError(f"Unknown reduction_method: '{reduction_method}'. Choose from 'pca', 'tsne', 'umap'.")


def plot_clusters(
    X: np.ndarray,
    cluster_labels: np.ndarray,
    cluster_centers: Optional[np.ndarray] = None,
    *,
    reduction_method: str = "pca",
    random_state: Optional[int] = None,
    ax: Optional[axes.Axes] = None,
    **kwargs,
) -> axes.Axes:
    """Create a 2D scatter plot of clustering results.

    Each cluster is shown in a distinct color. For high-dimensional data (more than 2 features),
    dimensionality reduction is applied automatically.

    :param X: array-like of shape (n_samples, n_features). The input data.
    :param cluster_labels: array-like of shape (n_samples,). Cluster labels assigned by the clustering algorithm.
    :param cluster_centers: array-like of shape (n_clusters, n_features), optional. Cluster centers to overlay.
    :param reduction_method: str, default='pca'. The method to use for dimensionality reduction if n_features > 2.
                             Options: 'pca', 'tsne', 'umap'.
    :param random_state: int, optional. Random state for the dimensionality reduction algorithm.
    :param ax: Axes object to draw the plot onto; if None, uses the current Axes.
    :param kwargs: Additional keyword arguments passed to `matplotlib.axes.Axes.scatter()` for cluster points.
    :return: The Axes object with the plot drawn onto it.
    :raises ValueError: If input arrays have inconsistent shapes or invalid reduction_method.
    :raises ImportError: If 'umap' is requested but umap-learn is not installed.
    """
    if ax is None:
        _, ax = plt.subplots()

    X, cluster_labels, cluster_centers, unique_labels = _validate_cluster_plot_inputs(
        X, cluster_labels, cluster_centers
    )

    X_2d, centers_2d, xlabel, ylabel, info_text = _reduce_dimensions(X, cluster_centers, reduction_method, random_state)

    if len(unique_labels) <= 10:
        colors = plt.get_cmap("tab10")
    else:
        colors = plt.get_cmap("tab20")

    _plot_cluster_scatter(ax, X_2d, cluster_labels, centers_2d, colors, unique_labels, **kwargs)

    if info_text:
        ax.text(
            0.02,
            0.98,
            info_text,
            transform=ax.transAxes,
            verticalalignment="top",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title("Cluster Visualization", fontsize=14)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    ax.grid(True, alpha=0.3)
    if ax.figure is not None:
        ax.figure.tight_layout()

    return ax


def plot_clusters_plotly(
    X: np.ndarray,
    cluster_labels: np.ndarray,
    cluster_centers: Optional[np.ndarray] = None,
    *,
    reduction_method: str = "pca",
    random_state: Optional[int] = None,
    fig: Optional[go.Figure] = None,
    show_legend: bool = True,
    **kwargs,
) -> go.Figure:
    """Create an interactive 2D scatter plot of clustering results using Plotly.

    Each cluster is shown in a distinct color. For high-dimensional data (more than 2 features),
    dimensionality reduction is applied automatically.

    :param X: array-like of shape (n_samples, n_features). The input data.
    :param cluster_labels: array-like of shape (n_samples,). Cluster labels assigned by the clustering algorithm.
    :param cluster_centers: array-like of shape (n_clusters, n_features), optional. Cluster centers to overlay.
    :param reduction_method: str, default='pca'. The method to use for dimensionality reduction if n_features > 2.
                             Options: 'pca', 'tsne', 'umap'.
    :param random_state: int, optional. Random state for the dimensionality reduction algorithm.
    :param fig: plotly's Figure object, optional. The figure to plot on.
    :param show_legend: bool, default=True. Whether to display legend in the plot.
    :param kwargs: Additional keyword arguments passed to the go.Scatter trace.
    :return: The Figure object with the plot drawn onto it.
    :raises ValueError: If input arrays have inconsistent shapes or invalid reduction_method.
    :raises ImportError: If 'umap' is requested but umap-learn is not installed.
    """
    if fig is None:
        fig = go.Figure()

    X, cluster_labels, cluster_centers, unique_labels = _validate_cluster_plot_inputs(
        X, cluster_labels, cluster_centers
    )

    X_2d, centers_2d, xlabel, ylabel, info_text = _reduce_dimensions(X, cluster_centers, reduction_method, random_state)

    # Note: For Plotly we just rely on its default color sequence (which is similar to tab10)
    # when iterating through the distinct labels.
    for cluster_label in unique_labels:
        cluster_data = X_2d[cluster_labels == cluster_label]
        fig.add_trace(
            go.Scatter(
                x=cluster_data[:, 0],
                y=cluster_data[:, 1],
                mode="markers",
                name=f"Cluster {cluster_label}",
                marker=dict(size=8, opacity=0.7),
                text=[f"Cluster {cluster_label}"] * len(cluster_data),
                hoverinfo="text+x+y",
                **kwargs,
            )
        )

    if centers_2d is not None:
        fig.add_trace(
            go.Scatter(
                x=centers_2d[:, 0],
                y=centers_2d[:, 1],
                mode="markers",
                name="Centroids",
                marker=dict(symbol="x", size=12, color="red", line=dict(width=2, color="red")),
                hoverinfo="name+x+y",
            )
        )

    title_text = "Cluster Visualization"
    if info_text:
        title_text += f"<br><sup>{info_text}</sup>"

    fig.update_layout(
        title=title_text,
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        showlegend=show_legend,
    )

    return fig
