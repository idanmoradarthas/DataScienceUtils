"""Per-sample silhouette analysis of a clustering."""

from typing import Any, Optional, Sequence
import warnings

import numpy as np
from matplotlib import pyplot as plt, axes
from sklearn.metrics import silhouette_samples


def _resolve_silhouette_colors(colors: Optional[Sequence[Any]], n_clusters: int) -> list:
    """Return one color per cluster, falling back to a qualitative palette when `colors` is missing or too short."""
    if colors is not None:
        if len(colors) >= n_clusters:
            return list(colors[:n_clusters])
        warnings.warn(
            f"colors has {len(colors)} entries for {n_clusters} clusters; falling back to the default palette.",
            UserWarning,
            stacklevel=3,
        )
    palette = plt.get_cmap("tab10" if n_clusters <= 10 else "tab20")
    return [palette(index % palette.N) for index in range(n_clusters)]


def plot_silhouette_analysis(
    X: np.ndarray,
    labels: np.ndarray,
    *,
    metric: str = "euclidean",
    colors: Optional[Sequence[Any]] = None,
    show_avg_line: bool = True,
    ax: Optional[axes.Axes] = None,
    **kwargs,
) -> axes.Axes:
    """Plot the silhouette coefficient of every sample, grouped by cluster.

    The silhouette coefficient of a sample compares its mean distance to the other samples of its own cluster with its
    mean distance to the samples of the nearest other cluster. Values close to 1 mean the sample is well matched to its
    cluster, values close to 0 mean it lies on the boundary between two clusters, and negative values suggest it may
    have been assigned to the wrong cluster. Each cluster is drawn as a horizontal band whose thickness is the cluster
    size and whose bars are sorted by score; the dashed vertical line marks the average silhouette score of all
    samples. Clusters with many samples to the left of that line, or much thicker than the others, deserve a closer
    look.

    Pairwise distances are computed, so time and memory grow quadratically with the number of samples; subsample large
    datasets before calling this function.

    :param X: Feature matrix of shape (n_samples, n_features), the same data the labels were computed on.
    :param labels: Cluster label of each sample. Every distinct label, including the DBSCAN noise label -1, is treated
           as a cluster.
    :param metric: Distance metric passed to `sklearn.metrics.silhouette_samples()`.
    :param colors: One color per cluster, in the order of the sorted labels. If None, or if fewer colors than clusters
           are given (a warning is emitted), a qualitative palette is used.
    :param show_avg_line: Whether to draw the average silhouette score as a dashed vertical line, with a legend entry
           in the lower right corner.
    :param ax: Axes object to draw the plot onto; if None, a new Axes is created.
    :param kwargs: Additional keyword arguments passed to `matplotlib.axes.Axes.fill_betweenx()`.
    :return: The Axes object with the plot drawn onto it.
    :raises ValueError: If X and labels differ in length, or if there are fewer than 2 clusters or not fewer clusters
            than samples.
    """
    X = np.asarray(X)
    labels = np.asarray(labels)

    if len(X) != len(labels):
        raise ValueError("X and labels must have the same length.")

    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    if n_clusters < 2:
        raise ValueError("Silhouette analysis requires at least 2 clusters.")
    if n_clusters >= len(labels):
        raise ValueError("Silhouette analysis requires fewer clusters than samples.")

    if ax is None:
        _, ax = plt.subplots()

    sample_scores = silhouette_samples(X, labels, metric=metric)
    average_score = float(np.mean(sample_scores))
    cluster_colors = _resolve_silhouette_colors(colors, n_clusters)

    gap = max(1, int(round(0.02 * len(labels))))
    y_lower = gap
    tick_positions = []
    for cluster_color, label in zip(cluster_colors, unique_labels, strict=True):
        cluster_scores = np.sort(sample_scores[labels == label])
        y_upper = y_lower + len(cluster_scores)
        ax.fill_betweenx(
            np.arange(y_lower, y_upper), 0, cluster_scores, **{"color": cluster_color, "alpha": 0.7, **kwargs}
        )
        tick_positions.append((y_lower + y_upper) / 2)
        y_lower = y_upper + gap

    if show_avg_line:
        ax.axvline(average_score, color="red", linestyle="--", label=f"Average: {average_score:.3f}")
        ax.legend(loc="lower right", framealpha=0.9)

    ax.set_yticks(tick_positions, labels=[str(label) for label in unique_labels])
    ax.set_xlim(min(-0.1, np.floor(sample_scores.min() * 10) / 10), 1.0)
    ax.set_ylim(0, y_lower)
    ax.set_xlabel("Silhouette Coefficient")
    ax.set_ylabel("Cluster Label")

    return ax
