"""Cluster cardinality and magnitude diagnostics."""

from typing import Callable, Optional

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, axes, lines

from ds_utils.unsupervised._common import _extract_cardinality, _extract_magnitude


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
