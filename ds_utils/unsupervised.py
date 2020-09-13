from typing import Optional, Callable

import numpy
import pandas
from matplotlib import axes, pyplot, lines


def plot_cluster_cardinality(labels: numpy.ndarray, *, ax: Optional[axes.Axes] = None, **kwargs) -> axes.Axes:
    """
    Cluster cardinality is the number of examples per cluster. This method plots the number of points per cluster as a
    bar chart.

    :param labels: Labels of each point.
    :param ax: Axes object to draw the plot onto, otherwise uses the current Axes.
    :param kwargs: other keyword arguments

                   All other keyword arguments are passed to ``matplotlib.axes.Axes.pcolormesh()``.
    :return: Returns the Axes object with the plot drawn onto it.
    """
    if ax is None:
        pyplot.figure()
        ax = pyplot.gca()

    cardinality = _extract_cardinality(labels)
    cardinality.plot(kind="bar", ax=ax, **kwargs)
    pyplot.xticks(rotation=0)
    ax.set_xlabel("Cluster Label")
    ax.set_ylabel("Points in Cluster")

    return ax


def _extract_cardinality(labels):
    labels_df = pandas.DataFrame(numpy.transpose(labels), columns=["labels"])
    cardinality = labels_df["labels"].value_counts().sort_index()
    return cardinality


def plot_cluster_magnitude(X: numpy.ndarray, labels: numpy.ndarray, cluster_centers: numpy.ndarray,
                           distance_function: Callable[[numpy.ndarray, numpy.ndarray], float], *,
                           ax: Optional[axes.Axes] = None, **kwargs) -> axes.Axes:
    """
    Cluster magnitude is the sum of distances from all examples to the centroid of the cluster.
    This method plots the Total Point-to-Centroid Distance per cluster as a bar chart.

    :param X: Training instances.
    :param labels: Labels of each point.
    :param cluster_centers: Coordinates of cluster centers.
    :param distance_function: The function used to calculate the distance between an instance to its cluster center.
    The function receives two ndarrays, one the instance and the second is the center and return a float number
    representing the distance between them.
    :param ax: Axes object to draw the plot onto, otherwise uses the current Axes.
    :param kwargs: other keyword arguments

                   All other keyword arguments are passed to ``matplotlib.axes.Axes.pcolormesh()``.
    :return: Returns the Axes object with the plot drawn onto it.
    """

    if ax is None:
        pyplot.figure()
        ax = pyplot.gca()

    magnitude = _extract_magnitude(X, labels, cluster_centers, distance_function)
    magnitude.sort_index().plot(kind="bar", ax=ax, **kwargs)
    pyplot.xticks(rotation=0)
    ax.set_xlabel("Cluster Label")
    ax.set_ylabel("Total Point-to-Centroid Distance")

    return ax


def _extract_magnitude(X, labels, cluster_centers, distance_function):
    data = pandas.DataFrame.from_records(numpy.expand_dims(X, axis=1), columns=["point"])
    data["label"] = labels
    data["center"] = data["label"].apply(lambda label: cluster_centers[label])
    data["distance"] = data.apply(lambda row: distance_function(row["point"], row["center"]), axis=1)
    magnitude = data.groupby(["label"])["distance"].sum()
    return magnitude


def plot_magnitude_vs_cardinality(X: numpy.ndarray, labels: numpy.ndarray, cluster_centers: numpy.ndarray,
                                  distance_function: Callable[[numpy.ndarray, numpy.ndarray], float], *,
                                  ax: Optional[axes.Axes] = None, **kwargs) -> axes.Axes:
    """
    Higher cluster cardinality tends to result in a higher cluster magnitude, which intuitively makes sense. Clusters
    are anomalous when cardinality doesn't correlate with magnitude relative to the other clusters. Find anomalous
    clusters by plotting magnitude against cardinality as a scatter plot.

    :param X: Training instances.
    :param labels: Labels of each point.
    :param cluster_centers: Coordinates of cluster centers.
    :param distance_function: The function used to calculate the distance between an instance to its cluster center.
    The function receives two ndarrays, one the instance and the second is the center and return a float number
    representing the distance between them.
    :param ax: Axes object to draw the plot onto, otherwise uses the current Axes.
    :param kwargs: other keyword arguments

                   All other keyword arguments are passed to ``matplotlib.axes.Axes.pcolormesh()``.
    :return: Returns the Axes object with the plot drawn onto it.
    """
    if ax is None:
        pyplot.figure()
        ax = pyplot.gca()

    cardinality = pandas.DataFrame(_extract_cardinality(labels))
    cardinality = cardinality.rename(columns={"labels": "Cardinality"})
    magnitude = pandas.DataFrame(_extract_magnitude(X, labels, cluster_centers, distance_function))
    magnitude = magnitude.rename(columns={"distance": "Magnitude"})
    merged = cardinality.merge(magnitude, left_index=True, right_index=True)

    merged.plot("Cardinality", "Magnitude", kind="scatter", ax=ax, **kwargs)
    for k, v in merged.iterrows():
        ax.annotate(k, v)

    line = lines.Line2D([0, 1], [0, 1])
    transform = ax.transAxes
    line.set_transform(transform)
    ax.add_line(line)

    pyplot.xticks(rotation=0)
    ax.set_xlabel("Cardinality")
    ax.set_ylabel("Magnitude")

    return ax
