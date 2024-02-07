from typing import Optional, Callable, Dict, Any

import numpy as np
import pandas as pd
from matplotlib import axes, lines
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans


def plot_cluster_cardinality(labels: np.ndarray, *, ax: Optional[axes.Axes] = None, **kwargs) -> axes.Axes:
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
        plt.figure()
        ax = plt.gca()

    cardinality = _extract_cardinality(labels)
    cardinality["count"].plot(kind="bar", ax=ax, **kwargs)
    plt.xticks(rotation=0)
    ax.set_xlabel("Cluster Label")
    ax.set_ylabel("Points in Cluster")

    return ax


def _extract_cardinality(labels):
    labels_df = pd.DataFrame(np.transpose(labels), columns=["labels"])
    cardinality = labels_df["labels"].value_counts().sort_index().reset_index()
    return cardinality


def plot_cluster_magnitude(X: np.ndarray, labels: np.ndarray, cluster_centers: np.ndarray,
                           distance_function: Callable[[np.ndarray, np.ndarray], float], *,
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
        plt.figure()
        ax = plt.gca()

    magnitude = _extract_magnitude(X, labels, cluster_centers, distance_function)
    magnitude.sort_index().plot(kind="bar", ax=ax, **kwargs)
    plt.xticks(rotation=0)
    ax.set_xlabel("Cluster Label")
    ax.set_ylabel("Total Point-to-Centroid Distance")

    return ax


def _extract_magnitude(X, labels, cluster_centers, distance_function):
    data = pd.DataFrame.from_records(np.expand_dims(X, axis=1), columns=["point"])
    data["label"] = labels
    data["center"] = data["label"].apply(lambda label: cluster_centers[label])
    data["distance"] = data.apply(lambda row: distance_function(row["point"], row["center"]), axis=1)
    magnitude = data.groupby(["label"])["distance"].sum()
    return magnitude


def plot_magnitude_vs_cardinality(X: np.ndarray, labels: np.ndarray, cluster_centers: np.ndarray,
                                  distance_function: Callable[[np.ndarray, np.ndarray], float], *,
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
        plt.figure()
        ax = plt.gca()

    cardinality = pd.DataFrame(_extract_cardinality(labels))
    cardinality = cardinality.rename(columns={"labels": "Cardinality"})
    magnitude = pd.DataFrame(_extract_magnitude(X, labels, cluster_centers, distance_function))
    magnitude = magnitude.rename(columns={"distance": "Magnitude"})
    merged = cardinality.merge(magnitude, left_index=True, right_index=True)

    merged.plot("Cardinality", "Magnitude", kind="scatter", ax=ax, **kwargs)
    [ax.annotate(index, [point["Cardinality"], point["Magnitude"]]) for index, point in merged.iterrows()]

    line = lines.Line2D([0, 1], [0, 1])
    transform = ax.transAxes
    line.set_transform(transform)
    ax.add_line(line)

    plt.xticks(rotation=0)
    ax.set_xlabel("Cardinality")
    ax.set_ylabel("Magnitude")

    return ax


def plot_loss_vs_cluster_number(X: np.ndarray, k_min: int, k_max: int,
                                distance_function: Callable[[np.ndarray, np.ndarray], float], *,
                                algorithm_parameters: Dict[str, Any] = None, ax: Optional[axes.Axes] = None,
                                **kwargs) -> axes.Axes:
    """
    k-means requires you to decide the number of clusters ``k`` beforehand. This method runs the KMean algorithm and
    increases the cluster number at each try. The Total magnitude or sum of distance is used as loss.

    Right now the method only works with ``sklearn.cluster.KMeans``.

    :param X: Training instances.
    :param k_min: The minimum cluster number.
    :param k_max: The maximum cluster number.
    :param distance_function: The function used to calculate the distance between an instance to its cluster center.
            The function receives two ndarrays, one the instance and the second is the center and return a float number
            representing the distance between them.
    :param algorithm_parameters: parameters to use for the algorithm. If None, deafult parameters of ``KMeans`` will
            be used.
    :param ax: Axes object to draw the plot onto, otherwise uses the current Axes.
    :param kwargs: other keyword arguments

                   All other keyword arguments are passed to ``matplotlib.axes.Axes.pcolormesh()``.
    :return: Returns the Axes object with the plot drawn onto it.
    """
    if algorithm_parameters is None:
        algorithm_parameters = KMeans().get_params()

    if "n_clusters" in algorithm_parameters:
        del algorithm_parameters["n_clusters"]

    if ax is None:
        plt.figure()
        ax = plt.gca()

    result = []

    for k in range(k_min, k_max + 1):
        estimator = KMeans(n_clusters=k)
        estimator.set_params(**algorithm_parameters)
        estimator.fit(X)
        magnitude = pd.DataFrame(
            _extract_magnitude(X, estimator.labels_, estimator.cluster_centers_, distance_function))
        result.append({"k": k, "magnitude": magnitude["distance"].sum()})

    pd.DataFrame(result).plot("k", "magnitude", kind="scatter", ax=ax, **kwargs)
    plt.xticks(range(max(0, k_min - 1), k_max + 2), rotation=0)
    ax.set_xlabel("Number of clusters")
    ax.set_ylabel("Total Point-to-Centroid Distance")
    ax.set_title("Loss vs Clusters Used")

    return ax
