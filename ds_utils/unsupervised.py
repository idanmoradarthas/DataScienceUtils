from typing import Optional, Callable, Dict, Any

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, axes, lines
from sklearn.cluster import KMeans


def _extract_cardinality(labels):
    labels_df = pd.DataFrame(np.transpose(labels), columns=["labels"])
    cardinality = labels_df["labels"].value_counts().sort_index().reset_index()
    return cardinality


def plot_cluster_cardinality(
        labels: np.ndarray,
        *,
        ax: Optional[axes.Axes] = None,
        **kwargs
) -> axes.Axes:
    """
    Plot the number of points per cluster as a bar chart.

    Cluster cardinality is the number of examples per cluster.

    :param labels: Labels of each point.
    :param ax: Axes object to draw the plot onto; if None, uses the current Axes.
    :param kwargs: Additional keyword arguments passed to `matplotlib.axes.Axes.bar()`.
    :return: The Axes object with the plot drawn onto it.
    :raises ValueError: If labels is empty.
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


def _extract_magnitude(
        X,
        labels,
        cluster_centers,
        distance_function
):
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
        **kwargs
) -> axes.Axes:
    """
    Plot the Total Point-to-Centroid Distance per cluster as a bar chart.

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
        **kwargs
) -> axes.Axes:
    """
    Plot magnitude against cardinality as a scatter plot to find anomalous clusters.

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

    line = lines.Line2D([0, 1], [0, 1], transform=ax.transAxes, color='r', linestyle='--')
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
        **kwargs
) -> axes.Axes:
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
    """
        Plot the Total magnitude (sum of distances) as loss against the number of clusters.

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
                _extract_magnitude(X, estimator.labels_, estimator.cluster_centers_, distance_function))
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
