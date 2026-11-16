"""Plots that help choose the number of clusters."""

from typing import Any, Callable, Dict, Iterable, Optional
import warnings

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, axes
from matplotlib.ticker import MaxNLocator
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from ds_utils.unsupervised._common import _extract_magnitude


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


def _resolve_k_values(k_range: Iterable[int], n_samples: int) -> list:
    """Return the sorted, de-duplicated cluster numbers of `k_range` after validating them against `n_samples`."""
    values = list(k_range)
    if not values:
        raise ValueError("k_range must contain at least one value.")
    if not all(isinstance(k, (int, np.integer)) and not isinstance(k, bool) and k >= 2 for k in values):
        raise ValueError("k_range must contain only integers of at least 2.")
    k_values = sorted({int(k) for k in values})
    if k_values[-1] >= n_samples:
        raise ValueError("k_range values must be smaller than the number of samples.")
    return k_values


def plot_silhouette_score_vs_cluster_number(
    X: np.ndarray,
    k_range: Iterable[int] = range(2, 11),
    *,
    algorithm: Optional[Callable[..., Any]] = None,
    algorithm_parameters: Optional[Dict[str, Any]] = None,
    metric: str = "euclidean",
    show_best: bool = True,
    annotate_scores: bool = True,
    ax: Optional[axes.Axes] = None,
    **kwargs,
) -> axes.Axes:
    """Plot the average silhouette score against the number of clusters to help choose the number of clusters.

    For every number of clusters in `k_range` the clustering algorithm is fitted on `X` and the average silhouette
    score of the resulting labels is computed. The scores are drawn as a line with one marker per k, a dashed vertical
    line marks the k with the highest score, and every point is annotated with its score. The best score is a
    starting point rather than a verdict: confirm the choice with `plot_silhouette_analysis` and the other cluster
    diagnostics. The algorithm is fitted once per k and the scores need pairwise distances, so time grows with the
    length of `k_range` and quadratically with the number of samples; subsample large datasets before calling this
    function.

    :param X: Feature matrix of shape (n_samples, n_features).
    :param k_range: Cluster numbers to evaluate, for example `range(2, 11)` for k = 2, ..., 10. Values are sorted and
           duplicates are ignored. Every value must be an integer of at least 2 and smaller than the number of samples.
    :param algorithm: Clustering class that accepts an `n_clusters` argument and implements `fit_predict()`, for
           example `AgglomerativeClustering`. If None, `sklearn.cluster.KMeans` is used.
    :param algorithm_parameters: Keyword arguments used to create the algorithm for every k. An `n_clusters` entry is
           ignored because it is set from `k_range`. If None, `{"random_state": 42, "n_init": 10}` is used for KMeans
           and no arguments for any other algorithm.
    :param metric: Distance metric passed to `sklearn.metrics.silhouette_score()`.
    :param show_best: Whether to draw a dashed vertical line at the k with the highest score, with a legend entry.
           When several values of k share the highest score, the smallest one is marked.
    :param annotate_scores: Whether to write the score next to every point. Turn it off for long `k_range` values.
    :param ax: Axes object to draw the plot onto; if None, a new Axes is created.
    :param kwargs: Additional keyword arguments passed to `matplotlib.axes.Axes.plot()`.
    :return: The Axes object with the plot drawn onto it. The score line is the first line this call adds, that is
             `ax.lines[0]` on a new or empty Axes.
    :raises ValueError: If `k_range` is empty, contains something other than integers of at least 2, contains a value
            that is not smaller than the number of samples, or if no silhouette score could be computed.
    """
    X = np.asarray(X)
    k_values = _resolve_k_values(k_range, len(X))

    if algorithm is None:
        algorithm = KMeans
    if algorithm_parameters is None:
        parameters = {"random_state": 42, "n_init": 10} if algorithm is KMeans else {}
    else:
        parameters = {key: value for key, value in algorithm_parameters.items() if key != "n_clusters"}

    scores = []
    undefined_k = []
    for k in k_values:
        labels = algorithm(n_clusters=k, **parameters).fit_predict(X)
        if 2 <= len(np.unique(labels)) < len(labels):
            scores.append(float(silhouette_score(X, labels, metric=metric)))
        else:
            scores.append(np.nan)
            undefined_k.append(k)

    if len(undefined_k) == len(k_values):
        raise ValueError("No silhouette score could be computed for any value in k_range.")
    if undefined_k:
        warnings.warn(
            f"The silhouette score is undefined for k={undefined_k} because the algorithm returned fewer than 2 "
            "clusters or one cluster per sample; these values are left out of the plot.",
            UserWarning,
            stacklevel=2,
        )

    if ax is None:
        _, ax = plt.subplots()

    plot_kwargs = {"marker": "o", **kwargs}
    if "linewidth" not in plot_kwargs and "lw" not in plot_kwargs:
        plot_kwargs["linewidth"] = 2
    ax.plot(k_values, scores, **plot_kwargs)

    if show_best:
        best_index = int(np.nanargmax(scores))
        ax.axvline(
            k_values[best_index],
            color="red",
            linestyle="--",
            alpha=0.7,
            label=f"Best: k={k_values[best_index]} ({scores[best_index]:.3f})",
        )
        ax.legend(loc="best", framealpha=0.9)

    if annotate_scores:
        for k, score in zip(k_values, scores, strict=True):
            if not np.isnan(score):
                ax.annotate(
                    f"{score:.3f}",
                    (k, score),
                    textcoords="offset points",
                    xytext=(0, -9),
                    ha="center",
                    va="top",
                    fontsize=9,
                    bbox={"boxstyle": "round,pad=0.15", "facecolor": "white", "edgecolor": "none", "alpha": 0.9},
                )

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.margins(x=0.05, y=0.12)
    ax.grid(alpha=0.3)
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Average silhouette score")
    ax.set_title("Silhouette Score vs Number of Clusters")

    return ax
