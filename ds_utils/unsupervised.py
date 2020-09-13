from typing import Optional

import numpy
import pandas
from matplotlib import axes, pyplot


def plot_cluster_cardinality(labels: numpy.ndarray, *, ax: Optional[axes.Axes] = None, **kwargs) -> axes.Axes:
    """
    Cluster cardinality is the number of examples per cluster. This method plots the number of points per cluster as a
    bar chart.

    Allow investigating clusters that are major outliers.

    :param labels: Labels of each point.
    :param ax: Axes object to draw the plot onto, otherwise uses the current Axes.
    :param kwargs: other keyword arguments

                   All other keyword arguments are passed to ``matplotlib.axes.Axes.pcolormesh()``.
    :return: Returns the Axes object with the plot drawn onto it.
    """
    if ax is None:
        pyplot.figure()
        ax = pyplot.gca()

    labels_df = pandas.DataFrame(numpy.transpose(labels), columns=["labels"])
    labels_df["labels"].value_counts().sort_index().plot(kind="bar", ax=ax, **kwargs)
    pyplot.xticks(rotation=0)
    ax.set_xlabel("Cluster Label")
    ax.set_ylabel("Points in Cluster")

    return ax
