from io import BytesIO
from typing import Optional, List, Union, Callable

import numpy
import pandas
import pydotplus
import seaborn
from matplotlib import axes, pyplot, image
from sklearn.tree import export_graphviz

try:
    from sklearn.tree import BaseDecisionTree
except ImportError:
    from sklearn.tree.tree import BaseDecisionTree


def draw_tree(tree: BaseDecisionTree, feature_names: Optional[List[str]] = None,
              class_names: Optional[List[str]] = None, *, ax: Optional[axes.Axes] = None,
              **kwargs) -> axes.Axes:
    """
    Receives a decision tree and return a plot graph of the tree for easy interpretation.

    :param tree: decision tree.
    :param feature_names: the features names.
    :param class_names: the classes names or labels.
    :param ax: Axes object to draw the plot onto, otherwise uses the current Axes.
    :param kwargs: other keyword arguments

                   All other keyword arguments are passed to ``matplotlib.axes.Axes.pcolormesh()``.
    :return: Returns the Axes object with the plot drawn onto it.
    """
    return draw_dot_data(export_graphviz(tree, feature_names=feature_names, out_file=None, filled=True, rounded=True,
                                         special_characters=True, class_names=class_names), ax=ax, **kwargs)


def draw_dot_data(dot_data: str, *, ax: Optional[axes.Axes] = None, **kwargs) -> axes.Axes:
    """
    Receives a Graphiz's Dot language string and return a plot graph of the data.

    :param dot_data: Graphiz's Dot language string.
    :param ax: Axes object to draw the plot onto, otherwise uses the current Axes.
    :param kwargs: other keyword arguments

                   All other keyword arguments are passed to ``matplotlib.axes.Axes.pcolormesh()``.
    :return: Returns the Axes object with the plot drawn onto it.
    """
    if ax is None:
        pyplot.figure()
        ax = pyplot.gca()
    sio = BytesIO()
    graph = pydotplus.graph_from_dot_data(dot_data)
    sio.write(graph.create_png())
    sio.seek(0)
    img = image.imread(sio, format="png")
    ax.imshow(img, **kwargs)
    ax.set_axis_off()
    return ax


def visualize_features(data: pandas.DataFrame, features: Optional[List[str]] = None, num_columns: int = 2,
                       remove_na: bool = False) -> axes.Axes:
    """
    Receives a data frame and visualize the features values on graphs.

    :param data: the data frame.
    :param features: list of feature to visualize.
    :param num_columns: number of columns in the grid.
    :param remove_na: True to ignore NA values when plotting; False otherwise.
    :return: Returns the Axes object with the plot drawn onto it.
    """
    if not features:
        features = data.columns

    if len(features) <= num_columns:
        nrows = 1
    else:
        nrows = int(len(features) / num_columns)
        if len(features) % num_columns > 0:
            nrows += 1
    figure, subplots = pyplot.subplots(nrows=nrows, ncols=num_columns)
    subplots = subplots.flatten()

    i = 0

    for feature in features:
        feature_series = data[feature]
        frame_reduced = data
        if remove_na:
            feature_series = feature_series.dropna()
            frame_reduced = data.dropna(subset=[feature])
        if str(feature_series.dtype).startswith("float"):
            plot = seaborn.distplot(feature_series, ax=subplots[i])
        elif str(feature_series.dtype).startswith("datetime"):
            plot = frame_reduced.groupby(feature).size().plot(ax=subplots[i])
        else:
            plot = seaborn.countplot(feature_series, ax=subplots[i])
        plot.set_title(f"{feature} ({feature_series.dtype})")
        plot.set_xlabel("")

        pyplot.setp(subplots[i].get_xticklabels(), rotation=45)
        i += 1

    if i < len(subplots):
        for j in range(i, len(subplots)):
            figure.delaxes(subplots[j])
    pyplot.subplots_adjust(hspace=0.5)
    return subplots


def visualize_correlations(data: pandas.DataFrame, method: Union[str, Callable] = 'pearson',
                           min_periods: Optional[int] = 1, *, ax: Optional[axes.Axes] = None,
                           **kwargs) -> axes.Axes:
    """
    Compute pairwise correlation of columns, excluding NA/null values, and visualize it with heat map.

    :param data: the data frame, were each feature is a column.
    :param method: {‘pearson’, ‘kendall’, ‘spearman’} or callable

                   Method of correlation:

                   * pearson : standard correlation coefficient
                   * kendall : Kendall Tau correlation coefficient
                   * spearman : Spearman rank correlation
                   * callable: callable with input two 1d ndarrays and returning a float. Note that the returned matrix from corr will have 1 along the diagonals and will be symmetric regardless of the callable’s behavior.
    :param min_periods: Minimum number of observations required per pair of columns to have a valid result. Currently only available for Pearson and Spearman correlation.
    :param ax: Axes in which to draw the plot, otherwise use the currently-active Axes.
    :param kwargs: other keyword arguments

                   All other keyword arguments are passed to ``matplotlib.axes.Axes.pcolormesh()``.
    :return: Returns the Axes object with the plot drawn onto it.
    """
    if ax is None:
        pyplot.figure()
        ax = pyplot.gca()

    corr = data.apply(lambda x: x.factorize()[0]).corr(method=method, min_periods=min_periods)
    mask = numpy.triu(numpy.ones_like(corr, dtype=numpy.bool))
    seaborn.heatmap(corr, mask=mask, annot=True, fmt=".3f", ax=ax, **kwargs)
    return ax
