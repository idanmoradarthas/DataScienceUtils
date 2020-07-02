from typing import Optional, Union, Callable

import numpy
import pandas
import seaborn
from matplotlib import axes, pyplot, dates
from scipy.cluster import hierarchy


def visualize_feature(series: pandas.Series, remove_na: bool = False, *, ax: Optional[axes.Axes] = None,
                      **kwargs) -> axes.Axes:
    """
    Visualize a feature series:

    * If the feature is float then the method plots the distribution plot.
    * If the feature is datetime then the method plots a line plot of progression of amount thought time.
    * If the feature is object, categorical, boolean or integer then the method plots count plot (histogram).

    :param series: the data series.
    :param remove_na: True to ignore NA values when plotting; False otherwise.
    :param ax: Axes in which to draw the plot, otherwise use the currently-active Axes.
    :param kwargs: other keyword arguments

                   All other keyword arguments are passed to ``matplotlib.axes.Axes.pcolormesh()``.
    :return: Returns the Axes object with the plot drawn onto it.
    """
    if ax is None:
        pyplot.figure()
        ax = pyplot.gca()

    if remove_na:
        feature_series = series.dropna()
    else:
        feature_series = series

    if str(feature_series.dtype).startswith("float"):
        seaborn.distplot(feature_series, ax=ax, hist_kws=kwargs)
        labels = ax.get_xticks()
    elif str(feature_series.dtype).startswith("datetime"):
        feature_series.value_counts().plot(kind="line", ax=ax, **kwargs)
        labels = ax.get_xticks()
    else:
        seaborn.countplot(_copy_series_or_keep_top_10(feature_series), ax=ax, **kwargs)
        labels = ax.get_xticklabels()

    if not ax.get_title():
        ax.set_title(f"{feature_series.name} ({feature_series.dtype})")
        ax.set_xlabel("")

    ax.set_xticklabels(labels, rotation=45, horizontalalignment='right')

    if str(feature_series.dtype).startswith("datetime"):
        ax.xaxis.set_major_formatter(_convert_numbers_to_dates)

    return ax


def visualize_correlations(data: pandas.DataFrame, method: Union[str, Callable] = 'pearson',
                           min_periods: Optional[int] = 1, *, ax: Optional[axes.Axes] = None,
                           **kwargs) -> axes.Axes:
    """
    Compute pairwise correlation of columns, excluding NA/null values, and visualize it with heat map.
    `Original code <https://seaborn.pydata.org/examples/many_pairwise_correlations.html>`_

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

    corr = _calc_corrections(data, method, min_periods)
    mask = numpy.triu(numpy.ones_like(corr, dtype=numpy.bool))
    seaborn.heatmap(corr, mask=mask, annot=True, fmt=".3f", ax=ax, **kwargs)
    return ax


def plot_correlation_dendrogram(data: pandas.DataFrame, correlation_method: Union[str, Callable] = 'pearson',
                                min_periods: Optional[int] = 1,
                                cluster_distance_method: Union[str, Callable] = "average", *,
                                ax: Optional[axes.Axes] = None,
                                **kwargs) -> axes.Axes:
    """
    Plot dendrogram of a correlation matrix. This consists of a chart that that shows hierarchically the variables that
    are most correlated by the connecting trees. The closer to the right that the connection is, the more correlated the features are.

    :param data: the data frame, were each feature is a column.
    :param correlation_method: {‘pearson’, ‘kendall’, ‘spearman’} or callable

                   Method of correlation:

                   * pearson : standard correlation coefficient
                   * kendall : Kendall Tau correlation coefficient
                   * spearman : Spearman rank correlation
                   * callable: callable with input two 1d ndarrays and returning a float. Note that the returned matrix from corr will have 1 along the diagonals and will be symmetric regardless of the callable’s behavior.

    :param min_periods: Minimum number of observations required per pair of columns to have a valid result. Currently only available for Pearson and Spearman correlation.
    :param cluster_distance_method: The following are methods for calculating the distance between the newly formed cluster.

            Methods of linkage:

            * single: This is also known as the Nearest Point Algorithm.
            * complete: This is also known by the Farthest Point Algorithm or Voor Hees Algorithm.
            * average:
            .. math:: d(u,v) = \\sum_{ij} \\frac{d(u[i], v[j])}{(|u|*|v|)}

            This is also called the UPGMA algorithm.

            * weighted:
            .. math:: d(u,v) = (dist(s,v) + dist(t,v))/2

            where cluster u was formed with cluster s and t and v
            is a remaining cluster in the forest. (also called WPGMA)

            * centroid: Euclidean distance between the centroids
            * median: This is also known as the WPGMC algorithm.
            * ward: uses the Ward variance minimization algorithm.

            see `scipy.cluster.hierarchy.linkage <https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html>`_ for more information.

    :param ax: Axes in which to draw the plot, otherwise use the currently-active Axes.
    :param kwargs: other keyword arguments

                   All other keyword arguments are passed to ``matplotlib.axes.Axes.pcolormesh()``.
    :return: Returns the Axes object with the plot drawn onto it.
    """

    if ax is None:
        pyplot.figure()
        ax = pyplot.gca()

    corr = _calc_corrections(data, correlation_method, min_periods)
    # reverse the distance
    corr_condensed = hierarchy.distance.squareform(1 - corr)
    z = hierarchy.linkage(corr_condensed, method=cluster_distance_method)
    ax.set(**kwargs)
    hierarchy.dendrogram(z, labels=data.columns.tolist(), orientation="left", ax=ax)
    return ax


def _calc_corrections(data: pandas.DataFrame, method: Union[str, Callable],
                      min_periods: Optional[int]) -> numpy.ndarray:
    return data.apply(lambda x: x.factorize()[0]).corr(method=method, min_periods=min_periods)


def plot_features_interaction(feature_1: str, feature_2: str, data: pandas.DataFrame, *,
                              ax: Optional[axes.Axes] = None, **kwargs) -> axes.Axes:
    """
    Plots the joint distribution between two features:

    * If both features are either categorical, boolean or object then the method plots the shared histogram.
    * If one feature is either categorical, boolean or object and the other is numeric then the method plots a boxplot chart.
    * If one feature is datetime and the other is numeric or datetime then the method plots a line plot graph.
    * If one feature is datetime and the other is either categorical, boolean or object the method plots a violin plot (combination of boxplot and kernel density estimate).
    * If both features are numeric then the method plots scatter graph.

    :param feature_1: the name of the first feature.
    :param feature_2: the name of the second feature.
    :param data: the data frame, were each feature is a column.
    :param ax: Axes in which to draw the plot, otherwise use the currently-active Axes.
    :param kwargs: other keyword arguments

                   All other keyword arguments are passed to ``matplotlib.axes.Axes.pcolormesh()``.
    :return: Returns the Axes object with the plot drawn onto it.
    """
    if ax is None:
        pyplot.figure()
        ax = pyplot.gca()

    dup_df = pandas.DataFrame()
    if str(data[feature_1].dtype) in ["object", "category", "bool"]:
        dup_df[feature_1] = _copy_series_or_keep_top_10(data[feature_1])
        if str(data[feature_2].dtype) in ["object", "category", "bool"]:
            # both features are categorical
            dup_df[feature_2] = _copy_series_or_keep_top_10(data[feature_2])
            group_feature_1 = dup_df[feature_1].unique().tolist()
            ax.hist([dup_df.loc[dup_df[feature_1] == value, feature_2] for value in group_feature_1],
                    label=group_feature_1, **kwargs)
            ax.set_xlabel(feature_1)
            ax.legend(title=feature_2)
        elif str(data[feature_2].dtype).startswith("datetime"):
            # first feature is categorical and the second is datetime
            dup_df[feature_2] = data[feature_2].apply(dates.date2num)
            chart = seaborn.violinplot(feature_2, feature_1, data=dup_df, ax=ax)
            chart.set_xticklabels(chart.get_xticklabels(), rotation=45, horizontalalignment='right')
            ax.xaxis.set_major_formatter(_convert_numbers_to_dates)
        else:
            # first feature is categorical and the second is numeric
            dup_df[feature_2] = data[feature_2]
            chart = seaborn.boxplot(feature_1, feature_2, data=dup_df, ax=ax, **kwargs)
            chart.set_xticklabels(chart.get_xticklabels(), rotation=45, horizontalalignment='right')
    elif str(data[feature_1].dtype).startswith("datetime"):
        if str(data[feature_2].dtype) in ["object", "category", "bool"]:
            # first feature is datetime and the second is categorical
            dup_df[feature_1] = data[feature_1].apply(dates.date2num)
            dup_df[feature_2] = _copy_series_or_keep_top_10(data[feature_2])
            chart = seaborn.violinplot(feature_1, feature_2, data=dup_df, ax=ax)
            chart.set_xticklabels(chart.get_xticklabels(), rotation=45, horizontalalignment='right')
            ax.xaxis.set_major_formatter(_convert_numbers_to_dates)
        else:
            # first feature is datetime and the second is numeric or datetime
            ax.plot(data[feature_1], data[feature_2], **kwargs)
            ax.set_xlabel(feature_1)
            ax.set_ylabel(feature_2)
    elif str(data[feature_2].dtype) in ["object", "category", "bool"]:
        # first feature is numeric and the second is categorical
        dup_df[feature_2] = _copy_series_or_keep_top_10(data[feature_2])
        dup_df[feature_1] = data[feature_1]
        chart = seaborn.boxplot(feature_2, feature_1, data=dup_df, ax=ax, **kwargs)
        chart.set_xticklabels(chart.get_xticklabels(), rotation=45, horizontalalignment='right')
    elif str(data[feature_2].dtype).startswith("datetime"):
        # first feature is numeric and the second is datetime
        ax.plot(data[feature_2], data[feature_1], **kwargs)
        ax.set_xlabel(feature_2)
        ax.set_ylabel(feature_1)
    else:
        # both feature are numeric
        ax.scatter(data[feature_1], data[feature_2], **kwargs)
        ax.set_xlabel(feature_1)
        ax.set_ylabel(feature_2)

    return ax


def _copy_series_or_keep_top_10(series: pandas.Series) -> pandas.Series:
    if str(series.dtype) == "bool":
        # avoiding RuntimeWarning from numpy (Converting input from bool to <class 'numpy.uint8'> for compatibility.)
        return series.apply(lambda val: "True" if val else "False")
    if len(series.unique().tolist()) > 10:
        top10 = series.value_counts()[:10].index.tolist()
        return series.apply(lambda val: val if val in top10 else "Other values")
    return series


@pyplot.FuncFormatter
def _convert_numbers_to_dates(x, pos):
    return dates.num2date(x).strftime('%Y-%m-%d %H:%M')
