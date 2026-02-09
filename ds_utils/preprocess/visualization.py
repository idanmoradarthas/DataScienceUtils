from typing import Optional, Union, List, Callable

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import axes, pyplot as plt, ticker
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform

from _plot_helpers import _plot_clean_violin_distribution, _plot_datetime_heatmap, _copy_series_or_keep_top_10, \
    _plot_count_bar, _is_categorical_like, _plot_categorical_feature1, _plot_datetime_feature1, \
    _plot_categorical_vs_numeric, _plot_datetime_vs_numeric, _plot_numeric_features


def visualize_feature(
    series: pd.Series,
    remove_na: bool = False,
    *,
    include_outliers: bool = True,
    outlier_iqr_multiplier: float = 1.5,
    first_day_of_week: str = "Monday",
    show_counts: bool = True,
    order: Optional[Union[List[str], str]] = None,
    ax: Optional[axes.Axes] = None,
    **kwargs,
) -> axes.Axes:
    """Visualize a pandas Series using an appropriate plot based on dtype.

    Behavior by dtype:

    - Float: draw a violin distribution. If ``include_outliers`` is False, values
      outside the IQR fence [Q1 - k*IQR, Q3 + k*IQR] with ``k=outlier_iqr_multiplier``
      are trimmed prior to plotting.
    - Datetime: draw a 2D heatmap showing day-of-week vs year-week patterns. The heatmap
      displays counts of records for each day of the week (X-axis) and year-week combination
      (Y-axis), making weekly and yearly patterns immediately visible.
    - Object/categorical/bool/int: draw a count plot. Extremely high-cardinality
      series may be reduced to their top categories internally.

    :param series: The data series to visualize.
    :param remove_na: If True, plot with NA values removed; otherwise include them.
    :param include_outliers: Whether to include outliers for float features.
    :param outlier_iqr_multiplier: IQR multiplier used to trim outliers for float features.
    :param first_day_of_week: First day of the week for the heatmap X-axis. Must be one of
                              "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday".
                              Default is "Monday".
    :param show_counts: If True, display count values on top of bars in count plots. Default is True.
    :param order: Order to plot categorical levels in count plots. Can be:

                  - None: Use default sorting (index order after value_counts)
                  - "count_desc": Sort by count in descending order (most frequent first)
                  - "count_asc": Sort by count in ascending order (least frequent first)
                  - "alpha_asc": Sort alphabetically in ascending order
                  - "alpha_desc": Sort alphabetically in descending order
                  - List: Explicit list of category names in desired order

                  Only applies to categorical/object/bool/int features.
    :param ax: Axes in which to draw the plot. If None, a new one is created.
    :param kwargs: Extra keyword arguments forwarded to the underlying plotting function
                   (``seaborn.violinplot``, ``seaborn.heatmap``, or ``matplotlib.pyplot.bar``).
    :return: The Axes object with the plot drawn onto it.
    """
    if ax is None:
        _, ax = plt.subplots()

    feature_series = series.dropna() if remove_na else series

    if pd.api.types.is_float_dtype(feature_series):
        ax = _plot_clean_violin_distribution(feature_series, include_outliers, outlier_iqr_multiplier, ax, **kwargs)
    elif pd.api.types.is_datetime64_any_dtype(feature_series):
        ax = _plot_datetime_heatmap(feature_series, first_day_of_week, ax, **kwargs)
        labels = ax.get_xticklabels()
    else:
        series_to_plot = _copy_series_or_keep_top_10(feature_series)
        value_counts = series_to_plot.value_counts(dropna=remove_na).sort_index()
        ax = _plot_count_bar(value_counts, order, show_counts, ax, **kwargs)
        labels = ax.get_xticklabels()

    if not ax.get_title():
        ax.set_title(f"{feature_series.name} ({feature_series.dtype})")
        # Only set empty xlabel for non-datetime plots
        if not pd.api.types.is_datetime64_any_dtype(feature_series):
            ax.set_xlabel("")

    # Skip tick relabeling for float (violin) plots where x-ticks are hidden
    # Also skip for datetime plots as they handle their own labels
    if not pd.api.types.is_float_dtype(feature_series) and not pd.api.types.is_datetime64_any_dtype(feature_series):
        ticks_loc = ax.get_xticks()
        ax.xaxis.set_major_locator(ticker.FixedLocator(ticks_loc))
        ax.set_xticklabels(labels, rotation=45, ha="right")

    return ax


def visualize_correlations(correlation_matrix: pd.DataFrame, *, ax: Optional[axes.Axes] = None, **kwargs) -> axes.Axes:
    """Compute and visualize pairwise correlations of columns, excluding NA/null values.

    `Original Seaborn code <https://seaborn.pydata.org/examples/many_pairwise_correlations.html>`_.

    :param correlation_matrix: The correlation matrix.
    :param ax: Axes in which to draw the plot. If None, use the currently active Axes.
    :param kwargs: Additional keyword arguments passed to seaborn's heatmap function.
    :return: The Axes object with the plot drawn onto it.
    """
    if ax is None:
        _, ax = plt.subplots()

    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, mask=mask, annot=True, fmt=".3f", ax=ax, **kwargs)
    return ax


def plot_correlation_dendrogram(
    correlation_matrix: pd.DataFrame,
    cluster_distance_method: Union[str, Callable] = "average",
    *,
    ax: Optional[axes.Axes] = None,
    **kwargs,
) -> axes.Axes:
    """Plot a dendrogram of the correlation matrix, showing hierarchically the most correlated variables.

    `Original XAI code <https://github.com/EthicalML/XAI>`_.

    :param correlation_matrix: The correlation matrix.
    :param cluster_distance_method: Method for calculating the distance between newly formed clusters.
                                    `Read more here <https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html>`_
    :param ax: Axes in which to draw the plot. If None, use the currently active Axes.
    :param kwargs: Additional keyword arguments passed to the dendrogram function.
    :return: The Axes object with the plot drawn onto it.
    """
    if ax is None:
        _, ax = plt.subplots()

    corr_condensed = squareform(1 - correlation_matrix)
    z = linkage(corr_condensed, method=cluster_distance_method)
    ax.set(**kwargs)
    dendrogram(z, labels=correlation_matrix.columns.tolist(), orientation="left", ax=ax)
    return ax


def plot_features_interaction(
    data: pd.DataFrame,
    feature_1: str,
    feature_2: str,
    *,
    remove_na: bool = False,
    include_outliers: bool = True,
    outlier_iqr_multiplier: float = 1.5,
    show_ratios: bool = False,
    ax: Optional[axes.Axes] = None,
    **kwargs,
) -> axes.Axes:
    """Plot the joint distribution between two features using type-aware defaults.

    Behavior by dtypes of ``feature_1`` and ``feature_2``:
    - If both are numeric: scatter plot.
    - If one is datetime and the other numeric: line/scatter over time.
    - If both are datetime: scatter plot with complete cases.
    - If both are categorical-like: overlaid histograms per category.
    - If one is categorical-like and the other numeric: violin plot by category.

    For the categorical-vs-numeric case, you can optionally trim outliers from the
    numeric feature using an IQR fence [Q1 - k*IQR, Q3 + k*IQR], where ``k`` is
    controlled by ``outlier_iqr_multiplier``.

    When ``remove_na`` is False, missing values are visualized:

    - Numeric vs Numeric: marginal rug plots showing missing values
    - Numeric vs Datetime: missing numeric values shown as markers on x-axis,
      missing datetime values shown as rug plot on right margin
    - Datetime vs Datetime: complete cases shown as scatter plot, missing values
      shown as rug plots on margins (x-axis for missing feature_2, y-axis for missing feature_1)
    - Categorical vs Numeric: missing numeric values shown with rug plots per category
    - Categorical vs Categorical: missing values included as "Missing" category
    - Categorical/Boolean vs Datetime: missing categorical values added as "Missing" category,
      missing datetime values shown as a separate violin at the edge of the plot

    :param data: The input DataFrame where each feature is a column.
    :param feature_1: Name of the first feature.
    :param feature_2: Name of the second feature.
    :param remove_na: If False (default), keep all data and visualize missingness patterns.
                      If True, remove rows where either feature is NA before plotting.
    :param include_outliers: Whether to include values outside the IQR fence for
                             categorical-vs-numeric violin plots (default True).
    :param outlier_iqr_multiplier: Multiplier ``k`` for the IQR fence when trimming
                                   outliers in categorical-vs-numeric plots (default 1.5).
    :param show_ratios: If True, display ratios (proportions) instead of absolute counts
                        for categorical vs categorical plots. Only applies when both
                        features are categorical-like (default False).
    :param ax: Axes in which to draw the plot. If None, a new one is created.
    :param kwargs: Additional keyword arguments forwarded to the underlying plotting
                   functions (e.g., ``seaborn.violinplot``, ``Axes.scatter``, ``Axes.plot``).
    :return: The Axes object with the plot drawn onto it.
    """
    if ax is None:
        _, ax = plt.subplots()

    if remove_na:
        plot_data = data[[feature_1, feature_2]].dropna()
    else:
        plot_data = data[[feature_1, feature_2]].copy()

    dtype1 = data[feature_1].dtype
    dtype2 = data[feature_2].dtype

    if _is_categorical_like(dtype1):
        ax = _plot_categorical_feature1(
            feature_1,
            feature_2,
            plot_data,
            dtype2,
            include_outliers,
            outlier_iqr_multiplier,
            show_ratios,
            remove_na,
            ax,
            **kwargs,
        )
    elif pd.api.types.is_datetime64_any_dtype(dtype1):
        ax = _plot_datetime_feature1(feature_1, feature_2, plot_data, dtype2, remove_na, ax, **kwargs)
    elif _is_categorical_like(dtype2):
        ax = _plot_categorical_vs_numeric(
            feature_2, feature_1, plot_data, outlier_iqr_multiplier, include_outliers, remove_na, ax, **kwargs
        )
    elif pd.api.types.is_datetime64_any_dtype(dtype2):
        ax = _plot_datetime_vs_numeric(feature_2, feature_1, plot_data, remove_na, ax, **kwargs)
    else:
        ax = _plot_numeric_features(feature_1, feature_2, plot_data, remove_na, ax, **kwargs)

    return ax
