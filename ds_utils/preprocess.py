import warnings
from typing import Optional, Union, Callable, List

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt, axes, dates, ticker
from scipy.cluster import hierarchy


def visualize_feature(
        series: pd.Series,
        remove_na: bool = False,
        *,
        ax: Optional[axes.Axes] = None,
        **kwargs
) -> axes.Axes:
    """
    Visualize a feature series:

    * For float features, plot a distribution plot.
    * For datetime features, plot a line plot of progression over time.
    * For object, categorical, boolean, or integer features, plot a count plot (histogram).

    :param series: The data series to visualize.
    :param remove_na: If True, ignore NA values when plotting; if False, include them.
    :param ax: Axes in which to draw the plot. If None, use the currently-active Axes.
    :param kwargs: Additional keyword arguments passed to the underlying plotting function.
    :return: The Axes object with the plot drawn onto it.
    """
    if ax is None:
        _, ax = plt.subplots()

    feature_series = series.dropna() if remove_na else series

    if pd.api.types.is_float_dtype(feature_series):
        sns.histplot(feature_series, ax=ax, kde=True, **kwargs)
        labels = ax.get_xticks()
    elif pd.api.types.is_datetime64_any_dtype(feature_series):
        feature_series.value_counts().sort_index().plot(kind="line", ax=ax, **kwargs)
        labels = ax.get_xticks()
    else:
        sns.countplot(x=_copy_series_or_keep_top_10(feature_series), ax=ax, **kwargs)
        labels = ax.get_xticklabels()

    if not ax.get_title():
        ax.set_title(f"{feature_series.name} ({feature_series.dtype})")
        ax.set_xlabel("")

    ticks_loc = ax.get_xticks()
    ax.xaxis.set_major_locator(ticker.FixedLocator(ticks_loc))
    ax.set_xticklabels(labels, rotation=45, ha='right')

    if pd.api.types.is_datetime64_any_dtype(feature_series):
        ax.xaxis.set_major_formatter(_convert_numbers_to_dates)

    return ax


def _calc_correlations(data, method, min_periods):
    return data.apply(lambda x: x.factorize()[0]).corr(method=method, min_periods=min_periods)


def get_correlated_features(
        data_frame: pd.DataFrame,
        features: List[str],
        target_feature: str,
        threshold: float = 0.95,
        method: Union[str, Callable] = 'pearson',
        min_periods: Optional[int] = 1
) -> pd.DataFrame:
    """
    Calculate features correlated above a threshold and extract a DataFrame with correlations and correlation
    to the target feature.

    :param data_frame: The input DataFrame.
    :param features: List of feature names to analyze.
    :param target_feature: Name of the target feature.
    :param threshold: Correlation threshold (default 0.95).
    :param method: Method of correlation: 'pearson', 'kendall', 'spearman', or a callable.
    :param min_periods: Minimum number of observations required per a pair of columns for a valid result.
    :return: DataFrame with correlations and correlation to the target feature.
    """
    correlations = _calc_correlations(data_frame[features + [target_feature]], method, min_periods)
    target_corr = correlations[target_feature]
    features_corr = correlations.loc[features, features]
    corr_matrix = features_corr.where(np.triu(np.ones(features_corr.shape), k=1).astype(bool))
    corr_matrix = corr_matrix[~np.isnan(corr_matrix)].stack().reset_index()
    corr_matrix = corr_matrix[corr_matrix[0].abs() >= threshold]

    if corr_matrix.empty:
        warnings.warn(f"Correlation threshold {threshold} was too high. An empty frame was returned", UserWarning)
        return pd.DataFrame(
            columns=['level_0', 'level_1', 'level_0_level_1_corr', 'level_0_target_corr', 'level_1_target_corr'])

    corr_matrix["level_0_target_corr"] = target_corr[corr_matrix["level_0"]].values
    corr_matrix["level_1_target_corr"] = target_corr[corr_matrix["level_1"]].values
    corr_matrix = corr_matrix.rename({0: "level_0_level_1_corr"}, axis=1).reset_index(drop=True)
    return corr_matrix


def visualize_correlations(
        data: pd.DataFrame,
        method: Union[str, Callable] = 'pearson',
        min_periods: Optional[int] = 1,
        *,
        ax: Optional[axes.Axes] = None,
        **kwargs
) -> axes.Axes:
    """
    Compute and visualize pairwise correlations of columns, excluding NA/null values.
    `Original code <https://seaborn.pydata.org/examples/many_pairwise_correlations.html>`_

    :param data: The input DataFrame, where each feature is a column.
    :param method: Method of correlation: 'pearson', 'kendall', 'spearman', or a callable.
    :param min_periods: Minimum number of observations required per a pair of columns for a valid result.
    :param ax: Axes in which to draw the plot. If None, use the currently-active Axes.
    :param kwargs: Additional keyword arguments passed to seaborn's heatmap function.
    :return: The Axes object with the plot drawn onto it.
    """
    if ax is None:
        _, ax = plt.subplots()

    corr = _calc_correlations(data, method, min_periods)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".3f", ax=ax, **kwargs)
    return ax


def plot_correlation_dendrogram(
        data: pd.DataFrame,
        correlation_method: Union[str, Callable] = 'pearson',
        min_periods: Optional[int] = 1,
        cluster_distance_method: Union[str, Callable] = "average",
        *,
        ax: Optional[axes.Axes] = None,
        **kwargs
) -> axes.Axes:
    """
    Plot a dendrogram of the correlation matrix, showing hierarchically the most correlated variables.
    `Original code <https://github.com/EthicalML/XAI>`_

    :param data: The input DataFrame, where each feature is a column.
    :param correlation_method: Method of correlation: 'pearson', 'kendall', 'spearman', or a callable.
    :param min_periods: Minimum number of observations required per a pair of columns for a valid result.
    :param cluster_distance_method: Method for calculating the distance between newly formed clusters.
    :param ax: Axes in which to draw the plot. If None, use the currently-active Axes.
    :param kwargs: Additional keyword arguments passed to the dendrogram function.
    :return: The Axes object with the plot drawn onto it.
    """
    if ax is None:
        _, ax = plt.subplots()

    corr = _calc_correlations(data, correlation_method, min_periods)
    corr_condensed = hierarchy.distance.squareform(1 - corr)
    z = hierarchy.linkage(corr_condensed, method=cluster_distance_method)
    ax.set(**kwargs)
    hierarchy.dendrogram(z, labels=data.columns.tolist(), orientation="left", ax=ax)
    return ax


def plot_features_interaction(
        feature_1: str,
        feature_2: str,
        data: pd.DataFrame,
        *,
        ax: Optional[axes.Axes] = None,
        **kwargs
) -> axes.Axes:
    """
    Plot the joint distribution between two features.

    :param feature_1: Name of the first feature.
    :param feature_2: Name of the second feature.
    :param data: The input DataFrame, where each feature is a column.
    :param ax: Axes in which to draw the plot. If None, use the currently-active Axes.
    :param kwargs: Additional keyword arguments passed to the underlying plotting function.
    :return: The Axes object with the plot drawn onto it.
    """
    if ax is None:
        _, ax = plt.subplots()

    dtype1 = data[feature_1].dtype
    dtype2 = data[feature_2].dtype

    if is_categorical_like(dtype1):
        plot_categorical_feature1(feature_1, feature_2, data, dtype2, ax, **kwargs)
    elif pd.api.types.is_datetime64_any_dtype(dtype1):
        plot_datetime_feature1(feature_1, feature_2, data, dtype2, ax, **kwargs)
    elif is_categorical_like(dtype2):
        plot_categorical_feature2(feature_1, feature_2, data, ax, **kwargs)
    elif pd.api.types.is_datetime64_any_dtype(dtype2):
        plot_datetime_feature2(feature_1, feature_2, data, ax, **kwargs)
    else:
        plot_numeric_features(feature_1, feature_2, data, ax, **kwargs)

    return ax


def is_categorical_like(dtype):
    """Check if the dtype is categorical-like (categorical, boolean, or object)."""
    return isinstance(dtype, pd.CategoricalDtype) or dtype == bool or dtype == object


def plot_categorical_feature1(feature_1, feature_2, data, dtype2, ax, **kwargs):
    """Plot when the first feature is categorical-like."""
    dup_df = pd.DataFrame()
    dup_df[feature_1] = _copy_series_or_keep_top_10(data[feature_1])

    if is_categorical_like(dtype2):
        plot_categorical_vs_categorical(feature_1, feature_2, dup_df, data, ax, **kwargs)
    elif pd.api.types.is_datetime64_any_dtype(dtype2):
        plot_categorical_vs_datetime(feature_1, feature_2, dup_df, data, ax, **kwargs)
    else:
        plot_categorical_vs_numeric(feature_1, feature_2, dup_df, data, ax, **kwargs)


def plot_datetime_feature1(feature_1, feature_2, data, dtype2, ax, **kwargs):
    """Plot when the first feature is datetime."""
    if is_categorical_like(dtype2):
        plot_datetime_vs_categorical(feature_1, feature_2, data, ax, **kwargs)
    else:
        ax.plot(data[feature_1], data[feature_2], **kwargs)
        ax.set_xlabel(feature_1)
        ax.set_ylabel(feature_2)


def plot_categorical_feature2(feature_1, feature_2, data, ax, **kwargs):
    """Plot when the second feature is categorical-like."""
    dup_df = pd.DataFrame()
    dup_df[feature_2] = _copy_series_or_keep_top_10(data[feature_2])
    dup_df[feature_1] = data[feature_1]
    chart = sns.boxplot(x=feature_2, y=feature_1, data=dup_df, ax=ax, **kwargs)
    ticks_loc = chart.get_xticks()  # Get the tick positions
    chart.xaxis.set_major_locator(ticker.FixedLocator(ticks_loc))  # Explicitly set the tick positions
    chart.set_xticklabels(chart.get_xticklabels(), rotation=45, ha='right')


def plot_datetime_feature2(feature_1, feature_2, data, ax, **kwargs):
    """Plot when the second feature is datetime."""
    ax.plot(data[feature_2], data[feature_1], **kwargs)
    ax.set_xlabel(feature_2)
    ax.set_ylabel(feature_1)


def plot_numeric_features(feature_1, feature_2, data, ax, **kwargs):
    """Plot when both features are numeric."""
    ax.scatter(data[feature_1], data[feature_2], **kwargs)
    ax.set_xlabel(feature_1)
    ax.set_ylabel(feature_2)


def plot_categorical_vs_categorical(feature_1, feature_2, dup_df, data, ax, **kwargs):
    """Plot when both features are categorical-like."""
    dup_df[feature_2] = _copy_series_or_keep_top_10(data[feature_2])
    group_feature_1 = dup_df[feature_1].unique().tolist()
    ax.hist([dup_df.loc[dup_df[feature_1] == value, feature_2] for value in group_feature_1],
            label=group_feature_1, **kwargs)
    ax.set_xlabel(feature_1)
    ax.legend(title=feature_2)


def plot_categorical_vs_datetime(feature_1, feature_2, dup_df, data, ax, **kwargs):
    """Plot when the first feature is categorical-like and the second is datetime."""
    dup_df[feature_2] = data[feature_2].apply(dates.date2num)
    chart = sns.violinplot(x=feature_2, y=feature_1, data=dup_df, ax=ax)
    ticks_loc = chart.get_xticks()
    chart.xaxis.set_major_locator(ticker.FixedLocator(ticks_loc))
    chart.set_xticklabels(chart.get_xticklabels(), rotation=45, ha='right')
    ax.xaxis.set_major_formatter(_convert_numbers_to_dates)


def plot_categorical_vs_numeric(feature_1, feature_2, dup_df, data, ax, **kwargs):
    """Plot when the first feature is categorical-like and the second is numeric."""
    dup_df[feature_2] = data[feature_2]
    chart = sns.boxplot(x=feature_1, y=feature_2, data=dup_df, ax=ax, **kwargs)
    ticks_loc = chart.get_xticks()  # Get the tick positions
    chart.xaxis.set_major_locator(ticker.FixedLocator(ticks_loc))  # Explicitly set the tick positions
    chart.set_xticklabels(chart.get_xticklabels(), rotation=45, ha='right')


def plot_datetime_vs_categorical(feature_1, feature_2, data, ax, **kwargs):
    """Plot when the first feature is datetime and the second is categorical-like."""
    dup_df = pd.DataFrame()
    dup_df[feature_1] = data[feature_1].apply(dates.date2num)
    dup_df[feature_2] = _copy_series_or_keep_top_10(data[feature_2])
    chart = sns.violinplot(x=feature_1, y=feature_2, data=dup_df, ax=ax)
    ticks_loc = chart.get_xticks()
    chart.xaxis.set_major_locator(ticker.FixedLocator(ticks_loc))
    chart.set_xticklabels(chart.get_xticklabels(), rotation=45, ha='right')
    ax.xaxis.set_major_formatter(_convert_numbers_to_dates)


def _copy_series_or_keep_top_10(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.map({True: "True", False: "False"})
    if len(series.unique()) > 10:
        top10 = series.value_counts().nlargest(10).index
        return series.map(lambda x: x if x in top10 else "Other values")
    return series


@plt.FuncFormatter
def _convert_numbers_to_dates(x, pos):
    return dates.num2date(x).strftime('%Y-%m-%d %H:%M')
