"""Data preprocessing utilities."""

import warnings
from typing import Optional, Union, Callable, List

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt, axes, dates, ticker
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform

from ds_utils.math_utils import safe_percentile


def visualize_feature(
    series: pd.Series, remove_na: bool = False, *, ax: Optional[axes.Axes] = None, **kwargs
) -> axes.Axes:
    """Visualize a feature series.

    * For float features, plot a distribution plot.
    * For datetime features, plot a line plot of progression over time.
    * For object, categorical, boolean, or integer features, plot a count plot (histogram).

    :param series: The data series to visualize.
    :param remove_na: If True, ignore NA values when plotting; if False, include them.
    :param ax: Axes in which to draw the plot. If None, use the currently active Axes.
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
    ax.set_xticklabels(labels, rotation=45, ha="right")

    if pd.api.types.is_datetime64_any_dtype(feature_series):
        ax.xaxis.set_major_formatter(_convert_numbers_to_dates)

    return ax


def get_correlated_features(
    correlation_matrix: pd.DataFrame, features: List[str], target_feature: str, threshold: float = 0.95
) -> pd.DataFrame:
    """Calculate features correlated above a threshold with target correlations.

    Calculate features correlated above a threshold and extract a DataFrame with correlations and correlation
    to the target feature.

    :param correlation_matrix: The correlation matrix.
    :param features: List of feature names to analyze.
    :param target_feature: Name of the target feature.
    :param threshold: Correlation threshold (default 0.95).
    :return: DataFrame with correlations and correlation to the target feature.
    """
    target_corr = correlation_matrix[target_feature]
    features_corr = correlation_matrix.loc[features, features]
    corr_matrix = features_corr.where(np.triu(np.ones(features_corr.shape), k=1).astype(bool))
    corr_matrix = corr_matrix[~np.isnan(corr_matrix)].stack().reset_index()
    corr_matrix = corr_matrix[corr_matrix[0].abs() >= threshold]

    if corr_matrix.empty:
        warnings.warn(f"Correlation threshold {threshold} was too high. An empty frame was returned", UserWarning)
        return pd.DataFrame(
            columns=["level_0", "level_1", "level_0_level_1_corr", "level_0_target_corr", "level_1_target_corr"]
        )

    corr_matrix["level_0_target_corr"] = target_corr[corr_matrix["level_0"]].values
    corr_matrix["level_1_target_corr"] = target_corr[corr_matrix["level_1"]].values
    corr_matrix = corr_matrix.rename({0: "level_0_level_1_corr"}, axis=1).reset_index(drop=True)
    return corr_matrix


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
    data: pd.DataFrame, feature_1: str, feature_2: str, *, ax: Optional[axes.Axes] = None, **kwargs
) -> axes.Axes:
    """Plot the joint distribution between two features.

    :param data: The input DataFrame, where each feature is a column.
    :param feature_1: Name of the first feature.
    :param feature_2: Name of the second feature.
    :param ax: Axes in which to draw the plot. If None, use the currently active Axes.
    :param kwargs: Additional keyword arguments passed to the underlying plotting function.
    :return: The Axes object with the plot drawn onto it.
    """
    if ax is None:
        _, ax = plt.subplots()

    dtype1 = data[feature_1].dtype
    dtype2 = data[feature_2].dtype

    if _is_categorical_like(dtype1):
        _plot_categorical_feature1(feature_1, feature_2, data, dtype2, ax, **kwargs)
    elif pd.api.types.is_datetime64_any_dtype(dtype1):
        _plot_datetime_feature1(feature_1, feature_2, data, dtype2, ax, **kwargs)
    elif _is_categorical_like(dtype2):
        _plot_categorical_feature2(feature_1, feature_2, data, ax, **kwargs)
    elif pd.api.types.is_datetime64_any_dtype(dtype2):
        _plot_datetime_feature2(feature_1, feature_2, data, ax, **kwargs)
    else:
        _plot_numeric_features(feature_1, feature_2, data, ax, **kwargs)

    return ax


def _is_categorical_like(dtype):
    """Check if the dtype is categorical-like (categorical, boolean, or object)."""
    return (
        isinstance(dtype, pd.CategoricalDtype)
        or pd.api.types.is_bool_dtype(dtype)
        or pd.api.types.is_object_dtype(dtype)
    )


def _plot_categorical_feature1(feature_1, feature_2, data, dtype2, ax, **kwargs):
    """Plot when the first feature is categorical-like."""
    dup_df = pd.DataFrame()
    dup_df[feature_1] = _copy_series_or_keep_top_10(data[feature_1])

    if _is_categorical_like(dtype2):
        _plot_categorical_vs_categorical(feature_1, feature_2, dup_df, data, ax, **kwargs)
    elif pd.api.types.is_datetime64_any_dtype(dtype2):
        _plot_categorical_vs_datetime(feature_1, feature_2, dup_df, data, ax, **kwargs)
    else:
        _plot_categorical_vs_numeric(feature_1, feature_2, dup_df, data, ax, **kwargs)


def _plot_datetime_feature1(feature_1, feature_2, data, dtype2, ax, **kwargs):
    """Plot when the first feature is datetime."""
    if _is_categorical_like(dtype2):
        _plot_datetime_vs_categorical(feature_1, feature_2, data, ax, **kwargs)
    else:
        ax.plot(data[feature_1], data[feature_2], **kwargs)
        ax.set_xlabel(feature_1)
        ax.set_ylabel(feature_2)


def _plot_categorical_feature2(feature_1, feature_2, data, ax, **kwargs):
    """Plot when the second feature is categorical-like."""
    dup_df = pd.DataFrame()
    dup_df[feature_2] = _copy_series_or_keep_top_10(data[feature_2])
    dup_df[feature_1] = data[feature_1]
    chart = sns.boxplot(x=feature_2, y=feature_1, data=dup_df, ax=ax, **kwargs)
    ticks_loc = chart.get_xticks()  # Get the tick positions
    chart.xaxis.set_major_locator(ticker.FixedLocator(ticks_loc))  # Explicitly set the tick positions
    chart.set_xticklabels(chart.get_xticklabels(), rotation=45, ha="right")


def _plot_datetime_feature2(feature_1, feature_2, data, ax, **kwargs):
    """Plot when the second feature is datetime."""
    ax.plot(data[feature_2], data[feature_1], **kwargs)
    ax.set_xlabel(feature_2)
    ax.set_ylabel(feature_1)


def _plot_numeric_features(feature_1, feature_2, data, ax, **kwargs):
    """Plot when both features are numeric."""
    ax.scatter(data[feature_1], data[feature_2], **kwargs)
    ax.set_xlabel(feature_1)
    ax.set_ylabel(feature_2)


def _plot_categorical_vs_categorical(feature_1, feature_2, dup_df, data, ax, **kwargs):
    """Plot when both features are categorical-like."""
    dup_df[feature_2] = _copy_series_or_keep_top_10(data[feature_2])
    group_feature_1 = dup_df[feature_1].unique().tolist()
    ax.hist(
        [dup_df.loc[dup_df[feature_1] == value, feature_2] for value in group_feature_1],
        label=group_feature_1,
        **kwargs,
    )
    ax.set_xlabel(feature_1)
    ax.legend(title=feature_2)


def _plot_categorical_vs_datetime(feature_1, feature_2, dup_df, data, ax, **kwargs):
    """Plot when the first feature is categorical-like, and the second is datetime."""
    dup_df[feature_2] = data[feature_2].apply(dates.date2num)
    chart = sns.violinplot(x=feature_2, y=feature_1, data=dup_df, ax=ax)
    ticks_loc = chart.get_xticks()
    chart.xaxis.set_major_locator(ticker.FixedLocator(ticks_loc))
    chart.set_xticklabels(chart.get_xticklabels(), rotation=45, ha="right")
    ax.xaxis.set_major_formatter(_convert_numbers_to_dates)


def _plot_categorical_vs_numeric(feature_1, feature_2, dup_df, data, ax, **kwargs):
    """Plot when the first feature is categorical-like and the second is numeric."""
    dup_df[feature_2] = data[feature_2]
    chart = sns.boxplot(x=feature_1, y=feature_2, data=dup_df, ax=ax, **kwargs)
    ticks_loc = chart.get_xticks()  # Get the tick positions
    chart.xaxis.set_major_locator(ticker.FixedLocator(ticks_loc))  # Explicitly set the tick positions
    chart.set_xticklabels(chart.get_xticklabels(), rotation=45, ha="right")


def _plot_datetime_vs_categorical(feature_1, feature_2, data, ax, **kwargs):
    """Plot when the first feature is datetime and the second is categorical-like."""
    dup_df = pd.DataFrame()
    dup_df[feature_1] = data[feature_1].apply(dates.date2num)
    dup_df[feature_2] = _copy_series_or_keep_top_10(data[feature_2])
    chart = sns.violinplot(x=feature_1, y=feature_2, data=dup_df, ax=ax)
    ticks_loc = chart.get_xticks()
    chart.xaxis.set_major_locator(ticker.FixedLocator(ticks_loc))
    chart.set_xticklabels(chart.get_xticklabels(), rotation=45, ha="right")
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
    return dates.num2date(x).strftime("%Y-%m-%d %H:%M")


def extract_statistics_dataframe_per_label(df: pd.DataFrame, feature_name: str, label_name: str) -> pd.DataFrame:
    """Calculate comprehensive statistical metrics for a specified feature grouped by label.

    This method computes various statistical measures for a given numerical feature, broken down by unique
    values in the specified label column. The statistics include count, null count,
    mean, standard deviation, min/max values and multiple percentiles.

    :param df: Input pandas DataFrame containing the data
    :param feature_name: Name of the column to calculate statistics on
    :param label_name: Name of the column to group by
    :return: DataFrame with statistical metrics for each unique label value, with columns:
            - count: Number of non-null observations
            - null_count: Number of null values
            - mean: Average value
            - min: Minimum value
            - 1_percentile: 1st percentile
            - 5_percentile: 5th percentile
            - 25_percentile: 25th percentile
            - median: 50th percentile
            - 75_percentile: 75th percentile
            - 95_percentile: 95th percentile
            - 99_percentile: 99th percentile
            - max: Maximum value

    :raises KeyError: If feature_name or label_name is not found in DataFrame
    :raises TypeError: If feature_name column is not numeric
    """
    if feature_name not in df.columns:
        raise KeyError(f"Feature column '{feature_name}' not found in DataFrame")
    if label_name not in df.columns:
        raise KeyError(f"Label column '{label_name}' not found in DataFrame")
    if not pd.api.types.is_numeric_dtype(df[feature_name]):
        raise TypeError(f"Feature column '{feature_name}' must be numeric")

    # Define percentile functions with consistent naming

    def percentile_1(x):
        return safe_percentile(x, 1)

    def percentile_5(x):
        return safe_percentile(x, 5)

    def percentile_25(x):
        return safe_percentile(x, 25)

    def percentile_75(x):
        return safe_percentile(x, 75)

    def percentile_95(x):
        return safe_percentile(x, 95)

    def percentile_99(x):
        return safe_percentile(x, 99)

    return df.groupby([label_name], observed=True)[feature_name].agg(
        [
            ("count", "count"),
            ("null_count", lambda x: x.isnull().sum()),
            ("mean", "mean"),
            ("min", "min"),
            ("1_percentile", percentile_1),
            ("5_percentile", percentile_5),
            ("25_percentile", percentile_25),
            ("median", "median"),
            ("75_percentile", percentile_75),
            ("95_percentile", percentile_95),
            ("99_percentile", percentile_99),
            ("max", "max"),
        ]
    )
