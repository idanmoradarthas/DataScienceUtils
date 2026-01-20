"""Data preprocessing utilities."""

from typing import Callable, List, Optional, Union
import warnings

from matplotlib import axes, dates, pyplot as plt, ticker
import numpy as np
from numpy.random import RandomState
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
import seaborn as sns
from sklearn.base import TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from ds_utils.math_utils import safe_percentile


def _plot_clean_violin_distribution(
    series: pd.Series, include_outliers: bool, outlier_iqr_multiplier: float, ax: Optional[axes.Axes] = None, **kwargs
) -> axes.Axes:
    """Plot a violin distribution for a numeric series with optional outlier trimming.

    When ``include_outliers`` is False, values outside the IQR fence are removed
    before plotting. The fence is defined as
    [Q1 - k * IQR, Q3 + k * IQR], where ``k`` is ``outlier_iqr_multiplier``, and
    the bounds are clipped to the observed min/max of the series.

    :param series: Numeric series to visualize. NA handling is expected upstream.
    :param include_outliers: Whether to include values outside the IQR fence.
    :param outlier_iqr_multiplier: Multiplier ``k`` used to compute the IQR fence.
    :param ax: Matplotlib Axes to draw on. If None, callers should provide one upstream.
    :param kwargs: Additional keyword arguments passed to ``seaborn.violinplot``.
    :return: The Axes object with the violin plot.
    """
    if include_outliers:
        series_plot = series.copy()
    else:
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        min_series_value = series.min()
        max_series_value = series.max()
        iqr = q3 - q1
        lower_bound = max(min_series_value, q1 - outlier_iqr_multiplier * iqr)
        upper_bound = min(max_series_value, q3 + outlier_iqr_multiplier * iqr)
        series_plot = series[(series >= lower_bound) & (series <= upper_bound)].copy()

    sns.violinplot(y=series_plot, hue=None, legend=False, ax=ax, **kwargs)

    ax.set_xticks([])
    ax.set_ylabel("Values")
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    return ax


def _plot_datetime_heatmap(feature_series: pd.Series, first_day_of_week: str, ax: axes.Axes, **kwargs) -> axes.Axes:
    """Plot a 2D heatmap for datetime features showing day-of-week vs year-week patterns.

    :param feature_series: The datetime series to visualize.
    :param first_day_of_week: First day of the week for the heatmap X-axis.
    :param ax: Matplotlib Axes to draw on.
    :param kwargs: Additional keyword arguments passed to seaborn's heatmap function.
    :return: The Axes object with the heatmap.
    """
    # Validate first_day_of_week parameter
    valid_days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    if first_day_of_week not in valid_days:
        raise ValueError(f"first_day_of_week must be one of {valid_days}, got '{first_day_of_week}'")

    # Create day of week order starting with first_day_of_week
    day_index = valid_days.index(first_day_of_week)
    day_order = valid_days[day_index:] + valid_days[:day_index]

    # Create DataFrame with date, day of week, year, and week number
    df = (
        feature_series.to_frame("date")
        .assign(
            day_of_week=lambda x: x["date"].dt.day_name(),
            year=lambda x: x["date"].dt.year,
            week_number=lambda x: x["date"].dt.isocalendar().week,
        )
        .assign(year_week=lambda x: x["year"].astype(str) + "-W" + x["week_number"].astype(str).str.zfill(2))
        .groupby(["year_week", "day_of_week"])
        .size()
        .unstack(fill_value=0)
    )

    # Ensure all days of the week are present as columns, reordered according to day_order
    for day in day_order:
        if day not in df.columns:
            df[day] = 0

    # Reorder columns to match day_order (columns = day of week, rows = year-week)
    df = df.reindex(columns=day_order)

    # Create heatmap with annotations to show numbers in cells
    sns.heatmap(df, cmap="Blues", ax=ax, annot=True, fmt="d", **kwargs)
    ax.set_xlabel("Day of Week")
    ax.set_ylabel("Year-Week")

    return ax


def _copy_series_or_keep_top_10(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.map({True: "True", False: "False"})
    if len(series.unique()) > 10:
        top10 = series.value_counts().nlargest(10).index
        return series.map(lambda x: x if x in top10 else "Other values")
    return series


def _plot_count_bar(
    value_counts: pd.Series, order: Optional[Union[List[str], str]], show_counts: bool, ax: axes.Axes, **kwargs
) -> axes.Axes:
    """Plot a bar chart for categorical data with optional ordering and count labels.

    :param value_counts: Series containing value counts to plot
    :param order: Order specification for categories (None, string, or list)
    :param show_counts: Whether to display count values on top of bars
    :param ax: Axes to draw on
    :param kwargs: Additional arguments passed to ax.bar
    :return: The Axes object with the bar plot
    """
    # Apply ordering based on the order parameter
    if order is None:
        value_counts = value_counts.sort_index()
    elif isinstance(order, str):
        if order == "count_desc":
            value_counts = value_counts.sort_values(ascending=False)
        elif order == "count_asc":
            value_counts = value_counts.sort_values(ascending=True)
        elif order == "alpha_asc":
            value_counts = value_counts.sort_index(ascending=True)
        elif order == "alpha_desc":
            value_counts = value_counts.sort_index(ascending=False)
        else:
            raise ValueError(
                f"Invalid order string: '{order}'. Must be one of: 'count_desc', 'count_asc', 'alpha_asc', 'alpha_desc'"
            )
    elif isinstance(order, list):
        # Filter to only include categories present in the data
        valid_order = [cat for cat in order if cat in value_counts.index]
        # Add any missing categories from value_counts that weren't in order
        missing_cats = [cat for cat in value_counts.index if cat not in valid_order]
        full_order = valid_order + missing_cats
        value_counts = value_counts.reindex(full_order)

    # Create bar plot using matplotlib
    bars = ax.bar(range(len(value_counts)), value_counts.values, **kwargs)
    ax.set_xticks(range(len(value_counts)))
    ax.set_xticklabels(value_counts.index)
    ax.set_ylabel("Count")

    # Add count labels if requested
    if show_counts:
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{int(height):,}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

    return ax


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
    data: pd.DataFrame,
    feature_1: str,
    feature_2: str,
    *,
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
    - If both are categorical-like: overlaid histograms per category.
    - If one is categorical-like and the other numeric: violin plot by category.

    For the categorical-vs-numeric case, you can optionally trim outliers from the
    numeric feature using an IQR fence [Q1 - k*IQR, Q3 + k*IQR], where ``k`` is
    controlled by ``outlier_iqr_multiplier``.

    :param data: The input DataFrame where each feature is a column.
    :param feature_1: Name of the first feature.
    :param feature_2: Name of the second feature.
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

    dtype1 = data[feature_1].dtype
    dtype2 = data[feature_2].dtype

    if _is_categorical_like(dtype1):
        ax = _plot_categorical_feature1(
            feature_1,
            feature_2,
            data,
            dtype2,
            include_outliers,
            outlier_iqr_multiplier,
            show_ratios,
            ax,
            **kwargs,
        )
    elif pd.api.types.is_datetime64_any_dtype(dtype1):
        ax = _plot_datetime_feature1(feature_1, feature_2, data, dtype2, ax, **kwargs)
    elif _is_categorical_like(dtype2):
        ax = _plot_categorical_vs_numeric(
            feature_2, feature_1, data, outlier_iqr_multiplier, include_outliers, ax, **kwargs
        )
    elif pd.api.types.is_datetime64_any_dtype(dtype2):
        ax = _plot_xy(feature_2, feature_1, data, ax, **kwargs)
    else:
        ax = _plot_numeric_features(feature_1, feature_2, data, ax, **kwargs)

    return ax


def _is_categorical_like(dtype):
    """Check if the dtype is categorical-like (categorical, boolean, or object)."""
    return (
        isinstance(dtype, pd.CategoricalDtype)
        or pd.api.types.is_bool_dtype(dtype)
        or pd.api.types.is_object_dtype(dtype)
    )


def _plot_categorical_feature1(
    categorical_feature,
    feature_2,
    data,
    dtype2,
    include_outliers,
    outlier_iqr_multiplier,
    show_ratios,
    ax,
    **kwargs,
):
    """Plot when the first feature is categorical-like."""
    if _is_categorical_like(dtype2):
        ax = _plot_categorical_vs_categorical(categorical_feature, feature_2, data, show_ratios, ax, **kwargs)
    elif pd.api.types.is_datetime64_any_dtype(dtype2):
        ax = _plot_categorical_vs_datetime(categorical_feature, feature_2, data, ax, **kwargs)
    else:
        ax = _plot_categorical_vs_numeric(
            categorical_feature,
            feature_2,
            data,
            outlier_iqr_multiplier,
            include_outliers,
            ax,
            **kwargs,
        )
    return ax


def _plot_xy(datetime_feature, other_feature, data, ax, **kwargs):
    ax.plot(data[datetime_feature], data[other_feature], **kwargs)
    ax.set_xlabel(datetime_feature)
    ax.set_ylabel(other_feature)
    return ax


def _plot_datetime_feature1(datetime_feature, feature_2, data, dtype2, ax, **kwargs):
    """Plot when the first feature is datetime."""
    if _is_categorical_like(dtype2):
        ax = _plot_categorical_vs_datetime(feature_2, datetime_feature, data, ax, **kwargs)
    else:
        ax = _plot_xy(datetime_feature, feature_2, data, ax, **kwargs)
    return ax


def _plot_numeric_features(feature_1, feature_2, data, ax, **kwargs):
    """Plot when both features are numeric."""
    ax.scatter(data[feature_1], data[feature_2], **kwargs)
    ax.set_xlabel(feature_1)
    ax.set_ylabel(feature_2)
    return ax


def _plot_categorical_vs_categorical(feature_1, feature_2, data, show_ratios, ax, **kwargs):
    """Plot when both features are categorical-like."""
    dup_df = pd.DataFrame()
    dup_df[feature_1] = _copy_series_or_keep_top_10(data[feature_1])
    dup_df[feature_2] = _copy_series_or_keep_top_10(data[feature_2])

    crosstab = pd.crosstab(dup_df[feature_1], dup_df[feature_2])

    if show_ratios:
        total = crosstab.sum().sum()
        crosstab_display = crosstab / total
        fmt = ".3f"
    else:
        crosstab_display = crosstab
        fmt = "d"

    sns.heatmap(crosstab_display, annot=True, fmt=fmt, ax=ax, **kwargs)
    ax.set_xlabel(feature_2)
    ax.set_ylabel(feature_1)

    if show_ratios:
        ax.set_title(f"{feature_1} vs {feature_2} (Proportions)")

    return ax


def _plot_categorical_vs_datetime(categorical_feature, datetime_feature, data, ax, **kwargs):
    """Plot when one feature is categorical-like and the other is datetime.

    Draws a violin plot across time buckets on the x-axis with categories on the
    y-axis. This unified function expects the categorical feature name first and
    the datetime feature name second.
    """
    dup_df = pd.DataFrame()
    dup_df[datetime_feature] = data[datetime_feature].apply(dates.date2num)
    dup_df[categorical_feature] = _copy_series_or_keep_top_10(data[categorical_feature])
    chart = sns.violinplot(x=datetime_feature, y=categorical_feature, data=dup_df, ax=ax, **kwargs)
    ticks_loc = chart.get_xticks()
    chart.xaxis.set_major_locator(ticker.FixedLocator(ticks_loc))
    chart.set_xticklabels(chart.get_xticklabels(), rotation=45, ha="right")
    ax.xaxis.set_major_formatter(_convert_numbers_to_dates)
    return ax


def _plot_categorical_vs_numeric(
    categorical_feature,
    numeric_feature,
    data,
    outlier_iqr_multiplier,
    include_outliers,
    ax,
    **kwargs,
):
    """Plot when the first feature is categorical-like and the second is numeric.

    Renders a violin plot of the numeric feature for each category. When
    ``include_outliers`` is False, numeric values outside the IQR fence
    [Q1 - k*IQR, Q3 + k*IQR] are trimmed, where ``k`` is ``outlier_iqr_multiplier``.
    """
    dup_df = pd.DataFrame()
    dup_df[categorical_feature] = _copy_series_or_keep_top_10(data[categorical_feature])
    dup_df[numeric_feature] = data[numeric_feature]

    if include_outliers:
        df_plot = dup_df.copy()
    else:
        q1 = dup_df[numeric_feature].quantile(0.25)
        q3 = dup_df[numeric_feature].quantile(0.75)
        min_series_value = dup_df[numeric_feature].min()
        max_series_value = dup_df[numeric_feature].max()
        iqr = q3 - q1
        lower_bound = max(min_series_value, q1 - outlier_iqr_multiplier * iqr)
        upper_bound = min(max_series_value, q3 + outlier_iqr_multiplier * iqr)
        df_plot = dup_df[(dup_df[numeric_feature] >= lower_bound) & (dup_df[numeric_feature] <= upper_bound)].copy()

    sns.violinplot(x=categorical_feature, y=numeric_feature, hue=categorical_feature, data=df_plot, ax=ax, **kwargs)

    ax.set_xlabel(categorical_feature.replace("_", " ").title())
    ax.set_ylabel(numeric_feature.replace("_", " ").title())
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    return ax


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


def compute_mutual_information(
    df: pd.DataFrame,
    features: List[str],
    label_col: str,
    *,
    n_neighbors: int = 3,
    random_state: Optional[Union[int, RandomState]] = None,
    n_jobs: Optional[int] = None,
    numerical_imputer: TransformerMixin = SimpleImputer(strategy="mean"),
    discrete_imputer: TransformerMixin = SimpleImputer(strategy="most_frequent"),
    discrete_encoder: TransformerMixin = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
) -> pd.DataFrame:
    """Compute mutual information scores between features and a target label.

    This function calculates mutual information scores for specified features with respect to a target
    label column. Features are automatically categorized as numerical or discrete (boolean/categorical)
    and preprocessed accordingly before computing mutual information.

    Mutual information measures the mutual dependence between two variables - higher scores indicate
    stronger relationships between the feature and the target label.

    :param df: Input pandas DataFrame containing the features and label
    :param features: List of column names to compute mutual information for
    :param label_col: Name of the target label column
    :param n_neighbors: Number of neighbors to use for MI estimation for continuous variables. Higher values
                        reduce variance of the estimation, but could introduce a bias.
    :param random_state: Random state for reproducible results. Can be int or RandomState instance
    :param n_jobs: The number of jobs to use for computing the mutual information. The parallelization is done
                   on the columns. `None` means 1 unless in a `joblib.parallel_backend` context. ``-1`` means
                   using all processors.
    :param numerical_imputer: Sklearn-compatible transformer for numerical features (default: mean imputation)
    :param discrete_imputer: Sklearn-compatible transformer for discrete features (default: most frequent imputation)
    :param discrete_encoder: Sklearn-compatible transformer for encoding discrete features (default: ordinal encoding
                            with unknown value handling)
    :return: DataFrame with columns 'feature_name' and 'mi_score', sorted by MI score (descending)

    :raises KeyError: If any feature or label_col is not found in DataFrame
    :raises ValueError: If features list is empty or label_col contains non-finite values
    """
    # Input validation
    if not features:
        raise ValueError("features list cannot be empty")

    if label_col not in df.columns:
        raise KeyError(f"Label column '{label_col}' not found in DataFrame")

    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        raise KeyError(f"Features not found in DataFrame: {missing_features}")

    if df[label_col].isnull().all():
        raise ValueError(f"Label column '{label_col}' contains only null values")

    # Identify and separate fully missing features
    fully_missing_features = [f for f in features if df[f].isnull().all()]
    if fully_missing_features:
        warnings.warn(f"Features {fully_missing_features} contain only null values and will be ignored.", UserWarning)
    features_to_process = [f for f in features if f not in fully_missing_features]

    # Create a DataFrame for missing features with MI score of 0
    missing_mi_df = pd.DataFrame({"feature_name": fully_missing_features, "mi_score": 0.0})

    # If all features were missing or no features to process, return the DataFrame of missing features
    if not features_to_process:
        return missing_mi_df.sort_values(by="feature_name").reset_index(drop=True)

    # Identify feature types for the features that will be processed
    df_processed = df[features_to_process].copy()
    numerical_features = df_processed.select_dtypes(include=[np.number]).columns.tolist()
    boolean_features = df_processed.select_dtypes(include=["bool", "boolean"]).columns.tolist()
    categorical_features = df_processed.select_dtypes(include=["object", "category"]).columns.tolist()

    # SimpleImputer does not support boolean dtype, so convert to object
    for col in boolean_features:
        df_processed[col] = df_processed[col].astype(object)

    # Create preprocessing pipelines
    numerical_transformer = Pipeline(steps=[("imputer", numerical_imputer)], memory=None, verbose=False)
    discrete_transformer = Pipeline(
        steps=[("imputer", discrete_imputer), ("encoder", discrete_encoder)], memory=None, verbose=False
    )

    # Setup column transformer
    transformers = []
    if numerical_features:
        transformers.append(("num", numerical_transformer, numerical_features))
    if boolean_features or categorical_features:
        transformers.append(("discrete", discrete_transformer, boolean_features + categorical_features))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0,
        n_jobs=n_jobs,
        transformer_weights=None,
        verbose=False,
        verbose_feature_names_out=True,
    )

    # Create discrete features mask for mutual_info_classif
    discrete_features_mask = [False] * len(numerical_features) + [True] * (
        len(boolean_features) + len(categorical_features)
    )

    # Create ordered feature names list matching the preprocessed data
    ordered_feature_names = numerical_features + boolean_features + categorical_features

    # Apply preprocessing
    x_preprocessed = preprocessor.fit_transform(df_processed[ordered_feature_names])
    y = df[label_col]

    # Compute mutual information scores
    mi_scores = mutual_info_classif(
        X=x_preprocessed,
        y=y,
        n_neighbors=n_neighbors,
        copy=True,
        random_state=random_state,
        n_jobs=n_jobs,
        discrete_features=discrete_features_mask,
    )

    # Create results DataFrame for processed features
    processed_mi_df = pd.DataFrame({"feature_name": ordered_feature_names, "mi_score": mi_scores})

    # Combine with missing features' results
    final_mi_df = pd.concat([processed_mi_df, missing_mi_df], ignore_index=True)

    return final_mi_df.sort_values(by="mi_score", ascending=False).reset_index(drop=True)
