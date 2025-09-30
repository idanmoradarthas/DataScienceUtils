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
        series_plot = series[(series >= lower_bound) & (series <= upper_bound)]

    sns.violinplot(y=series_plot, hue=None, legend=False, ax=ax, **kwargs)

    ax.set_xticks([])
    ax.set_ylabel("Values")
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    return ax


def visualize_feature(
    series: pd.Series,
    remove_na: bool = False,
    *,
    include_outliers: bool = True,
    outlier_iqr_multiplier: float = 1.5,
    ax: Optional[axes.Axes] = None,
    **kwargs,
) -> axes.Axes:
    """Visualize a pandas Series using an appropriate plot based on dtype.

    Behavior by dtype:
    - Float: draw a violin distribution. If ``include_outliers`` is False, values
      outside the IQR fence [Q1 - k*IQR, Q3 + k*IQR] with ``k=outlier_iqr_multiplier``
      are trimmed prior to plotting.
    - Datetime: draw a line plot of value counts over time (sorted by index).
    - Object/categorical/bool/int: draw a count plot. Extremely high-cardinality
      series may be reduced to their top categories internally.

    :param series: The data series to visualize.
    :param remove_na: If True, plot with NA values removed; otherwise include them.
    :param include_outliers: Whether to include outliers for float features.
    :param outlier_iqr_multiplier: IQR multiplier used to trim outliers for float features.
    :param ax: Axes in which to draw the plot. If None, a new one is created.
    :param kwargs: Extra keyword arguments forwarded to the underlying plotting function
                   (``seaborn.violinplot``, ``Series.plot``, or ``seaborn.countplot``).
    :return: The Axes object with the plot drawn onto it.
    """
    if ax is None:
        _, ax = plt.subplots()

    feature_series = series.dropna() if remove_na else series

    if pd.api.types.is_float_dtype(feature_series):
        ax = _plot_clean_violin_distribution(feature_series, include_outliers, outlier_iqr_multiplier, ax, **kwargs)
    elif pd.api.types.is_datetime64_any_dtype(feature_series):
        feature_series.value_counts().sort_index().plot(kind="line", ax=ax, **kwargs)
        labels = ax.get_xticks()
    else:
        sns.countplot(x=_copy_series_or_keep_top_10(feature_series), ax=ax, **kwargs)
        labels = ax.get_xticklabels()

    if not ax.get_title():
        ax.set_title(f"{feature_series.name} ({feature_series.dtype})")
        ax.set_xlabel("")

    # Skip tick relabeling for float (violin) plots where x-ticks are hidden
    if not pd.api.types.is_float_dtype(feature_series):
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
        _plot_categorical_vs_numeric(feature_2, feature_1, data, ax, **kwargs)
    elif pd.api.types.is_datetime64_any_dtype(dtype2):
        _plot_xy(feature_2, feature_1, data, ax, **kwargs)
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


def _plot_categorical_feature1(categorical_feature, feature_2, data, dtype2, ax, **kwargs):
    """Plot when the first feature is categorical-like."""
    if _is_categorical_like(dtype2):
        _plot_categorical_vs_categorical(categorical_feature, feature_2, data, ax, **kwargs)
    elif pd.api.types.is_datetime64_any_dtype(dtype2):
        _plot_categorical_vs_datetime(categorical_feature, feature_2, data, ax, **kwargs)
    else:
        _plot_categorical_vs_numeric(categorical_feature, feature_2, data, ax, **kwargs)


def _plot_xy(datetime_feature, other_feature, data, ax, **kwargs):
    ax.plot(data[datetime_feature], data[other_feature], **kwargs)
    ax.set_xlabel(datetime_feature)
    ax.set_ylabel(other_feature)

def _plot_datetime_feature1(datetime_feature, feature_2, data, dtype2, ax, **kwargs):
    """Plot when the first feature is datetime."""
    if _is_categorical_like(dtype2):
        _plot_categorical_vs_datetime(feature_2, datetime_feature, data, ax, **kwargs)
    else:
        _plot_xy(datetime_feature, feature_2, data, ax, **kwargs)


def _plot_numeric_features(feature_1, feature_2, data, ax, **kwargs):
    """Plot when both features are numeric."""
    ax.scatter(data[feature_1], data[feature_2], **kwargs)
    ax.set_xlabel(feature_1)
    ax.set_ylabel(feature_2)


def _plot_categorical_vs_categorical(feature_1, feature_2, data, ax, **kwargs):
    """Plot when both features are categorical-like."""
    dup_df = pd.DataFrame()
    dup_df[feature_1] = _copy_series_or_keep_top_10(data[feature_1])
    dup_df[feature_2] = _copy_series_or_keep_top_10(data[feature_2])
    group_feature_1 = dup_df[feature_1].unique().tolist()
    ax.hist(
        [dup_df.loc[dup_df[feature_1] == value, feature_2] for value in group_feature_1],
        label=group_feature_1,
        **kwargs,
    )
    ax.set_xlabel(feature_1)
    ax.legend(title=feature_2)


def _plot_categorical_vs_datetime(categorical_feature, datetime_feature, data, ax, **kwargs):
    """Plot when one feature is categorical-like and the other is datetime.

    This unified function expects the categorical feature name first and the
    datetime feature name second.
    """
    dup_df = pd.DataFrame()
    dup_df[datetime_feature] = data[datetime_feature].apply(dates.date2num)
    dup_df[categorical_feature] = _copy_series_or_keep_top_10(data[categorical_feature])
    chart = sns.violinplot(x=datetime_feature, y=categorical_feature, data=dup_df, ax=ax, **kwargs)
    ticks_loc = chart.get_xticks()
    chart.xaxis.set_major_locator(ticker.FixedLocator(ticks_loc))
    chart.set_xticklabels(chart.get_xticklabels(), rotation=45, ha="right")
    ax.xaxis.set_major_formatter(_convert_numbers_to_dates)


def _plot_categorical_vs_numeric(categorical_feature, numeric_feature, data, ax, **kwargs):
    """Plot when the first feature is categorical-like and the second is numeric."""
    dup_df = pd.DataFrame()
    dup_df[categorical_feature] = _copy_series_or_keep_top_10(data[categorical_feature])
    dup_df[numeric_feature] = data[numeric_feature]
    chart = sns.boxplot(x=categorical_feature, y=numeric_feature, data=dup_df, ax=ax, **kwargs)
    ticks_loc = chart.get_xticks()  # Get the tick positions
    chart.xaxis.set_major_locator(ticker.FixedLocator(ticks_loc))  # Explicitly set the tick positions
    chart.set_xticklabels(chart.get_xticklabels(), rotation=45, ha="right")


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

    # Identify feature types
    numerical_features = df[features].select_dtypes(include=[np.number]).columns.tolist()
    boolean_features = df[features].select_dtypes(include=[bool]).columns.tolist()
    categorical_features = df[features].select_dtypes(include=["object", "category"]).columns.tolist()

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
        remainder="drop",  # Drop any features not explicitly handled
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
    x_preprocessed = preprocessor.fit_transform(df[ordered_feature_names])
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

    # Create results DataFrame
    mi_df = pd.DataFrame({"feature_name": ordered_feature_names, "mi_score": mi_scores})

    return mi_df.sort_values(by="mi_score", ascending=False).reset_index(drop=True)
