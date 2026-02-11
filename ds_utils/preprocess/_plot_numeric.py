from typing import Optional

import pandas as pd
import seaborn as sns
from matplotlib import axes


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


def _plot_numeric_features(feature_1, feature_2, data, remove_na, ax, **kwargs):
    """Plot when both features are numeric.

    If remove_na is False, adds marginal rug plots showing where missing values occur.
    """
    # Get complete cases for main scatter plot
    complete_data = data.dropna(subset=[feature_1, feature_2])

    ax.scatter(complete_data[feature_1], complete_data[feature_2], **kwargs)
    ax.set_xlabel(feature_1)
    ax.set_ylabel(feature_2)

    # Add marginal rug plots for missing values if not removed
    if not remove_na:
        # Cases where feature_1 is present but feature_2 is missing
        missing_f2 = data[data[feature_2].isna() & data[feature_1].notna()]
        if len(missing_f2) > 0:
            y_min = ax.get_ylim()[0]
            ax.scatter(
                missing_f2[feature_1],
                [y_min] * len(missing_f2),
                marker="|",
                s=100,
                alpha=0.5,
                color="red",
                label=f"{feature_2} missing",
            )

        # Cases where feature_2 is present but feature_1 is missing
        missing_f1 = data[data[feature_1].isna() & data[feature_2].notna()]
        if len(missing_f1) > 0:
            x_min = ax.get_xlim()[0]
            ax.scatter(
                [x_min] * len(missing_f1),
                missing_f1[feature_2],
                marker="_",
                s=100,
                alpha=0.5,
                color="orange",
                label=f"{feature_1} missing",
            )

        # Add legend if there are any missing values
        if len(missing_f2) > 0 or len(missing_f1) > 0:
            ax.legend(loc="best", framealpha=0.9)

    return ax
