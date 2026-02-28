"""Utilities for plotting relationships involving categorical features.

This module contains helper functions used internally by the preprocess
package to visualize categorical variables against other types such as
categorical, datetime, and numeric features.
"""

from typing import List, Optional, Union

from matplotlib import axes, dates, ticker
import numpy as np
import pandas as pd
import seaborn as sns

from ds_utils.preprocess._plot_formatters import _convert_numbers_to_dates
from ds_utils.preprocess._plot_utils import _copy_series_or_keep_top_10, _is_categorical_like


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


def _plot_categorical_feature1(
    categorical_feature,
    feature_2,
    data,
    dtype2,
    include_outliers,
    outlier_iqr_multiplier,
    show_ratios,
    remove_na,
    ax,
    **kwargs,
):
    """Plot when the first feature is categorical-like."""
    if _is_categorical_like(dtype2):
        ax = _plot_categorical_vs_categorical(
            categorical_feature, feature_2, data, show_ratios, remove_na, ax, **kwargs
        )
    elif pd.api.types.is_datetime64_any_dtype(dtype2):
        ax = _plot_categorical_vs_datetime(categorical_feature, feature_2, data, remove_na, ax, **kwargs)
    else:
        ax = _plot_categorical_vs_numeric(
            categorical_feature,
            feature_2,
            data,
            outlier_iqr_multiplier,
            include_outliers,
            remove_na,
            ax,
            **kwargs,
        )
    return ax


def _plot_categorical_vs_categorical(feature_1, feature_2, data, show_ratios, remove_na, ax, **kwargs):
    """Plot when both features are categorical-like.

    When remove_na is False, missing values are handled by:
    - Adding a "Missing" category for any NaN values in either feature
    - Including these in the crosstab/heatmap display

    When remove_na is True, rows with missing values in either feature are excluded.
    """
    dup_df = pd.DataFrame()
    dup_df[feature_1] = _copy_series_or_keep_top_10(data[feature_1])
    dup_df[feature_2] = _copy_series_or_keep_top_10(data[feature_2])

    # Handle missing values based on remove_na parameter
    if not remove_na:
        # Replace NaN with "Missing" category for both features
        if dup_df[feature_1].isna().any():
            dup_df[feature_1] = dup_df[feature_1].fillna("Missing")
        if dup_df[feature_2].isna().any():
            dup_df[feature_2] = dup_df[feature_2].fillna("Missing")

        # Create crosstab with all values (including "Missing" categories)
        crosstab = pd.crosstab(dup_df[feature_1], dup_df[feature_2], dropna=False)
    else:
        # Remove rows where either feature is missing
        dup_df = dup_df.dropna(subset=[feature_1, feature_2])
        crosstab = pd.crosstab(dup_df[feature_1], dup_df[feature_2], dropna=True)

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


def _plot_categorical_vs_datetime(categorical_feature, datetime_feature, data, remove_na, ax, **kwargs):
    """Plot when one feature is categorical-like and the other is datetime.

    When remove_na is False, missing values are handled as follows:
    - Missing categorical values: added as "Missing" category (creates an extra violin)
    - Missing datetime values: shown as a separate violin at the edge of the plot
    """
    dup_df = pd.DataFrame()
    dup_df[categorical_feature] = _copy_series_or_keep_top_10(data[categorical_feature])

    # Handle missing categorical values by adding "Missing" category
    if not remove_na and dup_df[categorical_feature].isna().any():
        dup_df[categorical_feature] = dup_df[categorical_feature].fillna("Missing")

    # Initialize variables for missing datetime handling
    missing_datetime_value = None
    has_missing_datetime = False

    # Convert datetime to numeric, handling missing values
    if not remove_na:
        # For missing datetime values, we'll use a special marker value
        # First, convert non-missing datetimes to numeric
        datetime_numeric = data[datetime_feature].apply(lambda x: dates.date2num(x) if pd.notna(x) else np.nan)

        # Get the range of valid datetime values to place "Missing" at the edge
        valid_datetime_numeric = datetime_numeric.dropna()
        has_missing_datetime = datetime_numeric.isna().any()

        if len(valid_datetime_numeric) > 0:
            datetime_min = valid_datetime_numeric.min()
            datetime_max = valid_datetime_numeric.max()
            datetime_range = datetime_max - datetime_min
            # Place "Missing" at the right edge, slightly offset (at least 1 day or 10% of range)
            missing_datetime_value = datetime_max + max(datetime_range * 0.1, 1.0)
        elif has_missing_datetime:
            # If no valid datetimes but we have missing ones, use a default value
            missing_datetime_value = dates.date2num(pd.Timestamp.now())

        # Replace NaN datetime values with the special marker if we have missing values
        if has_missing_datetime and missing_datetime_value is not None:
            dup_df[datetime_feature] = datetime_numeric.fillna(missing_datetime_value)
        else:
            dup_df[datetime_feature] = datetime_numeric
    else:
        # Remove rows where either feature is missing
        dup_df = dup_df.dropna(subset=[categorical_feature])
        datetime_numeric = data[datetime_feature].apply(dates.date2num)
        dup_df[datetime_feature] = datetime_numeric
        dup_df = dup_df.dropna(subset=[datetime_feature])

    # Create violin plot with all data (complete + missing datetime)
    chart = sns.violinplot(x=datetime_feature, y=categorical_feature, data=dup_df, ax=ax, **kwargs)

    # Format x-axis ticks for datetime
    ticks_loc = chart.get_xticks()

    if not remove_na and has_missing_datetime and missing_datetime_value is not None:
        # Check if we have missing datetime data

        # Separate valid datetime ticks from the missing datetime position
        # Use a threshold to identify the missing datetime position
        valid_ticks = [t for t in ticks_loc if abs(t - missing_datetime_value) > 0.1]
        # Add the missing datetime position if it's not already in ticks
        if not any(abs(t - missing_datetime_value) < 0.1 for t in ticks_loc):
            valid_ticks.append(missing_datetime_value)
        valid_ticks = sorted(valid_ticks)

        chart.xaxis.set_major_locator(ticker.FixedLocator(valid_ticks))
        tick_labels = [
            dates.num2date(t).strftime("%Y-%m-%d %H:%M") if abs(t - missing_datetime_value) > 0.1 else "Missing"
            for t in valid_ticks
        ]
        chart.set_xticklabels(tick_labels, rotation=45, ha="right")
    else:
        # Standard datetime formatting
        chart.xaxis.set_major_locator(ticker.FixedLocator(ticks_loc))
        chart.set_xticklabels(chart.get_xticklabels(), rotation=45, ha="right")
        ax.xaxis.set_major_formatter(_convert_numbers_to_dates)

    ax.set_xlabel(datetime_feature)
    ax.set_ylabel(categorical_feature)

    return ax


def _plot_categorical_vs_numeric(
    categorical_feature,
    numeric_feature,
    data,
    outlier_iqr_multiplier,
    include_outliers,
    remove_na,
    ax,
    **kwargs,
):
    """Plot when the first feature is categorical-like and the second is numeric.

    Renders a violin plot of the numeric feature for each category. When
    ``include_outliers`` is False, numeric values outside the IQR fence
    [Q1 - k*IQR, Q3 + k*IQR] are trimmed, where ``k`` is ``outlier_iqr_multiplier``.

    When ``remove_na`` is False, missing values are handled as follows:
    - Missing categorical values get a "Missing" category
    - Missing numeric values are shown with rug plots at the bottom of each category
    """
    dup_df = pd.DataFrame()
    dup_df[categorical_feature] = _copy_series_or_keep_top_10(data[categorical_feature])
    dup_df[numeric_feature] = data[numeric_feature]

    # Handle missing categorical values by adding "Missing" category
    if not remove_na and dup_df[categorical_feature].isna().any():
        dup_df[categorical_feature] = dup_df[categorical_feature].fillna("Missing")

    # Apply outlier filtering if requested
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

    # Create main violin plot (only with non-NA numeric values)
    df_plot_complete = df_plot.dropna(subset=[numeric_feature])
    sns.violinplot(
        x=categorical_feature, y=numeric_feature, hue=categorical_feature, data=df_plot_complete, ax=ax, **kwargs
    )

    # If remove_na is False, add rug plots for missing numeric values
    if not remove_na:
        missing_numeric = df_plot[df_plot[numeric_feature].isna()]
        if len(missing_numeric) > 0:
            # Get the y-axis limits to place rug marks at the bottom
            y_min = ax.get_ylim()[0]

            # Get unique categories and their x-axis positions
            categories = df_plot_complete[categorical_feature].unique()
            cat_to_pos = {cat: i for i, cat in enumerate(categories)}

            # Plot rug marks for each category that has missing numeric values
            for cat in missing_numeric[categorical_feature].unique():
                if cat in cat_to_pos:
                    count = len(missing_numeric[missing_numeric[categorical_feature] == cat])
                    x_pos = cat_to_pos[cat]

                    # Add small horizontal jitter for visibility when there are multiple missing values
                    jitter = np.random.uniform(-0.1, 0.1, count)

                    ax.scatter(
                        [x_pos] * count + jitter,
                        [y_min] * count,
                        marker="|",
                        s=100,
                        alpha=0.6,
                        color="red",
                        linewidths=2,
                        label=f"{numeric_feature} missing"
                        if cat == missing_numeric[categorical_feature].unique()[0]
                        else "",
                    )

            # Add legend if we plotted any missing values
            if len(missing_numeric) > 0:
                ax.legend(loc="best", framealpha=0.9)

    ax.set_xlabel(categorical_feature.replace("_", " ").title())
    ax.set_ylabel(numeric_feature.replace("_", " ").title())
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    return ax
