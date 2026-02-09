from typing import Optional, Union, List

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import axes, dates, pyplot as plt, ticker


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


def _plot_datetime_vs_numeric(datetime_feature, other_feature, data, remove_na, ax, **kwargs):
    """Plot datetime vs numeric feature.

    When remove_na is False, missing values are handled as follows:
    - Missing numeric values: shown as markers on the x-axis (y=0 or bottom of plot)
    - Missing datetime values: shown as a rug plot on the right margin
    - Different colors/markers distinguish the two types of missingness
    """
    # Get complete cases for main plot
    complete_data = data.dropna(subset=[datetime_feature, other_feature])

    if len(complete_data) > 0:
        ax.plot(complete_data[datetime_feature], complete_data[other_feature], **kwargs)

    ax.set_xlabel(datetime_feature)
    ax.set_ylabel(other_feature)

    # Handle missing values if not removed
    # Skip missing value visualization if both features are the same column
    if not remove_na and datetime_feature != other_feature:
        has_plotted_missing = False
        missing_numeric = data[data[other_feature].isna() & data[datetime_feature].notna()]
        missing_datetime = data[data[datetime_feature].isna() & data[other_feature].notna()]

        # If no complete cases, we must establish limits manually so the two missing groups align
        if len(complete_data) == 0:
            # Handle Y limits from missing_datetime data
            if len(missing_datetime) > 0:
                y_vals = missing_datetime[other_feature].dropna()
                if len(y_vals) > 0:
                    y_min = y_vals.min()
                    y_max = y_vals.max()
                    if y_min == y_max:
                        y_min -= 1
                        y_max += 1
                    ax.set_ylim(y_min, y_max)

            # Handle X limits from missing_numeric data
            if len(missing_numeric) > 0:
                valid_dates = missing_numeric[datetime_feature].dropna()
                if len(valid_dates) > 0:
                    x_min = dates.date2num(valid_dates.min())
                    x_max = dates.date2num(valid_dates.max())
                    if x_min == x_max:
                        x_min -= 1.0  # 1 day
                        x_max += 1.0
                    ax.set_xlim(x_min, x_max)

        # Plot cases where datetime is present but numeric is missing
        if len(missing_numeric) > 0:
            # Filter out any rows where datetime_feature is also NaN
            missing_numeric_clean = missing_numeric[missing_numeric[datetime_feature].notna()]
            if len(missing_numeric_clean) > 0:
                y_min = ax.get_ylim()[0]
                ax.scatter(
                    missing_numeric_clean[datetime_feature],
                    [y_min] * len(missing_numeric_clean),
                    marker="|",
                    s=100,
                    alpha=0.6,
                    color="red",
                    label=f"{other_feature} missing",
                    zorder=5,
                )
                has_plotted_missing = True

        # Plot cases where numeric is present but datetime is missing
        if len(missing_datetime) > 0:
            # Determine logic for X limits fallback if not already set
            x_min, x_max = ax.get_xlim()

            # If limits are still default (0, 1) or invalid because no X data existed, set fallback
            # Simple check: if complete_data empty AND missing_numeric empty, we need defaults.
            if len(complete_data) == 0 and len(missing_numeric) == 0:
                x_min = dates.date2num(pd.Timestamp.now() - pd.Timedelta(days=30))
                x_max = dates.date2num(pd.Timestamp.now())
                ax.set_xlim(x_min, x_max)

            # Re-fetch limits in case they changed
            x_min, x_max = ax.get_xlim()
            y_min, y_max = ax.get_ylim()

            # Plot rug marks for missing datetime values at the right edge
            x_range = x_max - x_min
            x_rug = x_max + x_range * 0.02  # 2% offset from right edge
            ax.scatter(
                [x_rug] * len(missing_datetime),
                missing_datetime[other_feature],
                marker="_",
                s=100,
                alpha=0.6,
                color="orange",
                label=f"{datetime_feature} missing",
                zorder=5,
            )

            # Extend xlim slightly to accommodate the rug plot
            ax.set_xlim(x_min, x_max + x_range * 0.05)
            has_plotted_missing = True

        # Add legend if we plotted any missing values
        if has_plotted_missing:
            ax.legend(loc="best", framealpha=0.9)

    return ax


def _plot_datetime_vs_datetime(datetime_feature_1, datetime_feature_2, data, remove_na, ax, **kwargs):
    """Plot when both features are datetime.

    When remove_na is False, missing values are handled as follows:
    - Complete cases: shown as line/scatter plot
    - Missing datetime_feature_2 values: shown as rug plot on x-axis (bottom margin)
    - Missing datetime_feature_1 values: shown as rug plot on y-axis (left margin)
    - Different colors/markers distinguish the two types of missingness
    """
    # Get complete cases for main plot
    complete_data = data.dropna(subset=[datetime_feature_1, datetime_feature_2])

    if len(complete_data) > 0:
        # Use scatter plot for datetime vs datetime (can also use line plot)
        ax.scatter(complete_data[datetime_feature_1], complete_data[datetime_feature_2], **kwargs)

    ax.set_xlabel(datetime_feature_1)
    ax.set_ylabel(datetime_feature_2)

    # Format both axes as datetime
    ax.xaxis.set_major_formatter(dates.DateFormatter("%Y-%m-%d"))
    ax.yaxis.set_major_formatter(dates.DateFormatter("%Y-%m-%d"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
    plt.setp(ax.yaxis.get_majorticklabels(), rotation=0)

    # Handle missing values if not removed
    if not remove_na and datetime_feature_1 != datetime_feature_2:
        has_plotted_missing = False

        # Cases where datetime_feature_1 is present but datetime_feature_2 is missing
        missing_f2 = data[data[datetime_feature_2].isna() & data[datetime_feature_1].notna()]
        if len(missing_f2) > 0:
            # Get y-axis limits to place rug marks at the bottom
            if len(complete_data) > 0:
                y_min, y_max = ax.get_ylim()
            else:
                # If no complete data, use range from available datetime_feature_2 values
                available_f2 = data[datetime_feature_2].dropna()
                if len(available_f2) > 0:
                    y_min = dates.date2num(available_f2.min())
                    y_max = dates.date2num(available_f2.max())
                    if y_min == y_max:
                        y_min -= 1
                        y_max += 1
                else:
                    # Fallback to default range if no datetime_feature_2 values available
                    y_min = dates.date2num(pd.Timestamp.now() - pd.Timedelta(days=30))
                    y_max = dates.date2num(pd.Timestamp.now())
                ax.set_ylim(y_min, y_max)

            # Place rug marks slightly below the bottom of the plot
            y_range = y_max - y_min
            y_rug = y_min - y_range * 0.02  # 2% offset below bottom

            ax.scatter(
                missing_f2[datetime_feature_1],
                [y_rug] * len(missing_f2),
                marker="|",
                s=100,
                alpha=0.6,
                color="red",
                label=f"{datetime_feature_2} missing",
                zorder=5,
            )
            # Extend ylim slightly to accommodate the rug plot
            ax.set_ylim(y_min - y_range * 0.05, y_max)
            has_plotted_missing = True

        # Cases where datetime_feature_2 is present but datetime_feature_1 is missing
        missing_f1 = data[data[datetime_feature_1].isna() & data[datetime_feature_2].notna()]
        if len(missing_f1) > 0:
            # Get x-axis limits to place rug marks on the left
            if len(complete_data) > 0:
                x_min, x_max = ax.get_xlim()
            else:
                # If no complete data, use range from available datetime_feature_1 values
                available_f1 = data[datetime_feature_1].dropna()
                if len(available_f1) > 0:
                    x_min = dates.date2num(available_f1.min())
                    x_max = dates.date2num(available_f1.max())
                    if x_min == x_max:
                        x_min -= 1
                        x_max += 1
                else:
                    # Fallback to default range if no datetime_feature_1 values available
                    x_min = dates.date2num(pd.Timestamp.now() - pd.Timedelta(days=30))
                    x_max = dates.date2num(pd.Timestamp.now())
                ax.set_xlim(x_min, x_max)

            # Place rug marks slightly to the left of the plot
            x_range = x_max - x_min
            x_rug = x_min - x_range * 0.02  # 2% offset to the left

            ax.scatter(
                [x_rug] * len(missing_f1),
                missing_f1[datetime_feature_2],
                marker="_",
                s=100,
                alpha=0.6,
                color="orange",
                label=f"{datetime_feature_1} missing",
                zorder=5,
            )
            # Extend xlim slightly to accommodate the rug plot
            ax.set_xlim(x_min - x_range * 0.05, x_max)
            has_plotted_missing = True

        # Add legend if we plotted any missing values
        if has_plotted_missing:
            ax.legend(loc="best", framealpha=0.9)

    return ax


def _plot_datetime_feature1(datetime_feature, feature_2, data, dtype2, remove_na, ax, **kwargs):
    """Plot when the first feature is datetime."""
    if _is_categorical_like(dtype2):
        ax = _plot_categorical_vs_datetime(feature_2, datetime_feature, data, remove_na, ax, **kwargs)
    elif pd.api.types.is_datetime64_any_dtype(dtype2):
        # Both features are datetime - use specialized datetime vs datetime plot
        ax = _plot_datetime_vs_datetime(datetime_feature, feature_2, data, remove_na, ax, **kwargs)
    else:
        ax = _plot_datetime_vs_numeric(datetime_feature, feature_2, data, remove_na, ax, **kwargs)
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


@plt.FuncFormatter
def _convert_numbers_to_dates(x, pos):
    return dates.num2date(x).strftime("%Y-%m-%d %H:%M")
