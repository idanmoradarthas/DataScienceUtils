"""Utilities for plotting relationships involving datetime features.

This module provides helper functions used internally by the preprocess
package to visualize datetime variables alone and in combination with
categorical and numeric features, including handling of missing values.
"""

from matplotlib import axes, dates, pyplot as plt
import pandas as pd
import seaborn as sns

from ds_utils.preprocess._plot_categorical import _plot_categorical_vs_datetime
from ds_utils.preprocess._plot_utils import _is_categorical_like


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
