"""Tests for the visualize_feature function."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from matplotlib import pyplot as plt

from ds_utils.preprocess.visualization import visualize_feature

BASELINE_DIR = Path(__file__).parent.parent / "baseline_images" / "test_preprocess" / "test_visualize_feature"


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR)
@pytest.mark.parametrize(
    "feature",
    ["emp_length_int", "issue_d", "loan_condition_cat"],
    ids=["float", "datetime", "int"],
)
def test_visualize_feature_float_datetime_int(loan_data, feature, request):
    """Test visualize_feature function for float, datetime and int features."""
    visualize_feature(loan_data[feature])

    if request.node.callspec.id in ["datetime"]:
        plt.gcf().set_size_inches(10, 11)
    return plt.gcf()


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR)
@pytest.mark.parametrize(
    ("feature", "show_counts"),
    [
        ("income_category", True),
        ("home_ownership", True),
        ("purpose", True),
        ("income_category", False),
        ("home_ownership", False),
        ("purpose", False),
    ],
    ids=[
        "object_show_counts",
        "category_show_counts",
        "category_more_than_10_categories_show_counts",
        "object_no_show_counts",
        "category_no_show_counts",
        "category_more_than_10_categories_no_show_counts",
    ],
)
def test_visualize_feature_object(loan_data, feature, show_counts, request):
    """Test visualize_feature function for object and category features."""
    if request.node.callspec.id in ["object_show_counts", "object_no_show_counts"]:
        visualize_feature(loan_data[feature], order=["Low", "Medium", "High"], show_counts=show_counts)
    else:
        visualize_feature(loan_data[feature], show_counts=show_counts)

    if request.node.callspec.id in [
        "object_show_counts",
        "category_show_counts",
        "object_no_show_counts",
        "category_no_show_counts",
    ]:
        plt.gcf().set_size_inches(10, 9)
    elif request.node.callspec.id in [
        "category_more_than_10_categories_show_counts",
        "category_more_than_10_categories_no_show_counts",
    ]:
        plt.gcf().set_size_inches(11, 14)
    return plt.gcf()


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR)
def test_visualize_feature_float_exist_ax(loan_data):
    """Test visualize_feature with a float feature on an existing Axes."""
    fig, ax = plt.subplots()
    ax.set_title("My ax")

    visualize_feature(loan_data["emp_length_int"], ax=ax)

    assert ax.get_title() == "My ax"
    fig.set_size_inches(10, 8)
    return fig


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR)
def test_visualize_feature_float_exclude_outliers(loan_data):
    """Test visualize_feature function with outliers excluded."""
    visualize_feature(loan_data["emp_length_int"], include_outliers=False, outlier_iqr_multiplier=0.01)
    return plt.gcf()


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR)
@pytest.mark.parametrize("show_counts", [True, False], ids=["show_counts", "no_show_counts"])
def test_visualize_feature_bool(loan_data, show_counts):
    """Test visualize_feature function for boolean data."""
    loan_dup = pd.DataFrame()
    loan_dup["term 36 months"] = loan_data["term"].apply(lambda term: term == " 36 months").astype("bool")
    visualize_feature(loan_dup["term 36 months"], order=["True", "False"], show_counts=show_counts)
    return plt.gcf()


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR)
def test_visualize_feature_remove_na(loan_data):
    """Test visualize_feature function with NA values removed."""
    loan_data_dup = pd.concat(
        [loan_data[["emp_length_int"]], pd.DataFrame([np.nan] * 250, columns=["emp_length_int"])], ignore_index=True
    ).sample(frac=1, random_state=0)

    visualize_feature(loan_data_dup["emp_length_int"], remove_na=True)
    return plt.gcf()


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR)
def test_visualize_feature_datetime_heatmap_sunday_start(loan_data):
    """Test visualize_feature with datetime data starting week on Sunday."""
    visualize_feature(loan_data["issue_d"], first_day_of_week="Sunday")
    plt.gcf().set_size_inches(10, 8)
    return plt.gcf()


def test_visualize_feature_datetime_invalid_first_day():
    """Test visualize_feature with invalid first_day_of_week parameter."""
    dates = pd.date_range("2023-01-01", "2023-12-31", freq="D")
    series = pd.Series(dates, name="test_dates")

    with pytest.raises(ValueError, match="first_day_of_week must be one of"):
        visualize_feature(series, first_day_of_week="InvalidDay")


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR)
def test_visualize_feature_datetime_missing_days_adds_columns():
    """Ensure heatmap adds missing weekday columns when some days are absent in data."""
    dates_mon = pd.date_range("2024-01-01", periods=4, freq="W-MON")
    dates_tue = pd.date_range("2024-01-02", periods=4, freq="W-TUE")
    combined = np.concatenate([dates_mon.values, dates_tue.values])
    series = pd.Series(pd.to_datetime(combined), name="test_dates").sort_values().reset_index(drop=True)

    visualize_feature(series, first_day_of_week="Monday")
    plt.gcf().set_size_inches(10, 8)
    return plt.gcf()


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR)
@pytest.mark.parametrize("order", ["count_desc", "count_asc", "alpha_asc", "alpha_desc"])
def test_visualize_feature_object_order(loan_data, order):
    """Test visualize_feature function with different order parameters."""
    visualize_feature(loan_data["purpose"], order=order)
    plt.gcf().set_size_inches(11, 14)
    return plt.gcf()


def test_visualize_feature_object_order_error():
    """Test visualize_feature function with an invalid order parameter."""
    with pytest.raises(ValueError, match="Invalid order string: 'invalid_order'. Must be one of: "):
        visualize_feature(pd.Series(["A", "B", "C"]), order="invalid_order")


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR)
@pytest.mark.parametrize(
    "order", [["High", "Low", "Medium"], ["Medium", "High", "Low"]], ids=["high_low_medium", "medium_high_low"]
)
def test_visualize_feature_object_order_list(loan_data, order):
    """Test visualize_feature function with a list of order parameters."""
    visualize_feature(loan_data["income_category"], order=order)
    plt.gcf().set_size_inches(11, 14)
    return plt.gcf()


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR)
def test_visualize_feature_object_null_values(loan_data):
    """Test visualize_feature function with null values in an object feature."""
    series = pd.concat(
        [loan_data["income_category"], pd.Series([np.nan] * 500, name="income_category")], ignore_index=True
    )
    visualize_feature(series, remove_na=False)
    plt.gcf().set_size_inches(8, 7)
    return plt.gcf()
