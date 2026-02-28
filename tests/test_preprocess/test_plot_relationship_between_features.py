"""Test the plot_features_interaction function for visualizing relationships between features."""

from pathlib import Path

from matplotlib import pyplot as plt
import pandas as pd
import pytest

from ds_utils.preprocess.visualization import plot_features_interaction

BASELINE_DIR = Path(__file__).parents[1] / "baseline_images" / Path(__file__).parent.name / Path(__file__).stem


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR)
@pytest.mark.parametrize(
    ("feature1", "feature2", "data_fixture"),
    [
        ("x4", "x5", "data_1m"),
        ("x1", "x7", "data_1m"),
        ("x7", "x1", "data_1m"),
        ("x1", "x12", "data_1m"),
        ("x7", "x10", "data_1m"),
        ("x10", "x12", "data_1m"),
        ("Date", "Temp", "daily_min_temperatures"),
        ("Temp", "Date", "daily_min_temperatures"),
        ("issue_d", "issue_d", "loan_data"),
        ("issue_d", "home_ownership", "loan_data"),
        ("home_ownership", "issue_d", "loan_data"),
    ],
    ids=[
        "both_numeric",
        "numeric_categorical",
        "numeric_categorical_reverse",
        "numeric_boolean",
        "both_categorical",
        "categorical_bool",
        "datetime_numeric",
        "datetime_numeric_reverse",
        "datetime_datetime",
        "datetime_categorical",
        "datetime_categorical_reverse",
    ],
)
def test_plot_relationship_between_features(feature1, feature2, data_fixture, request):
    """Test plot_features_interaction function for various feature combinations."""
    data = request.getfixturevalue(data_fixture)
    plot_features_interaction(data, feature1, feature2)

    if request.node.callspec.id in ["numeric_categorical", "numeric_categorical_reverse"]:
        plt.gcf().set_size_inches(14, 9)
    elif request.node.callspec.id == "numeric_boolean":
        plt.gcf().set_size_inches(8, 7)
    elif request.node.callspec.id == "both_categorical":
        plt.gcf().set_size_inches(12, 5)
    elif request.node.callspec.id in ["datetime_numeric", "datetime_numeric_reverse"]:
        plt.gcf().set_size_inches(18, 8)
    elif request.node.callspec.id in ["datetime_categorical", "datetime_categorical_reverse"]:
        plt.gcf().set_size_inches(10, 11.5)

    return plt.gcf()


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR)
def test_plot_relationship_between_features_both_bool(loan_data):
    """Test interaction plot for two boolean features."""
    data = pd.DataFrame()
    data["is_home_ownership_rent"] = loan_data["home_ownership"] == "RENT"
    data["is_low_interest_payments"] = loan_data["interest_payments"] == "Low"
    plot_features_interaction(data, "is_home_ownership_rent", "is_low_interest_payments")
    return plt.gcf()


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR)
@pytest.mark.parametrize(
    ("feature1", "feature2"),
    [("issue_d", "loan_condition_cat"), ("loan_condition_cat", "issue_d")],
    ids=["default", "reverse"],
)
def test_plot_relationship_between_features_datetime_bool(loan_data, feature1, feature2):
    """Test interaction plot between a datetime and a boolean feature."""
    df = pd.DataFrame()
    df["loan_condition_cat"] = loan_data["loan_condition_cat"].astype("bool")
    df["issue_d"] = loan_data["issue_d"]

    plot_features_interaction(df, feature1, feature2)

    fig = plt.gcf()
    fig.set_size_inches(10, 11.5)
    return fig


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR)
def test_plot_relationship_between_features_both_numeric_exist_ax(data_1m):
    """Test interaction plot for two numeric features on an existing Axes."""
    fig, ax = plt.subplots()
    ax.set_title("My ax")

    plot_features_interaction(data_1m, "x4", "x5", ax=ax)
    assert ax.get_title() == "My ax"
    return fig


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR)
def test_plot_relationship_between_features_numeric_categorical_without_outliers(data_1m):
    """Test interaction plot for two numeric features without outliers."""
    plot_features_interaction(data_1m, "x1", "x7", include_outliers=False, outlier_iqr_multiplier=0.01)
    plt.gcf().set_size_inches(14, 9)
    return plt.gcf()
