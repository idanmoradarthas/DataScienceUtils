from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from matplotlib import pyplot as plt

from ds_utils.preprocess import (
    visualize_correlations,
    plot_features_interaction,
    plot_correlation_dendrogram,
    visualize_feature,
    get_correlated_features
)

RESOURCES_PATH = Path(__file__).parent / "resources"
BASELINE_DIR = Path(__file__).parent / "baseline_images" / "test_preprocess"


@pytest.fixture()
def loan_data():
    return pd.read_csv(RESOURCES_PATH.joinpath("loan_final313.csv"),
                       encoding="latin1", parse_dates=["issue_d"]).drop("id", axis=1)


@pytest.fixture()
def data_1m():
    return pd.read_csv(RESOURCES_PATH.joinpath("data.1M.zip"), compression='zip')


@pytest.fixture()
def daily_min_temperatures():
    return pd.read_csv(RESOURCES_PATH.joinpath("daily-min-temperatures.csv"), parse_dates=["Date"])


@pytest.fixture(autouse=True)
def setup_teardown():
    yield
    plt.cla()
    plt.close(plt.gcf())


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR)
@pytest.mark.parametrize("feature",
                         ["emp_length_int", "issue_d", "loan_condition_cat", "income_category", "home_ownership",
                          "purpose"],
                         ids=["float", "datetime", "int", "object", "category", "category_more_than_10_categories"])
def test_visualize_feature(loan_data, feature, request):
    """Test visualize_feature function for different feature types."""
    visualize_feature(loan_data[feature])

    if request.node.callspec.id in ["datetime", "object", "category"]:
        plt.gcf().set_size_inches(10, 8)
    elif request.node.callspec.id == "category_more_than_10_categories":
        plt.gcf().set_size_inches(11, 11)

    return plt.gcf()


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR)
def test_visualize_feature_float_exist_ax(loan_data):
    fig, ax = plt.subplots()
    ax.set_title("My ax")

    visualize_feature(loan_data["emp_length_int"], ax=ax)

    assert ax.get_title() == "My ax"
    fig.set_size_inches(10, 8)
    return fig


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR)
def test_visualize_feature_bool(loan_data):
    """Test visualize_feature function for boolean data."""
    loan_dup = pd.DataFrame()
    loan_dup["term 36 months"] = loan_data["term"].apply(lambda term: term == " 36 months").astype("bool")
    visualize_feature(loan_dup["term 36 months"])
    return plt.gcf()


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR)
def test_visualize_feature_remove_na(loan_data):
    """Test visualize_feature function with NA values removed."""
    loan_data_dup = pd.concat([
        loan_data[["emp_length_int"]],
        pd.DataFrame([np.nan] * 250, columns=["emp_length_int"])
    ], ignore_index=True).sample(frac=1, random_state=0)

    visualize_feature(loan_data_dup["emp_length_int"], remove_na=True)
    return plt.gcf()


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR)
@pytest.mark.parametrize("use_existing_ax", [False, True], ids=["default", "exist_ax"])
def test_visualize_correlations(data_1m, use_existing_ax):
    """Test visualize_correlations function with and without existing axes."""
    if use_existing_ax:
        _, ax = plt.subplots()
        ax.set_title("My ax")
        visualize_correlations(data_1m, ax=ax)
        assert ax.get_title() == "My ax"
    else:
        visualize_correlations(data_1m)

    fig = plt.gcf()
    fig.set_size_inches(14, 9)
    return fig


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR)
@pytest.mark.parametrize("feature1, feature2, data_fixture", [
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
    ("x12", "x12", "data_1m")
], ids=["both_numeric", "numeric_categorical", "numeric_categorical_reverse", "numeric_boolean", "both_categorical",
        "categorical_bool", "datetime_numeric", "datetime_numeric_reverse", "datetime_datetime", "datetime_categorical",
        "datetime_categorical_reverse", "both_bool"])
def test_plot_relationship_between_features(feature1, feature2, data_fixture, request):
    """Test plot_features_interaction function for various feature combinations."""
    data = request.getfixturevalue(data_fixture)
    plot_features_interaction(feature1, feature2, data)

    if request.node.callspec.id in ["numeric_categorical", "numeric_categorical_reverse"]:
        plt.gcf().set_size_inches(14, 9)
    elif request.node.callspec.id == "numeric_boolean":
        plt.gcf().set_size_inches(8, 7)
    elif request.node.callspec.id == "both_categorical":
        plt.gcf().set_size_inches(9, 5)
    elif request.node.callspec.id in ["datetime_numeric", "datetime_numeric_reverse"]:
        plt.gcf().set_size_inches(18, 8)
    elif request.node.callspec.id in ["datetime_categorical", "datetime_categorical_reverse"]:
        plt.gcf().set_size_inches(10, 11.5)

    return plt.gcf()


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR)
@pytest.mark.parametrize("feature1, feature2",
                         [("issue_d", "loan_condition_cat"), ("loan_condition_cat", "issue_d")],
                         ids=["default", "reverse"])
def test_plot_relationship_between_features_datetime_bool(loan_data, feature1, feature2):
    df = pd.DataFrame()
    df["loan_condition_cat"] = loan_data["loan_condition_cat"].astype("bool")
    df["issue_d"] = loan_data["issue_d"]

    plot_features_interaction(feature1, feature2, df)

    fig = plt.gcf()
    fig.set_size_inches(10, 11.5)
    return fig


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR)
def test_plot_relationship_between_features_both_numeric_exist_ax(data_1m):
    fig, ax = plt.subplots()
    ax.set_title("My ax")

    plot_features_interaction("x4", "x5", data_1m, ax=ax)
    assert ax.get_title() == "My ax"
    return fig


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR)
@pytest.mark.parametrize("use_existing_ax", [False, True], ids=["default", "exist_ax"])
def test_plot_correlation_dendrogram(data_1m, use_existing_ax):
    """Test plot_correlation_dendrogram function with and without existing axes."""
    if use_existing_ax:
        _, ax = plt.subplots()
        ax.set_title("My ax")
        plot_correlation_dendrogram(data_1m, ax=ax)
        assert ax.get_title() == "My ax"
    else:
        plot_correlation_dendrogram(data_1m)

    return plt.gcf()


def test_get_correlated_features():
    """Test get_correlated_features function."""
    data_frame = pd.read_csv(RESOURCES_PATH.joinpath("loan_final313_small.csv"))
    correlation = get_correlated_features(data_frame, data_frame.columns.drop("loan_condition_cat").tolist(),
                                          "loan_condition_cat", 0.95)
    correlation_expected = pd.DataFrame([
        {'level_0': 'income_category_Low', 'level_1': 'income_category_Medium',
         'level_0_level_1_corr': 1.0,
         'level_0_target_corr': 0.11821656093586508,
         'level_1_target_corr': 0.11821656093586504},
        {'level_0': 'term_ 36 months', 'level_1': 'term_ 60 months',
         'level_0_level_1_corr': 1.0,
         'level_0_target_corr': 0.223606797749979,
         'level_1_target_corr': 0.223606797749979},
        {'level_0': 'interest_payments_High', 'level_1': 'interest_payments_Low',
         'level_0_level_1_corr': 1.0, 'level_0_target_corr': 0.13363062095621223,
         'level_1_target_corr': 0.13363062095621223}
    ])
    pd.testing.assert_frame_equal(correlation_expected, correlation)


def test_get_correlated_features_empty_result():
    """Test get_correlated_features function with an empty result."""
    data_frame = pd.read_csv(RESOURCES_PATH.joinpath("clothing_classification_train.csv"))
    expected_warning = "Correlation threshold 0.95 was too high. An empty frame was returned"
    with pytest.warns(UserWarning, match=expected_warning):
        correlation = get_correlated_features(data_frame,
                                              ["Clothing ID", "Age", "Title", "Review Text", "Rating",
                                               "Recommended IND", "Positive Feedback Count", "Division Name",
                                               "Department Name"],
                                              "Class Name", 0.95)
    correlation_expected = pd.DataFrame(
        columns=['level_0', 'level_1', 'level_0_level_1_corr', 'level_0_target_corr', 'level_1_target_corr'])
    pd.testing.assert_frame_equal(correlation_expected, correlation)
