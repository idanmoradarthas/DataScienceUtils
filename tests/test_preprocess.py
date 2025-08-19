"""Tests for data preprocessing functions and visualizations."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from matplotlib import pyplot as plt

from ds_utils.math_utils import safe_percentile
from ds_utils.preprocess import (
    visualize_correlations,
    plot_features_interaction,
    plot_correlation_dendrogram,
    visualize_feature,
    get_correlated_features,
    extract_statistics_dataframe_per_label, compute_mutual_information,
)

RESOURCES_PATH = Path(__file__).parent / "resources"
BASELINE_DIR = Path(__file__).parent / "baseline_images" / "test_preprocess"


@pytest.fixture
def loan_data():
    """Load and return loan dataset for testing."""
    return pd.read_csv(RESOURCES_PATH.joinpath("loan_final313.csv"), encoding="latin1", parse_dates=["issue_d"]).drop(
        "id", axis=1
    )


@pytest.fixture
def data_1m():
    """Load and return 1M dataset for testing."""
    return pd.read_csv(RESOURCES_PATH.joinpath("data.1M.zip"), compression="zip")


@pytest.fixture
def daily_min_temperatures():
    """Load and return daily minimum temperatures dataset for testing."""
    return pd.read_csv(RESOURCES_PATH.joinpath("daily-min-temperatures.csv"), parse_dates=["Date"])


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Create a sample DataFrame for testing."""
    return pd.DataFrame(
        {
            "value": [1.0, 2.0, 3.0, 4.0, 5.0, np.nan, 7.0, 8.0, 9.0, 10.0],
            "category": ["A", "A", "A", "B", "B", "B", "C", "C", "C", "C"],
            "text_col": ["x", "y", "z", "x", "y", "z", "x", "y", "z", "x"],
        }
    )


@pytest.fixture(autouse=True)
def setup_teardown():
    """Set up and tear down for each test function in this module."""
    yield
    plt.cla()
    plt.close(plt.gcf())


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR)
@pytest.mark.parametrize(
    "feature",
    ["emp_length_int", "issue_d", "loan_condition_cat", "income_category", "home_ownership", "purpose"],
    ids=["float", "datetime", "int", "object", "category", "category_more_than_10_categories"],
)
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
    """Test visualize_feature with a float feature on an existing Axes."""
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
    loan_data_dup = pd.concat(
        [loan_data[["emp_length_int"]], pd.DataFrame([np.nan] * 250, columns=["emp_length_int"])], ignore_index=True
    ).sample(frac=1, random_state=0)

    visualize_feature(loan_data_dup["emp_length_int"], remove_na=True)
    return plt.gcf()


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR)
@pytest.mark.parametrize("use_existing_ax", [False, True], ids=["default", "exist_ax"])
def test_visualize_correlations(data_1m, use_existing_ax):
    """Test visualize_correlations function with and without existing axes."""
    corr = data_1m.apply(lambda x: x.factorize()[0]).corr(method="pearson", min_periods=1)
    if use_existing_ax:
        _, ax = plt.subplots()
        ax.set_title("My ax")
        visualize_correlations(corr, ax=ax)
        assert ax.get_title() == "My ax"
    else:
        visualize_correlations(corr)

    fig = plt.gcf()
    fig.set_size_inches(14, 9)
    return fig


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
        ("x12", "x12", "data_1m"),
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
        "both_bool",
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
        plt.gcf().set_size_inches(9, 5)
    elif request.node.callspec.id in ["datetime_numeric", "datetime_numeric_reverse"]:
        plt.gcf().set_size_inches(18, 8)
    elif request.node.callspec.id in ["datetime_categorical", "datetime_categorical_reverse"]:
        plt.gcf().set_size_inches(10, 11.5)

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
@pytest.mark.parametrize("use_existing_ax", [False, True], ids=["default", "exist_ax"])
def test_plot_correlation_dendrogram(data_1m, use_existing_ax):
    """Test plot_correlation_dendrogram function with and without existing axes."""
    corr = data_1m.apply(lambda x: x.factorize()[0]).corr(method="pearson", min_periods=1)
    if use_existing_ax:
        _, ax = plt.subplots()
        ax.set_title("My ax")
        plot_correlation_dendrogram(corr, ax=ax)
        assert ax.get_title() == "My ax"
    else:
        plot_correlation_dendrogram(corr)

    return plt.gcf()


def test_get_correlated_features():
    """Test get_correlated_features function."""
    correlations = pd.read_feather(RESOURCES_PATH.joinpath("loan_final313_small_corr.feather"))
    correlation = get_correlated_features(
        correlations, correlations.columns.drop("loan_condition_cat").tolist(), "loan_condition_cat", 0.95
    )
    correlation_expected = pd.DataFrame(
        [
            {
                "level_0": "income_category_Low",
                "level_1": "income_category_Medium",
                "level_0_level_1_corr": 1.0,
                "level_0_target_corr": 0.11821656093586508,
                "level_1_target_corr": 0.11821656093586504,
            },
            {
                "level_0": "term_ 36 months",
                "level_1": "term_ 60 months",
                "level_0_level_1_corr": 1.0,
                "level_0_target_corr": 0.223606797749979,
                "level_1_target_corr": 0.223606797749979,
            },
            {
                "level_0": "interest_payments_High",
                "level_1": "interest_payments_Low",
                "level_0_level_1_corr": 1.0,
                "level_0_target_corr": 0.13363062095621223,
                "level_1_target_corr": 0.13363062095621223,
            },
        ]
    )
    pd.testing.assert_frame_equal(correlation_expected, correlation)


def test_get_correlated_features_empty_result():
    """Test get_correlated_features function with an empty result."""
    correlations = pd.read_feather(RESOURCES_PATH.joinpath("clothing_classification_train_corr.feather"))
    expected_warning = "Correlation threshold 0.95 was too high. An empty frame was returned"
    with pytest.warns(UserWarning, match=expected_warning):
        correlation = get_correlated_features(
            correlations,
            [
                "Clothing ID",
                "Age",
                "Title",
                "Review Text",
                "Rating",
                "Recommended IND",
                "Positive Feedback Count",
                "Division Name",
                "Department Name",
            ],
            "Class Name",
            0.95,
        )
    correlation_expected = pd.DataFrame(
        columns=["level_0", "level_1", "level_0_level_1_corr", "level_0_target_corr", "level_1_target_corr"]
    )
    pd.testing.assert_frame_equal(correlation_expected, correlation)


def assert_series_called_with(mock_calls, expected_series, percentile):
    """Check if a pandas' Series was called with specific values."""
    for args, _ in mock_calls:
        series, p = args
        if p == percentile and isinstance(series, pd.Series) and series.equals(expected_series):
            return True
    return False


def test_extract_statistics_dataframe_per_label_basic_functionality(sample_df, mocker):
    """Test basic functionality and verify safe_percentile calls."""
    mock_safe_percentile = mocker.patch("ds_utils.preprocess.safe_percentile", wraps=safe_percentile)

    result = extract_statistics_dataframe_per_label(sample_df, "value", "category")

    # Check if all expected columns are present
    expected_columns = [
        "count",
        "null_count",
        "mean",
        "min",
        "1_percentile",
        "5_percentile",
        "25_percentile",
        "median",
        "75_percentile",
        "95_percentile",
        "99_percentile",
        "max",
    ]
    assert all(col in result.columns for col in expected_columns)

    # Verify safe_percentile was called correct number of times with right arguments
    assert mock_safe_percentile.call_count == 18  # 6 percentiles * 3 categories

    # Verify some specific calls
    expected_series_a = pd.Series([1.0, 2.0, 3.0])
    assert assert_series_called_with(
        mock_safe_percentile.call_args_list, expected_series_a, 1
    )  # Category A, 1st percentile
    assert assert_series_called_with(
        mock_safe_percentile.call_args_list, expected_series_a, 99
    )  # Category A, 99th percentile


@pytest.mark.parametrize(
    ("feature_name", "label_name", "exception", "message"),
    [
        ("invalid_col", "category", KeyError, "Feature column 'invalid_col' not found"),
        ("value", "invalid_col", KeyError, "Label column 'invalid_col' not found"),
        ("text_col", "category", TypeError, "Feature column 'text_col' must be numeric"),
    ],
    ids=["test_invalid_feature_name", "test_invalid_label_name", "test_non_numeric_feature"],
)
def test_extract_statistics_dataframe_per_label_exceptions(sample_df, feature_name, label_name, exception, message):
    """Test exceptions for extract_statistics_dataframe_per_label."""
    with pytest.raises(exception, match=message):
        extract_statistics_dataframe_per_label(sample_df, feature_name, label_name)

def test_compute_mutual_information(data_1m):
    """Test basic functionality for compute_mutual_information"""
    df = data_1m.copy()
    features = df.columns.tolist()
    rng = np.random.default_rng(seed=42)
    df["target"] = rng.choice(['class_1', 'class_2', 'class_3'], size=len(df))

    expected = pd.DataFrame([['x3', 0.001766052134277718], ['x2', 0.001479947096839851], ['x1', 0.0009494384943877776], ['x8', 0.00036423417047570794], ['x4', 0.00014870988232429383], ['x5', 0.00013023297539671574], ['x7', 9.545793450264087e-05], ['x10', 3.387523951139948e-06], ['x12', 7.467128754767849e-07], ['x6', 0.0], ['x9', 0.0], ['x11', 0.0]], columns=["feature_name", "mi_score"])
    results = compute_mutual_information(df, features, "target", random_state=42)
    pd.testing.assert_frame_equal(expected, results)


def test_compute_mutual_information_empty_features_list():
    """Test compute_mutual_information with empty features list."""
    df = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5],
        'target': [0, 1, 0, 1, 0]
    })
    with pytest.raises(ValueError, match="features list cannot be empty"):
        compute_mutual_information(df, [], 'target')


def test_compute_mutual_information_missing_label_column():
    """Test compute_mutual_information with missing label column."""
    df = pd.DataFrame({
        'num_high_corr': [1, 2, 3, 4, 5],
        'target': [0, 1, 0, 1, 0]
    })
    with pytest.raises(KeyError, match="Label column 'nonexistent' not found"):
        compute_mutual_information(df, ['num_high_corr'], 'nonexistent')


def test_compute_mutual_information_missing_feature_columns():
    """Test compute_mutual_information with missing feature columns."""
    df = pd.DataFrame({
        'num_high_corr': [1, 2, 3, 4, 5],
        'target': [0, 1, 0, 1, 0]
    })
    with pytest.raises(KeyError, match="Features not found in DataFrame: \\['nonexistent1', 'nonexistent2'\\]"):
        compute_mutual_information(
            df,
            ['num_high_corr', 'nonexistent1', 'nonexistent2'],
            'target'
        )


def test_compute_mutual_information_all_null_target():
    """Test compute_mutual_information when target column has only null values."""
    df = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5],
        'target': [np.nan, np.nan, np.nan, np.nan, np.nan]
    })

    with pytest.raises(ValueError, match="Label column 'target' contains only null values"):
        compute_mutual_information(df, ['feature1'], 'target')
