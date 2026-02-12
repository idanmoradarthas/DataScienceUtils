"""Tests for the preprocess statistics utilities."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ds_utils.math_utils import safe_percentile
from ds_utils.preprocess.statistics import (
    compute_mutual_information,
    extract_statistics_dataframe_per_label,
    get_correlated_features,
)

RESOURCES_PATH = Path(__file__).parent.parent / "resources"


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
    mock_safe_percentile = mocker.patch("ds_utils.preprocess.statistics.safe_percentile", wraps=safe_percentile)

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
    """Test basic functionality for compute_mutual_information."""
    df = data_1m.copy()
    features = df.columns.tolist()
    rng = np.random.default_rng(seed=42)
    df["target"] = rng.choice(["class_1", "class_2", "class_3"], size=len(df))

    expected = pd.DataFrame(
        [
            ["x3", 0.001766052134277718],
            ["x2", 0.001479947096839851],
            ["x1", 0.0009494384943877776],
            ["x8", 0.00036423417047570794],
            ["x4", 0.00014870988232429383],
            ["x5", 0.00013023297539671574],
            ["x7", 9.545793450264087e-05],
            ["x10", 3.387523951139948e-06],
            ["x12", 7.467128754767849e-07],
            ["x6", 0.0],
            ["x9", 0.0],
            ["x11", 0.0],
        ],
        columns=["feature_name", "mi_score"],
    )
    results = compute_mutual_information(df, features, "target", random_state=42)
    pd.testing.assert_frame_equal(expected, results)


def test_compute_mutual_information_empty_features_list():
    """Test compute_mutual_information with empty features list."""
    df = pd.DataFrame({"feature1": [1, 2, 3, 4, 5], "target": [0, 1, 0, 1, 0]})
    with pytest.raises(ValueError, match="features list cannot be empty"):
        compute_mutual_information(df, [], "target")


def test_compute_mutual_information_missing_label_column():
    """Test compute_mutual_information with missing label column."""
    df = pd.DataFrame({"num_high_corr": [1, 2, 3, 4, 5], "target": [0, 1, 0, 1, 0]})
    with pytest.raises(KeyError, match="Label column 'nonexistent' not found"):
        compute_mutual_information(df, ["num_high_corr"], "nonexistent")


def test_compute_mutual_information_missing_feature_columns():
    """Test compute_mutual_information with missing feature columns."""
    df = pd.DataFrame({"num_high_corr": [1, 2, 3, 4, 5], "target": [0, 1, 0, 1, 0]})
    with pytest.raises(KeyError, match="Features not found in DataFrame: \\['nonexistent1', 'nonexistent2'\\]"):
        compute_mutual_information(df, ["num_high_corr", "nonexistent1", "nonexistent2"], "target")


def test_compute_mutual_information_all_null_target():
    """Test compute_mutual_information when target column has only null values."""
    df = pd.DataFrame({"feature1": [1, 2, 3, 4, 5], "target": [np.nan, np.nan, np.nan, np.nan, np.nan]})

    with pytest.raises(ValueError, match="Label column 'target' contains only null values"):
        compute_mutual_information(df, ["feature1"], "target")


@pytest.mark.parametrize(
    ("valid_feature_name", "valid_feature_data", "missing_feature_name", "missing_dtype"),
    [
        ("numerical_feature", [1, 2, 3, 4, 5], "missing_numerical", float),
        ("categorical_feature", ["A", "B", "A", "B", "A"], "missing_categorical", object),
        ("boolean_feature", [True, False, True, False, True], "missing_boolean", "boolean"),
    ],
    ids=["numerical", "categorical", "boolean"],
)
def test_compute_mutual_information_fully_missing_feature(
    valid_feature_name, valid_feature_data, missing_feature_name, missing_dtype
):
    """Test that fully missing features are handled, get a score of 0, and raise a warning."""
    df = pd.DataFrame(
        {
            valid_feature_name: valid_feature_data,
            missing_feature_name: [np.nan] * 5,
            "target": [0, 1, 0, 1, 0],
        }
    )
    df[missing_feature_name] = df[missing_feature_name].astype(missing_dtype)
    features = [valid_feature_name, missing_feature_name]

    expected_warning = f"Features \\['{missing_feature_name}'\\] contain only null values and will be ignored."
    with pytest.warns(UserWarning, match=expected_warning):
        mi_scores = compute_mutual_information(df, features, "target", random_state=42)

    assert missing_feature_name in mi_scores["feature_name"].values
    assert mi_scores.loc[mi_scores["feature_name"] == missing_feature_name, "mi_score"].iloc[0] == 0.0


def test_compute_mutual_information_all_features_fully_missing():
    """Test compute_mutual_information when ALL features contain only null values.

    This tests the edge case where all features are missing, which should return
    a DataFrame with all features having MI score of 0, sorted by feature_name.
    """
    df = pd.DataFrame(
        {
            "missing_feature1": [np.nan, np.nan, np.nan, np.nan, np.nan],
            "missing_feature2": [np.nan, np.nan, np.nan, np.nan, np.nan],
            "missing_feature3": [np.nan, np.nan, np.nan, np.nan, np.nan],
            "target": [0, 1, 0, 1, 0],
        }
    )

    features = ["missing_feature1", "missing_feature2", "missing_feature3"]

    expected_warning = (
        r"Features \['missing_feature1', 'missing_feature2', 'missing_feature3'\] "
        r"contain only null values and will be ignored."
    )

    with pytest.warns(UserWarning, match=expected_warning):
        mi_scores = compute_mutual_information(df, features, "target", random_state=42)

    # Verify the exact DataFrame structure (checks length, values, and ordering)
    expected_df = pd.DataFrame(
        {
            "feature_name": ["missing_feature1", "missing_feature2", "missing_feature3"],
            "mi_score": [0.0, 0.0, 0.0],
        }
    )
    pd.testing.assert_frame_equal(mi_scores, expected_df)
