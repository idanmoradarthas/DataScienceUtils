"""Tests for the generate_error_analysis_report function."""

import numpy as np
import pandas as pd
import pytest
from ds_utils.xai import generate_error_analysis_report


def test_generate_error_analysis_report_categorical():
    """Test basic functionality with categorical feature."""
    X = pd.DataFrame({"cat": ["A", "A", "B", "B", "C"]})
    y_true = np.array([0, 0, 1, 1, 0])
    y_pred = np.array([0, 1, 1, 0, 0])
    # A: 2 samples, 1 error -> error_rate 0.5
    # B: 2 samples, 1 error -> error_rate 0.5
    # C: 1 sample, 0 error -> error_rate 0.0

    report = generate_error_analysis_report(X, y_true, y_pred, sort_metric="group", ascending=True)

    expected = pd.DataFrame(
        {
            "feature": ["cat", "cat", "cat"],
            "group": ["A", "B", "C"],
            "count": [2, 2, 1],
            "error_count": [1, 1, 0],
            "error_rate": [0.5, 0.5, 0.0],
            "accuracy": [0.5, 0.5, 1.0],
        }
    )
    pd.testing.assert_frame_equal(report, expected)


def test_generate_error_analysis_report_numerical_binning():
    """Test numerical feature binning."""
    X = pd.DataFrame({"num": [1, 2, 11, 12]})
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    # bins=2 -> [0.989, 6.5] and [6.5, 12.0]
    # Group 1 (1, 2): 2 samples, 1 error (2 is error)
    # Group 2 (11, 12): 2 samples, 0 errors

    report = generate_error_analysis_report(X, y_true, y_pred, bins=2, sort_metric="group", ascending=True)

    assert report.shape[0] == 2
    assert report.iloc[0]["count"] == 2
    assert report.iloc[0]["error_count"] == 1
    assert report.iloc[1]["count"] == 2
    assert report.iloc[1]["error_count"] == 0
    assert isinstance(report.iloc[0]["group"], pd.Interval)


def test_generate_error_analysis_report_feature_subset():
    """Test passing a subset of columns."""
    X = pd.DataFrame({"cat1": ["A", "A"], "cat2": ["B", "B"]})
    y_true = np.array([0, 0])
    y_pred = np.array([0, 0])

    report = generate_error_analysis_report(X, y_true, y_pred, feature_columns=["cat1"])

    assert "cat1" in report["feature"].values
    assert "cat2" not in report["feature"].values


def test_generate_error_analysis_report_min_count():
    """Test min_count filtering."""
    X = pd.DataFrame({"cat": ["A", "A", "B"]})
    y_true = np.array([0, 0, 0])
    y_pred = np.array([0, 0, 0])

    report = generate_error_analysis_report(X, y_true, y_pred, min_count=2)

    assert "A" in report["group"].values
    assert "B" not in report["group"].values


def test_generate_error_analysis_report_sorting():
    """Test sort_metric and ascending parameters."""
    X = pd.DataFrame({"cat": ["A", "B", "C"]})
    y_true = np.array([0, 0, 0])
    y_pred = np.array([1, 0, 0])
    # A: error_rate 1.0, count 1
    # B: error_rate 0.0, count 1
    # C: error_rate 0.0, count 1

    # Sort by error_rate descending (default)
    report = generate_error_analysis_report(X, y_true, y_pred, sort_metric="error_rate", ascending=False)
    assert report.iloc[0]["group"] == "A"

    # Sort by error_rate ascending
    report = generate_error_analysis_report(X, y_true, y_pred, sort_metric="error_rate", ascending=True)
    assert report.iloc[-1]["group"] == "A"


def test_generate_error_analysis_report_invalid_sort_metric():
    """Test sort_metric validation."""
    X = pd.DataFrame({"cat": ["A"]})
    y_true = np.array([0])
    y_pred = np.array([0])
    with pytest.raises(ValueError, match="sort_metric must be one of"):
        generate_error_analysis_report(X, y_true, y_pred, sort_metric="invalid")


def test_generate_error_analysis_report_invalid_feature_columns():
    """Test feature_columns validation."""
    X = pd.DataFrame({"cat": ["A"]})
    y_true = np.array([0])
    y_pred = np.array([0])
    with pytest.raises(KeyError, match="The following columns are missing from X"):
        generate_error_analysis_report(X, y_true, y_pred, feature_columns=["missing"])


def test_generate_error_analysis_report_invalid_bins():
    """Test bins validation."""
    X = pd.DataFrame({"cat": ["A"]})
    y_true = np.array([0])
    y_pred = np.array([0])
    with pytest.raises(ValueError, match="bins must be at least 1"):
        generate_error_analysis_report(X, y_true, y_pred, bins=0)


def test_generate_error_analysis_report_invalid_threshold():
    """Test threshold validation."""
    X = pd.DataFrame({"cat": ["A"]})
    y_true = np.array([0])
    y_pred = np.array([0])
    with pytest.raises(ValueError, match="threshold must be between 0 and 1 inclusive"):
        generate_error_analysis_report(X, y_true, y_pred, threshold=1.5)


def test_generate_error_analysis_report_invalid_min_count():
    """Test min_count validation."""
    X = pd.DataFrame({"cat": ["A"]})
    y_true = np.array([0])
    y_pred = np.array([0])
    with pytest.raises(ValueError, match="min_count must be at least 1"):
        generate_error_analysis_report(X, y_true, y_pred, min_count=0)


def test_generate_error_analysis_report_multiple_features():
    """Test report with multiple features."""
    X = pd.DataFrame({"cat": ["A", "B"], "num": [1, 2]})
    y_true = np.array([0, 1])
    y_pred = np.array([0, 1])

    report = generate_error_analysis_report(X, y_true, y_pred)

    assert set(report["feature"]) == {"cat", "num"}


def test_generate_error_analysis_report_perfect_predictions():
    """Test perfect predictions (no errors)."""
    X = pd.DataFrame({"cat": ["A", "B"]})
    y_true = np.array([0, 1])
    y_pred = np.array([0, 1])

    report = generate_error_analysis_report(X, y_true, y_pred)

    assert (report["error_count"] == 0).all()
    assert (report["error_rate"] == 0.0).all()
    assert (report["accuracy"] == 1.0).all()


def test_generate_error_analysis_report_all_wrong_predictions():
    """Test all wrong predictions."""
    X = pd.DataFrame({"cat": ["A", "B"]})
    y_true = np.array([0, 1])
    y_pred = np.array([1, 0])

    report = generate_error_analysis_report(X, y_true, y_pred)

    assert (report["error_count"] == report["count"]).all()
    assert (report["error_rate"] == 1.0).all()
    assert (report["accuracy"] == 0.0).all()


def test_generate_error_analysis_report_mismatched_lengths():
    """Test X, y_true, y_pred mismatched lengths."""
    X = pd.DataFrame({"cat": ["A"]})
    y_true = np.array([0, 1])
    y_pred = np.array([0])
    with pytest.raises(ValueError, match="X, y_true, and y_pred must have the same number of samples"):
        generate_error_analysis_report(X, y_true, y_pred)


def test_generate_error_analysis_report_empty_report():
    """Test empty report when no columns are used (should not happen normally but for coverage)."""
    X = pd.DataFrame({"cat": ["A"]})
    y_true = np.array([0])
    y_pred = np.array([0])
    report = generate_error_analysis_report(X, y_true, y_pred, feature_columns=[])
    assert report.empty
    assert list(report.columns) == ["feature", "group", "count", "error_count", "error_rate", "accuracy"]
