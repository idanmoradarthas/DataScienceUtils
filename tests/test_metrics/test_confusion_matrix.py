"""Tests for Confusion Matrix Visualization."""

from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np
import pytest

from ds_utils.metrics.confusion_matrix import plot_confusion_matrix

BASELINE_DIR = Path(__file__).parents[1] / "baseline_images" / Path(__file__).parent.name / Path(__file__).stem


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR)
@pytest.mark.parametrize(
    ("custom_y_test", "custom_y_pred", "labels"),
    [
        (
            "1 1 1 1 1 0 1 0 1 0 0 1 0 0 1 0 0 0 1 0 1 0 0 1 1 1 1 1 0 0 0 1 0 1 0 1 0 0 0 0 1 1 1 1 0 1 1 0 1 0",
            "0 1 1 1 1 0 0 0 1 0 0 0 0 1 1 0 1 1 0 0 1 0 0 1 1 0 1 1 0 0 0 1 0 1 0 1 1 0 1 0 1 1 1 1 0 1 1 1 1 1",
            [1, 0],
        ),
        (
            "1 0 1 1 0 0 0 0 2 2 1 1 1 2 2 0 1 0 0 1 1 2 2 2 2 1 1 0 1 1 0 0 2 0 1 1 0 2 1 2 2 1 2 1 0 0 0 1 0 2 1 0 "
            "1 2 2 2 1 1 2 2 1 2 1 0 1 1 2 0 0 2 0 2 1 2 0",
            "0 0 2 2 2 0 1 0 1 2 2 2 2 2 2 0 2 1 2 2 0 2 2 2 1 1 2 0 1 2 0 2 2 0 2 2 2 2 2 2 2 0 2 1 0 0 1 1 1 0 1 1 2 "
            "0 1 2 0 0 0 2 2 2 2 0 0 2 2 1 0 2 0 0 2 0 2",
            [0, 1, 2],
        ),
    ],
    ids=["binary", "multiclass"],
)
def test_plot_confusion_matrix(custom_y_test, custom_y_pred, labels):
    """Test plotting of confusion matrix for binary and multiclass cases."""
    y_test = np.fromstring(custom_y_test, dtype=int, sep=" ")
    y_pred = np.fromstring(custom_y_pred, dtype=int, sep=" ")

    ax = plot_confusion_matrix(y_test, y_pred, labels)

    # Assert that the confusion matrix is correctly calculated
    cm = ax[1].get_children()[0].get_array().data[: len(labels), : len(labels)]
    np.testing.assert_array_equal(
        cm, np.array([[np.sum((y_test == i) & (y_pred == j)) for j in labels] for i in labels])
    )

    # Assert that the accuracy and F1 score are correctly calculated
    accuracy = float(ax[2].texts[0].get_text().split(": ")[1])
    assert accuracy == np.mean(y_test == y_pred)

    return plt.gcf()


def test_print_confusion_matrix_exception():
    """Test plot_confusion_matrix raises ValueError for invalid labels."""
    with pytest.raises(ValueError, match="Number of labels must be greater than 1"):
        plot_confusion_matrix(np.array([]), np.array([]), [])


@pytest.mark.parametrize(
    ("y_test", "y_pred", "labels", "expected_error_pattern"),
    [
        (
            np.array([0, 1, 2, 1]),
            np.array([0, 1, 1, 1]),
            [0, 1],
            r"Values in data but not in labels: \[.*2.*\]",
        ),
        (
            np.array([0, 1, 1, 0]),
            np.array([0, 1, 0, 1]),
            [0, 1, 2],
            r"Values in labels but not in data: \[2\]",
        ),
        (
            np.array([0, 1, 3, 0]),
            np.array([0, 1, 0, 1]),
            [0, 1, 2],
            r"Values in data but not in labels: \[.*3.*\].*Values in labels but not in data: \[2\]",
        ),
    ],
    ids=["extra_in_data", "missing_from_data", "both_issues"],
)
def test_plot_confusion_matrix_label_mismatch(y_test, y_pred, labels, expected_error_pattern):
    """Test that plot_confusion_matrix raises ValueError when labels don't match data."""
    with pytest.raises(ValueError, match=expected_error_pattern):
        plot_confusion_matrix(y_test, y_pred, labels)
