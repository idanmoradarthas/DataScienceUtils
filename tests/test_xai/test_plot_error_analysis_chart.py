"""Tests for the plot_error_analysis_chart function in ds_utils.xai."""

from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np
import pytest

from ds_utils.xai import plot_error_analysis_chart

BASELINE_DIR = Path(__file__).parent.parent / "baseline_images" / "test_xai" / "test_plot_error_analysis_chart"


@pytest.fixture
def binary_data():
    """Fixture for binary classification data."""
    rng = np.random.RandomState(42)
    n = 100
    y_true = np.array([1] * 50 + [0] * 50)
    # Introduce some errors: ~10 FP and ~10 FN
    y_pred = y_true.copy()
    y_pred[40:50] = 0  # 10 FN (true=1, pred=0)
    y_pred[50:60] = 1  # 10 FP (true=0, pred=1)
    y_proba = rng.beta(2, 5, size=n)
    # Make probabilities roughly consistent with predictions
    y_proba[y_pred == 1] = rng.beta(5, 2, size=(y_pred == 1).sum())
    y_proba[y_pred == 0] = rng.beta(2, 5, size=(y_pred == 0).sum())
    return y_true, y_pred, y_proba


@pytest.fixture
def multiclass_data():
    """Fixture for multi-class classification data."""
    rng = np.random.RandomState(42)
    classes = [0, 1, 2]
    y_true = np.array([0] * 30 + [1] * 30 + [2] * 40)
    y_pred = y_true.copy()
    # Introduce some errors
    y_pred[25:30] = 1  # 5 class-0 predicted as class-1
    y_pred[55:60] = 0  # 5 class-1 predicted as class-0
    y_pred[90:95] = 1  # 5 class-2 predicted as class-1
    y_proba = rng.dirichlet([1, 1, 1], size=100)
    # Adjust probabilities to be somewhat consistent
    for i in range(100):
        y_proba[i, y_pred[i]] += 0.3
    y_proba = y_proba / y_proba.sum(axis=1, keepdims=True)
    return y_true, y_pred, y_proba, classes


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR)
def test_plot_error_analysis_chart_binary(binary_data):
    """Test error analysis chart for binary classification."""
    y_true, y_pred, y_proba = binary_data

    plot_error_analysis_chart(y_true, y_pred, y_proba, positive_class=1)

    figure = plt.gcf()
    figure.set_size_inches(10, 8)
    return figure


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR)
def test_plot_error_analysis_chart_multiclass(multiclass_data):
    """Test error analysis chart for multi-class classification (one-vs-rest)."""
    y_true, y_pred, y_proba, classes = multiclass_data

    plot_error_analysis_chart(y_true, y_pred, y_proba, positive_class=1, classes=classes)

    figure = plt.gcf()
    figure.set_size_inches(10, 8)
    return figure


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR)
def test_plot_error_analysis_chart_existing_ax(binary_data):
    """Test plotting error analysis chart on an existing Axes object."""
    y_true, y_pred, y_proba = binary_data
    fig, ax = plt.subplots()

    ax.set_title("My Error Analysis")
    plot_error_analysis_chart(y_true, y_pred, y_proba, positive_class=1, ax=ax)
    assert ax.get_title() == "My Error Analysis"

    fig.set_size_inches(10, 8)
    return fig


def test_plot_error_analysis_chart_multiclass_infer_classes():
    """Test that classes are inferred from y_true when not provided."""
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 1, 0, 2])
    y_proba = np.array(
        [
            [0.7, 0.2, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
            [0.3, 0.5, 0.2],
            [0.5, 0.3, 0.2],
            [0.2, 0.1, 0.7],
        ]
    )

    # Should not raise when classes is not provided
    ax = plot_error_analysis_chart(y_true, y_pred, y_proba, positive_class=1)
    assert ax is not None


def test_plot_error_analysis_chart_mismatched_lengths():
    """Test that ValueError is raised for mismatched y_true and y_pred lengths."""
    with pytest.raises(ValueError, match="y_true and y_pred must have the same length"):
        plot_error_analysis_chart([1, 0, 1], [1, 0], [0.9, 0.1, 0.8], positive_class=1)


def test_plot_error_analysis_chart_mismatched_proba_length():
    """Test that ValueError is raised for mismatched y_true and y_proba lengths."""
    with pytest.raises(ValueError, match="y_true and y_proba must have the same length"):
        plot_error_analysis_chart([1, 0, 1], [1, 0, 1], [0.9, 0.1], positive_class=1)


def test_plot_error_analysis_chart_invalid_proba_dimensions():
    """Test that ValueError is raised for 3-D y_proba."""
    with pytest.raises(ValueError, match="y_proba must be 1-D or 2-D"):
        plot_error_analysis_chart([1, 0], [1, 0], np.ones((2, 2, 2)), positive_class=1)


def test_plot_error_analysis_chart_positive_class_not_in_classes():
    """Test that ValueError is raised when positive_class is not in classes."""
    y_true = np.array([0, 1, 2])
    y_pred = np.array([0, 1, 2])
    y_proba = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])

    with pytest.raises(ValueError, match="positive_class 99 not found in classes"):
        plot_error_analysis_chart(y_true, y_pred, y_proba, positive_class=99, classes=[0, 1, 2])
