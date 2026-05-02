"""Tests for regression metrics module."""

from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np
from plotly import graph_objects as go
import pytest

from ds_utils.metrics.regression import (
    plot_rec_curve_with_annotations,
    regression_auc_score,
)
from tests.utils import save_plotly_figure_and_return_matplot

BASELINE_DIR = Path(__file__).parents[1] / "baseline_images" / Path(__file__).parent.name / Path(__file__).stem
RESULT_DIR = Path(__file__).parents[1] / "result_images" / Path(__file__).parent.name / Path(__file__).stem

RESULT_DIR.mkdir(exist_ok=True, parents=True)


# ---------------------------------------------------------------------------
# regression_auc_score tests
# ---------------------------------------------------------------------------


def test_regression_auc_score_perfect_predictions():
    """Test that perfect predictions give AUC of 0."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = y_true.copy()
    auc_val = regression_auc_score(y_true, y_pred)
    assert auc_val == pytest.approx(0.0, abs=1e-6)


def test_regression_auc_score_with_errors_normalized():
    """Test AUC calculation with known errors (normalized)."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.5, 2.5, 3.5, 4.5, 5.5])
    auc_val = regression_auc_score(y_true, y_pred)
    assert 0 < auc_val < 1


def test_regression_auc_score_with_errors_unnormalized():
    """Test AUC calculation with known errors (unnormalized)."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.5, 2.5, 3.5, 4.5, 5.5])
    auc_normalized = regression_auc_score(y_true, y_pred, normalize=True)
    auc_raw = regression_auc_score(y_true, y_pred, normalize=False)
    # Raw AOC should be in error-scale units, not [0,1]
    assert auc_raw != auc_normalized


def test_regression_auc_score_shape_mismatch():
    """Test that shape mismatch raises ValueError."""
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0])
    with pytest.raises(ValueError, match="Shape mismatch"):
        regression_auc_score(y_true, y_pred)


def test_regression_auc_score_with_sample_weights():
    """Test AUC calculation with sample weights."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    # Non-uniform errors: [0.1, 0.3, 0.5, 0.7, 1.0]
    y_pred = np.array([1.1, 2.3, 3.5, 4.7, 6.0])
    # Weight heavily the samples with large errors
    weights = np.array([1.0, 1.0, 1.0, 5.0, 5.0])
    auc_weighted = regression_auc_score(y_true, y_pred, sample_weight=weights)
    auc_unweighted = regression_auc_score(y_true, y_pred)
    assert auc_weighted != auc_unweighted


def test_regression_auc_score_worse_model_has_higher_aoc():
    """Test that a worse model produces a higher AOC."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    # Good model: varied small errors [0.1, 0.0, 0.2, 0.1, 0.3]
    y_pred_good = np.array([1.1, 2.0, 3.2, 4.1, 5.3])
    # Bad model: varied large errors [1.0, 2.0, 0.5, 1.5, 2.0]
    y_pred_bad = np.array([2.0, 4.0, 2.5, 5.5, 7.0])
    auc_good = regression_auc_score(y_true, y_pred_good, normalize=False)
    auc_bad = regression_auc_score(y_true, y_pred_bad, normalize=False)
    assert auc_good < auc_bad


# ---------------------------------------------------------------------------
# plot_rec_curve_with_annotations tests
# ---------------------------------------------------------------------------


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR, tolerance=18)
def test_plot_rec_curve_basic(request):
    """Test basic REC curve plotting with 2 models."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    predictions = {
        "Good Model": np.array([1.1, 2.2, 2.8, 4.1, 5.0, 5.9, 7.2, 7.8, 9.1, 10.0]),
        "Bad Model": np.array([2.0, 3.5, 1.5, 5.5, 3.0, 8.0, 5.5, 9.5, 7.0, 12.0]),
    }
    fig = plot_rec_curve_with_annotations(y_true, predictions)

    return save_plotly_figure_and_return_matplot(fig, RESULT_DIR / f"{request.node.name}.png")


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR, tolerance=18)
def test_plot_rec_curve_existing_figure(request):
    """Test plotting REC curve on an existing figure."""
    fig = go.Figure()
    fig.update_layout(title="Custom REC Curves")

    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    predictions = {
        "Model A": np.array([1.1, 2.2, 2.8, 4.1, 5.0, 5.9, 7.2, 7.8, 9.1, 10.0]),
    }
    fig = plot_rec_curve_with_annotations(y_true, predictions, fig=fig)

    return save_plotly_figure_and_return_matplot(fig, RESULT_DIR / f"{request.node.name}.png")


def test_plot_rec_curve_shape_mismatch():
    """Test that shape mismatch raises ValueError with regressor name."""
    y_true = np.array([1.0, 2.0, 3.0])
    predictions = {
        "Model A": np.array([1.0, 2.0]),
    }
    with pytest.raises(ValueError, match=r"Shape mismatch.*Model A"):
        plot_rec_curve_with_annotations(y_true, predictions)


def test_plot_rec_curve_calc_failure(mocker):
    """Test that REC curve calculation failure is re-raised with regressor name."""
    mocker.patch(
        "ds_utils.metrics.regression._calculate_rec_curve",
        side_effect=ValueError("test error"),
    )
    y_true = np.array([1.0, 2.0, 3.0])
    predictions = {"Model A": np.array([1.5, 2.5, 3.5])}
    with pytest.raises(ValueError, match="Error calculating REC curve for regressor Model A:"):
        plot_rec_curve_with_annotations(y_true, predictions)


def test_plot_rec_curve_auc_score_calc_failure(mocker):
    """Test that AUC score calculation failure is re-raised with regressor name."""
    mocker.patch(
        "ds_utils.metrics.regression.regression_auc_score",
        side_effect=ValueError("test error"),
    )
    y_true = np.array([1.0, 2.0, 3.0])
    predictions = {"Model A": np.array([1.5, 2.5, 3.5])}
    with pytest.raises(ValueError, match="Error calculating AUC score for regressor Model A:"):
        plot_rec_curve_with_annotations(y_true, predictions)
