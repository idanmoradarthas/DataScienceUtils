"""Tests for the time_series metrics."""

import numpy as np
import pytest
from ds_utils.metrics.time_series import directional_accuracy_score, directional_bias_score


def test_directional_accuracy_perfect_prediction():
    """Test directional_accuracy_score with perfect predictions."""
    y_true = np.array([100, 102, 98, 101, 99, 103])
    y_pred = np.array([100.5, 102.5, 97.5, 101.5, 98.5, 103.5])
    assert directional_accuracy_score(y_true, y_pred) == pytest.approx(1.0)


def test_directional_accuracy_random_prediction():
    """Test directional_accuracy_score with random predictions against baseline."""
    # Corrected data that genuinely yields 0.5:
    # true dirs: [+, -, +, -]; pred dirs: [+, -, -, +] -> correct at 0, 1; wrong at 2, 3 -> 2/4 = 0.5
    y_true = np.array([102.0, 98.0, 101.0, 99.0])
    baseline = np.array([100.0, 100.0, 100.0, 100.0])
    y_pred = np.array([101.0, 99.0, 99.0, 101.0])
    assert directional_accuracy_score(y_true, y_pred, baseline=baseline) == pytest.approx(0.5)


def test_directional_accuracy_all_wrong():
    """Test directional_accuracy_score with all wrong predictions."""
    y_true = np.array([100, 102, 98, 101, 99])
    baseline = np.array([100, 100, 100, 100, 100])
    y_pred = np.array([99, 97, 101, 98, 102])
    assert directional_accuracy_score(y_true, y_pred, baseline=baseline) == pytest.approx(0.0)


def test_directional_accuracy_time_series_default():
    """Test directional_accuracy_score in default time-series mode."""
    # true changes from prev: +2, -4, +3, -2
    # pred changes from prev: +2.5, -6, +5, -4  -> all correct
    y_true = np.array([100, 102, 98, 101, 99])
    y_pred = np.array([100.5, 103, 97, 102, 98])
    assert directional_accuracy_score(y_true, y_pred) == pytest.approx(1.0)


def test_directional_accuracy_insufficient_samples():
    """Test directional_accuracy_score raises error for insufficient samples."""
    with pytest.raises(ValueError, match="at least 2 samples"):
        directional_accuracy_score(np.array([100]), np.array([101]))


def test_directional_accuracy_shape_mismatch():
    """Test directional_accuracy_score raises error for shape mismatch."""
    with pytest.raises(ValueError, match="Shape mismatch"):
        directional_accuracy_score(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0]))


def test_directional_accuracy_baseline_shape_mismatch():
    """Test directional_accuracy_score raises error for baseline shape mismatch."""
    with pytest.raises(ValueError, match="Shape mismatch"):
        directional_accuracy_score(
            np.array([1.0, 2.0, 3.0]),
            np.array([1.0, 2.0, 3.0]),
            baseline=np.array([1.0, 1.0]),
        )


def test_directional_accuracy_invalid_handle_equal():
    """Test directional_accuracy_score raises error for invalid handle_equal."""
    with pytest.raises(ValueError, match="handle_equal must be"):
        directional_accuracy_score(
            np.array([1.0, 2.0, 3.0]),
            np.array([1.0, 2.0, 3.0]),
            baseline=np.array([1.0, 1.0, 1.0]),
            handle_equal="invalid",
        )


@pytest.mark.parametrize(
    ("handle_equal", "expected"),
    [
        ("exclude", 1.0),
        ("correct", 0.6),
        ("incorrect", 0.4),
    ],
)
def test_directional_accuracy_handle_equal(handle_equal, expected):
    """Test directional_accuracy_score handle_equal logic."""
    y_true = np.array([100.0, 100.0, 102.0, 100.0, 98.0])
    baseline = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
    y_pred = np.array([100.0, 101.0, 103.0, 99.0, 97.0])
    assert directional_accuracy_score(y_true, y_pred, baseline=baseline, handle_equal=handle_equal) == pytest.approx(
        expected
    )


def test_directional_accuracy_with_weights_all_correct():
    """Test directional_accuracy_score with sample weights and all correct predictions."""
    y_true = np.array([100, 102, 98, 101])
    baseline = np.array([100, 100, 100, 100])
    y_pred = np.array([101, 103, 97, 102])
    weights = np.array([2.0, 1.0, 1.0, 1.0])
    assert directional_accuracy_score(y_true, y_pred, baseline=baseline, sample_weight=weights) == pytest.approx(1.0)


def test_directional_accuracy_with_weights_partial():
    """Test directional_accuracy_score with sample weights and partial correct predictions."""
    # true directions: +2, -2; pred: +1 (correct), +1 (wrong)
    # weights 3.0, 1.0 -> correct weight = 3.0, total = 4.0 -> 0.75
    y_true = np.array([102.0, 98.0])
    baseline = np.array([100.0, 100.0])
    y_pred = np.array([101.0, 101.0])
    weights = np.array([3.0, 1.0])
    assert directional_accuracy_score(y_true, y_pred, baseline=baseline, sample_weight=weights) == pytest.approx(0.75)


def test_directional_accuracy_with_weights_time_series():
    """Test directional_accuracy_score with sample weights in time-series mode."""
    y_true = np.array([100, 102, 98])
    y_pred = np.array([100, 103, 97])
    weights = np.array([10, 2, 3])
    # time series mode:
    # baseline = [100, 102]
    # y_true_sliced = [102, 98]
    # y_pred_sliced = [103, 97]
    # weights_sliced = [2, 3]
    # true dirs: [+1, -1]
    # pred dirs: [+1, -1] -> all correct
    assert directional_accuracy_score(y_true, y_pred, sample_weight=weights) == pytest.approx(1.0)


def test_directional_accuracy_weight_mismatch_with_baseline():
    """Test directional_accuracy_score raises error for weight length mismatch with explicit baseline."""
    with pytest.raises(ValueError, match="Sample weight length"):
        directional_accuracy_score(
            np.array([1, 2]), np.array([1, 2]), baseline=np.array([0, 0]), sample_weight=np.array([1])
        )


def test_directional_accuracy_weight_mismatch_time_series():
    """Test directional_accuracy_score raises error for weight length mismatch in time-series mode."""
    with pytest.raises(ValueError, match="Sample weight length"):
        directional_accuracy_score(np.array([1, 2]), np.array([1, 2]), sample_weight=np.array([1]))


def test_directional_accuracy_no_valid_samples_after_exclude():
    """Test directional_accuracy_score raises error when no samples remain after exclude."""
    # All y_true == baseline -> exclude removes everything
    y_true = np.array([100.0, 100.0, 100.0])
    baseline = np.array([100.0, 100.0, 100.0])
    y_pred = np.array([101.0, 99.0, 100.0])
    with pytest.raises(ValueError, match="No valid samples remain"):
        directional_accuracy_score(y_true, y_pred, baseline=baseline, handle_equal="exclude")


def test_directional_bias_no_bias():
    """Test directional_bias_score with balanced predictions."""
    # 2 over, 2 under, 1 equal -> exclude the equal -> (2-2)/4 = 0.0
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.1, 1.9, 3.1, 3.9, 5.0])
    assert directional_bias_score(y_true, y_pred, handle_equal="exclude") == pytest.approx(0.0)


def test_directional_bias_complete_over_prediction():
    """Test directional_bias_score with complete over-prediction."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.1, 2.1, 3.1, 4.1, 5.1])
    assert directional_bias_score(y_true, y_pred) == pytest.approx(1.0)


def test_directional_bias_complete_under_prediction():
    """Test directional_bias_score with complete under-prediction."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([0.9, 1.9, 2.9, 3.9, 4.9])
    assert directional_bias_score(y_true, y_pred) == pytest.approx(-1.0)


def test_directional_bias_mostly_over():
    """Test directional_bias_score with mostly over-predictions."""
    # 3 over, 2 under -> (3-2)/5 = 0.2
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.1, 2.1, 3.1, 3.9, 4.9])
    assert directional_bias_score(y_true, y_pred) == pytest.approx(0.2)


def test_directional_bias_shape_mismatch():
    """Test directional_bias_score raises error for shape mismatch."""
    with pytest.raises(ValueError, match="Shape mismatch"):
        directional_bias_score(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0]))


def test_directional_bias_invalid_handle_equal():
    """Test directional_bias_score raises error for invalid handle_equal."""
    with pytest.raises(ValueError, match="handle_equal must be"):
        directional_bias_score(
            np.array([1.0, 2.0, 3.0]),
            np.array([1.0, 2.0, 3.0]),
            handle_equal="invalid",
        )


@pytest.mark.parametrize(
    ("handle_equal", "expected"),
    [
        ("exclude", 1.0),
        ("neutral", 0.6),
    ],
)
def test_directional_bias_handle_equal(handle_equal, expected):
    """Test directional_bias_score handle_equal logic."""
    # 3 over, 0 under, 2 equal -> exclude -> (3-0)/3 = 1.0
    # 3 over, 0 under, 2 equal -> neutral -> (3-0)/5 = 0.6
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.1, 2.0, 3.1, 4.0, 5.1])
    assert directional_bias_score(y_true, y_pred, handle_equal=handle_equal) == pytest.approx(expected)


def test_directional_bias_with_weights_balanced():
    """Test directional_bias_score with sample weights and balanced predictions."""
    # 2 over, 2 under, equal weights -> 0.0
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.1, 2.1, 2.9, 3.9])
    assert directional_bias_score(y_true, y_pred) == pytest.approx(0.0)


def test_directional_bias_with_weights_skewed():
    """Test directional_bias_score with sample weights and skewed predictions."""
    # 2 over (weights 2, 2), 2 under (weights 1, 1)
    # prop_over = 4/6, prop_under = 2/6 -> bias = 2/6 ≈ 0.3333
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.1, 2.1, 2.9, 3.9])
    weights = np.array([2.0, 2.0, 1.0, 1.0])
    assert directional_bias_score(y_true, y_pred, sample_weight=weights) == pytest.approx(1 / 3, rel=1e-4)


def test_directional_bias_with_weights_and_exclusion_filtering():
    """Test directional_bias_score with weights when exclude removes zero-error samples."""
    # errors: [+0.1, 0.0, -0.1, +0.2]
    # handle_equal='exclude' removes index 1 (error == 0)
    # remaining: [+0.1, -0.1, +0.2] with weights [1.0, 2.0, 3.0] (original [1.0, 5.0, 2.0, 3.0])
    # normalized weights: [1/6, 2/6, 3/6]
    # prop_over = (1 + 3) / 6 = 4/6
    # prop_under = 2/6
    # bias = (4-2)/6 = 2/6 ≈ 0.3333
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.1, 2.0, 2.9, 4.2])
    weights = np.array([1.0, 5.0, 2.0, 3.0])
    result = directional_bias_score(y_true, y_pred, sample_weight=weights, handle_equal="exclude")
    assert result == pytest.approx(1 / 3, rel=1e-4)


def test_directional_bias_weight_mismatch():
    """Test directional_bias_score raises error for weight length mismatch."""
    with pytest.raises(ValueError, match="Sample weight length"):
        directional_bias_score(np.array([1, 2]), np.array([1, 2]), sample_weight=np.array([1]))


def test_directional_bias_all_equal_raises():
    """Test directional_bias_score raises error when no samples remain after exclude."""
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="No valid samples remain"):
        directional_bias_score(y_true, y_pred, handle_equal="exclude")


def test_directional_metrics_on_simulated_time_series():
    """Integration test: model performance on simulated time-series."""
    rng = np.random.default_rng(42)
    n = 100
    true_prices = 100 + np.cumsum(rng.standard_normal(n) * 2)
    good_pred = true_prices + rng.standard_normal(n) * 0.5  # tight errors
    bad_pred = 100 + np.cumsum(rng.standard_normal(n) * 2)  # independent walk

    da_good = directional_accuracy_score(true_prices, good_pred)
    da_bad = directional_accuracy_score(true_prices, bad_pred)
    assert da_good > da_bad
    assert da_good > 0.5

    bias_good = directional_bias_score(true_prices, good_pred)
    assert abs(bias_good) < 0.3


def test_consistent_over_predictor():
    """Integration test: uniformly shifted predictor."""
    y_true = np.array([100.0, 102.0, 98.0, 101.0, 99.0, 103.0])
    y_pred = y_true + 1.0
    assert directional_bias_score(y_true, y_pred) == pytest.approx(1.0)
    assert directional_accuracy_score(y_true, y_pred) == pytest.approx(1.0)
