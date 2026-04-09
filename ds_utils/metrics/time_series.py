"""Time series and forecasting metrics.

This module provides scalar metric functions for time-series forecasting,
financial modeling, and other use cases where trend direction and bias
are of primary interest.
"""

from typing import Optional

import numpy as np


def directional_accuracy_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    baseline: Optional[np.ndarray] = None,
    sample_weight: Optional[np.ndarray] = None,
    handle_equal: str = "exclude",
) -> float:
    """Calculate the directional accuracy score.

    Directional accuracy (DA) measures the proportion of time steps for which
    the predicted direction of change matches the true direction of change,
    relative to a baseline.

    The formula is:
    DA = (1/n) * Σ I(sign(y_true - baseline) == sign(y_pred - baseline))

    :param y_true: Ground truth (correct) target values.
    :param y_pred: Estimated target values.
    :param baseline: Baseline values to compare against. If None, uses the previous
                     value of y_true (time-series mode). If provided, must have
                     the same shape as y_true.
    :param sample_weight: Sample weights.
    :param handle_equal: How to treat samples where y_true == baseline.
                         - 'exclude': Filter out these samples (default).
                         - 'correct': Count as correct if y_pred == baseline, else incorrect.
                         - 'incorrect': Always count as incorrect.
    :return: Directional accuracy score as a float.
    :raises ValueError: If handle_equal is invalid, if shapes mismatch, or if
                        insufficient samples are provided for time-series mode.

    Time series example:
    >>> import numpy as np
    >>> y_true = np.array([100, 102, 98, 101, 99])
    >>> y_pred = np.array([100.5, 103, 97, 102, 98])
    >>> directional_accuracy_score(y_true, y_pred)
    1.0

    Custom baseline example:
    >>> y_true = np.array([102, 98, 101, 99, 102])
    >>> baseline = np.array([100, 100, 100, 100, 100])
    >>> y_pred = np.array([101, 99, 99, 101, 99])
    >>> directional_accuracy_score(y_true, y_pred, baseline=baseline)
    0.4
    """
    if handle_equal not in ["exclude", "correct", "incorrect"]:
        raise ValueError(f"handle_equal must be 'exclude', 'correct', or 'incorrect', got '{handle_equal}'")

    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape} and y_pred {y_pred.shape} must match.")

    if baseline is None:
        if len(y_true) < 2:
            raise ValueError("Time-series mode (baseline=None) requires at least 2 samples.")
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight).flatten()
            if len(sample_weight) != len(y_true):
                raise ValueError(
                    f"Sample weight length ({len(sample_weight)}) does not match sample count ({len(y_true)})"
                )
        baseline = y_true[:-1]
        y_true = y_true[1:]
        y_pred = y_pred[1:]
        if sample_weight is not None:
            sample_weight = sample_weight[1:]
    else:
        baseline = np.asarray(baseline).flatten()
        if baseline.shape != y_true.shape:
            raise ValueError(f"Shape mismatch: baseline {baseline.shape} and y_true {y_true.shape} must match.")
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight).flatten()
            if len(sample_weight) != len(y_true):
                raise ValueError(
                    f"Sample weight length ({len(sample_weight)}) does not match sample count ({len(y_true)})"
                )

    true_direction = np.sign(y_true - baseline)
    pred_direction = np.sign(y_pred - baseline)

    if handle_equal == "exclude":
        mask = true_direction != 0
        true_direction = true_direction[mask]
        pred_direction = pred_direction[mask]
        if sample_weight is not None:
            sample_weight = sample_weight[mask]
        if len(true_direction) == 0:
            raise ValueError("No valid samples remain after filtering")
        correct_direction = true_direction == pred_direction
    elif handle_equal == "correct":
        equal_mask = true_direction == 0
        correct_direction = np.where(
            equal_mask,
            pred_direction == 0,
            true_direction == pred_direction,
        )
    else:  # handle_equal == "incorrect"
        equal_mask = true_direction == 0
        correct_direction = np.where(
            equal_mask,
            False,
            true_direction == pred_direction,
        )

    if sample_weight is None:
        accuracy = correct_direction.mean()
    else:
        w = sample_weight / sample_weight.sum()
        accuracy = (correct_direction * w).sum()

    return float(accuracy)


def directional_bias_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
    handle_equal: str = "exclude",
) -> float:
    """Calculate the directional bias score.

    Directional bias (DB) measures the systematic tendency of a model to
    over-predict or under-predict the target values.

    The formula is:
    DB = (n_over - n_under) / n

    :param y_true: Ground truth (correct) target values.
    :param y_pred: Estimated target values.
    :param sample_weight: Sample weights.
    :param handle_equal: How to treat samples where y_pred == y_true.
                         - 'exclude': Filter out these samples (default).
                         - 'neutral': Keep samples; they contribute 0 to the bias.
    :return: Directional bias score as a float in [-1, 1].
             1.0 = complete over-prediction, -1.0 = complete under-prediction,
             0.0 = balanced predictions.
    :raises ValueError: If handle_equal is invalid, if shapes mismatch, or if
                        no samples remain after filtering.

    Complete over-prediction example:
    >>> import numpy as np
    >>> y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> y_pred = np.array([1.1, 2.1, 3.1, 4.1, 5.1])
    >>> directional_bias_score(y_true, y_pred)
    1.0

    Complete under-prediction example:
    >>> y_pred = np.array([0.9, 1.9, 2.9, 3.9, 4.9])
    >>> directional_bias_score(y_true, y_pred)
    -1.0

    Balanced example:
    >>> y_pred = np.array([1.1, 1.9, 3.1, 3.9, 5.0])
    >>> directional_bias_score(y_true, y_pred)
    0.0
    """
    if handle_equal not in ["exclude", "neutral"]:
        raise ValueError(f"handle_equal must be 'exclude' or 'neutral', got '{handle_equal}'")

    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape} and y_pred {y_pred.shape} must match.")

    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight).flatten()
        if len(sample_weight) != len(y_true):
            raise ValueError(f"Sample weight length ({len(sample_weight)}) does not match sample count ({len(y_true)})")

    errors = y_pred - y_true

    if handle_equal == "exclude":
        mask = errors != 0
        errors = errors[mask]
        if sample_weight is not None:
            sample_weight = sample_weight[mask]
        if len(errors) == 0:
            raise ValueError("No valid samples remain after filtering")

    over_predictions = errors > 0
    under_predictions = errors < 0

    if sample_weight is None:
        prop_over = over_predictions.mean()
        prop_under = under_predictions.mean()
    else:
        w = sample_weight / sample_weight.sum()
        prop_over = (over_predictions * w).sum()
        prop_under = (under_predictions * w).sum()

    return float(prop_over - prop_under)
