"""Regression evaluation metrics and visualization.

This module provides the Regression Error Characteristic (REC) curve and
the associated Area Over the Curve (AOC) metric for evaluating and
comparing regression models.
"""

from typing import Dict, Optional, Tuple

import numpy as np
from plotly import graph_objects as go
from sklearn.metrics import auc


def _calculate_rec_curve(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
    n_points: int = 1000,
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate REC curve coordinates.

    :param y_true: True target values.
    :param y_pred: Predicted target values.
    :param sample_weight: Sample weights.
    :param n_points: Number of points to calculate on the curve.
    :return: Tuple of (error_tolerances, accuracies).
    """
    errors = np.abs(y_true - y_pred)

    if sample_weight is not None:
        sample_weight = np.array(sample_weight)
        sample_weight = sample_weight / sample_weight.sum()
    else:
        sample_weight = np.ones(len(errors)) / len(errors)

    max_error = np.max(errors)
    error_tolerances = np.linspace(0, max_error, n_points)

    accuracies = np.array([sample_weight[errors <= tolerance].sum() for tolerance in error_tolerances])

    return error_tolerances, accuracies


def regression_auc_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
    normalize: bool = True,
) -> float:
    """Calculate Area Over the REC Curve (AOC) / Regression AUC.

    This is the standalone version of the AUC calculation used in
    :func:`plot_rec_curve_with_annotations`. Lower values indicate
    better model performance. The AOC is calculated as the area between
    the REC curve and the y=1 line.

    When normalized, it is divided by the maximum absolute error to give
    a score in [0, 1].

    :param y_true: array-like of shape (n_samples,). True target values.
    :param y_pred: array-like of shape (n_samples,). Predicted target values.
    :param sample_weight: array-like of shape (n_samples,), default=None. Sample weights.
    :param normalize: bool, default=True. If True, normalize by maximum absolute error
                      to get a score in [0, 1] range.
    :return: The regression AUC score. Lower is better (0 = perfect, 1 = worst possible).
    :raises ValueError: If shapes of y_true and y_pred do not match.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape} and y_pred {y_pred.shape}")

    error_tolerances, accuracies = _calculate_rec_curve(y_true, y_pred, sample_weight)

    max_error = np.max(np.abs(y_true - y_pred))

    # Area under the REC curve using sklearn.metrics.auc
    auc_under_curve = auc(error_tolerances, accuracies)

    # AOC = total rectangle area (max_error * 1.0) minus area under curve
    aoc = max_error * 1.0 - auc_under_curve

    if normalize:
        if max_error > 0:
            aoc = aoc / max_error

    return float(aoc)


def plot_rec_curve_with_annotations(
    y_true: np.ndarray,
    regressors_names_and_predictions_dict: Dict[str, np.ndarray],
    *,
    sample_weight: Optional[np.ndarray] = None,
    normalize_auc: bool = True,
    fig: Optional[go.Figure] = None,
    mode: Optional[str] = "lines+markers",
    show_legend: bool = True,
    **kwargs,
) -> go.Figure:
    """Plot Regression Error Characteristic (REC) curves with AUC annotations.

    The REC curve shows the cumulative distribution of absolute errors,
    allowing comparison of regression model performance. The Area Over
    the Curve (AOC) is calculated and displayed in the legend for each
    regressor.

    :param y_true: array-like of shape (n_samples,). True target values.
    :param regressors_names_and_predictions_dict: mapping from regressor name to predictions.
    :param sample_weight: array-like of shape (n_samples,), default=None. Sample weights.
    :param normalize_auc: bool, default=True. If True, normalize AOC by maximum absolute error
                          to get a score in [0, 1] range.
    :param fig: plotly's Figure object, optional. The figure to plot on.
    :param mode: str, default='lines+markers'. Determines the drawing mode for this scatter trace.
    :param show_legend: bool, default=True. Whether to display legend in the plot.
    :param kwargs: additional keyword arguments to be passed to the plot function.
    :return: The Figure object with the plot drawn onto it.
    :raises ValueError: If the input data is invalid or inconsistent.
    """
    if fig is None:
        fig = go.Figure()

    y_true = np.asarray(y_true)

    for regressor_name, y_pred in regressors_names_and_predictions_dict.items():
        y_pred = np.asarray(y_pred)

        if y_true.shape != y_pred.shape:
            raise ValueError(
                f"Shape mismatch: y_true {y_true.shape} and y_pred {y_pred.shape} for regressor {regressor_name}"
            )

        try:
            error_tolerances, accuracies = _calculate_rec_curve(y_true, y_pred, sample_weight)
        except ValueError as e:
            raise ValueError(f"Error calculating REC curve for regressor {regressor_name}: {str(e)}")

        try:
            auc_score = regression_auc_score(y_true, y_pred, sample_weight=sample_weight, normalize=normalize_auc)
        except ValueError as e:
            raise ValueError(f"Error calculating AUC score for regressor {regressor_name}: {str(e)}")

        hover_text = [f"Tolerance: {tol:.4f}<br>Accuracy: {acc:.4f}" for tol, acc in zip(error_tolerances, accuracies)]

        fig.add_trace(
            go.Scatter(
                x=error_tolerances,
                y=accuracies,
                mode=mode,
                text=hover_text,
                hoverinfo="text",
                name=f"{regressor_name} (AOC = {auc_score:.4f})",
                **kwargs,
            )
        )

    fig.update_layout(
        xaxis_title="Error Tolerance",
        yaxis_title="Accuracy (Proportion within tolerance)",
        showlegend=show_legend,
    )

    return fig
