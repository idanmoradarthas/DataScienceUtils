"""Confusion Matrix Metrics Module.

This module provides functions to compute and plot confusion matrices along with
various classification metrics such as False Positive Rate, False Negative Rate,
Accuracy, and F1 score. It supports both binary and multiclass classification.
"""

from typing import List, Union, Optional, Dict, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import axes, pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score


def plot_confusion_matrix(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    labels: List[Union[str, int]],
    sample_weight: Optional[List[float]] = None,
    annot_kws: Optional[Dict] = None,
    cbar: bool = True,
    cbar_kws: Optional[Dict] = None,
    **kwargs,
) -> axes.Axes:
    """Compute and plot confusion matrix with classification metrics.

    Computes and plots confusion matrix, False Positive Rate, False Negative Rate, Accuracy, and F1 score of a
    classification. Before plotting, it validates that the unique values in `y_test`, `y_pred`, and `labels`
    are identical.

    :param y_test: array, shape = [n_samples]. Ground truth (correct) target values.
    :param y_pred: array, shape = [n_samples]. Estimated targets as returned by a classifier.
    :param labels: List of labels (strings or integers) used to index the matrix, corresponding to n_classes.
    :param sample_weight: array-like of shape = [n_samples], optional. Optional sample weights for weighting the
                          samples.
    :param annot_kws: dict of key, value mappings, optional. Keyword arguments for ``ax.text``.
    :param cbar: boolean, optional. Whether to draw a colorbar.
    :param cbar_kws: dict of key, value mappings, optional. Keyword arguments for ``figure.colorbar``.
    :param kwargs: other keyword arguments. All other keyword arguments are passed
                   to ``matplotlib.axes.Axes.pcolormesh()``.
    :return: Returns the Axes object with the matrix drawn onto it.

    :raises ValueError: If number of labels is lower than 2, or if there is a mismatch between the unique
                        values in `y_test`, `y_pred`, and `labels`.
    """
    if len(labels) < 2:
        raise ValueError("Number of labels must be greater than 1")

    # Get unique values from each input
    unique_y_test = set(np.unique(y_test))
    unique_y_pred = set(np.unique(y_pred))
    unique_labels = set(labels)

    # Check if all values in y_test and y_pred are in labels
    extra_in_data = (unique_y_test | unique_y_pred) - unique_labels
    missing_from_data = unique_labels - (unique_y_test | unique_y_pred)

    if extra_in_data or missing_from_data:
        error_parts = []
        if extra_in_data:
            error_parts.append(f"Values in data but not in labels: {sorted(extra_in_data)}")
        if missing_from_data:
            error_parts.append(f"Values in labels but not in data: {sorted(missing_from_data)}")
        raise ValueError(f"Mismatch between labels and data. {'. '.join(error_parts)}")

    cnf_matrix = confusion_matrix(y_test, y_pred, labels=labels, sample_weight=sample_weight)
    if len(labels) == 2:
        df, tnr, tpr = _create_binary_confusion_matrix(cnf_matrix, labels)
    else:
        df, tnr, tpr = _create_multiclass_confusion_matrix(cnf_matrix, labels)

    subplots = _plot_confusion_matrix_helper(
        df, tnr, tpr, labels, y_pred, y_test, sample_weight, annot_kws, cbar, cbar_kws, kwargs
    )
    return subplots


def _calc_precision_recall(
    fn: Union[float, np.ndarray],
    fp: Union[float, np.ndarray],
    tn: Union[float, np.ndarray],
    tp: Union[float, np.ndarray],
) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray], Union[float, np.ndarray], Union[float, np.ndarray]]:
    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)
    npv = tn / (tn + fn)
    ppv = tp / (tp + fp)
    return npv, ppv, tnr, tpr


def _create_multiclass_confusion_matrix(
    cnf_matrix: np.ndarray, labels: List[Union[str, int]]
) -> Tuple[pd.DataFrame, Union[float, np.ndarray], Union[float, np.ndarray]]:
    fp = (cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)).astype(float)
    fn = (cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)).astype(float)
    tp = (np.diag(cnf_matrix)).astype(float)
    tn = (cnf_matrix.sum() - (fp + fn + tp)).astype(float)
    _, ppv, tnr, tpr = _calc_precision_recall(fn, fp, tn, tp)
    df = pd.DataFrame(
        cnf_matrix,
        columns=[f"{label} - Predicted" for label in labels],
        index=[f"{label} - Actual" for label in labels],
    )
    df["Recall"] = tpr
    df = pd.concat(
        [df, pd.DataFrame([ppv], columns=[f"{label} - Predicted" for label in labels], index=["Precision"])], sort=False
    )
    return df, tnr, tpr


def _create_binary_confusion_matrix(
    cnf_matrix: np.ndarray, labels: List[Union[str, int]]
) -> Tuple[pd.DataFrame, float, float]:
    tn, fp, fn, tp = cnf_matrix.ravel()
    npv, ppv, tnr, tpr = _calc_precision_recall(fn, fp, tn, tp)
    table = np.array([[tn, fp, tnr], [fn, tp, tpr], [npv, ppv, np.nan]], dtype=np.float64)
    df = pd.DataFrame(
        table,
        columns=[f"{labels[0]} - Predicted", f"{labels[1]} - Predicted", "Recall"],
        index=[f"{labels[0]} - Actual", f"{labels[1]} - Actual", "Precision"],
    )
    return df, tnr, tpr


def _plot_confusion_matrix_helper(
    df: pd.DataFrame,
    tnr: Union[float, np.ndarray],
    tpr: Union[float, np.ndarray],
    labels: List[Union[str, int]],
    y_pred: np.ndarray,
    y_test: np.ndarray,
    sample_weight: Optional[List[float]],
    annot_kws: Optional[Dict],
    cbar: bool,
    cbar_kws: Optional[Dict],
    kwargs,
) -> axes.Axes:
    figure, subplots = plt.subplots(nrows=3, ncols=1, gridspec_kw={"height_ratios": [1, 8, 1]})
    subplots = subplots.flatten()
    subplots[0].set_axis_off()
    if len(labels) == 2:
        subplots[0].text(0, 0.85, f"False Positive Rate: {1 - tnr:.4f}")
        subplots[0].text(0, 0.35, f"False Negative Rate: {1 - tpr:.4f}")
    else:
        subplots[0].text(0, 0.85, f"False Positive Rate: {np.array2string(1 - tnr, precision=2, separator=',')}")
        subplots[0].text(0, 0.35, f"False Negative Rate: {np.array2string(1 - tpr, precision=2, separator=',')}")
    subplots[0].text(0, -0.5, "Confusion Matrix:")
    sns.heatmap(df, annot=True, fmt=".3f", ax=subplots[1], annot_kws=annot_kws, cbar=cbar, cbar_kws=cbar_kws, **kwargs)
    subplots[2].set_axis_off()
    subplots[2].text(0, 0.15, f"Accuracy: {accuracy_score(y_test, y_pred, sample_weight=sample_weight):.4f}")
    if len(labels) == 2:
        f_score = f1_score(
            y_test, y_pred, labels=labels, pos_label=labels[1], average="binary", sample_weight=sample_weight
        )
    else:
        f_score = f1_score(y_test, y_pred, labels=labels, average="micro", sample_weight=sample_weight)
    subplots[2].text(0, -0.5, f"F1 Score: {f_score:.4f}")
    return subplots
