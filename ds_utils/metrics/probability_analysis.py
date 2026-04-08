"""Probability Analysis Module.

This module provides functions to visualize and analyze the performance of classifiers
based on their predicted probabilities. It includes tools for grouping results by
probability bins, plotting accuracy distributions, and error analysis charts.
"""

from typing import List, Optional, Sequence, Union

from matplotlib import axes, pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def visualize_accuracy_grouped_by_probability(
    y_test: np.ndarray,
    labeled_class: Union[str, int],
    probabilities: np.ndarray,
    threshold: float = 0.5,
    display_breakdown: bool = False,
    bins: Optional[Union[int, Sequence[float], pd.IntervalIndex]] = None,
    *,
    ax: Optional[axes.Axes] = None,
    **kwargs,
) -> axes.Axes:
    """Plot a stacked bar chart of classifier results by probability bins.

    Receives true test labels and classifier probability predictions, divides and classifies the results,
    and finally plots a stacked bar chart with the results.
    `Original code <https://github.com/EthicalML/XAI>`_.

    :param y_test: array, shape = [n_samples]. Ground truth (correct) target values.
    :param labeled_class: the class to inquire about.
    :param probabilities: array, shape = [n_samples]. Classifier probabilities for the labeled class.
    :param threshold: the probability threshold for classifying the labeled class.
    :param display_breakdown: if True, the results will be displayed as "correct" and "incorrect";
                              otherwise as "true-positives", "true-negatives", "false-positives" and "false-negatives".
    :param bins: int, sequence of scalars, or IntervalIndex. The criteria to bin by.
    :param ax: matplotlib Axes object, optional. The axes to plot on.
    :param kwargs: additional keyword arguments to be passed to the plot function.
    :return: The Axes object with the plot drawn onto it.
    """
    if bins is None:
        bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

    if ax is None:
        _, ax = plt.subplots()

    df_results = pd.DataFrame(
        {
            "Actual Class": np.vectorize(lambda x: x == labeled_class)(y_test),
            "Probability": probabilities,
            "Prediction": probabilities > threshold,
        }
    )

    df_results["True Positive"] = (df_results["Actual Class"] & df_results["Prediction"]).astype(int)
    df_results["True Negative"] = (~df_results["Actual Class"] & ~df_results["Prediction"]).astype(int)
    df_results["False Positive"] = (~df_results["Actual Class"] & df_results["Prediction"]).astype(int)
    df_results["False Negative"] = (df_results["Actual Class"] & ~df_results["Prediction"]).astype(int)

    if display_breakdown:
        df_results["Correct"] = df_results["True Positive"] + df_results["True Negative"]
        df_results["Incorrect"] = df_results["False Positive"] + df_results["False Negative"]
        display_columns = ["Correct", "Incorrect"]
    else:
        display_columns = ["True Positive", "True Negative", "False Positive", "False Negative"]

    # Group the results by probability bins
    df_results["Probability Bin"] = pd.cut(df_results["Probability"], bins=bins)
    grouped_results = df_results.groupby("Probability Bin", observed=False)[display_columns].sum()

    # Plot the results
    grouped_results.plot(kind="bar", stacked=True, ax=ax, **kwargs)

    # Customize the plot
    ax.set_xlabel("Probability Range")
    ax.set_ylabel("Count")
    if not ax.get_title():
        ax.set_title(f"Accuracy Distribution for {labeled_class} Class")
    ax.legend(title="Prediction Type")

    # Rotate x-axis labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    # Adjust layout to prevent cutting off labels
    plt.tight_layout()

    return ax


def plot_error_analysis_chart(
    y_true: Union[np.ndarray, List],
    y_pred: Union[np.ndarray, List],
    y_proba: Union[np.ndarray, List],
    positive_class: Union[int, str],
    *,
    classes: Optional[List] = None,
    ax: Optional[axes.Axes] = None,
    **kwargs,
) -> axes.Axes:
    """Plot an error analysis chart showing prediction errors relative to predicted probabilities.

    Builds an internal DataFrame with the predicted probability for the ``positive_class``
    and an ``error_type`` column (correct, false_positive, false_negative), and draws
    a violin plot showing the distribution of predicted probabilities across these error types.

    For **binary** classification ``y_proba`` should be 1-D (probability of the positive class).
    For **multi-class** classification ``y_proba`` should be 2-D with shape ``(n_samples, n_classes)``;
    the column corresponding to ``positive_class`` is determined via ``classes``
    (or inferred from ``np.unique(y_true)`` when ``classes`` is ``None``).
    Error types are computed using a one-vs-rest scheme against ``positive_class``
    by comparing ``y_true`` and ``y_pred`` directly. The original spec included a
    ``threshold`` float parameter for re-thresholding raw probabilities; this
    implementation intentionally omits it — callers should apply any desired
    threshold to produce ``y_pred`` before calling this function.

    :param y_true: Array-like of true labels.
    :param y_pred: Array-like of predicted labels.
    :param y_proba: Array-like of predicted probabilities.  1-D for binary or
                    2-D with shape ``(n_samples, n_classes)`` for multi-class.
    :param positive_class: The class to treat as the positive class.
    :param classes: Ordered list of class labels matching the columns of ``y_proba``
                    when it is 2-D.  If ``None``, inferred from ``np.unique(y_true)``.
    :param ax: Axes object to draw the plot onto, otherwise uses the current Axes.
    :param kwargs: Additional keyword arguments passed to :func:`seaborn.violinplot`.
    :return: Axes object with the plot drawn onto it.
    :raises ValueError: If inputs have mismatched lengths, ``y_proba`` has an invalid
                        number of dimensions, or ``positive_class`` is not found in ``classes``.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_proba = np.asarray(y_proba)

    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("y_true and y_pred must have the same number of samples")
    if y_true.shape[0] != y_proba.shape[0]:
        raise ValueError("y_true and y_proba must have the same number of samples")

    # Determine probability column for the positive class
    if y_proba.ndim == 1:
        proba = y_proba
    elif y_proba.ndim == 2:
        if classes is None:
            classes = list(np.unique(y_true))
        if positive_class not in classes:
            raise ValueError(f"positive_class {positive_class!r} not found in classes {classes}")
        class_index = classes.index(positive_class)
        proba = y_proba[:, class_index]
    else:
        raise ValueError("y_proba must be 1-D or 2-D")

    # Compute error_type using one-vs-rest logic on positive_class
    is_positive_true = y_true == positive_class
    is_positive_pred = y_pred == positive_class

    error_type = np.select(
        [
            is_positive_true & ~is_positive_pred,
            ~is_positive_true & is_positive_pred,
        ],
        ["false_negative", "false_positive"],
        default="correct",
    )

    error_df = pd.DataFrame({"y_proba": proba, "error_type": error_type})

    if ax is None:
        _, ax = plt.subplots()

    # order is hardcoded so the axis layout is consistent even when one
    # error category is absent (e.g. a perfect model produces no FP/FN).
    sns.violinplot(
        x="error_type",
        y="y_proba",
        data=error_df,
        order=["correct", "false_positive", "false_negative"],
        ax=ax,
        **kwargs,
    )
    if not ax.get_title():
        ax.set_title(f"Error Analysis — positive class: {positive_class!r} (one-vs-rest)")
    ax.set_xlabel("Error Type")
    ax.set_ylabel("Predicted Probability")
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    return ax
