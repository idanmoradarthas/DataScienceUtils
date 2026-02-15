"""Probability Analysis Module.

This module provides functions to visualize and analyze the performance of classifiers
based on their predicted probabilities. It includes tools for grouping results by
probability bins and plotting accuracy distributions.
"""

from typing import Union, Optional, Sequence

import numpy as np
import pandas as pd
from matplotlib import axes, pyplot as plt


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
