"""Module containing functions for plotting evaluation curves."""

from typing import Dict, Optional, Union
import warnings

import numpy as np
from plotly import graph_objects as go
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve


def plot_roc_curve_with_thresholds_annotations(
    y_true: np.ndarray,
    classifiers_names_and_scores_dict: Dict[str, np.ndarray],
    *,
    positive_label: Optional[Union[int, float, bool, str]] = None,
    sample_weight: Optional[np.ndarray] = None,
    drop_intermediate: bool = True,
    average: Optional[str] = "macro",
    max_fpr: Optional[float] = None,
    multi_class: str = "raise",
    labels: Optional[np.ndarray] = None,
    fig: Optional[go.Figure] = None,
    mode: Optional[str] = "lines+markers",
    add_random_classifier_line: bool = True,
    random_classifier_line_kw: Optional[Dict] = None,
    show_legend: bool = True,
    **kwargs,
) -> go.Figure:
    """Plot ROC curves with threshold annotations for multiple classifiers.

    :param y_true: array-like of shape (n_samples,). True binary labels.
    :param classifiers_names_and_scores_dict: mapping from classifier name to classifier's score.
    :param positive_label: int, float, bool or str, default=None. The label of the positive class.
    :param sample_weight: array-like of shape (n_samples,), default=None. Sample weights.
    :param drop_intermediate: bool, default=True. Whether to drop some suboptimal thresholds which would not appear on
                              a plotted ROC curve.
    :param average: {'micro', 'macro', 'samples', 'weighted'} or None, default='macro'. If not None, this determines
                    the type of averaging performed on the data.
    :param max_fpr: float > 0 and <= 1, default=None. If not None, the standardized partial AUC over the range
                    [0, max_fpr] is returned.
    :param multi_class: {'raise', 'ovr', 'ovo'}, default='raise'. Determines the type of configuration to use for
                        multiclass targets.
    :param labels: array-like of shape (n_classes,), default=None. Only used for multiclass targets. List of labels
                   that index the classes in y_score.
    :param fig: plotly's Figure object, optional. The figure to plot on.
    :param mode: str, default='lines+markers'. Determines the drawing mode for this scatter trace.
    :param add_random_classifier_line: bool, default=True. Whether to plot a diagonal dashed black line which
                                       represents a random classifier.
    :param random_classifier_line_kw: dict, default=None. Keyword arguments to be passed to plotly's Scatter
                                      for rendering the random classifier line (e.g., line color, style).
    :param show_legend: bool, default=True. Whether to display legend in the plot.
    :param kwargs: additional keyword arguments to be passed to the plot function.
    :return: The Figure object with the plot drawn onto it.
    :raises ValueError: If the input data is invalid or inconsistent.
    """
    if fig is None:
        fig = go.Figure()  # Create a new figure if none is provided

    for classifier_name, y_scores in classifiers_names_and_scores_dict.items():
        if y_true.shape != y_scores.shape:
            raise ValueError(
                f"Shape mismatch: y_true {y_true.shape} and y_scores {y_scores.shape} for classifier {classifier_name}"
            )

        try:
            fpr_array, tpr_array, thresholds = roc_curve(
                y_true,
                y_scores,
                pos_label=positive_label,
                sample_weight=sample_weight,
                drop_intermediate=drop_intermediate,
            )
        except ValueError as e:
            raise ValueError(f"Error calculating ROC curve for classifier {classifier_name}: {str(e)}")

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
                auc_score = roc_auc_score(
                    y_true,
                    y_scores,
                    average=average,
                    sample_weight=sample_weight,
                    max_fpr=max_fpr,
                    multi_class=multi_class,
                    labels=labels,
                )
        except ValueError as e:
            raise ValueError(f"Error calculating AUC score for classifier {classifier_name}: {str(e)}")

        fig.add_trace(
            go.Scatter(
                x=fpr_array,
                y=tpr_array,
                mode=mode,
                text=[
                    f"Prob: {threshold:.2f}<br>FPR: {fpr:.2f}<br>TPR: {tpr:.2f}"
                    for fpr, tpr, threshold in zip(fpr_array, tpr_array, thresholds)
                ],
                hoverinfo="text",
                name=f"{classifier_name} (AUC = {auc_score:.2f})",
                **kwargs,
            )
        )

    if add_random_classifier_line:  # Add dashed line for random classifier
        default_random_classifier_kw = {
            "line": dict(dash="dash", color="black"),
            "name": "Random Classifier (AUC = 0.50)"
        }
        if random_classifier_line_kw is not None:
            default_random_classifier_kw.update(random_classifier_line_kw)

        fig.add_trace(
            go.Scatter(
                x=[0, 1], y=[0, 1], mode="lines", hoverinfo="name", **default_random_classifier_kw
            )
        )

    fig.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", showlegend=show_legend)

    return fig


def plot_precision_recall_curve_with_thresholds_annotations(
    y_true: np.ndarray,
    classifiers_names_and_scores_dict: Dict[str, np.ndarray],
    *,
    positive_label: Optional[Union[int, float, bool, str]] = None,
    sample_weight: Optional[np.ndarray] = None,
    drop_intermediate: bool = True,
    fig: Optional[go.Figure] = None,
    mode: Optional[str] = "lines+markers",
    plot_chance_level: bool = False,
    chance_level_kw: Optional[Dict] = None,
    show_legend: bool = True,
    **kwargs,
) -> go.Figure:
    """Plot Precision-Recall curves with threshold annotations for multiple classifiers.

    :param y_true: array-like of shape (n_samples,). True binary labels.
    :param classifiers_names_and_scores_dict: mapping from classifier name to classifier's score.
    :param positive_label: int, float, bool or str, default=None. The label of the positive class.
    :param sample_weight: array-like of shape (n_samples,), default=None. Sample weights.
    :param drop_intermediate: bool, default=True. Whether to drop some suboptimal thresholds that don't change the
                              precision. This is useful to create lighter Precision-Recall curves.
    :param fig: plotly's Figure object, optional. The figure to plot on.
    :param mode: str, default='lines+markers'. Determines the drawing mode for this scatter trace.
    :param plot_chance_level: bool, default=False. Whether to plot the chance level. The chance level is the prevalence
                              of the positive label computed from the data passed. Behavior is like sklearn:
                              https://scikit-learn.org/stable/modules/generated/sklearn.metrics.PrecisionRecallDisplay.html
                              When positive_label is None, the positive class is inferred as the larger of the two
                              unique labels in y_true, consistent with scikit-learn's convention.
    :param chance_level_kw: dict, default=None. Keyword arguments to be passed to plotly's Scatter for rendering the
                            chance level line (e.g., line color, style).
    :param show_legend: bool, default=True. Whether to display legend in the plot.
    :param kwargs: additional keyword arguments to be passed to the plot function.
    :return: The Figure object with the plot drawn onto it.
    :raises ValueError: If the input data is invalid or inconsistent.
    """
    if fig is None:
        fig = go.Figure()  # Create a new figure if none is provided

    # When plot_chance_level=True and positive_label=None, resolve to the larger unique label
    # (sklearn convention) and validate binary input. Otherwise, keep positive_label as-is —
    # sklearn will apply its own default when pos_label=None.
    effective_positive_label = positive_label
    if plot_chance_level and effective_positive_label is None:
        unique_labels = np.unique(y_true)
        if len(unique_labels) != 2:
            raise ValueError("y_true must be binary for plotting chance level")
        # Use the convention that the larger label is positive (consistent with sklearn)
        effective_positive_label = unique_labels[1]

    for classifier_name, y_scores in classifiers_names_and_scores_dict.items():
        if y_true.shape != y_scores.shape:
            raise ValueError(
                f"Shape mismatch: y_true {y_true.shape} and y_scores {y_scores.shape} for classifier {classifier_name}"
            )
        try:
            precision_array, recall_array, thresholds = precision_recall_curve(
                y_true,
                y_scores,
                pos_label=effective_positive_label,
                sample_weight=sample_weight,
                drop_intermediate=drop_intermediate,
            )
        except ValueError as e:
            raise ValueError(f"Error calculating Precision-Recall curve for classifier {classifier_name}: {str(e)}")

        try:
            ap_kwargs = {}
            if effective_positive_label is not None:
                ap_kwargs["pos_label"] = effective_positive_label
            if sample_weight is not None:
                ap_kwargs["sample_weight"] = sample_weight

            ap = average_precision_score(y_true, y_scores, **ap_kwargs)
        except ValueError as e:
            raise ValueError(f"Error calculating Average Precision for classifier {classifier_name}: {str(e)}")

        display_thresholds = np.append(thresholds, np.nan)
        fig.add_trace(
            go.Scatter(
                x=recall_array,
                y=precision_array,
                mode=mode,
                text=[
                    f"Prob: {'N/A' if np.isnan(t) else f'{t:.2f}'}<br>Precision: {p:.2f}<br>Recall: {r:.2f}"
                    for p, r, t in zip(precision_array, recall_array, display_thresholds)
                ],
                hoverinfo="text",
                name=f"{classifier_name} (AP = {ap:0.2f})",
                **kwargs,
            )
        )

    if plot_chance_level:
        # Note: AP here refers to the chance-level Average Precision, which equals prevalence.
        # This naming convention is consistent with scikit-learn's PrecisionRecallDisplay.
        if sample_weight is not None:
            prevalence = np.sum(sample_weight[y_true == effective_positive_label]) / np.sum(sample_weight)
        else:
            prevalence = np.sum(y_true == effective_positive_label) / len(y_true)

        # Default styling for chance level line
        default_chance_level_kw = {
            "line": dict(dash="dash", color="black"),
            "name": f"Chance level (AP = {prevalence:0.2f})"
        }
        if chance_level_kw is not None:
            default_chance_level_kw.update(chance_level_kw)

        # Plot horizontal line at prevalence
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[prevalence, prevalence],
                mode="lines",
                hoverinfo="name",
                **default_chance_level_kw,
            )
        )

    fig.update_layout(xaxis_title="Recall", yaxis_title="Precision", showlegend=show_legend)

    return fig
