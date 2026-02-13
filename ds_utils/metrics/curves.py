import warnings
from typing import Dict, Optional, Union

import numpy as np
from plotly import graph_objects as go
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve


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
    show_legend: bool = True,
    **kwargs,
) -> go.Figure:
    """Plot ROC curves with threshold annotations for multiple classifiers.

    :param y_true: array-like of shape (n_samples,). True binary labels.
    :param classifiers_names_and_scores_dict: mapping from classifier name to classifier's score.
    :param positive_label: int, float, bool or str, default=None. The label of the positive class.
    :param sample_weight: array-like of shape (n_samples,), default=None. Sample weights.
    :param drop_intermediate: bool, default=True. Whether to drop some suboptimal thresholds that would not appear on
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
                name=f"{classifier_name} (AUC = {auc_score:.2f})",
                **kwargs,
            )
        )

    if add_random_classifier_line:  # Add dashed line for random classifier
        fig.add_trace(
            go.Scatter(
                x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="black"), name="Random Classifier"
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
    add_random_classifier_line: bool = False,
    show_legend: bool = True,
    **kwargs,
) -> go.Figure:
    """Plot Precision-Recall curves with threshold annotations for multiple classifiers.

    :param y_true: array-like of shape (n_samples,). True binary labels.
    :param classifiers_names_and_scores_dict: mapping from classifier name to classifier's score.
    :param positive_label: int, float, bool or str, default=None. The label of the positive class.
    :param sample_weight: array-like of shape (n_samples,), default=None. Sample weights.
    :param drop_intermediate: bool, default=True. Whether to drop some suboptimal thresholds that would not appear on
                              a plotted Precision-Recall curve.
    :param fig: plotly's Figure object, optional. The figure to plot on.
    :param mode: str, default='lines+markers'. Determines the drawing mode for this scatter trace.
    :param add_random_classifier_line: bool, default=False. Whether to plot a diagonal dashed black line which
                                       represents a random classifier.
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
            precision_array, recall_array, thresholds = precision_recall_curve(
                y_true,
                y_scores,
                pos_label=positive_label,
                sample_weight=sample_weight,
                drop_intermediate=drop_intermediate,
            )
        except ValueError as e:
            raise ValueError(f"Error calculating Precision-Recall curve for classifier {classifier_name}: {str(e)}")

        fig.add_trace(
            go.Scatter(
                x=recall_array,
                y=precision_array,
                mode=mode,
                text=[
                    f"Prob: {threshold:.2f}<br>Precision: {precision:.2f}<br>Recall: {recall:.2f}"
                    for precision, recall, threshold in zip(precision_array, recall_array, thresholds)
                ],
                name=classifier_name,
                **kwargs,
            )
        )

    if add_random_classifier_line:  # Add dashed line for random classifier
        fig.add_trace(
            go.Scatter(
                x=[1, 0], y=[0, 1], mode="lines", line=dict(dash="dash", color="black"), name="Random Classifier"
            )
        )

    fig.update_layout(xaxis_title="Recall", yaxis_title="Precision", showlegend=show_legend)

    return fig
