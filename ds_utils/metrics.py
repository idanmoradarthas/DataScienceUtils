import warnings
from typing import Union, List, Optional, Callable, Dict, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from joblib import Parallel, delayed
from matplotlib import pyplot as plt, axes
from numpy.random.mtrand import RandomState
from sklearn.base import ClassifierMixin, clone
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    f1_score,
    roc_curve,
    roc_auc_score,
    precision_recall_curve
)
from sklearn.utils import shuffle


def plot_confusion_matrix(
        y_test: np.ndarray,
        y_pred: np.ndarray,
        labels: List[Union[str, int]],
        sample_weight: Optional[List[float]] = None,
        annot_kws: Optional[Dict] = None,
        cbar: bool = True,
        cbar_kws: Optional[Dict] = None,
        **kwargs
) -> axes.Axes:
    """
    Computes and plots confusion matrix, False Positive Rate, False Negative Rate, Accuracy and F1 score of a
    classification.

    :param y_test: array, shape = [n_samples]. Ground truth (correct) target values.
    :param y_pred: array, shape = [n_samples]. Estimated targets as returned by a classifier.
    :param labels: array, shape = [n_classes]. List of labels used to index the matrix.
    :param sample_weight: array-like of shape = [n_samples], optional. Optional sample weights for weighting the samples.
    :param annot_kws: dict of key, value mappings, optional. Keyword arguments for ``ax.text``.
    :param cbar: boolean, optional. Whether to draw a colorbar.
    :param cbar_kws: dict of key, value mappings, optional. Keyword arguments for ``figure.colorbar``.
    :param kwargs: other keyword arguments. All other keyword arguments are passed to ``matplotlib.axes.Axes.pcolormesh()``.
    :return: Returns the Axes object with the matrix drawn onto it.
    """
    if len(labels) < 2:
        raise ValueError("Number of labels must be greater than 1")

    cnf_matrix = confusion_matrix(y_test, y_pred, labels=labels, sample_weight=sample_weight)
    if len(labels) == 2:
        df, tnr, tpr = _create_binary_confusion_matrix(cnf_matrix, labels)
    else:
        df, tnr, tpr = _create_multiclass_confusion_matrix(cnf_matrix, labels)

    subplots = _plot_confusion_matrix_helper(df, tnr, tpr, labels, y_pred, y_test, sample_weight, annot_kws, cbar,
                                             cbar_kws, kwargs)
    return subplots


def _calc_precision_recall(
        fn: Union[float, np.ndarray],
        fp: Union[float, np.ndarray],
        tn: Union[float, np.ndarray],
        tp: Union[float, np.ndarray]
) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray], Union[float, np.ndarray], Union[float, np.ndarray]]:
    tpr = (tp / (tp + fn))
    tnr = (tn / (tn + fp))
    npv = (tn / (tn + fn))
    ppv = (tp / (tp + fp))
    return npv, ppv, tnr, tpr


def _create_multiclass_confusion_matrix(
        cnf_matrix: np.ndarray,
        labels: List[Union[str, int]]
) -> Tuple[pd.DataFrame, Union[float, np.ndarray], Union[float, np.ndarray]]:
    fp = (cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)).astype(float)
    fn = (cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)).astype(float)
    tp = (np.diag(cnf_matrix)).astype(float)
    tn = (cnf_matrix.sum() - (fp + fn + tp)).astype(float)
    _, ppv, tnr, tpr = _calc_precision_recall(fn, fp, tn, tp)
    df = pd.DataFrame(cnf_matrix, columns=[f"{label} - Predicted" for label in labels],
                      index=[f"{label} - Actual" for label in labels])
    df["Recall"] = tpr
    df = pd.concat(
        [df, pd.DataFrame([ppv], columns=[f"{label} - Predicted" for label in labels], index=["Precision"])],
        sort=False)
    return df, tnr, tpr


def _create_binary_confusion_matrix(
        cnf_matrix: np.ndarray,
        labels: List[Union[str, int]]
) -> Tuple[pd.DataFrame, float, float]:
    tn, fp, fn, tp = cnf_matrix.ravel()
    npv, ppv, tnr, tpr = _calc_precision_recall(fn, fp, tn, tp)
    table = np.array([[tn, fp, tnr], [fn, tp, tpr], [npv, ppv, np.nan]], dtype=np.float64)
    df = pd.DataFrame(table, columns=[f"{labels[0]} - Predicted", f"{labels[1]} - Predicted", "Recall"],
                      index=[f"{labels[0]} - Actual", f"{labels[1]} - Actual", "Precision"])
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
        kwargs
) -> axes.Axes:
    figure, subplots = plt.subplots(nrows=3, ncols=1, gridspec_kw={'height_ratios': [1, 8, 1]})
    subplots = subplots.flatten()
    subplots[0].set_axis_off()
    if len(labels) == 2:
        subplots[0].text(0, 0.85, f"False Positive Rate: {1 - tnr:.4f}")
        subplots[0].text(0, 0.35, f"False Negative Rate: {1 - tpr:.4f}")
    else:
        subplots[0].text(0, 0.85, f"False Positive Rate: {np.array2string(1 - tnr, precision=2, separator=',')}")
        subplots[0].text(0, 0.35, f"False Negative Rate: {np.array2string(1 - tpr, precision=2, separator=',')}")
    subplots[0].text(0, -0.5, "Confusion Matrix:")
    sns.heatmap(df, annot=True, fmt=".3f", ax=subplots[1], annot_kws=annot_kws, cbar=cbar, cbar_kws=cbar_kws,
                **kwargs)
    subplots[2].set_axis_off()
    subplots[2].text(0, 0.15, f"Accuracy: {accuracy_score(y_test, y_pred, sample_weight=sample_weight):.4f}")
    if len(labels) == 2:
        f_score = f1_score(y_test, y_pred, labels=labels, pos_label=labels[1], average="binary",
                           sample_weight=sample_weight)
    else:
        f_score = f1_score(y_test, y_pred, labels=labels, average="micro", sample_weight=sample_weight)
    subplots[2].text(0, -0.5, f"F1 Score: {f_score:.4f}")
    return subplots


def plot_metric_growth_per_labeled_instances(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        classifiers_dict: Dict[str, ClassifierMixin],
        n_samples: Optional[List[int]] = None,
        quantiles: Optional[List[float]] = np.linspace(0.05, 1, 20).tolist(),
        metric: Callable[[np.ndarray, np.ndarray], float] = accuracy_score,
        random_state: Optional[Union[int, RandomState]] = None,
        n_jobs: Optional[int] = None,
        verbose: int = 0,
        pre_dispatch: Optional[Union[int, str]] = "2*n_jobs",
        *,
        ax: Optional[axes.Axes] = None,
        **kwargs
) -> axes.Axes:
    """
    Receives train and test sets, and plots the change in the given metric with increasing amounts of trained instances.

    :param X_train: array-like or sparse matrix of shape (n_samples, n_features). The training input samples.
    :param y_train: array-like of shape (n_samples,). The target values (class labels) as integers or strings.
    :param X_test: array-like or sparse matrix of shape (n_samples, n_features). The test or evaluation input samples.
    :param y_test: array-like of shape (n_samples,). The true labels for X_test.
    :param classifiers_dict: mapping from classifier name to classifier object.
    :param n_samples: List of numbers of samples for training batches, optional (default=None).
    :param quantiles: List of percentages of samples for training batches, optional (default=[0.05, 0.1, ..., 0.95, 1]).
                      Used when n_samples=None.
    :param metric: sklearn.metrics API function which receives y_true and y_pred and returns float.
    :param random_state: int, RandomState instance or None, optional (default=None).
                         Controls the shuffling applied to the data before applying the split.
    :param n_jobs: int or None, optional (default=None). The number of jobs to run in parallel.
    :param verbose: int, optional (default=0). Controls the verbosity when fitting and predicting.
    :param pre_dispatch: int or string, optional. Controls the number of jobs that get dispatched during parallel execution.
    :param ax: matplotlib Axes object, optional. The axes to plot on.
    :param kwargs: additional keyword arguments to be passed to the plot function.
    :return: The Axes object with the plot drawn onto it.
    """
    if ax is None:
        _, ax = plt.subplots()

    random_state = RandomState(random_state) if not isinstance(random_state, RandomState) else random_state

    if n_samples is None:
        if quantiles is not None:
            n_samples = [int(quantile * X_train.shape[0]) for quantile in quantiles]
        else:
            raise ValueError("n_samples must be specified if quantiles is None")

    parallel = Parallel(n_jobs=n_jobs, verbose=verbose, pre_dispatch=pre_dispatch)

    classifiers_dict_results = dict()
    samples_list = [shuffle(X_train, y_train, random_state=random_state, n_samples=n_sample) for n_sample in n_samples]

    with parallel:
        for classifier_name, classifier in classifiers_dict.items():
            if verbose > 0:
                print(f"Fitting classifier {classifier_name} for {len(n_samples)} times")
            scores = parallel(
                delayed(_perform_data_partition_and_evaluation)(
                    x_train_part, y_train_part, X_test, y_test, clone(classifier), metric
                ) for x_train_part, y_train_part in samples_list)
            classifiers_dict_results[classifier_name] = scores

    for classifier_name, scores in classifiers_dict_results.items():
        ax.plot(n_samples, scores, label=classifier_name, **kwargs)

    ax.legend(loc="lower right", **kwargs)
    ax.set_xlabel("Number of training samples")
    ax.set_ylabel("Metric score")

    return ax


def _perform_data_partition_and_evaluation(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        classifier: ClassifierMixin,
        metric: Callable[[np.ndarray, np.ndarray], float]
) -> float:
    if y_train.shape[1] == 1:
        classifier.fit(X_train, y_train.ravel())
    else:
        classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    return metric(y_test, y_pred)


def visualize_accuracy_grouped_by_probability(
        y_test: np.ndarray,
        labeled_class: Union[str, int],
        probabilities: np.ndarray,
        threshold: float = 0.5,
        display_breakdown: bool = False,
        bins: Optional[Union[int, Sequence[float], pd.IntervalIndex]] = None,
        *,
        ax: Optional[axes.Axes] = None,
        **kwargs
) -> axes.Axes:
    """
    Receives true test labels and classifier probability predictions, divides and classifies the results,
    and finally plots a stacked bar chart with the results. `Original code <https://github.com/EthicalML/XAI>`_

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

    df_results = pd.DataFrame({
        "Actual Class": np.vectorize(lambda x: x == labeled_class)(y_test),
        "Probability": probabilities,
        "Prediction": probabilities > threshold
    })

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
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    # Adjust layout to prevent cutting off labels
    plt.tight_layout()

    return ax


def plot_roc_curve_with_thresholds_annotations(
        y_true: np.ndarray,
        classifiers_names_and_scores_dict: Dict[str, np.ndarray],
        *,
        positive_label: Optional[Union[int, float, bool, str]] = None,
        sample_weight: Optional[np.ndarray] = None,
        drop_intermediate: bool = True,
        average: Optional[str] = 'macro',
        max_fpr: Optional[float] = None,
        multi_class: str = 'raise',
        labels: Optional[np.ndarray] = None,
        fig: Optional[go.Figure] = None,
        mode: Optional[str] = 'lines+markers',
        add_random_classifier_line: bool = True,
        show_legend: bool = True,
        **kwargs
) -> go.Figure:
    """
    Plot ROC curves with threshold annotations for multiple classifiers.

    :param y_true: array-like of shape (n_samples,). True binary labels.
    :param classifiers_names_and_scores_dict: mapping from classifier name to classifier's score.
    :param positive_label: int, float, bool or str, default=None. The label of the positive class.
    :param sample_weight: array-like of shape (n_samples,), default=None. Sample weights.
    :param drop_intermediate: bool, default=True. Whether to drop some suboptimal thresholds which would not appear on a plotted ROC curve.
    :param average: {'micro', 'macro', 'samples', 'weighted'} or None, default='macro'. If not None, this determines the type of averaging performed on the data.
    :param max_fpr: float > 0 and <= 1, default=None. If not None, the standardized partial AUC over the range [0, max_fpr] is returned.
    :param multi_class: {'raise', 'ovr', 'ovo'}, default='raise'. Determines the type of configuration to use for multiclass targets.
    :param labels: array-like of shape (n_classes,), default=None. Only used for multiclass targets. List of labels that index the classes in y_score.
    :param fig: plotly's Figure object, optional. The figure to plot on.
    :param mode: str, default='lines+markers'. Determines the drawing mode for this scatter trace.
    :param add_random_classifier_line: bool, default=True. Whether to plot a diagonal dashed black line which represents a random classifier.
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
                f"Shape mismatch: y_true {y_true.shape} and y_scores {y_scores.shape} for classifier {classifier_name}")

        try:
            fpr_array, tpr_array, thresholds = roc_curve(y_true, y_scores, pos_label=positive_label,
                                                         sample_weight=sample_weight,
                                                         drop_intermediate=drop_intermediate)
        except ValueError as e:
            raise ValueError(f"Error calculating ROC curve for classifier {classifier_name}: {str(e)}")

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
                auc_score = roc_auc_score(y_true, y_scores, average=average, sample_weight=sample_weight,
                                          max_fpr=max_fpr,
                                          multi_class=multi_class, labels=labels)
        except ValueError as e:
            raise ValueError(f"Error calculating AUC score for classifier {classifier_name}: {str(e)}")

        fig.add_trace(
            go.Scatter(
                x=fpr_array,
                y=tpr_array,
                mode=mode,
                text=[f'Prob: {threshold:.2f}<br>FPR: {fpr:.2f}<br>TPR: {tpr:.2f}' for fpr, tpr, threshold in
                      zip(fpr_array, tpr_array, thresholds)],
                name=f"{classifier_name} (AUC = {auc_score:.2f})",
                **kwargs
            )
        )

    if add_random_classifier_line:  # Add dashed line for random classifier
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode='lines',
                line=dict(dash='dash', color='black'),
                name="Random Classifier"
            )
        )

    fig.update_layout(
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        showlegend=show_legend
    )

    return fig


def plot_precision_recall_curve_with_thresholds_annotations(
        y_true: np.ndarray,
        classifiers_names_and_scores_dict: Dict[str, np.ndarray],
        *,
        positive_label: Optional[Union[int, float, bool, str]] = None,
        sample_weight: Optional[np.ndarray] = None,
        drop_intermediate: bool = True,
        fig: Optional[go.Figure] = None,
        mode: Optional[str] = 'lines+markers',
        add_random_classifier_line: bool = False,
        show_legend: bool = True,
        **kwargs
) -> go.Figure:
    """
    Plot Precision-Recall curves with threshold annotations for multiple classifiers.

    :param y_true: array-like of shape (n_samples,). True binary labels.
    :param classifiers_names_and_scores_dict: mapping from classifier name to classifier's score.
    :param positive_label: int, float, bool or str, default=None. The label of the positive class.
    :param sample_weight: array-like of shape (n_samples,), default=None. Sample weights.
    :param drop_intermediate: bool, default=True. Whether to drop some suboptimal thresholds which would not appear on a plotted Precision-Recall curve.
    :param fig: plotly's Figure object, optional. The figure to plot on.
    :param mode: str, default='lines+markers'. Determines the drawing mode for this scatter trace.
    :param add_random_classifier_line: bool, default=False. Whether to plot a diagonal dashed black line which represents a random classifier.
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
                f"Shape mismatch: y_true {y_true.shape} and y_scores {y_scores.shape} for classifier {classifier_name}")
        try:
            precision_array, recall_array, thresholds = precision_recall_curve(y_true, y_scores,
                                                                               pos_label=positive_label,
                                                                               sample_weight=sample_weight,
                                                                               drop_intermediate=drop_intermediate)
        except ValueError as e:
            raise ValueError(f"Error calculating Precision-Recall curve for classifier {classifier_name}: {str(e)}")

        fig.add_trace(
            go.Scatter(
                x=recall_array,
                y=precision_array,
                mode=mode,
                text=[f'Prob: {threshold:.2f}<br>Precision: {precision:.2f}<br>Recall: {recall:.2f}' for
                      precision, recall, threshold in zip(precision_array, recall_array, thresholds)],
                name=classifier_name,
                **kwargs
            )
        )

    if add_random_classifier_line:  # Add dashed line for random classifier
        fig.add_trace(
            go.Scatter(
                x=[1, 0],
                y=[0, 1],
                mode='lines',
                line=dict(dash='dash', color='black'),
                name="Random Classifier"
            )
        )

    fig.update_layout(
        xaxis_title="Recall",
        yaxis_title="Precision",
        showlegend=show_legend
    )

    return fig
