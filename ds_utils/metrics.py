from typing import Union, List, Optional

import numpy
import pandas
import seaborn
from matplotlib import pyplot
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score


def plot_confusion_matrix(y_test: numpy.ndarray, y_pred: numpy.ndarray, labels: List[Union[str, int]],
                          sample_weight: Optional[List[float]] = None) -> pyplot.Figure:
    """
    Computes and plot confusion matrix, False Positive Rate, False Negative Rate, Accuracy and F1 score of a
    classification.

    :param y_test: array, shape = [n_samples]. Ground truth (correct) target values.
    :param y_pred: array, shape = [n_samples]. Estimated targets as returned by a classifier.
    :param labels: array, shape = [n_classes]. List of labels to index the matrix. This may be used to reorder or select a subset of labels.
    :param sample_weight: array-like of shape = [n_samples], optional. Sample weights.
    :return: matplotlib Figure object with the metrics plotted out.
    """
    if len(labels) < 2:
        raise ValueError("Number of labels must be greater than 1")

    cnf_matrix = confusion_matrix(y_test, y_pred, labels, sample_weight)
    if len(labels) == 2:
        tn, fp, fn, tp = cnf_matrix.ravel()
        npv, ppv, tnr, tpr = _calc_precision_recall(fn, fp, tn, tp)

        table = numpy.array([[tn, fp, tnr], [fn, tp, tpr], [npv, ppv, numpy.NaN]], dtype=numpy.float)
        df = pandas.DataFrame(table, columns=[f"{labels[0]} - Predicted", f"{labels[1]} - Predicted", "Recall"],
                              index=[f"{labels[0]} - Actual", f"{labels[1]} - Actual", "Precision"])
    else:
        fp = (cnf_matrix.sum(axis=0) - numpy.diag(cnf_matrix)).astype(float)
        fn = (cnf_matrix.sum(axis=1) - numpy.diag(cnf_matrix)).astype(float)
        tp = (numpy.diag(cnf_matrix)).astype(float)
        tn = (cnf_matrix.sum() - (fp + fn + tp)).astype(float)
        _, ppv, tnr, tpr = _calc_precision_recall(fn, fp, tn, tp)
        df = pandas.DataFrame(cnf_matrix, columns=[f"{label} - Predicted" for label in labels],
                              index=[f"{label} - Actual" for label in labels])
        df["Recall"] = tpr
        df = df.append(
            pandas.DataFrame([ppv], columns=[f"{label} - Predicted" for label in labels], index=["Precision"]),
            sort=False)

    figure, axes = pyplot.subplots(nrows=3, ncols=1, gridspec_kw={'height_ratios': [1, 8, 1]})
    axes = axes.flatten()

    axes[0].set_axis_off()
    if len(labels) == 2:
        axes[0].text(0, 0.85, f"False Positive Rate: {1 - tnr:.4f}")
        axes[0].text(0, 0.35, f"False Negative Rate: {1 - tpr:.4f}")
    else:
        axes[0].text(0, 0.85, f"False Positive Rate: {numpy.array2string(1 - tnr, precision=2, separator=',')}")
        axes[0].text(0, 0.35, f"False Negative Rate: {numpy.array2string(1 - tpr, precision=2, separator=',')}")
    axes[0].text(0, -0.5, "Confusion Matrix:")
    seaborn.heatmap(df, annot=True, fmt=".3f", ax=axes[1])
    axes[2].set_axis_off()
    axes[2].text(0, 0.15, f"Accuracy: {accuracy_score(y_test, y_pred, sample_weight=sample_weight):.4f}")
    if len(labels) == 2:
        f_score = f1_score(y_test, y_pred, labels, labels[1], "binary", sample_weight)
    else:
        f_score = f1_score(y_test, y_pred, labels, average="micro", sample_weight=sample_weight)
    axes[2].text(0, -0.5, f"F1 Score: {f_score:.4f}")
    return figure


def _calc_precision_recall(fn, fp, tn, tp):
    tpr = (tp / (tp + fn))
    tnr = (tn / (tn + fp))
    npv = (tn / (tn + fn))
    ppv = (tp / (tp + fp))
    return npv, ppv, tnr, tpr
