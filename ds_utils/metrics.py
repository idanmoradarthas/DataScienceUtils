from typing import Union

import numpy
import pandas
import seaborn
from matplotlib import pyplot
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score


def print_confusion_matrix_binary(y_test: numpy.ndarray, y_pred: numpy.ndarray, positive_label: Union[int, float, str],
                                  negative_label: Union[int, float, str]) -> pyplot.Figure:
    """
    Prints the confusion matrix using seaborn.
    :param y_test: true labels.
    :param y_pred: predicted labels.
    :param positive_label: what is the positive label.
    :param negative_label: what is the negative label.
    """
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[negative_label, positive_label]).ravel()
    tpr = (tp / (tp + fn)) if (tp + fn) > 0 else numpy.NaN
    tnr = (tn / (tn + fp)) if (tn + fp) > 0 else numpy.NaN
    npv = (tn / (tn + fn)) if (tn + fn) > 0 else numpy.NaN
    ppv = (tp / (tp + fp)) if (tp + fp) > 0 else numpy.NaN

    figure, axes = pyplot.subplots(nrows=3, ncols=1, gridspec_kw={'height_ratios': [1, 8, 1]})
    axes = axes.flatten()

    axes[0].set_axis_off()
    axes[0].text(0, 0.85, f"False Positive Rate: {1 - tnr:.4f}")
    axes[0].text(0, 0.35, f"False Negative Rate: {1 - tpr:.4f}")
    axes[0].text(0, -0.5, "Confusion Matrix:")
    matrix = numpy.array([[tn, fp, tnr], [fn, tp, tpr], [npv, ppv, numpy.NaN]], dtype=numpy.float)
    seaborn.heatmap(
        pandas.DataFrame(matrix, columns=[f"{negative_label} - Predicted", f"{positive_label} - Predicted", "Recall"],
                         index=[f"{negative_label} - Actual", f"{positive_label} - Actual", "Precision"]), annot=True,
        fmt="f", ax=axes[1])
    axes[2].set_axis_off()
    axes[2].text(0, 0.15, f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    axes[2].text(0, -0.5,
                 f"F1 Score: {f1_score(y_test, y_pred, labels=[negative_label, positive_label], pos_label=positive_label):.4f}")
    return figure
