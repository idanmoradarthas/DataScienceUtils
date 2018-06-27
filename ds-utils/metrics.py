from typing import Union

import matplotlib.pyplot as plt
import numpy
import pandas
import seaborn
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, roc_curve, auc


def print_confusion_matrix(y_test: numpy.ndarray, y_pred: numpy.ndarray, positive_label: Union[int, float, str],
                           negative_label: Union[int, float, str]) -> None:
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

    print(f"False Positive Rate: {1-tnr:.4f}")
    print(f"False Negative Rate: {1-tpr:.4f}")
    print()
    print("Confusion Matrix:")
    matrix = numpy.array([[tn, fp, tnr], [fn, tp, tpr], [npv, ppv, numpy.NaN]], dtype=numpy.float)
    seaborn.heatmap(
        pandas.DataFrame(matrix, columns=[f"{negative_label} - Predicted", f"{positive_label} - Predicted", "Recall"],
                         index=[f"{negative_label} - Actual", f"{positive_label} - Actual", "Precision"]), annot=True,
        fmt="f")
    plt.show()
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(
        f"F1 Score: {f1_score(y_test, y_pred, labels=[negative_label, positive_label], pos_label=positive_label):.4f}")


def plot_roc_curve(y_test: numpy.ndarray, y_score: numpy.ndarray, positive_label: Union[int, float, str]) -> None:
    """
    Prints the roc curve using matplotlib.
    :param y_test: true labels.
    :param y_score: predicted probability for positive_label.
    :param positive_label: what is the positive label.
    """
    fpr, tpr, _ = roc_curve(y_test, y_score, pos_label=positive_label)
    area = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"Base model ROC curve (area = {area:.2f})")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.01])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic (ROC)")
    plt.legend(loc="lower right")
    plt.show()
