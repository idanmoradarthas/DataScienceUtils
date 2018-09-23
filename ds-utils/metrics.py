from typing import Union, Mapping

import numpy
import pandas
import seaborn
from matplotlib import pyplot
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
    pyplot.show()
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(
        f"F1 Score: {f1_score(y_test, y_pred, labels=[negative_label, positive_label], pos_label=positive_label):.4f}")


def plot_roc_curve(y_test: numpy.ndarray, classifiers_scores_dict: Mapping[str, numpy.ndarray],
                   positive_label: Union[int, float, str]) -> None:
    """
    Prints the roc curve using matplotlib.
    :param y_test: true labels.
    :param classifiers_scores_dict: dictionary of classifier name and predicted probability for positive_label.
    :param positive_label: what is the positive label.
    """
    for classifier_name, y_score in classifiers_scores_dict.items():
        fpr, tpr, _ = roc_curve(y_test, y_score, pos_label=positive_label)
        area = auc(fpr, tpr)
        pyplot.plot(fpr, tpr, label=f"{classifier_name} ROC curve (area = {area:.2f})")
    # base line (random classifier)
    pyplot.plot([0, 1], [0, 1], linestyle="--")
    pyplot.xlim([0.0, 1.0])
    pyplot.ylim([0.0, 1.01])
    pyplot.xlabel("False Positive Rate")
    pyplot.ylabel("True Positive Rate")
    pyplot.title("Receiver operating characteristic (ROC)")
    pyplot.legend(loc="lower right")
    pyplot.show()
