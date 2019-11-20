import numpy
import pytest
from matplotlib import pyplot
from matplotlib.testing.compare import compare_images
from sklearn import datasets, svm
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize

from ds_utils.metrics import plot_precision_recall


def _compare_images(first: str, second: str) -> None:
    results = compare_images(first, second, 10)
    if results is not None:  # the images compare favorably
        assert False


def test_plot_precision_recall_multi_class():
    iris = datasets.load_iris()
    x = iris.data
    y = iris.target

    # Add noisy features
    random_state = numpy.random.RandomState(0)
    n_samples, n_features = x.shape
    x = numpy.c_[x, random_state.randn(n_samples, 200 * n_features)]

    # Use label_binarize to be multi-label like settings
    y = label_binarize(y, classes=[0, 1, 2])
    n_classes = y.shape[1]

    # Limit to the two first classes, and split into training and test
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.5, random_state=random_state)

    # Create a simple classifier
    classifier = OneVsRestClassifier(svm.LinearSVC(random_state=random_state))
    classifier.fit(x_train, y_train)
    y_score = classifier.decision_function(x_test)

    plot = plot_precision_recall(y_test, y_score, n_classes)
    pyplot.savefig("result_images/test_metrics/test_plot_precision_recall_multi_class.png")

    _compare_images("baseline_images/test_metrics/test_plot_precision_recall_multi_class.png",
                    "result_images/test_metrics/test_plot_precision_recall_multi_class.png")


def test_plot_precision_recall_binary_class():
    iris = datasets.load_iris()
    x = iris.data
    y = iris.target

    # Add noisy features
    random_state = numpy.random.RandomState(0)
    n_samples, n_features = x.shape
    x = numpy.c_[x, random_state.randn(n_samples, 200 * n_features)]

    # Limit to the two first classes, and split into training and test
    x_train, x_test, y_train, y_test = train_test_split(x[y < 2], y[y < 2], test_size=.5, random_state=random_state)

    # Create a simple classifier
    classifier = svm.LinearSVC(random_state=random_state)
    classifier.fit(x_train, y_train)
    y_score = classifier.decision_function(x_test)

    plot = plot_precision_recall(y_test, y_score, 2)
    _compare_images("baseline_images/test_metrics/test_plot_precision_recall_binary_class.png",
                    "result_images/test_metrics/test_plot_precision_recall_binary_class.png")


def test_plot_precision_recall_exception():
    with pytest.raises(ValueError):
        plot_precision_recall(numpy.array([]), numpy.array([]), 1)
