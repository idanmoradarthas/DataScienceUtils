from pathlib import Path

import numpy
import pytest
from matplotlib.testing.compare import compare_images
from sklearn import datasets, svm
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize

from ds_utils.metrics import plot_precision_recall, plot_roc_curve_binary_class

IRIS = datasets.load_iris()
RANDOM_STATE = numpy.random.RandomState(0)


def _add_noisy_features(random_state, x):
    n_samples, n_features = x.shape
    return numpy.c_[x, random_state.randn(n_samples, 200 * n_features)]


def _compare_images(first: str, second: str) -> None:
    results = compare_images(first, second, 10)
    if results is not None:  # the images compare favorably
        assert False


def test_plot_precision_recall_multi_class():
    x = IRIS.data
    y = IRIS.target

    # Add noisy features
    x = _add_noisy_features(RANDOM_STATE, x)

    # Use label_binarize to be multi-label like settings
    y = label_binarize(y, classes=[0, 1, 2])
    n_classes = y.shape[1]

    # Limit to the two first classes, and split into training and test
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.5, random_state=RANDOM_STATE)

    # Create a simple classifier
    classifier = OneVsRestClassifier(svm.LinearSVC(random_state=RANDOM_STATE))
    classifier.fit(x_train, y_train)
    y_score = classifier.decision_function(x_test)

    plot = plot_precision_recall(y_test, y_score, n_classes)
    Path("result_images").mkdir(exist_ok=True)
    Path("result_images").joinpath("test_metrics").mkdir(exist_ok=True)
    result_path = Path(__file__).parents[0].absolute().joinpath("result_images").joinpath("test_metrics").joinpath(
        "test_plot_precision_recall_multi_class.png")
    plot.savefig(str(result_path))

    baseline_path = Path(__file__).parents[0].absolute().joinpath("baseline_images").joinpath(
        "test_metrics").joinpath("test_plot_precision_recall_multi_class.png")
    _compare_images(str(baseline_path), str(result_path))


def test_plot_precision_recall_binary_class():
    x = IRIS.data
    y = IRIS.target

    # Add noisy features
    x = _add_noisy_features(RANDOM_STATE, x)

    # Limit to the two first classes, and split into training and test
    x_train, x_test, y_train, y_test = train_test_split(x[y < 2], y[y < 2], test_size=.5, random_state=RANDOM_STATE)

    # Create a simple classifier
    classifier = svm.LinearSVC(random_state=RANDOM_STATE)
    classifier.fit(x_train, y_train)
    y_score = classifier.decision_function(x_test)

    plot = plot_precision_recall(y_test, y_score, 2)
    Path("result_images").mkdir(exist_ok=True)
    Path("result_images").joinpath("test_metrics").mkdir(exist_ok=True)
    result_path = Path(__file__).parents[0].absolute().joinpath("result_images").joinpath("test_metrics").joinpath(
        "test_plot_precision_recall_binary_class.png")
    plot.savefig(str(result_path))

    baseline_path = Path(__file__).parents[0].absolute().joinpath("baseline_images").joinpath(
        "test_metrics").joinpath("test_plot_precision_recall_binary_class.png")
    _compare_images(str(baseline_path), str(result_path))


def test_plot_precision_recall_exception():
    with pytest.raises(ValueError):
        plot_precision_recall(numpy.array([]), numpy.array([]), 1)


def test_plot_roc_curve_binary_class():
    x = IRIS.data
    y = IRIS.target

    # Binarize the output
    y = label_binarize(y, classes=[0, 1, 2])
    n_classes = y.shape[1]

    # Add noisy features to make the problem harder
    x = _add_noisy_features(RANDOM_STATE, x)
    # shuffle and split training and test sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.5, random_state=0)

    # Learn to predict each class against the other
    classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True, random_state=RANDOM_STATE))
    y_score = classifier.fit(x_train, y_train).decision_function(x_test)

    plot = plot_roc_curve_binary_class(y_test[:, 0], {"Base Classifier": y_score[:, 0]}, 1)
    Path("result_images").mkdir(exist_ok=True)
    Path("result_images").joinpath("test_metrics").mkdir(exist_ok=True)
    result_path = Path(__file__).parents[0].absolute().joinpath("result_images").joinpath("test_metrics").joinpath(
        "test_plot_roc_curve_binary_class.png")
    plot.savefig(str(result_path))

    baseline_path = Path(__file__).parents[0].absolute().joinpath("baseline_images").joinpath(
        "test_metrics").joinpath("test_plot_roc_curve_binary_class.png")
    _compare_images(str(baseline_path), str(result_path))
