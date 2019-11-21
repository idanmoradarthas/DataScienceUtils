from pathlib import Path

import numpy
import pytest
from matplotlib.testing.compare import compare_images
from sklearn import datasets, svm
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier

from ds_utils.metrics import plot_confusion_matrix

IRIS = datasets.load_iris()
RANDOM_STATE = numpy.random.RandomState(0)


def _add_noisy_features(x, random_state):
    n_samples, n_features = x.shape
    return numpy.c_[x, random_state.randn(n_samples, 200 * n_features)]


def _compare_images(first: str, second: str) -> None:
    results = compare_images(first, second, 10)
    if results is not None:  # the images compare favorably
        assert False


def test_print_confusion_matrix_binary():
    x = IRIS.data
    y = IRIS.target

    # Add noisy features
    x = _add_noisy_features(x, RANDOM_STATE)

    # Limit to the two first classes, and split into training and test
    x_train, x_test, y_train, y_test = train_test_split(x[y < 2], y[y < 2], test_size=.5, random_state=RANDOM_STATE)

    # Create a simple classifier
    classifier = svm.LinearSVC(random_state=RANDOM_STATE)
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)

    plot = plot_confusion_matrix(y_test, y_pred, [1, 0])

    Path(__file__).parents[0].absolute().joinpath("result_images").mkdir(exist_ok=True)
    Path(__file__).parents[0].absolute().joinpath("result_images").joinpath("test_metrics").mkdir(exist_ok=True)
    result_path = Path(__file__).parents[0].absolute().joinpath("result_images").joinpath("test_metrics").joinpath(
        "test_print_confusion_matrix_binary.png")
    plot.savefig(str(result_path))

    baseline_path = Path(__file__).parents[0].absolute().joinpath("baseline_images").joinpath("test_metrics").joinpath(
        "test_print_confusion_matrix_binary.png")
    _compare_images(str(baseline_path), str(result_path))


def test_print_confusion_matrix():
    x = IRIS.data
    y = IRIS.target

    # Add noisy features
    x = _add_noisy_features(x, RANDOM_STATE)

    # Limit to the two first classes, and split into training and test
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.5, random_state=RANDOM_STATE)

    # Create a simple classifier
    classifier = OneVsRestClassifier(svm.LinearSVC(random_state=RANDOM_STATE))
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)

    plot = plot_confusion_matrix(y_test, y_pred, [0, 1, 2])
    Path(__file__).parents[0].absolute().joinpath("result_images").mkdir(exist_ok=True)
    Path(__file__).parents[0].absolute().joinpath("result_images").joinpath("test_metrics").mkdir(exist_ok=True)
    result_path = Path(__file__).parents[0].absolute().joinpath("result_images").joinpath("test_metrics").joinpath(
        "test_print_confusion_matrix.png")
    plot.savefig(str(result_path))

    baseline_path = Path(__file__).parents[0].absolute().joinpath("baseline_images").joinpath("test_metrics").joinpath(
        "test_print_confusion_matrix.png")
    _compare_images(str(baseline_path), str(result_path))


def test_print_confusion_matrix_exception():
    with pytest.raises(ValueError):
        plot_confusion_matrix(numpy.array([]), numpy.array([]), [])
