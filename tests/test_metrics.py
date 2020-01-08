from pathlib import Path

import matplotlib
from numpy.random.mtrand import RandomState
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

matplotlib.use('agg')
import numpy
import pytest
from matplotlib import pyplot
from sklearn import datasets, svm
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier

from ds_utils.metrics import plot_confusion_matrix, plot_metric_growth_per_labeled_instances
from tests.utils import compare_images_paths

iris = datasets.load_iris()
x = iris.data
y = iris.target
RANDOM_STATE = numpy.random.RandomState(0)


def _add_noisy_features(x, random_state):
    n_samples, n_features = x.shape
    return numpy.c_[x, random_state.randn(n_samples, 200 * n_features)]


def test_print_confusion_matrix_binary():
    # Add noisy features
    x_noisy = _add_noisy_features(x, RANDOM_STATE)

    # Limit to the two first classes, and split into training and test
    x_train, x_test, y_train, y_test = train_test_split(x_noisy[y < 2], y[y < 2], test_size=.5,
                                                        random_state=RANDOM_STATE)

    # Create a simple classifier
    classifier = svm.LinearSVC(random_state=RANDOM_STATE)
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)

    plot_confusion_matrix(y_test, y_pred, [1, 0])

    Path(__file__).parents[0].absolute().joinpath("result_images").mkdir(exist_ok=True)
    Path(__file__).parents[0].absolute().joinpath("result_images").joinpath("test_metrics").mkdir(exist_ok=True)
    result_path = Path(__file__).parents[0].absolute().joinpath("result_images").joinpath("test_metrics").joinpath(
        "test_print_confusion_matrix_binary.png")
    pyplot.savefig(str(result_path))

    baseline_path = Path(__file__).parents[0].absolute().joinpath("baseline_images").joinpath("test_metrics").joinpath(
        "test_print_confusion_matrix_binary.png")
    compare_images_paths(str(baseline_path), str(result_path))


def test_print_confusion_matrix():
    # Add noisy features
    x_noisy = _add_noisy_features(x, RANDOM_STATE)

    x_train, x_test, y_train, y_test = train_test_split(x_noisy, y, test_size=.5, random_state=RANDOM_STATE)

    # Create a simple classifier
    classifier = OneVsRestClassifier(svm.LinearSVC(random_state=RANDOM_STATE))
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)

    plot_confusion_matrix(y_test, y_pred, [0, 1, 2])
    Path(__file__).parents[0].absolute().joinpath("result_images").mkdir(exist_ok=True)
    Path(__file__).parents[0].absolute().joinpath("result_images").joinpath("test_metrics").mkdir(exist_ok=True)
    result_path = Path(__file__).parents[0].absolute().joinpath("result_images").joinpath("test_metrics").joinpath(
        "test_print_confusion_matrix.png")
    pyplot.savefig(str(result_path))

    baseline_path = Path(__file__).parents[0].absolute().joinpath("baseline_images").joinpath("test_metrics").joinpath(
        "test_print_confusion_matrix.png")
    compare_images_paths(str(baseline_path), str(result_path))


def test_print_confusion_matrix_exception():
    with pytest.raises(ValueError):
        plot_confusion_matrix(numpy.array([]), numpy.array([]), [])


def test_plot_metric_growth_per_labeled_instances_no_n_samples():
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.3, random_state=0)
    plot_metric_growth_per_labeled_instances(x_train, y_train, x_test, y_test,
                                             {"DecisionTreeClassifier": DecisionTreeClassifier(random_state=0),
                                              "RandomForestClassifier": RandomForestClassifier(random_state=0,
                                                                                               n_estimators=5)})
    result_path = Path(__file__).parents[0].absolute().joinpath("result_images").joinpath(
        "test_metrics").joinpath("test_plot_metric_growth_per_labeled_instances_no_n_samples.png")
    pyplot.savefig(str(result_path))

    baseline_path = Path(__file__).parents[0].absolute().joinpath("baseline_images").joinpath(
        "test_metrics").joinpath("test_plot_metric_growth_per_labeled_instances_no_n_samples.png")
    compare_images_paths(str(baseline_path), str(result_path))


def test_plot_metric_growth_per_labeled_instances_with_n_samples():
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.3, random_state=0)
    plot_metric_growth_per_labeled_instances(x_train, y_train, x_test, y_test,
                                             {"DecisionTreeClassifier": DecisionTreeClassifier(random_state=0),
                                              "RandomForestClassifier": RandomForestClassifier(random_state=0,
                                                                                               n_estimators=5)},
                                             n_samples=list(range(10, x_train.shape[0], 10)))
    result_path = Path(__file__).parents[0].absolute().joinpath("result_images").joinpath(
        "test_metrics").joinpath("test_plot_metric_growth_per_labeled_instances_with_n_samples.png")
    pyplot.savefig(str(result_path))

    baseline_path = Path(__file__).parents[0].absolute().joinpath("baseline_images").joinpath(
        "test_metrics").joinpath("test_plot_metric_growth_per_labeled_instances_with_n_samples.png")
    compare_images_paths(str(baseline_path), str(result_path))


def test_plot_metric_growth_per_labeled_instances_no_n_samples_no_quantiles():
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.3, random_state=0)
    with pytest.raises(ValueError):
        plot_metric_growth_per_labeled_instances(x_train, y_train, x_test, y_test,
                                                 {"DecisionTreeClassifier": DecisionTreeClassifier(random_state=0),
                                                  "RandomForestClassifier": RandomForestClassifier(random_state=0,
                                                                                                   n_estimators=5)},
                                                 n_samples=None, quantiles=None)


def test_plot_metric_growth_per_labeled_instances_given_random_state_int():
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.3, random_state=0)
    plot_metric_growth_per_labeled_instances(x_train, y_train, x_test, y_test,
                                             {"DecisionTreeClassifier": DecisionTreeClassifier(random_state=0),
                                              "RandomForestClassifier": RandomForestClassifier(random_state=0,
                                                                                               n_estimators=5)},
                                             random_state=1)
    result_path = Path(__file__).parents[0].absolute().joinpath("result_images").joinpath(
        "test_metrics").joinpath("test_plot_metric_growth_per_labeled_instances_given_random_state_int.png")
    pyplot.savefig(str(result_path))

    baseline_path = Path(__file__).parents[0].absolute().joinpath("baseline_images").joinpath(
        "test_metrics").joinpath("test_plot_metric_growth_per_labeled_instances_given_random_state_int.png")
    compare_images_paths(str(baseline_path), str(result_path))


def test_plot_metric_growth_per_labeled_instances_given_random_state():
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.3, random_state=0)
    plot_metric_growth_per_labeled_instances(x_train, y_train, x_test, y_test,
                                             {"DecisionTreeClassifier": DecisionTreeClassifier(random_state=0),
                                              "RandomForestClassifier": RandomForestClassifier(random_state=0,
                                                                                               n_estimators=5)},
                                             random_state=RandomState(5))
    result_path = Path(__file__).parents[0].absolute().joinpath("result_images").joinpath(
        "test_metrics").joinpath("test_plot_metric_growth_per_labeled_instances_given_random_state.png")
    pyplot.savefig(str(result_path))

    baseline_path = Path(__file__).parents[0].absolute().joinpath("baseline_images").joinpath(
        "test_metrics").joinpath("test_plot_metric_growth_per_labeled_instances_given_random_state.png")
    compare_images_paths(str(baseline_path), str(result_path))


def test_plot_metric_growth_per_labeled_instances_exists_ax():
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.3, random_state=0)
    pyplot.figure()
    ax = pyplot.gca()

    ax.set_title("My ax")
    plot_metric_growth_per_labeled_instances(x_train, y_train, x_test, y_test,
                                             {"DecisionTreeClassifier": DecisionTreeClassifier(random_state=0),
                                              "RandomForestClassifier": RandomForestClassifier(random_state=0,
                                                                                               n_estimators=5)},
                                             ax=ax)
    result_path = Path(__file__).parents[0].absolute().joinpath("result_images").joinpath(
        "test_metrics").joinpath("test_plot_metric_growth_per_labeled_instances_exists_ax.png")
    pyplot.savefig(str(result_path))

    baseline_path = Path(__file__).parents[0].absolute().joinpath("baseline_images").joinpath(
        "test_metrics").joinpath("test_plot_metric_growth_per_labeled_instances_exists_ax.png")
    compare_images_paths(str(baseline_path), str(result_path))


def test_plot_metric_growth_per_labeled_instances_verbose(capsys):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.3, random_state=0)
    plot_metric_growth_per_labeled_instances(x_train, y_train, x_test, y_test,
                                             {"DecisionTreeClassifier": DecisionTreeClassifier(random_state=0),
                                              "RandomForestClassifier": RandomForestClassifier(random_state=0,
                                                                                               n_estimators=5)},
                                             verbose=1)
    captured = capsys.readouterr().out
    expected = "Fitting classifier DecisionTreeClassifier for 20 times\nFitting classifier RandomForestClassifier for 20 times\n"

    assert expected == captured
