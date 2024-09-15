from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from matplotlib import pyplot as plt
from numpy.random.mtrand import RandomState
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from ds_utils.metrics import plot_confusion_matrix, plot_metric_growth_per_labeled_instances, \
    visualize_accuracy_grouped_by_probability
from tests.utils import compare_images_from_paths


@pytest.fixture
def iris_data():
    base_path = Path(__file__).parent.joinpath("resources")
    return {
        "x_train": pd.read_csv(base_path.joinpath("iris_x_train.csv")).values,
        "x_test": pd.read_csv(base_path.joinpath("iris_x_test.csv")).values,
        "y_train": pd.read_csv(base_path.joinpath("iris_y_train.csv")).values,
        "y_test": pd.read_csv(base_path.joinpath("iris_y_test.csv")).values
    }


@pytest.fixture
def classifiers():
    return {
        "DecisionTreeClassifier": DecisionTreeClassifier(random_state=0),
        "RandomForestClassifier": RandomForestClassifier(random_state=0, n_estimators=5)
    }


@pytest.fixture
def result_path(request):
    return Path(__file__).parent.joinpath("result_images", "test_metrics", f"{request.node.name}.png")


@pytest.fixture
def baseline_path(request):
    return Path(__file__).parent.joinpath("baseline_images", "test_metrics", f"{request.node.name}.png")


Path(__file__).parents[0].absolute().joinpath("result_images").mkdir(exist_ok=True)
Path(__file__).parents[0].absolute().joinpath("result_images").joinpath("test_metrics").mkdir(exist_ok=True)


@pytest.mark.parametrize("custom_y_test, custom_y_pred, labels", [
    ("1 1 1 1 1 0 1 0 1 0 0 1 0 0 1 0 0 0 1 0 1 0 0 1 1 1 1 1 0 0 0 1 0 1 0 1 0 0 0 0 1 1 1 1 0 1 1 0 1 0",
     "0 1 1 1 1 0 0 0 1 0 0 0 0 1 1 0 1 1 0 0 1 0 0 1 1 0 1 1 0 0 0 1 0 1 0 1 1 0 1 0 1 1 1 1 0 1 1 1 1 1",
     [1, 0]),
    ("1 0 1 1 0 0 0 0 2 2 1 1 1 2 2 0 1 0 0 1 1 2 2 2 2 1 1 0 1 1 0 0 2 0 1 1 0 2 1 2 2 1 2 1 0 0 0 1 0 2 1 0 "
     "1 2 2 2 1 1 2 2 1 2 1 0 1 1 2 0 0 2 0 2 1 2 0",
     "0 0 2 2 2 0 1 0 1 2 2 2 2 2 2 0 2 1 2 2 0 2 2 2 1 1 2 0 1 2 0 2 2 0 2 2 2 2 2 2 2 0 2 1 0 0 1 1 1 0 1 1 2 "
     "0 1 2 0 0 0 2 2 2 2 0 0 2 2 1 0 2 0 0 2 0 2",
     [0, 1, 2])
], ids=["binary", "multiclass"])
def test_plot_confusion_matrix(custom_y_test, custom_y_pred, labels, result_path, baseline_path):
    y_test = np.fromstring(custom_y_test, dtype=int, sep=' ')
    y_pred = np.fromstring(custom_y_pred, dtype=int, sep=' ')

    ax = plot_confusion_matrix(y_test, y_pred, labels)

    # Assert that the confusion matrix is correctly calculated
    cm = ax[1].get_children()[0].get_array().data[:len(labels), :len(labels)]
    np.testing.assert_array_equal(cm,
                                  np.array([[np.sum((y_test == i) & (y_pred == j)) for j in labels] for i in labels]))

    # Assert that the accuracy and F1 score are correctly calculated
    accuracy = float(ax[2].texts[0].get_text().split(': ')[1])
    assert accuracy == np.mean(y_test == y_pred)

    plt.savefig(str(result_path))
    plt.cla()
    plt.close(plt.gcf())
    compare_images_from_paths(str(baseline_path), str(result_path))


def test_print_confusion_matrix_exception():
    with pytest.raises(ValueError):
        plot_confusion_matrix(np.array([]), np.array([]), [])


@pytest.mark.parametrize("test_case, n_samples, quantiles, random_state", [
    ("no_n_samples", None, np.linspace(0.05, 1, 20).tolist(), None),
    ("y_shape_n_outputs", None, np.linspace(0.05, 1, 20).tolist(), None),
    ("with_n_samples", list(range(10, 100, 10)), None, None),
    ("given_random_state_int", None, np.linspace(0.05, 1, 20).tolist(), 1),
    ("given_random_state", None, np.linspace(0.05, 1, 20).tolist(), RandomState(5))
], ids=["no_n_samples", "y_shape_n_outputs", "with_n_samples", "given_random_state_int", "given_random_state"])
def test_plot_metric_growth_per_labeled_instances(iris_data, classifiers, test_case, n_samples, quantiles, random_state,
                                                  result_path, baseline_path):
    if test_case == "y_shape_n_outputs":
        y_train = pd.get_dummies(pd.DataFrame(iris_data["y_train"]).astype(str))
        y_test = pd.get_dummies(pd.DataFrame(iris_data["y_test"]).astype(str))
    else:
        y_train, y_test = iris_data["y_train"], iris_data["y_test"]

    ax = plot_metric_growth_per_labeled_instances(
        iris_data["x_train"], y_train, iris_data["x_test"], y_test,
        classifiers, n_samples=n_samples, quantiles=quantiles, random_state=random_state
    )

    # Assert that the number of lines in the plot matches the number of classifiers
    assert len(ax.lines) == len(classifiers)

    # Assert that the x-axis label is correct
    assert ax.get_xlabel() == "Number of training samples"

    # Assert that the y-axis label is correct
    assert ax.get_ylabel() == "Metric score"

    plt.savefig(str(result_path))
    plt.cla()
    plt.close(plt.gcf())
    compare_images_from_paths(str(baseline_path), str(result_path))


def test_plot_metric_growth_per_labeled_instances_no_n_samples_no_quantiles(iris_data, classifiers):
    with pytest.raises(ValueError):
        plot_metric_growth_per_labeled_instances(
            iris_data["x_train"], iris_data["y_train"],
            iris_data["x_test"], iris_data["y_test"],
            classifiers, n_samples=None, quantiles=None
        )


def test_plot_metric_growth_per_labeled_instances_exists_ax(iris_data, classifiers, baseline_path, result_path):
    fig, ax = plt.subplots()
    ax.set_title("My ax")
    plot_metric_growth_per_labeled_instances(
        iris_data["x_train"], iris_data["y_train"],
        iris_data["x_test"], iris_data["y_test"],
        classifiers, ax=ax
    )
    plt.savefig(str(result_path))

    assert ax.get_title() == "My ax"
    plt.cla()
    plt.close(fig)
    compare_images_from_paths(str(baseline_path), str(result_path))


def test_plot_metric_growth_per_labeled_instances_verbose(iris_data, classifiers, capsys):
    plot_metric_growth_per_labeled_instances(
        iris_data["x_train"], iris_data["y_train"],
        iris_data["x_test"], iris_data["y_test"],
        classifiers, verbose=1
    )
    captured = capsys.readouterr().out
    expected = ("Fitting classifier DecisionTreeClassifier for 20 times\nFitting classifier RandomForestClassifier"
                " for 20 times\n")
    assert captured == expected


@pytest.mark.parametrize("display_breakdown, bins, threshold", [
    (False, None, 0.5),
    (True, None, 0.5),
    (False, [0, 0.3, 0.5, 0.8, 1], 0.5),
    (False, None, 0.3)
], ids=["default", "with_breakdown", "custom_bins", "custom_threshold"])
def test_visualize_accuracy_grouped_by_probability(display_breakdown, bins, threshold, result_path,
                                                   baseline_path):
    class_with_probabilities = pd.read_csv(Path(__file__).parent.joinpath("resources", "class_with_probabilities.csv"))
    ax = visualize_accuracy_grouped_by_probability(
        class_with_probabilities["loan_condition_cat"], 1,
        class_with_probabilities["probabilities"],
        display_breakdown=display_breakdown, bins=bins, threshold=threshold
    )

    # Assert that the x-axis label is correct
    assert ax.get_xlabel() == "Probability Range"

    # Assert that the y-axis label is correct
    assert ax.get_ylabel() == "Count"

    # Assert that the title is correct
    assert ax.get_title() == "Accuracy Distribution for 1 Class"

    plt.gcf().set_size_inches(10, 8)
    plt.savefig(str(result_path))
    plt.cla()
    plt.close(plt.gcf())
    compare_images_from_paths(str(baseline_path), str(result_path))


def test_visualize_accuracy_grouped_by_probability_exists_ax(baseline_path, result_path):
    fig, ax = plt.subplots()
    ax.set_title("My ax")

    class_with_probabilities = pd.read_csv(Path(__file__).parent.joinpath("resources", "class_with_probabilities.csv"))
    visualize_accuracy_grouped_by_probability(
        class_with_probabilities["loan_condition_cat"], 1,
        class_with_probabilities["probabilities"], ax=ax
    )

    assert ax.get_title() == "My ax"

    plt.gcf().set_size_inches(10, 8)
    plt.savefig(str(result_path))

    plt.cla()
    plt.close(fig)
    compare_images_from_paths(str(baseline_path), str(result_path))
