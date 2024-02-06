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

RANDOM_STATE = np.random.RandomState(0)

x_train = pd.read_csv(Path(__file__).parents[0].absolute().joinpath("resources").joinpath("iris_x_train.csv"))
x_test = pd.read_csv(Path(__file__).parents[0].absolute().joinpath("resources").joinpath("iris_x_test.csv"))
y_train = pd.read_csv(Path(__file__).parents[0].absolute().joinpath("resources").joinpath("iris_y_train.csv"))
y_test = pd.read_csv(Path(__file__).parents[0].absolute().joinpath("resources").joinpath("iris_y_test.csv"))

Path(__file__).parents[0].absolute().joinpath("result_images").mkdir(exist_ok=True)
Path(__file__).parents[0].absolute().joinpath("result_images").joinpath("test_metrics").mkdir(exist_ok=True)


def test_print_confusion_matrix_binary():
    custom_y_test = "1 1 1 1 1 0 1 0 1 0 0 1 0 0 1 0 0 0 1 0 1 0 0 1 1 1 1 1 0 0 0 1 0 1 0 1 0 0 0 0 1 1 1 1 0 1 1 " \
                    "0 1 0"
    custom_y_pred = "0 1 1 1 1 0 0 0 1 0 0 0 0 1 1 0 1 1 0 0 1 0 0 1 1 0 1 1 0 0 0 1 0 1 0 1 1 0 1 0 1 1 1 1 0 1 1 " \
                    "1 1 1"

    plot_confusion_matrix(np.fromstring(custom_y_test, dtype=int, sep=' '),
                          np.fromstring(custom_y_pred, dtype=int, sep=' '),
                          [1, 0])

    result_path = Path(__file__).parents[0].absolute().joinpath("result_images").joinpath("test_metrics").joinpath(
        "test_print_confusion_matrix_binary.png")
    plt.savefig(str(result_path))

    baseline_path = Path(__file__).parents[0].absolute().joinpath("baseline_images").joinpath("test_metrics").joinpath(
        "test_print_confusion_matrix_binary.png")
    plt.cla()
    plt.close(plt.gcf())
    compare_images_from_paths(str(baseline_path), str(result_path))


def test_print_confusion_matrix():
    custom_y_test = "1 0 1 1 0 0 0 0 2 2 1 1 1 2 2 0 1 0 0 1 1 2 2 2 2 1 1 0 1 1 0 0 2 0 1 1 0 2 1 2 2 1 2 1 0 0 0 1 " \
                    "0 2 1 0 1 2 2 2 1 1 2 2 1 2 1 0 1 1 2 0 0 2 0 2 1 2 0"
    y_pred = "0 0 2 2 2 0 1 0 1 2 2 2 2 2 2 0 2 1 2 2 0 2 2 2 1 1 2 0 1 2 0 2 2 0 2 2 2 2 2 2 2 0 2 1 0 0 1 1 1 0 1 1" \
             " 2 0 1 2 0 0 0 2 2 2 2 0 0 2 2 1 0 2 0 0 2 0 2"

    plot_confusion_matrix(np.fromstring(custom_y_test, dtype=int, sep=' '),
                          np.fromstring(y_pred, dtype=int, sep=' '),
                          [0, 1, 2])

    result_path = Path(__file__).parents[0].absolute().joinpath("result_images").joinpath("test_metrics").joinpath(
        "test_print_confusion_matrix.png")
    plt.savefig(str(result_path))

    baseline_path = Path(__file__).parents[0].absolute().joinpath("baseline_images").joinpath("test_metrics").joinpath(
        "test_print_confusion_matrix.png")
    plt.cla()
    plt.close(plt.gcf())
    compare_images_from_paths(str(baseline_path), str(result_path))


def test_print_confusion_matrix_exception():
    with pytest.raises(ValueError):
        plot_confusion_matrix(np.array([]), np.array([]), [])


def test_plot_metric_growth_per_labeled_instances_no_n_samples():
    plot_metric_growth_per_labeled_instances(x_train, y_train, x_test, y_test,
                                             {"DecisionTreeClassifier": DecisionTreeClassifier(random_state=0),
                                              "RandomForestClassifier": RandomForestClassifier(random_state=0,
                                                                                               n_estimators=5)})
    result_path = Path(__file__).parents[0].absolute().joinpath("result_images").joinpath(
        "test_metrics").joinpath("test_plot_metric_growth_per_labeled_instances_no_n_samples.png")
    plt.savefig(str(result_path))

    baseline_path = Path(__file__).parents[0].absolute().joinpath("baseline_images").joinpath(
        "test_metrics").joinpath("test_plot_metric_growth_per_labeled_instances_no_n_samples.png")
    plt.cla()
    plt.close(plt.gcf())
    compare_images_from_paths(str(baseline_path), str(result_path))


def test_plot_metric_growth_per_labeled_instances_y_shape_n_outputs():
    custom_y_train = pd.get_dummies(y_train.astype(str))
    custom_y_test = pd.get_dummies(y_test.astype(str))
    plot_metric_growth_per_labeled_instances(x_train.values, custom_y_train.values, x_test.values, custom_y_test.values,
                                             {"DecisionTreeClassifier": DecisionTreeClassifier(random_state=0),
                                              "RandomForestClassifier": RandomForestClassifier(random_state=0,
                                                                                               n_estimators=5)})
    result_path = Path(__file__).parents[0].absolute().joinpath("result_images").joinpath(
        "test_metrics").joinpath("test_plot_metric_growth_per_labeled_instances_y_shape_n_outputs.png")
    plt.savefig(str(result_path))

    baseline_path = Path(__file__).parents[0].absolute().joinpath("baseline_images").joinpath(
        "test_metrics").joinpath("test_plot_metric_growth_per_labeled_instances_y_shape_n_outputs.png")
    plt.cla()
    plt.close(plt.gcf())
    compare_images_from_paths(str(baseline_path), str(result_path))


def test_plot_metric_growth_per_labeled_instances_with_n_samples():
    plot_metric_growth_per_labeled_instances(x_train, y_train, x_test, y_test,
                                             {"DecisionTreeClassifier": DecisionTreeClassifier(random_state=0),
                                              "RandomForestClassifier": RandomForestClassifier(random_state=0,
                                                                                               n_estimators=5)},
                                             n_samples=list(range(10, x_train.shape[0], 10)))
    result_path = Path(__file__).parents[0].absolute().joinpath("result_images").joinpath(
        "test_metrics").joinpath("test_plot_metric_growth_per_labeled_instances_with_n_samples.png")
    plt.savefig(str(result_path))

    baseline_path = Path(__file__).parents[0].absolute().joinpath("baseline_images").joinpath(
        "test_metrics").joinpath("test_plot_metric_growth_per_labeled_instances_with_n_samples.png")
    plt.cla()
    plt.close(plt.gcf())
    compare_images_from_paths(str(baseline_path), str(result_path))


def test_plot_metric_growth_per_labeled_instances_no_n_samples_no_quantiles():
    with pytest.raises(ValueError):
        plot_metric_growth_per_labeled_instances(np.array([]), np.array([]), np.array([]), np.array([]),
                                                 {"DecisionTreeClassifier": DecisionTreeClassifier(random_state=0),
                                                  "RandomForestClassifier": RandomForestClassifier(random_state=0,
                                                                                                   n_estimators=5)},
                                                 n_samples=None, quantiles=None)


def test_plot_metric_growth_per_labeled_instances_given_random_state_int():
    plot_metric_growth_per_labeled_instances(x_train, y_train, x_test, y_test,
                                             {"DecisionTreeClassifier": DecisionTreeClassifier(random_state=0),
                                              "RandomForestClassifier": RandomForestClassifier(random_state=0,
                                                                                               n_estimators=5)},
                                             random_state=1)
    result_path = Path(__file__).parents[0].absolute().joinpath("result_images").joinpath(
        "test_metrics").joinpath("test_plot_metric_growth_per_labeled_instances_given_random_state_int.png")
    plt.savefig(str(result_path))

    baseline_path = Path(__file__).parents[0].absolute().joinpath("baseline_images").joinpath(
        "test_metrics").joinpath("test_plot_metric_growth_per_labeled_instances_given_random_state_int.png")
    plt.cla()
    plt.close(plt.gcf())
    compare_images_from_paths(str(baseline_path), str(result_path))


def test_plot_metric_growth_per_labeled_instances_given_random_state():
    plot_metric_growth_per_labeled_instances(x_train, y_train, x_test, y_test,
                                             {"DecisionTreeClassifier": DecisionTreeClassifier(random_state=0),
                                              "RandomForestClassifier": RandomForestClassifier(random_state=0,
                                                                                               n_estimators=5)},
                                             random_state=RandomState(5))
    result_path = Path(__file__).parents[0].absolute().joinpath("result_images").joinpath(
        "test_metrics").joinpath("test_plot_metric_growth_per_labeled_instances_given_random_state.png")
    plt.savefig(str(result_path))

    baseline_path = Path(__file__).parents[0].absolute().joinpath("baseline_images").joinpath(
        "test_metrics").joinpath("test_plot_metric_growth_per_labeled_instances_given_random_state.png")
    plt.cla()
    plt.close(plt.gcf())
    compare_images_from_paths(str(baseline_path), str(result_path))


def test_plot_metric_growth_per_labeled_instances_exists_ax():
    plt.figure()
    ax = plt.gca()

    ax.set_title("My ax")
    plot_metric_growth_per_labeled_instances(x_train, y_train, x_test, y_test,
                                             {"DecisionTreeClassifier": DecisionTreeClassifier(random_state=0),
                                              "RandomForestClassifier": RandomForestClassifier(random_state=0,
                                                                                               n_estimators=5)},
                                             ax=ax)
    result_path = Path(__file__).parents[0].absolute().joinpath("result_images").joinpath(
        "test_metrics").joinpath("test_plot_metric_growth_per_labeled_instances_exists_ax.png")
    plt.savefig(str(result_path))

    baseline_path = Path(__file__).parents[0].absolute().joinpath("baseline_images").joinpath(
        "test_metrics").joinpath("test_plot_metric_growth_per_labeled_instances_exists_ax.png")
    plt.cla()
    plt.close(plt.gcf())
    compare_images_from_paths(str(baseline_path), str(result_path))


def test_plot_metric_growth_per_labeled_instances_verbose(capsys):
    plot_metric_growth_per_labeled_instances(x_train, y_train, x_test, y_test,
                                             {"DecisionTreeClassifier": DecisionTreeClassifier(random_state=0),
                                              "RandomForestClassifier": RandomForestClassifier(random_state=0,
                                                                                               n_estimators=5)},
                                             verbose=1)
    captured = capsys.readouterr().out
    expected = "Fitting classifier DecisionTreeClassifier for 20 times\nFitting classifier RandomForestClassifier " \
               "for 20 times\n"

    assert expected == captured


def test_visualize_accuracy_grouped_by_probability():
    class_with_probabilities = pd.read_csv(
        Path(__file__).parents[0].absolute().joinpath("resources").joinpath("class_with_probabilities.csv"))
    visualize_accuracy_grouped_by_probability(class_with_probabilities["loan_condition_cat"], 1,
                                              class_with_probabilities["probabilities"])
    result_path = Path(__file__).parents[0].absolute().joinpath("result_images").joinpath(
        "test_metrics").joinpath("test_visualize_accuracy_grouped_by_probability.png")
    plt.gcf().set_size_inches(10, 8)
    plt.savefig(str(result_path))

    baseline_path = Path(__file__).parents[0].absolute().joinpath("baseline_images").joinpath(
        "test_metrics").joinpath("test_visualize_accuracy_grouped_by_probability.png")
    plt.cla()
    plt.close(plt.gcf())
    compare_images_from_paths(str(baseline_path), str(result_path))


def test_visualize_accuracy_grouped_by_probability_exists_ax():
    plt.figure()
    ax = plt.gca()

    ax.set_title("My ax")

    class_with_probabilities = pd.read_csv(
        Path(__file__).parents[0].absolute().joinpath("resources").joinpath("class_with_probabilities.csv"))
    visualize_accuracy_grouped_by_probability(class_with_probabilities["loan_condition_cat"], 1,
                                              class_with_probabilities["probabilities"], ax=ax)
    result_path = Path(__file__).parents[0].absolute().joinpath("result_images").joinpath(
        "test_metrics").joinpath("test_visualize_accuracy_grouped_by_probability_exists_ax.png")
    plt.gcf().set_size_inches(10, 8)
    plt.savefig(str(result_path))

    baseline_path = Path(__file__).parents[0].absolute().joinpath("baseline_images").joinpath(
        "test_metrics").joinpath("test_visualize_accuracy_grouped_by_probability_exists_ax.png")
    plt.cla()
    plt.close(plt.gcf())
    compare_images_from_paths(str(baseline_path), str(result_path))


def test_visualize_accuracy_grouped_by_probability_with_breakdown():
    class_with_probabilities = pd.read_csv(
        Path(__file__).parents[0].absolute().joinpath("resources").joinpath("class_with_probabilities.csv"))
    visualize_accuracy_grouped_by_probability(class_with_probabilities["loan_condition_cat"], 1,
                                              class_with_probabilities["probabilities"], display_breakdown=True)
    result_path = Path(__file__).parents[0].absolute().joinpath("result_images").joinpath(
        "test_metrics").joinpath("test_visualize_accuracy_grouped_by_probability_with_breakdown.png")
    plt.gcf().set_size_inches(10, 8)
    plt.savefig(str(result_path))

    baseline_path = Path(__file__).parents[0].absolute().joinpath("baseline_images").joinpath(
        "test_metrics").joinpath("test_visualize_accuracy_grouped_by_probability_with_breakdown.png")
    plt.cla()
    plt.close(plt.gcf())
    compare_images_from_paths(str(baseline_path), str(result_path))


def test_visualize_accuracy_grouped_by_probability_custom_bins():
    class_with_probabilities = pd.read_csv(
        Path(__file__).parents[0].absolute().joinpath("resources").joinpath("class_with_probabilities.csv"))
    visualize_accuracy_grouped_by_probability(class_with_probabilities["loan_condition_cat"], 1,
                                              class_with_probabilities["probabilities"], bins=[0, 0.3, 0.5, 0.8, 1])
    result_path = Path(__file__).parents[0].absolute().joinpath("result_images").joinpath(
        "test_metrics").joinpath("test_visualize_accuracy_grouped_by_probability_custom_bins.png")
    plt.gcf().set_size_inches(10, 8)
    plt.savefig(str(result_path))

    baseline_path = Path(__file__).parents[0].absolute().joinpath("baseline_images").joinpath(
        "test_metrics").joinpath("test_visualize_accuracy_grouped_by_probability_custom_bins.png")
    plt.cla()
    plt.close(plt.gcf())
    compare_images_from_paths(str(baseline_path), str(result_path))


def test_visualize_accuracy_grouped_by_probability_custom_threshold():
    class_with_probabilities = pd.read_csv(
        Path(__file__).parents[0].absolute().joinpath("resources").joinpath("class_with_probabilities.csv"))
    visualize_accuracy_grouped_by_probability(class_with_probabilities["loan_condition_cat"], 1,
                                              class_with_probabilities["probabilities"], threshold=0.3)
    result_path = Path(__file__).parents[0].absolute().joinpath("result_images").joinpath(
        "test_metrics").joinpath("test_visualize_accuracy_grouped_by_probability_custom_threshold.png")
    plt.gcf().set_size_inches(10, 8)
    plt.savefig(str(result_path))

    baseline_path = Path(__file__).parents[0].absolute().joinpath("baseline_images").joinpath(
        "test_metrics").joinpath("test_visualize_accuracy_grouped_by_probability_custom_threshold.png")
    plt.cla()
    plt.close(plt.gcf())
    compare_images_from_paths(str(baseline_path), str(result_path))
