import os
from pathlib import Path

import numpy
import pandas
from matplotlib import pyplot
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier

from ds_utils.visualization_aids import draw_tree, visualize_features, print_decision_paths
from tests.utils import compare_images_paths

iris = datasets.load_iris()
x = iris.data
y = iris.target

Path(__file__).parents[0].absolute().joinpath("result_images").mkdir(exist_ok=True)
Path(__file__).parents[0].absolute().joinpath("result_images").joinpath("test_visualization_aids").mkdir(
    exist_ok=True)


def test_draw_tree():
    # Create decision tree classifier object
    clf = DecisionTreeClassifier(random_state=0)

    # Train model
    clf.fit(x, y)

    draw_tree(clf, iris.feature_names, iris.target_names)
    result_path = Path(__file__).parents[0].absolute().joinpath("result_images").joinpath(
        "test_visualization_aids").joinpath("test_draw_tree.png")
    pyplot.savefig(str(result_path))

    baseline_path = Path(__file__).parents[0].absolute().joinpath("baseline_images").joinpath(
        "test_visualization_aids").joinpath("test_draw_tree.png")
    compare_images_paths(str(baseline_path), str(result_path))


def test_draw_tree_exists_ax():
    # Create decision tree classifier object
    clf = DecisionTreeClassifier(random_state=0)

    # Train model
    clf.fit(x, y)

    pyplot.figure()
    ax = pyplot.gca()

    ax.set_title("My ax")

    draw_tree(clf, iris.feature_names, iris.target_names, ax=ax)

    result_path = Path(__file__).parents[0].absolute().joinpath("result_images").joinpath(
        "test_visualization_aids").joinpath("test_draw_tree_exists_ax.png")
    pyplot.savefig(str(result_path))

    baseline_path = Path(__file__).parents[0].absolute().joinpath("baseline_images").joinpath(
        "test_visualization_aids").joinpath("test_draw_tree_exists_ax.png")
    compare_images_paths(str(baseline_path), str(result_path))


def test_visualize_features():
    file_path = Path(__file__).parents[0].joinpath("resources").joinpath("loan_final313.csv")
    loan_frame = pandas.read_csv(file_path, encoding="latin1", parse_dates=["issue_d"])
    loan_frame = loan_frame.drop("id", axis=1)

    visualize_features(loan_frame)

    result_path = Path(__file__).parents[0].absolute().joinpath("result_images").joinpath(
        "test_visualization_aids").joinpath("test_visualize_features.png")
    pyplot.gcf().set_size_inches(20, 30)
    pyplot.savefig(str(result_path))

    baseline_path = Path(__file__).parents[0].absolute().joinpath("baseline_images").joinpath(
        "test_visualization_aids").joinpath("test_visualize_features.png")
    compare_images_paths(str(baseline_path), str(result_path))


def test_visualize_features_list_of_features():
    frame = pandas.DataFrame(x, columns=iris.feature_names)
    visualize_features(frame, iris.feature_names[:2])

    result_path = Path(__file__).parents[0].absolute().joinpath("result_images").joinpath(
        "test_visualization_aids").joinpath("test_visualize_features_list_of_features.png")
    pyplot.savefig(str(result_path))

    baseline_path = Path(__file__).parents[0].absolute().joinpath("baseline_images").joinpath(
        "test_visualization_aids").joinpath("test_visualize_features_list_of_features.png")
    compare_images_paths(str(baseline_path), str(result_path))


def test_visualize_features_remove_na():
    file_path = Path(__file__).parents[0].joinpath("resources").joinpath("loan_final313.csv")
    loan_frame = pandas.read_csv(file_path, encoding="latin1", parse_dates=["issue_d"])
    loan_frame = loan_frame.drop("id", axis=1)
    loan_frame = loan_frame.sample(1000, random_state=0)
    loan_frame = pandas.concat(
        [loan_frame, pandas.DataFrame([[numpy.nan] * len(loan_frame.columns)] * 250, columns=loan_frame.columns)],
        ignore_index=True).sample(frac=1, random_state=0)

    visualize_features(loan_frame, remove_na=True)

    result_path = Path(__file__).parents[0].absolute().joinpath("result_images").joinpath(
        "test_visualization_aids").joinpath("test_visualize_features_remove_na.png")
    pyplot.gcf().set_size_inches(20, 30)
    pyplot.savefig(str(result_path))

    baseline_path = Path(__file__).parents[0].absolute().joinpath("baseline_images").joinpath(
        "test_visualization_aids").joinpath("test_visualize_features_remove_na.png")
    compare_images_paths(str(baseline_path), str(result_path))


def test_print_decision_paths():
    # Create decision tree classifier object
    clf = DecisionTreeClassifier(random_state=0, max_depth=3)

    # Train model
    clf.fit(x, y)

    result = print_decision_paths(clf, iris.feature_names, iris.target_names.tolist(), "iris_tree")

    expected = 'def iris_tree(petal width (cm), petal length (cm)):' + os.linesep + \
               '  if petal width (cm) <= 0.8000:' + os.linesep + \
               '    # return class setosa with probability 0.9804' + os.linesep + \
               '    return ("setosa", 0.9804)' + os.linesep + \
               '  else:  # if petal width (cm) > 0.8000' + os.linesep + \
               '    if petal width (cm) <= 1.7500:' + os.linesep + \
               '      if petal length (cm) <= 4.9500:' + os.linesep + \
               '        # return class versicolor with probability 0.9792' + os.linesep + \
               '        return ("versicolor", 0.9792)' + os.linesep + \
               '      else:  # if petal length (cm) > 4.9500' + os.linesep + \
               '        # return class virginica with probability 0.6667' + os.linesep + \
               '        return ("virginica", 0.6667)' + os.linesep + \
               '    else:  # if petal width (cm) > 1.7500' + os.linesep + \
               '      if petal length (cm) <= 4.8500:' + os.linesep + \
               '        # return class virginica with probability 0.6667' + os.linesep + \
               '        return ("virginica", 0.6667)' + os.linesep + \
               '      else:  # if petal length (cm) > 4.8500' + os.linesep + \
               '        # return class virginica with probability 0.9773' + os.linesep + \
               '        return ("virginica", 0.9773)' + os.linesep

    assert result == expected


def test_print_decision_paths_no_tree_name():
    # Create decision tree classifier object
    clf = DecisionTreeClassifier(random_state=0, max_depth=3)

    # Train model
    clf.fit(x, y)

    result = print_decision_paths(clf, iris.feature_names, iris.target_names.tolist())

    expected = 'def tree(petal width (cm), petal length (cm)):' + os.linesep + \
               '  if petal width (cm) <= 0.8000:' + os.linesep + \
               '    # return class setosa with probability 0.9804' + os.linesep + \
               '    return ("setosa", 0.9804)' + os.linesep + \
               '  else:  # if petal width (cm) > 0.8000' + os.linesep + \
               '    if petal width (cm) <= 1.7500:' + os.linesep + \
               '      if petal length (cm) <= 4.9500:' + os.linesep + \
               '        # return class versicolor with probability 0.9792' + os.linesep + \
               '        return ("versicolor", 0.9792)' + os.linesep + \
               '      else:  # if petal length (cm) > 4.9500' + os.linesep + \
               '        # return class virginica with probability 0.6667' + os.linesep + \
               '        return ("virginica", 0.6667)' + os.linesep + \
               '    else:  # if petal width (cm) > 1.7500' + os.linesep + \
               '      if petal length (cm) <= 4.8500:' + os.linesep + \
               '        # return class virginica with probability 0.6667' + os.linesep + \
               '        return ("virginica", 0.6667)' + os.linesep + \
               '      else:  # if petal length (cm) > 4.8500' + os.linesep + \
               '        # return class virginica with probability 0.9773' + os.linesep + \
               '        return ("virginica", 0.9773)' + os.linesep

    assert result == expected


def test_print_decision_paths_no_feature_names():
    # Create decision tree classifier object
    clf = DecisionTreeClassifier(random_state=0, max_depth=3)

    # Train model
    clf.fit(x, y)

    result = print_decision_paths(clf, None, iris.target_names.tolist(), "iris_tree")

    expected = 'def iris_tree(feature_3, feature_2):' + os.linesep + \
               '  if feature_3 <= 0.8000:' + os.linesep + \
               '    # return class setosa with probability 0.9804' + os.linesep + \
               '    return ("setosa", 0.9804)' + os.linesep + \
               '  else:  # if feature_3 > 0.8000' + os.linesep + \
               '    if feature_3 <= 1.7500:' + os.linesep + \
               '      if feature_2 <= 4.9500:' + os.linesep + \
               '        # return class versicolor with probability 0.9792' + os.linesep + \
               '        return ("versicolor", 0.9792)' + os.linesep + \
               '      else:  # if feature_2 > 4.9500' + os.linesep + \
               '        # return class virginica with probability 0.6667' + os.linesep + \
               '        return ("virginica", 0.6667)' + os.linesep + \
               '    else:  # if feature_3 > 1.7500' + os.linesep + \
               '      if feature_2 <= 4.8500:' + os.linesep + \
               '        # return class virginica with probability 0.6667' + os.linesep + \
               '        return ("virginica", 0.6667)' + os.linesep + \
               '      else:  # if feature_2 > 4.8500' + os.linesep + \
               '        # return class virginica with probability 0.9773' + os.linesep + \
               '        return ("virginica", 0.9773)' + os.linesep

    assert result == expected


def test_print_decision_paths_no_class_names():
    # Create decision tree classifier object
    clf = DecisionTreeClassifier(random_state=0, max_depth=3)

    # Train model
    clf.fit(x, y)

    result = print_decision_paths(clf, iris.feature_names, None, "iris_tree")

    expected = 'def iris_tree(petal width (cm), petal length (cm)):' + os.linesep + \
               '  if petal width (cm) <= 0.8000:' + os.linesep + \
               '    # return class class_0 with probability 0.9804' + os.linesep + \
               '    return ("class_0", 0.9804)' + os.linesep + \
               '  else:  # if petal width (cm) > 0.8000' + os.linesep + \
               '    if petal width (cm) <= 1.7500:' + os.linesep + \
               '      if petal length (cm) <= 4.9500:' + os.linesep + \
               '        # return class class_1 with probability 0.9792' + os.linesep + \
               '        return ("class_1", 0.9792)' + os.linesep + \
               '      else:  # if petal length (cm) > 4.9500' + os.linesep + \
               '        # return class class_2 with probability 0.6667' + os.linesep + \
               '        return ("class_2", 0.6667)' + os.linesep + \
               '    else:  # if petal width (cm) > 1.7500' + os.linesep + \
               '      if petal length (cm) <= 4.8500:' + os.linesep + \
               '        # return class class_2 with probability 0.6667' + os.linesep + \
               '        return ("class_2", 0.6667)' + os.linesep + \
               '      else:  # if petal length (cm) > 4.8500' + os.linesep + \
               '        # return class class_2 with probability 0.9773' + os.linesep + \
               '        return ("class_2", 0.9773)' + os.linesep

    assert result == expected
