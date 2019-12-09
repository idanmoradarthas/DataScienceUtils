import os
from pathlib import Path

import pandas
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier

from ds_utils.visualization_aids import draw_tree, visualize_features, print_decision_paths
from tests.utils import compare_images_paths


def test_draw_tree():
    iris = datasets.load_iris()
    x = iris.data
    y = iris.target

    # Create decision tree classifier object
    clf = DecisionTreeClassifier(random_state=0)

    # Train model
    clf.fit(x, y)

    plot = draw_tree(clf, iris.feature_names, iris.target_names)

    Path(__file__).parents[0].absolute().joinpath("result_images").mkdir(exist_ok=True)
    Path(__file__).parents[0].absolute().joinpath("result_images").joinpath("test_visualization_aids").mkdir(
        exist_ok=True)
    result_path = Path(__file__).parents[0].absolute().joinpath("result_images").joinpath(
        "test_visualization_aids").joinpath("test_draw_tree.png")
    plot.savefig(str(result_path))

    baseline_path = Path(__file__).parents[0].absolute().joinpath("baseline_images").joinpath(
        "test_visualization_aids").joinpath("test_draw_tree.png")
    compare_images_paths(str(baseline_path), str(result_path))


def test_visualize_features():
    file_path = Path(__file__).parents[0].joinpath("resources").joinpath("loan_final313.csv")
    loan_frame = pandas.read_csv(file_path, encoding="latin1", parse_dates=["issue_d"])
    loan_frame = loan_frame.drop("id", axis=1)

    plot = visualize_features(loan_frame)
    Path(__file__).parents[0].absolute().joinpath("result_images").mkdir(exist_ok=True)
    Path(__file__).parents[0].absolute().joinpath("result_images").joinpath("test_visualization_aids").mkdir(
        exist_ok=True)
    result_path = Path(__file__).parents[0].absolute().joinpath("result_images").joinpath(
        "test_visualization_aids").joinpath("test_visualize_features.png")
    plot.savefig(str(result_path))

    baseline_path = Path(__file__).parents[0].absolute().joinpath("baseline_images").joinpath(
        "test_visualization_aids").joinpath("test_visualize_features.png")
    compare_images_paths(str(baseline_path), str(result_path))


def test_print_decision_paths():
    iris = datasets.load_iris()
    x = iris.data
    y = iris.target

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
    iris = datasets.load_iris()
    x = iris.data
    y = iris.target

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
    iris = datasets.load_iris()
    x = iris.data
    y = iris.target

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
    iris = datasets.load_iris()
    x = iris.data
    y = iris.target

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
