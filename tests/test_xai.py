import os

from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier

from ds_utils.xai import generate_decision_paths

iris = datasets.load_iris()
x = iris.data
y = iris.target


def test_print_decision_paths():
    # Create decision tree classifier object
    clf = DecisionTreeClassifier(random_state=0, max_depth=3)

    # Train model
    clf.fit(x, y)

    result = generate_decision_paths(clf, iris.feature_names, iris.target_names.tolist(), "iris_tree", "  ")

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

    result = generate_decision_paths(clf, iris.feature_names, iris.target_names.tolist(), indent_char="  ")

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

    result = generate_decision_paths(clf, None, iris.target_names.tolist(), "iris_tree", "  ")

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

    result = generate_decision_paths(clf, iris.feature_names, None, "iris_tree", "  ")

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
