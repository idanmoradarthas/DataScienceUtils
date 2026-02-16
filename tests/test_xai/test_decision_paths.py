"""Tests for the decision paths generation functionality in ds_utils.xai."""

import os

import numpy as np
import pytest

from ds_utils.xai import generate_decision_paths


@pytest.fixture
def decision_tree_generate_decision_paths(mocker):
    """Fixture for a mocked decision tree for testing generate_decision_paths."""
    mock_tree = mocker.Mock()
    mock_tree.tree_.feature = np.array([3, -2, 3, 2, -2, -2, 2, -2, -2])
    mock_tree.tree_.threshold = np.array([0.80000001, -2.0, 1.75, 4.95000005, -2.0, -2.0, 4.85000014, -2.0, -2.0])
    mock_tree.tree_.children_left = np.array([1, -1, 3, 4, -1, -1, 7, -1, -1])
    mock_tree.tree_.children_right = np.array([2, -1, 6, 5, -1, -1, 8, -1, -1])
    mock_tree.tree_.value = np.array(
        [
            [[0.33333333, 0.33333333, 0.33333333]],
            [[1.0, 0.0, 0.0]],
            [[0.0, 0.5, 0.5]],
            [[0.0, 0.90740741, 0.09259259]],
            [[0.0, 0.97916667, 0.02083333]],
            [[0.0, 0.33333333, 0.66666667]],
            [[0.0, 0.02173913, 0.97826087]],
            [[0.0, 0.33333333, 0.66666667]],
            [[0.0, 0.0, 1.0]],
        ]
    )
    mock_tree.n_features_in_ = 4
    return mock_tree


@pytest.mark.parametrize(
    ("tree_name", "expected_name"),
    [
        ("iris_tree", "def iris_tree(petal width (cm), petal length (cm)):"),
        (None, "def tree(petal width (cm), petal length (cm)):"),
    ],
    ids=["default", "no_tree_name"],
)
def test_print_decision_paths(tree_name, expected_name, decision_tree_generate_decision_paths):
    """Test generation of decision paths from a tree."""
    with pytest.warns(DeprecationWarning, match="This module is deprecated. Use sklearn.tree.export_text instead"):
        result = generate_decision_paths(
            decision_tree_generate_decision_paths,
            ["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"],
            ["setosa", "versicolor", "virginica"],
            tree_name,
            "  ",
        )

    assert result.startswith(expected_name)

    expected = (
        f"def {tree_name if tree_name else 'tree'}(petal width (cm), petal length (cm)):"
        + os.linesep
        + "  if petal width (cm) <= 0.8000:"
        + os.linesep
        + "    # return class setosa with probability 0.5000"
        + os.linesep
        + '    return ("setosa", 0.5000)'
        + os.linesep
        + "  else:  # if petal width (cm) > 0.8000"
        + os.linesep
        + "    if petal width (cm) <= 1.7500:"
        + os.linesep
        + "      if petal length (cm) <= 4.9500:"
        + os.linesep
        + "        # return class versicolor with probability 0.9792"
        + os.linesep
        + '        return ("versicolor", 0.9792)'
        + os.linesep
        + "      else:  # if petal length (cm) > 4.9500"
        + os.linesep
        + "        # return class virginica with probability 0.6667"
        + os.linesep
        + '        return ("virginica", 0.6667)'
        + os.linesep
        + "    else:  # if petal width (cm) > 1.7500"
        + os.linesep
        + "      if petal length (cm) <= 4.8500:"
        + os.linesep
        + "        # return class virginica with probability 0.6667"
        + os.linesep
        + '        return ("virginica", 0.6667)'
        + os.linesep
        + "      else:  # if petal length (cm) > 4.8500"
        + os.linesep
        + "        # return class virginica with probability 0.5000"
        + os.linesep
        + '        return ("virginica", 0.5000)'
        + os.linesep
    )

    assert result == expected


def test_print_decision_paths_no_feature_names(decision_tree_generate_decision_paths):
    """Test decision path generation when feature_names are not provided."""
    with pytest.warns(DeprecationWarning, match="This module is deprecated. Use sklearn.tree.export_text instead"):
        result = generate_decision_paths(
            decision_tree_generate_decision_paths, None, ["setosa", "versicolor", "virginica"], "iris_tree", "  "
        )

    expected = (
        "def iris_tree(feature_3, feature_2):"
        + os.linesep
        + "  if feature_3 <= 0.8000:"
        + os.linesep
        + "    # return class setosa with probability 0.5000"
        + os.linesep
        + '    return ("setosa", 0.5000)'
        + os.linesep
        + "  else:  # if feature_3 > 0.8000"
        + os.linesep
        + "    if feature_3 <= 1.7500:"
        + os.linesep
        + "      if feature_2 <= 4.9500:"
        + os.linesep
        + "        # return class versicolor with probability 0.9792"
        + os.linesep
        + '        return ("versicolor", 0.9792)'
        + os.linesep
        + "      else:  # if feature_2 > 4.9500"
        + os.linesep
        + "        # return class virginica with probability 0.6667"
        + os.linesep
        + '        return ("virginica", 0.6667)'
        + os.linesep
        + "    else:  # if feature_3 > 1.7500"
        + os.linesep
        + "      if feature_2 <= 4.8500:"
        + os.linesep
        + "        # return class virginica with probability 0.6667"
        + os.linesep
        + '        return ("virginica", 0.6667)'
        + os.linesep
        + "      else:  # if feature_2 > 4.8500"
        + os.linesep
        + "        # return class virginica with probability 0.5000"
        + os.linesep
        + '        return ("virginica", 0.5000)'
        + os.linesep
    )

    assert result == expected


def test_print_decision_paths_no_class_names(decision_tree_generate_decision_paths):
    """Test decision path generation when class_names are not provided."""
    with pytest.warns(DeprecationWarning, match="This module is deprecated. Use sklearn.tree.export_text instead"):
        result = generate_decision_paths(
            decision_tree_generate_decision_paths,
            ["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"],
            None,
            "iris_tree",
            "  ",
        )

    expected = (
        "def iris_tree(petal width (cm), petal length (cm)):"
        + os.linesep
        + "  if petal width (cm) <= 0.8000:"
        + os.linesep
        + "    # return class class_0 with probability 0.5000"
        + os.linesep
        + '    return ("class_0", 0.5000)'
        + os.linesep
        + "  else:  # if petal width (cm) > 0.8000"
        + os.linesep
        + "    if petal width (cm) <= 1.7500:"
        + os.linesep
        + "      if petal length (cm) <= 4.9500:"
        + os.linesep
        + "        # return class class_1 with probability 0.9792"
        + os.linesep
        + '        return ("class_1", 0.9792)'
        + os.linesep
        + "      else:  # if petal length (cm) > 4.9500"
        + os.linesep
        + "        # return class class_2 with probability 0.6667"
        + os.linesep
        + '        return ("class_2", 0.6667)'
        + os.linesep
        + "    else:  # if petal width (cm) > 1.7500"
        + os.linesep
        + "      if petal length (cm) <= 4.8500:"
        + os.linesep
        + "        # return class class_2 with probability 0.6667"
        + os.linesep
        + '        return ("class_2", 0.6667)'
        + os.linesep
        + "      else:  # if petal length (cm) > 4.8500"
        + os.linesep
        + "        # return class class_2 with probability 0.5000"
        + os.linesep
        + '        return ("class_2", 0.5000)'
        + os.linesep
    )

    assert result == expected
