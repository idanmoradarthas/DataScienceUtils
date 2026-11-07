"""Tests for the decision paths generation functionality in ds_utils.xai."""

import inspect
import os

import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

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
        ("iris_tree", "def iris_tree(petal_width_cm, petal_length_cm):"),
        (None, "def tree(petal_width_cm, petal_length_cm):"),
    ],
    ids=["default", "no_tree_name"],
)
def test_print_decision_paths(tree_name, expected_name, decision_tree_generate_decision_paths):
    """Test generation of decision paths from a tree."""
    with pytest.warns(DeprecationWarning, match="This module is deprecated. Use sklearn.tree.export_text instead"):
        result = generate_decision_paths(
            decision_tree_generate_decision_paths,
            ["sepal_length_cm", "sepal_width_cm", "petal_length_cm", "petal_width_cm"],
            ["setosa", "versicolor", "virginica"],
            tree_name,
            "  ",
        )

    assert result.startswith(expected_name)

    expected = (
        f"def {tree_name if tree_name else 'tree'}(petal_width_cm, petal_length_cm):"
        + os.linesep
        + "  if petal_width_cm <= 0.8000:"
        + os.linesep
        + "    # return class setosa with probability 0.5000"
        + os.linesep
        + '    return ("setosa", 0.5000)'
        + os.linesep
        + "  else:  # if petal_width_cm > 0.8000"
        + os.linesep
        + "    if petal_width_cm <= 1.7500:"
        + os.linesep
        + "      if petal_length_cm <= 4.9500:"
        + os.linesep
        + "        # return class versicolor with probability 0.9792"
        + os.linesep
        + '        return ("versicolor", 0.9792)'
        + os.linesep
        + "      else:  # if petal_length_cm > 4.9500"
        + os.linesep
        + "        # return class virginica with probability 0.6667"
        + os.linesep
        + '        return ("virginica", 0.6667)'
        + os.linesep
        + "    else:  # if petal_width_cm > 1.7500"
        + os.linesep
        + "      if petal_length_cm <= 4.8500:"
        + os.linesep
        + "        # return class virginica with probability 0.6667"
        + os.linesep
        + '        return ("virginica", 0.6667)'
        + os.linesep
        + "      else:  # if petal_length_cm > 4.8500"
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
            ["sepal_length_cm", "sepal_width_cm", "petal_length_cm", "petal_width_cm"],
            None,
            "iris_tree",
            "  ",
        )

    expected = (
        "def iris_tree(petal_width_cm, petal_length_cm):"
        + os.linesep
        + "  if petal_width_cm <= 0.8000:"
        + os.linesep
        + "    # return class class_0 with probability 0.5000"
        + os.linesep
        + '    return ("class_0", 0.5000)'
        + os.linesep
        + "  else:  # if petal_width_cm > 0.8000"
        + os.linesep
        + "    if petal_width_cm <= 1.7500:"
        + os.linesep
        + "      if petal_length_cm <= 4.9500:"
        + os.linesep
        + "        # return class class_1 with probability 0.9792"
        + os.linesep
        + '        return ("class_1", 0.9792)'
        + os.linesep
        + "      else:  # if petal_length_cm > 4.9500"
        + os.linesep
        + "        # return class class_2 with probability 0.6667"
        + os.linesep
        + '        return ("class_2", 0.6667)'
        + os.linesep
        + "    else:  # if petal_width_cm > 1.7500"
        + os.linesep
        + "      if petal_length_cm <= 4.8500:"
        + os.linesep
        + "        # return class class_2 with probability 0.6667"
        + os.linesep
        + '        return ("class_2", 0.6667)'
        + os.linesep
        + "      else:  # if petal_length_cm > 4.8500"
        + os.linesep
        + "        # return class class_2 with probability 0.5000"
        + os.linesep
        + '        return ("class_2", 0.5000)'
        + os.linesep
    )

    assert result == expected


@pytest.mark.parametrize(
    "invalid_name",
    ["mean radius", "mean-radius", "mean.radius", "1st", "class", ""],
)
def test_generate_decision_paths_invalid_feature_name_raises(invalid_name, decision_tree_generate_decision_paths):
    """Test that invalid feature names used by the tree raise ValueError."""
    feature_names = [invalid_name] * 4
    with (
        pytest.warns(DeprecationWarning, match="This module is deprecated. Use sklearn.tree.export_text instead"),
        pytest.raises(ValueError, match="valid Python identifiers") as exc_info,
    ):
        generate_decision_paths(
            decision_tree_generate_decision_paths,
            feature_names,
            ["setosa", "versicolor", "virginica"],
            "tree",
        )

    assert repr(invalid_name) in str(exc_info.value)


def test_generate_decision_paths_unused_invalid_feature_does_not_raise(decision_tree_generate_decision_paths):
    """Test that an invalid feature name at an index the tree never splits on does not raise ValueError."""
    # The fixture tree splits only on features 2 and 3; features 0 and 1 are unused.
    feature_names = ["invalid-name 0", "invalid-name 1", "petal_length_cm", "petal_width_cm"]
    with pytest.warns(DeprecationWarning, match="This module is deprecated. Use sklearn.tree.export_text instead"):
        result = generate_decision_paths(
            decision_tree_generate_decision_paths,
            feature_names,
            ["setosa", "versicolor", "virginica"],
            "valid_tree",
        )
    assert result.startswith("def valid_tree(petal_width_cm, petal_length_cm):")


@pytest.mark.parametrize(
    "invalid_tree_name",
    ["my tree", "1tree", "def"],
)
def test_generate_decision_paths_invalid_tree_name_raises(invalid_tree_name, decision_tree_generate_decision_paths):
    """Test that invalid tree_name values raise ValueError."""
    feature_names = ["feat_0", "feat_1", "feat_2", "feat_3"]
    with (
        pytest.warns(DeprecationWarning, match="This module is deprecated. Use sklearn.tree.export_text instead"),
        pytest.raises(ValueError, match="tree_name"),
    ):
        generate_decision_paths(
            decision_tree_generate_decision_paths,
            feature_names,
            ["setosa", "versicolor", "virginica"],
            invalid_tree_name,
        )


def test_generate_decision_paths_escaped_class_names():
    """Test that special class names with quotes/slashes are properly escaped and produce valid executable code."""
    iris = load_iris()
    X = iris.data
    y = iris.target
    clf = DecisionTreeClassifier(max_depth=2, random_state=42)
    clf.fit(X, y)

    feature_names = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    class_names = ['he said "hi"', "back\\slash", "plain"]

    with pytest.warns(DeprecationWarning, match="This module is deprecated. Use sklearn.tree.export_text instead"):
        code_str = generate_decision_paths(
            clf,
            feature_names=feature_names,
            class_names=class_names,
            tree_name="predict_special_classes",
        )

    compiled = compile(code_str, "<test_tree>", "exec")
    local_env = {}
    exec(compiled, {}, local_env)
    tree_fn = local_env["predict_special_classes"]

    sig = inspect.signature(tree_fn)
    # Call the generated function over samples spanning all classes
    predicted_classes = set()
    for row in X[::5]:
        kwargs = {param_name: row[feature_names.index(param_name)] for param_name in sig.parameters}
        predicted_class, prob = tree_fn(**kwargs)
        assert predicted_class in class_names
        assert 0.0 <= prob <= 1.0
        predicted_classes.add(predicted_class)
    assert predicted_classes == set(class_names)
