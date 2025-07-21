"""Tests for Explainable AI (XAI) utility functions."""

import os
from pathlib import Path

import numpy as np
import pytest
from matplotlib import pyplot as plt

from ds_utils.xai import generate_decision_paths, draw_tree, draw_dot_data, plot_features_importance

BASELINE_DIR = Path(__file__).parent / "baseline_images" / "test_xai"


@pytest.fixture
def decision_tree_generate_decision_paths(mocker):
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


@pytest.fixture
def decision_tree_draw_tree(mocker):
    mock = mocker.Mock()
    mock.tree_.node_count = 17
    mock.tree_.children_left = np.array([1, -1, 3, 4, 5, -1, -1, 8, -1, 10, -1, -1, 13, 14, -1, -1, -1])
    mock.tree_.children_right = np.array([2, -1, 12, 7, 6, -1, -1, 9, -1, 11, -1, -1, 16, 15, -1, -1, -1])
    mock.tree_.feature = np.array([3, -2, 3, 2, 3, -2, -2, 3, -2, 2, -2, -2, 2, 1, -2, -2, -2])
    mock.tree_.threshold = np.array(
        [
            0.80000001,
            -2.0,
            1.75,
            4.95000005,
            1.65000004,
            -2.0,
            -2.0,
            1.55000001,
            -2.0,
            5.45000005,
            -2.0,
            -2.0,
            4.85000014,
            3.10000002,
            -2.0,
            -2.0,
            -2.0,
        ]
    )
    mock.tree_.value = np.array(
        [
            [[0.33333333, 0.33333333, 0.33333333]],
            [[1.0, 0.0, 0.0]],
            [[0.0, 0.5, 0.5]],
            [[0.0, 0.90740741, 0.09259259]],
            [[0.0, 0.97916667, 0.02083333]],
            [[0.0, 1.0, 0.0]],
            [[0.0, 0.0, 1.0]],
            [[0.0, 0.33333333, 0.66666667]],
            [[0.0, 0.0, 1.0]],
            [[0.0, 0.66666667, 0.33333333]],
            [[0.0, 1.0, 0.0]],
            [[0.0, 0.0, 1.0]],
            [[0.0, 0.02173913, 0.97826087]],
            [[0.0, 0.33333333, 0.66666667]],
            [[0.0, 0.0, 1.0]],
            [[0.0, 1.0, 0.0]],
            [[0.0, 0.0, 1.0]],
        ]
    )
    mock.tree_.impurity = np.array(
        [
            0.66666667,
            0.0,
            0.5,
            0.16803841,
            0.04079861,
            0.0,
            0.0,
            0.44444444,
            0.0,
            0.44444444,
            0.0,
            0.0,
            0.04253308,
            0.44444444,
            0.0,
            0.0,
            0.0,
        ]
    )
    mock.tree_.n_node_samples = np.array([150, 50, 100, 54, 48, 47, 1, 6, 3, 3, 2, 1, 46, 3, 2, 1, 43])
    mock.tree_.n_classes = np.array([3])
    mock.tree_.weighted_n_node_samples = np.array(
        [150.0, 50.0, 100.0, 54.0, 48.0, 47.0, 1.0, 6.0, 3.0, 3.0, 2.0, 1.0, 46.0, 3.0, 2.0, 1.0, 43.0]
    )

    mock.classes_ = np.array([0, 1, 2])
    mock.n_classes_ = 3
    mock.n_features_in_ = 4
    mock.n_outputs_ = 1
    mock.max_depth = None

    mock.get_depth.return_value = 5
    mock.get_n_leaves.return_value = 9

    # Create a mock tags object with the required attributes
    mock_tags = mocker.Mock()
    mock_tags.requires_fit = True
    mock_tags.requires_y = True
    mock_tags.requires_positive_X = False
    mock_tags.requires_positive_y = False
    mock_tags.no_validation = False
    mock_tags.poor_score = False
    mock_tags.allow_nan = False
    mock_tags.stateless = False
    mock_tags.binary_only = False
    mock_tags._xfail_checks = {}
    mock_tags.multiclass_only = False
    mock_tags.multilabel = False
    mock_tags.multioutput_only = False
    mock_tags.multioutput = False
    mock_tags.pairwise = False
    mock_tags.preserves_dtype = []
    mock_tags.X_types = ["2darray"]
    mock_tags.y_types = ["1dlabels"]
    mock_tags.target_tags = ["multiclass"]
    mock_tags._estimator_type = "classifier"

    # Add the __sklearn_tags__ method that returns the mock_tags object
    mock.__sklearn_tags__ = mocker.Mock(return_value=mock_tags)

    # Add some fitted attributes to make the estimator appear fitted
    # mock.feature_importances_ = np.random.Generator(4)
    mock.n_features_ = 4

    return mock


@pytest.fixture
def importance():
    return [
        0.047304175084187376,
        0.011129476233187116,
        0.01289095487553893,
        0.015528563988219685,
        0.010904371085026893,
        0.03088871541039015,
        0.03642466650007851,
        0.005758395681352302,
        0.3270373201417306,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        3.2562763221434323e-08,
        0,
        0,
        1.3846860504742983e-05,
        0,
        0,
        0,
        0,
        0,
        0,
        7.31917344936033e-07,
        3.689321224943123e-05,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0.5019726768895573,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0.00010917955786877644,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ]


@pytest.fixture
def features():
    return [
        "x1",
        "x2",
        "x3",
        "x4",
        "x5",
        "x6",
        "x8",
        "x9",
        "x11",
        "x7_value_0",
        "x7_value_1",
        "x7_value_10",
        "x7_value_11",
        "x7_value_12",
        "x7_value_13",
        "x7_value_14",
        "x7_value_15",
        "x7_value_16",
        "x7_value_17",
        "x7_value_18",
        "x7_value_19",
        "x7_value_2",
        "x7_value_20",
        "x7_value_21",
        "x7_value_22",
        "x7_value_23",
        "x7_value_24",
        "x7_value_25",
        "x7_value_26",
        "x7_value_27",
        "x7_value_28",
        "x7_value_29",
        "x7_value_3",
        "x7_value_30",
        "x7_value_31",
        "x7_value_32",
        "x7_value_33",
        "x7_value_34",
        "x7_value_35",
        "x7_value_36",
        "x7_value_37",
        "x7_value_38",
        "x7_value_39",
        "x7_value_4",
        "x7_value_40",
        "x7_value_41",
        "x7_value_42",
        "x7_value_43",
        "x7_value_44",
        "x7_value_45",
        "x7_value_46",
        "x7_value_47",
        "x7_value_48",
        "x7_value_49",
        "x7_value_5",
        "x7_value_50",
        "x7_value_51",
        "x7_value_52",
        "x7_value_53",
        "x7_value_54",
        "x7_value_55",
        "x7_value_56",
        "x7_value_57",
        "x7_value_58",
        "x7_value_59",
        "x7_value_6",
        "x7_value_60",
        "x7_value_61",
        "x7_value_62",
        "x7_value_63",
        "x7_value_64",
        "x7_value_65",
        "x7_value_66",
        "x7_value_67",
        "x7_value_68",
        "x7_value_69",
        "x7_value_7",
        "x7_value_70",
        "x7_value_71",
        "x7_value_8",
        "x7_value_9",
        "x10_value_0",
        "x10_value_1",
        "x10_value_2",
    ]


@pytest.fixture(autouse=True)
def setup_teardown():
    yield
    plt.cla()
    plt.close(plt.gcf())


@pytest.mark.parametrize(
    "tree_name, expected_name",
    [
        ("iris_tree", "def iris_tree(petal width (cm), petal length (cm)):"),
        (None, "def tree(petal width (cm), petal length (cm)):"),
    ],
    ids=["default", "no_tree_name"],
)
def test_print_decision_paths(tree_name, expected_name, decision_tree_generate_decision_paths):
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


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR, tolerance=53)
def test_draw_tree(decision_tree_draw_tree):
    with pytest.warns(DeprecationWarning, match="This module is deprecated. Use sklearn.tree.plot_tree instead"):
        draw_tree(
            decision_tree_draw_tree,
            ["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"],
            ["setosa", "versicolor", "virginica"],
        )
    return plt.gcf()


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR, tolerance=54)
def test_draw_tree_exists_ax(decision_tree_draw_tree):
    fig, ax = plt.subplots()
    ax.set_title("My ax")
    with pytest.warns(DeprecationWarning, match="This module is deprecated. Use sklearn.tree.plot_tree instead"):
        draw_tree(
            decision_tree_draw_tree,
            ["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"],
            ["setosa", "versicolor", "virginica"],
            ax=ax,
        )

    assert ax.get_title() == "My ax"
    return fig


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR, tolerance=11)
def test_draw_dot_data():
    dot_data = (
        "digraph D{\n"
        "\tA [shape=diamond]\n"
        "\tB [shape=box]\n"
        "\tC [shape=circle]\n"
        "\n"
        "\tA -> B [style=dashed, color=grey]\n"
        '\tA -> C [color="black:invis:black"]\n'
        "\tA -> D [penwidth=5, arrowhead=none]\n"
        "\n"
        "}"
    )

    draw_dot_data(dot_data)
    return plt.gcf()


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR, tolerance=11)
def test_draw_dot_data_exist_ax():
    dot_data = (
        "digraph D{\n"
        "\tA [shape=diamond]\n"
        "\tB [shape=box]\n"
        "\tC [shape=circle]\n"
        "\n"
        "\tA -> B [style=dashed, color=grey]\n"
        '\tA -> C [color="black:invis:black"]\n'
        "\tA -> D [penwidth=5, arrowhead=none]\n"
        "\n"
        "}"
    )

    fig, ax = plt.subplots()
    ax.set_title("My ax")

    draw_dot_data(dot_data, ax=ax)
    assert ax.get_title() == "My ax"
    return fig


def test_draw_dot_data_empty_input():
    with pytest.raises(ValueError, match="dot_data must not be empty"):
        draw_dot_data("")


def test_draw_dot_data_invalid_input(mocker):
    mock_pydotplus = mocker.patch("ds_utils.xai.pydotplus")
    mock_pydotplus.graph_from_dot_data.side_effect = Exception("Invalid dot data")

    with pytest.raises(ValueError, match="Failed to create graph from dot data"):
        draw_dot_data("invalid dot data")


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR)
def test_plot_features_importance(importance, features):
    plot_features_importance(features, importance)

    figure = plt.gcf()
    figure.set_size_inches(17, 10)
    return figure


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR)
def test_plot_features_importance_exists_ax(importance, features):
    fig, ax = plt.subplots()

    ax.set_title("My ax")
    plot_features_importance(features, importance, ax=ax)
    assert ax.get_title() == "My ax"

    fig.set_size_inches(17, 10)
    return fig


def test_plot_features_importance_mismatched_lengths():
    """Test that ValueError is raised for mismatched feature and importance lengths."""
    with pytest.raises(ValueError, match="feature_names and feature_importance must have the same length"):
        plot_features_importance(["feature1", "feature2"], [0.5, 0.3, 0.2])


def test_plot_features_importance_invalid_input_dimensions():
    """Test that ValueError is raised for multidimensional inputs."""
    with pytest.raises(ValueError, match="feature_names and feature_importance must be 1-dimensional"):
        plot_features_importance(np.array([["f1", "f2"], ["f3", "f4"]]), np.array([[0.5, 0.3], [0.2, 0.1]]))
