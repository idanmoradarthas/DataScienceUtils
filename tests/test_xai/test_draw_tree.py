"""Tests for the draw_tree function in ds_utils.xai."""

from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np
import pytest

from ds_utils.xai import draw_tree

BASELINE_DIR = Path(__file__).parent.parent / "baseline_images" / "test_xai" / "test_draw_tree"


@pytest.fixture
def decision_tree_draw_tree(mocker):
    """Fixture for a mocked decision tree for testing draw_tree."""
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

    mock.n_features_ = 4

    return mock


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR, tolerance=53)
def test_draw_tree(decision_tree_draw_tree):
    """Test drawing a decision tree."""
    with pytest.warns(DeprecationWarning, match="This module is deprecated. Use sklearn.tree.plot_tree instead"):
        draw_tree(
            decision_tree_draw_tree,
            ["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"],
            ["setosa", "versicolor", "virginica"],
        )
    return plt.gcf()


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR, tolerance=54)
def test_draw_tree_exists_ax(decision_tree_draw_tree):
    """Test drawing a decision tree on an existing Axes object."""
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
