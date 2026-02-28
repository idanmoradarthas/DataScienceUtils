"""Tests for the plot_features_importance function in ds_utils.xai."""

from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np
import pytest

from ds_utils.xai import plot_features_importance

BASELINE_DIR = Path(__file__).parent.parent / "baseline_images" / "test_xai" / "test_plot_features_importance"


@pytest.fixture
def importance():
    """Fixture for sample feature importances."""
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
    """Fixture for sample feature names."""
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


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR)
def test_plot_features_importance(importance, features):
    """Test plotting feature importance."""
    plot_features_importance(features, importance)

    figure = plt.gcf()
    figure.set_size_inches(17, 10)
    return figure


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR)
def test_plot_features_importance_exists_ax(importance, features):
    """Test plotting feature importance on an existing Axes object."""
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
