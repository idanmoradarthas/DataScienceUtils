"""Tests for unsupervised learning utility functions."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from matplotlib import pyplot as plt
from scipy.spatial.distance import euclidean

from ds_utils.unsupervised import (
    plot_cluster_cardinality,
    plot_cluster_magnitude,
    plot_magnitude_vs_cardinality,
    plot_loss_vs_cluster_number,
)

BASELINE_DIR = Path(__file__).parent / "baseline_images" / "test_unsupervised"


@pytest.fixture
def iris_data():
    """Load and return Iris dataset components for testing."""
    iris_x = pd.read_csv(Path(__file__).parents[0].joinpath("resources").joinpath("iris_x_full.csv"))
    labels = np.asarray(
        [
            1,
            6,
            6,
            6,
            1,
            1,
            6,
            1,
            6,
            6,
            1,
            6,
            6,
            6,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            6,
            1,
            6,
            6,
            1,
            1,
            1,
            6,
            6,
            1,
            1,
            1,
            6,
            6,
            1,
            1,
            6,
            1,
            1,
            6,
            6,
            1,
            1,
            6,
            1,
            6,
            1,
            6,
            0,
            0,
            0,
            3,
            0,
            3,
            0,
            5,
            0,
            3,
            5,
            3,
            3,
            0,
            3,
            0,
            3,
            3,
            4,
            3,
            4,
            3,
            4,
            0,
            0,
            0,
            0,
            0,
            0,
            3,
            3,
            3,
            3,
            4,
            3,
            0,
            0,
            0,
            3,
            3,
            3,
            0,
            3,
            5,
            3,
            3,
            3,
            0,
            5,
            3,
            7,
            4,
            2,
            7,
            7,
            2,
            3,
            2,
            7,
            2,
            7,
            4,
            7,
            4,
            4,
            7,
            7,
            2,
            2,
            4,
            7,
            4,
            2,
            4,
            7,
            2,
            4,
            4,
            7,
            2,
            2,
            2,
            7,
            4,
            4,
            2,
            7,
            7,
            4,
            7,
            7,
            7,
            4,
            7,
            7,
            7,
            4,
            7,
            7,
            4,
        ]
    )
    cluster_centers = np.asarray(
        [
            [6.442105263157895223e00, 2.978947368421052566e00, 4.594736842105263008e00, 1.431578947368421062e00],
            [5.242857142857142883e00, 3.667857142857142705e00, 1.499999999999999556e00, 2.821428571428574728e-01],
            [7.474999999999999645e00, 3.125000000000000000e00, 6.299999999999999822e00, 2.049999999999999822e00],
            [5.620833333333333570e00, 2.691666666666666430e00, 4.075000000000000178e00, 1.262499999999999956e00],
            [6.036842105263158231e00, 2.705263157894736814e00, 5.000000000000000000e00, 1.778947368421052611e00],
            [5.000000000000000000e00, 2.299999999999999822e00, 3.274999999999999911e00, 1.024999999999999911e00],
            [4.704545454545455030e00, 3.122727272727272574e00, 1.413636363636363136e00, 2.000000000000001776e-01],
            [6.568181818181818343e00, 3.086363636363636420e00, 5.536363636363637042e00, 2.163636363636363580e00],
        ]
    )
    return iris_x, labels, cluster_centers


@pytest.fixture
def distance_wrapper_plot_magnitude_vs_cardinality(mocker):
    """Create a mock distance function that returns preloaded distances."""
    with (
        Path(__file__)
        .parents[0]
        .joinpath("resources")
        .joinpath("euclidean_distances_plot_magnitude_vs_cardinality.txt")
        .open("r") as file
    ):
        loaded_distances = [float(line.strip()) for line in file]

    mock_distance = mocker.Mock(side_effect=loaded_distances)

    # Wrap the mock function to handle numpy arrays
    def wrapper(a, b):
        return mock_distance(tuple(a), tuple(b))

    return wrapper


@pytest.fixture(autouse=True)
def setup_teardown():
    """Set up and tear down for each test in this module."""
    yield
    plt.cla()
    plt.close(plt.gcf())


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR)
def test_cluster_cardinality(iris_data):
    """Test plotting cluster cardinality."""
    _, labels, _ = iris_data
    plot_cluster_cardinality(np.asarray(labels))
    return plt.gcf()


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR)
def test_cluster_cardinality_exist_ax(iris_data):
    """Test plotting cluster cardinality on an existing Axes object."""
    fig, ax = plt.subplots()
    ax.set_title("My ax")

    _, labels, _ = iris_data
    plot_cluster_cardinality(np.asarray(labels), ax=ax)
    assert ax.get_title() == "My ax"
    return fig


def test_cluster_cardinality_empty_labels():
    """Test plot_cluster_cardinality raises ValueError for empty labels."""
    with pytest.raises(ValueError, match="Labels array is empty."):
        plot_cluster_cardinality(np.array([]))


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR)
def test_plot_cluster_magnitude(iris_data, distance_wrapper_plot_magnitude_vs_cardinality):
    """Test plotting cluster magnitude."""
    iris_x, labels, cluster_centers = iris_data

    plot_cluster_magnitude(iris_x.values, labels, cluster_centers, distance_wrapper_plot_magnitude_vs_cardinality)
    return plt.gcf()


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR)
def test_plot_cluster_magnitude_exist_ax(iris_data, distance_wrapper_plot_magnitude_vs_cardinality):
    """Test plotting cluster magnitude on an existing Axes object."""
    fig, ax = plt.subplots()
    ax.set_title("My ax")

    iris_x, labels, cluster_centers = iris_data
    plot_cluster_magnitude(
        iris_x.values, labels, cluster_centers, distance_wrapper_plot_magnitude_vs_cardinality, ax=ax
    )
    assert ax.get_title() == "My ax"
    return fig


def test_cluster_magnitude_inconsistent_shapes(mocker, iris_data):
    """Test plot_cluster_magnitude with inconsistent X and labels shapes."""
    iris_x, labels, cluster_centers = iris_data
    with pytest.raises(ValueError, match="X and labels must have the same length."):
        plot_cluster_magnitude(iris_x.values[:-1], labels, cluster_centers, mocker.Mock())


def test_cluster_magnitude_invalid_distance_function(mocker, iris_data):
    """Test plot_cluster_magnitude with an invalid distance_function."""
    iris_x, labels, cluster_centers = iris_data
    with pytest.raises(ValueError, match="Invalid distance_function provided."):
        plot_cluster_magnitude(iris_x.values, labels, cluster_centers, mocker.Mock(side_effect=TypeError))


def test_cluster_magnitude_invalid_cluster_number_vs_labels(mocker, iris_data):
    """Test plot_cluster_magnitude with mismatch between cluster_centers and labels."""
    iris_x, labels, cluster_centers = iris_data
    with pytest.raises(ValueError, match="Number of cluster centers must match the number of unique labels."):
        plot_cluster_magnitude(np.array([1]), np.array([1]), cluster_centers, mocker.Mock())


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR)
def test_plot_magnitude_vs_cardinality(iris_data, distance_wrapper_plot_magnitude_vs_cardinality):
    """Test plotting magnitude vs. cardinality."""
    iris_x, labels, cluster_centers = iris_data
    plot_magnitude_vs_cardinality(
        iris_x.values, labels, cluster_centers, distance_wrapper_plot_magnitude_vs_cardinality
    )
    return plt.gcf()


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR)
def test_plot_magnitude_vs_cardinality_exist_ax(iris_data, distance_wrapper_plot_magnitude_vs_cardinality):
    """Test plotting magnitude vs. cardinality on an existing Axes object."""
    fig, ax = plt.subplots()
    ax.set_title("My ax")

    iris_x, labels, cluster_centers = iris_data
    plot_magnitude_vs_cardinality(
        iris_x.values, labels, cluster_centers, distance_wrapper_plot_magnitude_vs_cardinality, ax=ax
    )
    assert ax.get_title() == "My ax"
    return fig


def test_plot_magnitude_vs_cardinality_inconsistent_shapes(mocker, iris_data):
    """Test plot_magnitude_vs_cardinality with inconsistent X and labels shapes."""
    iris_x, labels, cluster_centers = iris_data
    with pytest.raises(ValueError, match="X and labels must have the same length."):
        plot_magnitude_vs_cardinality(iris_x.values[:-1], labels, cluster_centers, mocker.Mock())


def test_magnitude_vs_cardinality_inconsistent_centers(mocker, iris_data):
    """Test plot_magnitude_vs_cardinality with inconsistent cluster_centers."""
    iris_x, labels, cluster_centers = iris_data
    with pytest.raises(ValueError, match="Number of cluster centers must match the number of unique labels."):
        plot_magnitude_vs_cardinality(iris_x.values, labels, cluster_centers[:-1], mocker.Mock())


def test_magnitude_vs_cardinality_invalid_distance_function(mocker, iris_data):
    """Test plot_magnitude_vs_cardinality with an invalid distance_function."""
    iris_x, labels, cluster_centers = iris_data
    with pytest.raises(ValueError, match="Invalid distance_function provided."):
        plot_magnitude_vs_cardinality(iris_x.values, labels, cluster_centers, mocker.Mock(side_effect=TypeError))


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR)
def test_plot_loss_vs_cluster_number(iris_data):
    """Test plotting loss vs. number of clusters."""
    iris_x, _, _ = iris_data
    plot_loss_vs_cluster_number(iris_x.values, 3, 20, euclidean)
    return plt.gcf()


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR)
def test_plot_loss_vs_cluster_number_exist_ax(iris_data):
    """Test plotting loss vs. number of clusters on an existing Axes object."""
    fig, ax = plt.subplots()
    ax.set_facecolor("tab:red")

    iris_x, _, _ = iris_data
    plot_loss_vs_cluster_number(iris_x.values, 3, 20, euclidean, ax=ax)
    return fig


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR)
def test_plot_loss_vs_cluster_number_given_parameters(iris_data):
    """Test plotting loss vs. number of clusters with specific algorithm parameters."""
    iris_x, _, _ = iris_data
    plot_loss_vs_cluster_number(
        iris_x.values,
        3,
        20,
        euclidean,
        algorithm_parameters={"random_state": 42, "algorithm": "lloyd", "n_clusters": 3},
    )
    return plt.gcf()


def test_loss_vs_cluster_number_invalid_k_range(mocker, iris_data):
    """Test plot_loss_vs_cluster_number with an invalid k_min > k_max."""
    iris_x, _, _ = iris_data
    with pytest.raises(ValueError, match="k_min must be less than or equal to k_max."):
        plot_loss_vs_cluster_number(iris_x.values, 10, 5, mocker.Mock())


def test_loss_vs_cluster_number_invalid_algorithm_parameters(mocker, iris_data):
    """Test plot_loss_vs_cluster_number with invalid algorithm parameters."""
    iris_x, _, _ = iris_data
    with pytest.raises(ValueError, match="No valid results were obtained. Check your input data and parameters."):
        plot_loss_vs_cluster_number(
            iris_x.values, 3, 20, mocker.Mock(), algorithm_parameters={"invalid_param": "value"}
        )
