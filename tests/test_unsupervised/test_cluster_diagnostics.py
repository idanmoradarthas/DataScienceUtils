"""Tests for cluster cardinality and magnitude diagnostics."""

from pathlib import Path

import numpy as np
import pytest
from matplotlib import pyplot as plt

from ds_utils.unsupervised.cluster_diagnostics import (
    plot_cluster_cardinality,
    plot_cluster_magnitude,
    plot_magnitude_vs_cardinality,
)

RESOURCES_DIR = Path(__file__).parents[1] / "resources"
BASELINE_DIR = Path(__file__).parents[1] / "baseline_images" / Path(__file__).parent.name / Path(__file__).stem


@pytest.fixture
def distance_wrapper_plot_magnitude_vs_cardinality(mocker):
    """Create a mock distance function that returns preloaded distances."""
    with (RESOURCES_DIR / "euclidean_distances_plot_magnitude_vs_cardinality.txt").open("r") as file:
        loaded_distances = [float(line.strip()) for line in file]

    mock_distance = mocker.Mock(side_effect=loaded_distances)

    # Wrap the mock function to handle numpy arrays
    def wrapper(a, b):
        return mock_distance(tuple(a), tuple(b))

    return wrapper


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
