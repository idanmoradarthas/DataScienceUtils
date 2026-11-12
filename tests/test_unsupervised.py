"""Tests for unsupervised learning utility functions."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from matplotlib import pyplot as plt
from scipy.spatial.distance import euclidean
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_samples, silhouette_score

from ds_utils.unsupervised import (
    plot_cluster_cardinality,
    plot_cluster_magnitude,
    plot_magnitude_vs_cardinality,
    plot_loss_vs_cluster_number,
    plot_silhouette_analysis,
    plot_clusters,
    plot_clusters_plotly,
)
from tests.utils import save_plotly_figure_and_return_matplot
from plotly import graph_objects as go

BASELINE_DIR = Path(__file__).parent / "baseline_images" / "test_unsupervised"
RESULT_DIR = Path(__file__).parent / "result_images" / "test_unsupervised"

RESULT_DIR.mkdir(exist_ok=True, parents=True)


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


# ---------------------------------------------------------------------------
# plot_silhouette_analysis
# ---------------------------------------------------------------------------


@pytest.fixture
def blobs_data():
    """Return three well separated 2D blobs with their generating labels."""
    X, labels = make_blobs(n_samples=60, centers=3, cluster_std=1.0, random_state=42)
    return X, labels


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR)
def test_plot_silhouette_analysis(iris_data):
    """Test plotting the silhouette analysis with default parameters."""
    iris_x, labels, _ = iris_data
    plot_silhouette_analysis(iris_x.values, labels)
    return plt.gcf()


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR)
def test_plot_silhouette_analysis_exist_ax(iris_data):
    """Test plotting the silhouette analysis on an existing Axes object."""
    fig, ax = plt.subplots()
    ax.set_title("My ax")

    iris_x, labels, _ = iris_data
    returned_ax = plot_silhouette_analysis(iris_x.values, labels, ax=ax)
    assert returned_ax is ax
    assert ax.get_title() == "My ax"
    return fig


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR)
def test_plot_silhouette_analysis_custom_options(blobs_data):
    """Test plotting the silhouette analysis with custom colors, metric, no average line and Matplotlib kwargs."""
    X, labels = blobs_data
    plot_silhouette_analysis(
        X,
        labels,
        metric="manhattan",
        colors=["tab:blue", "tab:orange", "tab:green"],
        show_avg_line=False,
        alpha=0.4,
        edgecolor="black",
    )
    return plt.gcf()


def test_plot_silhouette_analysis_scores_and_ticks(blobs_data):
    """Test that the average line equals the silhouette score and that every cluster gets a band and a tick."""
    X, labels = blobs_data
    ax = plot_silhouette_analysis(X, labels)

    average_score = silhouette_score(X, labels)
    assert ax.lines[0].get_xdata()[0] == pytest.approx(average_score)
    assert ax.get_legend().get_texts()[0].get_text() == f"Average: {average_score:.3f}"
    assert len(ax.collections) == 3
    assert [tick.get_text() for tick in ax.get_yticklabels()] == ["0", "1", "2"]
    assert ax.get_xlabel() == "Silhouette Coefficient"
    assert ax.get_ylabel() == "Cluster Label"
    assert ax.get_xlim()[1] == 1.0


def test_plot_silhouette_analysis_no_average_line(blobs_data):
    """Test that show_avg_line=False draws neither the line nor the legend."""
    X, labels = blobs_data
    ax = plot_silhouette_analysis(X, labels, show_avg_line=False)
    assert len(ax.lines) == 0
    assert ax.get_legend() is None


def test_plot_silhouette_analysis_negative_scores_are_not_clipped(blobs_data):
    """Test that samples with a silhouette score below -0.1 stay inside the x limits."""
    X, _ = blobs_data
    shuffled_labels = np.random.default_rng(0).integers(0, 3, size=len(X))
    lowest_score = silhouette_samples(X, shuffled_labels).min()
    assert lowest_score < -0.1

    ax = plot_silhouette_analysis(X, shuffled_labels)
    assert ax.get_xlim()[0] <= lowest_score


def test_plot_silhouette_analysis_kwargs_forwarded(blobs_data):
    """Test that kwargs reach fill_betweenx and that the default alpha is 0.7."""
    X, labels = blobs_data
    default_ax = plot_silhouette_analysis(X, labels)
    assert [collection.get_alpha() for collection in default_ax.collections] == [0.7, 0.7, 0.7]

    custom_ax = plot_silhouette_analysis(X, labels, alpha=0.3)
    assert [collection.get_alpha() for collection in custom_ax.collections] == [0.3, 0.3, 0.3]


def test_plot_silhouette_analysis_dataframe_and_string_labels(blobs_data):
    """Test that a DataFrame and non-integer labels are supported."""
    X, labels = blobs_data
    string_labels = np.array(["low", "mid", "high"])[labels]
    ax = plot_silhouette_analysis(pd.DataFrame(X, columns=["a", "b"]), string_labels)
    assert [tick.get_text() for tick in ax.get_yticklabels()] == ["high", "low", "mid"]


def test_plot_silhouette_analysis_noise_label_is_a_cluster(blobs_data):
    """Test that the DBSCAN noise label -1 gets its own band."""
    X, labels = blobs_data
    labels = labels.copy()
    labels[:5] = -1
    ax = plot_silhouette_analysis(X, labels)
    assert [tick.get_text() for tick in ax.get_yticklabels()] == ["-1", "0", "1", "2"]
    assert len(ax.collections) == 4


def test_plot_silhouette_analysis_colors_too_short_falls_back_to_default(blobs_data):
    """Test that too few colors emit a warning and the plot is still drawn with the default palette."""
    X, labels = blobs_data
    with pytest.warns(UserWarning, match="colors has 2 entries for 3 clusters; falling back to the default palette."):
        ax = plot_silhouette_analysis(X, labels, colors=["red", "blue"])
    assert len(ax.collections) == 3


def test_plot_silhouette_analysis_more_than_ten_clusters():
    """Test that more than ten clusters are drawn with a distinct color per band."""
    X, labels = make_blobs(n_samples=120, centers=12, random_state=0)
    ax = plot_silhouette_analysis(X, labels)
    assert len(ax.collections) == 12
    assert len({tuple(collection.get_facecolor()[0]) for collection in ax.collections}) == 12


def test_plot_silhouette_analysis_inconsistent_lengths(blobs_data):
    """Test plot_silhouette_analysis raises ValueError when X and labels differ in length."""
    X, labels = blobs_data
    with pytest.raises(ValueError, match="X and labels must have the same length."):
        plot_silhouette_analysis(X, labels[:-1])


def test_plot_silhouette_analysis_single_cluster(blobs_data):
    """Test plot_silhouette_analysis raises ValueError for fewer than two clusters."""
    X, _ = blobs_data
    with pytest.raises(ValueError, match="Silhouette analysis requires at least 2 clusters."):
        plot_silhouette_analysis(X, np.zeros(len(X), dtype=int))


def test_plot_silhouette_analysis_as_many_clusters_as_samples():
    """Test plot_silhouette_analysis raises ValueError when every sample is its own cluster."""
    X = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    with pytest.raises(ValueError, match="Silhouette analysis requires fewer clusters than samples."):
        plot_silhouette_analysis(X, np.array([0, 1, 2]))


# ---------------------------------------------------------------------------
# plot_clusters tests (matplotlib)
# ---------------------------------------------------------------------------


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR, tolerance=18)
def test_plot_clusters_2d():
    """Test plot_clusters with 2D data (no PCA needed)."""
    # Create simple 2D dataset
    X = np.array(
        [
            [1.0, 2.0],
            [1.5, 1.8],
            [1.2, 2.2],
            [8.0, 8.0],
            [8.5, 8.2],
            [8.2, 8.5],
            [1.0, 8.0],
            [1.5, 8.2],
            [1.2, 7.8],
        ]
    )
    labels = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    plot_clusters(X, labels, reduction_method="pca")
    return plt.gcf()


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR, tolerance=18)
def test_plot_clusters_2d_with_centroids():
    """Test plot_clusters with 2D data and centroids overlay."""
    X = np.array(
        [
            [1.0, 2.0],
            [1.5, 1.8],
            [1.2, 2.2],
            [8.0, 8.0],
            [8.5, 8.2],
            [8.2, 8.5],
        ]
    )
    labels = np.array([0, 0, 0, 1, 1, 1])
    centers = np.array([[1.23, 2.0], [8.23, 8.23]])
    plot_clusters(X, labels, centers)
    return plt.gcf()


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR, tolerance=18)
def test_plot_clusters_pca(iris_data):
    """Test plot_clusters with high-dimensional data using PCA."""
    iris_x, labels, _ = iris_data
    plot_clusters(iris_x.values, labels, reduction_method="pca", random_state=42)
    return plt.gcf()


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR, tolerance=18)
def test_plot_clusters_pca_with_centroids(iris_data):
    """Test plot_clusters with high-dimensional data, PCA, and centroids."""
    iris_x, labels, cluster_centers = iris_data
    plot_clusters(iris_x.values, labels, cluster_centers, reduction_method="pca", random_state=42)
    return plt.gcf()


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR, tolerance=18)
def test_plot_clusters_tsne(mocker, iris_data):
    """Test plot_clusters with high-dimensional data using t-SNE."""
    iris_x, labels, _ = iris_data
    fixed_2d = np.random.RandomState(42).normal(size=(len(iris_x), 2))
    mocker.patch("ds_utils.unsupervised.TSNE.fit_transform", return_value=fixed_2d)
    plot_clusters(iris_x.values, labels, reduction_method="tsne", random_state=42)
    return plt.gcf()


def test_plot_clusters_tsne_with_centroids(iris_data):
    """Test plot_clusters with t-SNE and centroids, should warn."""
    iris_x, labels, cluster_centers = iris_data
    with pytest.warns(UserWarning, match="t-SNE cannot transform new data; centroids will not be plotted."):
        ax = plot_clusters(iris_x.values, labels, cluster_centers, reduction_method="tsne", random_state=42)
    assert ax is not None


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR, tolerance=18)
def test_plot_clusters_exist_ax(iris_data):
    """Test plot_clusters works with a pre-existing Axes."""
    fig, ax = plt.subplots()
    ax.set_facecolor("lightgray")

    iris_x, labels, _ = iris_data
    plot_clusters(iris_x.values, labels, reduction_method="pca", random_state=42, ax=ax)
    return fig


def test_plot_clusters_empty_x():
    """Test plot_clusters raises ValueError for empty X."""
    with pytest.raises(ValueError, match="X must be a 2D array with at least 1 feature."):
        plot_clusters(np.array([]), np.array([]))


def test_plot_clusters_1d_x():
    """Test plot_clusters raises ValueError for 1D X."""
    with pytest.raises(ValueError, match="X must be a 2D array with at least 1 feature."):
        plot_clusters(np.array([1, 2, 3]), np.array([0, 0, 0]))


def test_plot_clusters_single_feature():
    """Test plot_clusters raises ValueError for a single-feature 2D array."""
    X = np.array([[1.0], [2.0], [3.0]])
    labels = np.array([0, 0, 1])
    with pytest.raises(ValueError, match="X must have at least 2 features for 2D scatter visualization."):
        plot_clusters(X, labels)


def test_plot_clusters_inconsistent_lengths():
    """Test plot_clusters raises ValueError for X and labels length mismatch."""
    X = np.array([[1, 2], [3, 4]])
    labels = np.array([0, 0, 1])
    with pytest.raises(ValueError, match="X and cluster_labels must have the same length."):
        plot_clusters(X, labels)


def test_plot_clusters_inconsistent_centers():
    """Test plot_clusters raises ValueError for cluster_centers length mismatch."""
    X = np.array([[1, 2], [3, 4]])
    labels = np.array([0, 1])
    centers = np.array([[1.5, 2.5]])  # Only 1 center, but 2 unique labels
    with pytest.raises(ValueError, match="Number of cluster centers must match the number of unique labels."):
        plot_clusters(X, labels, centers)


def test_plot_clusters_invalid_reduction_method():
    """Test plot_clusters raises ValueError for invalid reduction_method."""
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    labels = np.array([0, 1, 0])
    with pytest.raises(ValueError, match="Unknown reduction_method: 'invalid'"):
        plot_clusters(X, labels, reduction_method="invalid")


def test_plot_clusters_umap_not_installed(mocker):
    """Test plot_clusters raises ImportError if UMAP is requested but not installed."""
    mocker.patch("ds_utils.unsupervised.UMAP", None)
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    labels = np.array([0, 1, 0])
    with pytest.raises(ImportError, match="UMAP is not installed"):
        plot_clusters(X, labels, reduction_method="umap")


def test_plot_clusters_umap_installed(mocker):
    """Test plot_clusters executes UMAP branch when UMAP is available."""
    mock_umap = mocker.Mock()
    mock_umap_instance = mocker.Mock()
    mock_umap.return_value = mock_umap_instance
    mock_umap_instance.fit_transform.return_value = np.array([[1, 2], [3, 4]])
    mock_umap_instance.transform.return_value = np.array([[1.5, 2.5], [3.5, 4.5]])
    mocker.patch("ds_utils.unsupervised.UMAP", mock_umap)

    X = np.array([[1, 2, 3], [4, 5, 6]])
    labels = np.array([0, 1])
    centers = np.array([[1, 2, 3], [4, 5, 6]])
    ax = plot_clusters(X, labels, centers, reduction_method="umap")
    assert ax is not None
    mock_umap_instance.fit_transform.assert_called_once()
    mock_umap_instance.transform.assert_called_once()


def test_plot_clusters_many_clusters():
    """Test plot_clusters with > 10 clusters to trigger tab20 colormap branch."""
    X = np.random.rand(20, 2)
    labels = np.arange(20)  # 20 unique clusters
    ax = plot_clusters(X, labels)
    assert ax is not None


# ---------------------------------------------------------------------------
# plot_clusters_plotly tests
# ---------------------------------------------------------------------------


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR, tolerance=18)
def test_plot_clusters_plotly_2d(request):
    """Test plot_clusters_plotly with 2D data."""
    X = np.array(
        [
            [1.0, 2.0],
            [1.5, 1.8],
            [1.2, 2.2],
            [8.0, 8.0],
            [8.5, 8.2],
            [8.2, 8.5],
            [1.0, 8.0],
            [1.5, 8.2],
            [1.2, 7.8],
        ]
    )
    labels = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    fig = plot_clusters_plotly(X, labels)
    return save_plotly_figure_and_return_matplot(fig, RESULT_DIR / f"{request.node.name}.png")


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR, tolerance=18)
def test_plot_clusters_plotly_pca_with_centroids(request, iris_data):
    """Test plot_clusters_plotly with high-dimensional data, PCA, and centroids."""
    iris_x, labels, cluster_centers = iris_data
    fig = plot_clusters_plotly(iris_x.values, labels, cluster_centers, reduction_method="pca", random_state=42)
    return save_plotly_figure_and_return_matplot(fig, RESULT_DIR / f"{request.node.name}.png")


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR, tolerance=18)
def test_plot_clusters_plotly_existing_fig(request, iris_data):
    """Test plot_clusters_plotly works with a pre-existing Figure."""
    fig = go.Figure()
    fig.update_layout(title="My Custom Plotly Title")

    iris_x, labels, _ = iris_data
    fig = plot_clusters_plotly(iris_x.values, labels, reduction_method="pca", random_state=42, fig=fig)
    return save_plotly_figure_and_return_matplot(fig, RESULT_DIR / f"{request.node.name}.png")


def test_plot_clusters_plotly_empty_x():
    """Test plot_clusters_plotly raises ValueError for empty X."""
    with pytest.raises(ValueError, match="X must be a 2D array with at least 1 feature."):
        plot_clusters_plotly(np.array([]), np.array([]))


def test_plot_clusters_plotly_inconsistent_lengths():
    """Test plot_clusters_plotly raises ValueError for X and labels length mismatch."""
    X = np.array([[1, 2], [3, 4]])
    labels = np.array([0, 0, 1])
    with pytest.raises(ValueError, match="X and cluster_labels must have the same length."):
        plot_clusters_plotly(X, labels)


def test_plot_clusters_plotly_single_feature():
    """Test plot_clusters_plotly raises ValueError for a single-feature 2D array."""
    X = np.array([[1.0], [2.0], [3.0]])
    labels = np.array([0, 0, 1])
    with pytest.raises(ValueError, match="X must have at least 2 features for 2D scatter visualization."):
        plot_clusters_plotly(X, labels)


def test_plot_clusters_plotly_inconsistent_centers():
    """Test plot_clusters_plotly raises ValueError for cluster_centers length mismatch."""
    X = np.array([[1, 2], [3, 4]])
    labels = np.array([0, 1])
    centers = np.array([[1.5, 2.5]])  # Only 1 center, but 2 unique labels
    with pytest.raises(ValueError, match="Number of cluster centers must match the number of unique labels."):
        plot_clusters_plotly(X, labels, centers)
