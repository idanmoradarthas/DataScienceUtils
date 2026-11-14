"""Tests for unsupervised learning utility functions."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from matplotlib import pyplot as plt
from scipy.spatial.distance import euclidean
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_samples, silhouette_score

from ds_utils.unsupervised import (
    plot_silhouette_score_vs_cluster_number,
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
# plot_silhouette_score_vs_cluster_number
# ---------------------------------------------------------------------------


class _FakeClustering:
    """Clustering stand-in that records its construction arguments and assigns samples to clusters in turn."""

    created = []

    def __init__(self, n_clusters, **parameters):
        self.n_clusters = n_clusters
        self.parameters = parameters
        _FakeClustering.created.append((n_clusters, parameters))

    def fit_predict(self, X):
        """Return labels that cycle through the requested number of clusters."""
        return np.arange(len(X)) % self.n_clusters


@pytest.fixture
def fake_clustering():
    """Return the recording clustering class with an empty call history."""
    _FakeClustering.created = []
    return _FakeClustering


def _kmeans_scores(X, k_values, **parameters):
    """Compute the reference silhouette scores of KMeans fits for every k."""
    return [silhouette_score(X, KMeans(n_clusters=k, **parameters).fit_predict(X)) for k in k_values]


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR)
def test_plot_silhouette_score_vs_cluster_number(iris_data):
    """Test comparing the silhouette scores of the default KMeans over the default range of k."""
    iris_x, _, _ = iris_data
    plot_silhouette_score_vs_cluster_number(iris_x.values)
    return plt.gcf()


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR)
def test_plot_silhouette_score_vs_cluster_number_exist_ax(iris_data):
    """Test comparing the silhouette scores on an existing Axes object."""
    fig, ax = plt.subplots()
    ax.set_facecolor("tab:red")

    iris_x, _, _ = iris_data
    returned_ax = plot_silhouette_score_vs_cluster_number(iris_x.values, range(2, 7), ax=ax)
    assert returned_ax is ax
    return fig


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR)
def test_plot_silhouette_score_vs_cluster_number_custom_algorithm(blobs_data):
    """Test comparing the silhouette scores of a custom algorithm, metric, and Matplotlib line options."""
    X, _ = blobs_data
    plot_silhouette_score_vs_cluster_number(
        X,
        range(2, 8),
        algorithm=AgglomerativeClustering,
        algorithm_parameters={"linkage": "average"},
        metric="manhattan",
        annotate_scores=False,
        color="tab:green",
        linestyle=":",
    )
    return plt.gcf()


def test_plot_silhouette_score_vs_cluster_number_matches_manual_computation(iris_data):
    """Test that the plotted scores are the silhouette scores of KMeans with the default parameters."""
    iris_x, _, _ = iris_data
    X = iris_x.values
    ax = plot_silhouette_score_vs_cluster_number(X, range(2, 7))

    k_values = list(range(2, 7))
    expected = _kmeans_scores(X, k_values, random_state=42, n_init=10)
    assert list(ax.lines[0].get_xdata()) == k_values
    assert list(ax.lines[0].get_ydata()) == pytest.approx(expected)


def test_plot_silhouette_score_vs_cluster_number_default_range_is_two_to_ten(blobs_data):
    """Test that the default k_range covers k = 2, ..., 10."""
    X, _ = blobs_data
    ax = plot_silhouette_score_vs_cluster_number(X)
    assert list(ax.lines[0].get_xdata()) == list(range(2, 11))


def test_plot_silhouette_score_vs_cluster_number_is_deterministic_by_default(blobs_data):
    """Test that two calls with default parameters draw identical scores."""
    X, _ = blobs_data
    first = plot_silhouette_score_vs_cluster_number(X, range(2, 6)).lines[0].get_ydata()
    second = plot_silhouette_score_vs_cluster_number(X, range(2, 6)).lines[0].get_ydata()
    assert list(first) == list(second)


def test_plot_silhouette_score_vs_cluster_number_labels_title_and_ticks(blobs_data):
    """Test the axis labels, the title, the markers, and that the x ticks are whole cluster numbers."""
    X, _ = blobs_data
    ax = plot_silhouette_score_vs_cluster_number(X, range(2, 5))
    ax.figure.canvas.draw()

    assert ax.get_xlabel() == "Number of clusters (k)"
    assert ax.get_ylabel() == "Average silhouette score"
    assert ax.get_title() == "Silhouette Score vs Number of Clusters"
    assert ax.lines[0].get_marker() == "o"
    assert ax.lines[0].get_linewidth() == 2
    visible_ticks = [tick for tick in ax.get_xticks() if ax.get_xlim()[0] <= tick <= ax.get_xlim()[1]]
    assert visible_ticks == [2, 3, 4]


def test_plot_silhouette_score_vs_cluster_number_marks_best_k(blobs_data):
    """Test that the dashed vertical line and the legend mark the k with the highest score."""
    X, _ = blobs_data
    ax = plot_silhouette_score_vs_cluster_number(X, range(2, 8))

    scores = ax.lines[0].get_ydata()
    best_k = int(np.argmax(scores)) + 2
    assert best_k == 3
    assert ax.lines[1].get_xdata()[0] == best_k
    assert ax.lines[1].get_linestyle() == "--"
    assert ax.get_legend().get_texts()[0].get_text() == f"Best: k={best_k} ({max(scores):.3f})"


def test_plot_silhouette_score_vs_cluster_number_best_k_tie_uses_smallest_k(mocker, blobs_data):
    """Test that the smallest k is marked when several values of k share the highest score."""
    X, _ = blobs_data
    mocker.patch("ds_utils.unsupervised.silhouette_score", side_effect=[0.4, 0.7, 0.7, 0.5])
    ax = plot_silhouette_score_vs_cluster_number(X, range(2, 6))
    assert ax.lines[1].get_xdata()[0] == 3
    assert ax.get_legend().get_texts()[0].get_text() == "Best: k=3 (0.700)"


def test_plot_silhouette_score_vs_cluster_number_no_best_marker(blobs_data):
    """Test that show_best=False draws only the score line and no legend."""
    X, _ = blobs_data
    ax = plot_silhouette_score_vs_cluster_number(X, range(2, 6), show_best=False)
    assert len(ax.lines) == 1
    assert ax.get_legend() is None


def test_plot_silhouette_score_vs_cluster_number_annotations(blobs_data):
    """Test that every point is annotated with its score and that annotate_scores=False removes the labels."""
    X, _ = blobs_data
    ax = plot_silhouette_score_vs_cluster_number(X, range(2, 6))
    scores = ax.lines[0].get_ydata()
    assert [text.get_text() for text in ax.texts] == [f"{score:.3f}" for score in scores]

    plt.close(ax.figure)
    ax = plot_silhouette_score_vs_cluster_number(X, range(2, 6), annotate_scores=False)
    assert len(ax.texts) == 0


def test_plot_silhouette_score_vs_cluster_number_kwargs_forwarded(blobs_data):
    """Test that kwargs reach Axes.plot() and override its defaults."""
    X, _ = blobs_data
    ax = plot_silhouette_score_vs_cluster_number(
        X, range(2, 5), color="tab:purple", marker="s", linewidth=4, linestyle=":"
    )
    line = ax.lines[0]
    assert (line.get_color(), line.get_marker(), line.get_linewidth(), line.get_linestyle()) == (
        "tab:purple",
        "s",
        4,
        ":",
    )


def test_plot_silhouette_score_vs_cluster_number_line_width_alias_overrides_default(blobs_data):
    """Test that the lw alias is accepted and replaces the default line width instead of clashing with it."""
    X, _ = blobs_data
    ax = plot_silhouette_score_vs_cluster_number(X, range(2, 5), lw=4)
    assert ax.lines[0].get_linewidth() == 4


def test_plot_silhouette_score_vs_cluster_number_score_line_is_first_line_added_to_existing_ax(blobs_data):
    """Test that on an Axes that already holds lines the score line is the first one added by the call."""
    X, _ = blobs_data
    _, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    lines_before = len(ax.lines)

    plot_silhouette_score_vs_cluster_number(X, range(2, 5), ax=ax)
    assert list(ax.lines[lines_before].get_xdata()) == [2, 3, 4]


def test_plot_silhouette_score_vs_cluster_number_metric_forwarded(mocker, blobs_data):
    """Test that metric is passed to silhouette_score."""
    X, _ = blobs_data
    silhouette_mock = mocker.patch("ds_utils.unsupervised.silhouette_score", return_value=0.5)
    plot_silhouette_score_vs_cluster_number(X, range(2, 4), metric="cosine")
    assert [call.kwargs for call in silhouette_mock.call_args_list] == [{"metric": "cosine"}] * 2


def test_plot_silhouette_score_vs_cluster_number_custom_algorithm_receives_k_and_parameters(
    blobs_data, fake_clustering
):
    """Test that the algorithm is created once per k with n_clusters from k_range and the given parameters."""
    X, _ = blobs_data
    plot_silhouette_score_vs_cluster_number(
        X, range(2, 5), algorithm=fake_clustering, algorithm_parameters={"linkage": "ward"}
    )
    assert fake_clustering.created == [(2, {"linkage": "ward"}), (3, {"linkage": "ward"}), (4, {"linkage": "ward"})]


def test_plot_silhouette_score_vs_cluster_number_n_clusters_parameter_is_ignored(blobs_data, fake_clustering):
    """Test that an n_clusters entry in algorithm_parameters is replaced by the values of k_range."""
    X, _ = blobs_data
    plot_silhouette_score_vs_cluster_number(
        X, [2, 3], algorithm=fake_clustering, algorithm_parameters={"n_clusters": 99, "linkage": "ward"}
    )
    assert fake_clustering.created == [(2, {"linkage": "ward"}), (3, {"linkage": "ward"})]


def test_plot_silhouette_score_vs_cluster_number_non_kmeans_gets_no_default_parameters(blobs_data, fake_clustering):
    """Test that the KMeans default parameters are not forced onto other algorithms."""
    X, _ = blobs_data
    plot_silhouette_score_vs_cluster_number(X, [2, 3], algorithm=fake_clustering)
    assert fake_clustering.created == [(2, {}), (3, {})]


def test_plot_silhouette_score_vs_cluster_number_agglomerative_clustering_with_defaults(blobs_data):
    """Test that an algorithm without random_state works with the default parameters."""
    X, _ = blobs_data
    ax = plot_silhouette_score_vs_cluster_number(X, range(2, 6), algorithm=AgglomerativeClustering)
    expected = [silhouette_score(X, AgglomerativeClustering(n_clusters=k).fit_predict(X)) for k in range(2, 6)]
    assert list(ax.lines[0].get_ydata()) == pytest.approx(expected)


def test_plot_silhouette_score_vs_cluster_number_algorithm_parameters_replace_kmeans_defaults(iris_data):
    """Test that algorithm_parameters replaces the default KMeans parameters instead of extending them."""
    iris_x, _, _ = iris_data
    X = iris_x.values
    ax = plot_silhouette_score_vs_cluster_number(X, range(2, 5), algorithm_parameters={"random_state": 7, "n_init": 1})
    expected = _kmeans_scores(X, range(2, 5), random_state=7, n_init=1)
    assert list(ax.lines[0].get_ydata()) == pytest.approx(expected)


def test_plot_silhouette_score_vs_cluster_number_k_range_is_sorted_and_deduplicated(blobs_data):
    """Test that unsorted k_range values with duplicates are plotted once each, in ascending order."""
    X, _ = blobs_data
    ax = plot_silhouette_score_vs_cluster_number(X, [5, 3, 3, 2])
    assert list(ax.lines[0].get_xdata()) == [2, 3, 5]


@pytest.mark.parametrize("k_range", [np.arange(2, 5), (2, 3, 4), {2, 3, 4}])
def test_plot_silhouette_score_vs_cluster_number_accepts_any_iterable_of_integers(blobs_data, k_range):
    """Test that arrays, tuples, and sets of integers are accepted as k_range."""
    X, _ = blobs_data
    ax = plot_silhouette_score_vs_cluster_number(X, k_range)
    assert list(ax.lines[0].get_xdata()) == [2, 3, 4]


def test_plot_silhouette_score_vs_cluster_number_accepts_a_generator_as_k_range(blobs_data):
    """Test that a generator of integers is consumed once and accepted as k_range."""
    X, _ = blobs_data
    ax = plot_silhouette_score_vs_cluster_number(X, (k for k in (2, 3, 4)))
    assert list(ax.lines[0].get_xdata()) == [2, 3, 4]


def test_plot_silhouette_score_vs_cluster_number_dataframe_input(blobs_data):
    """Test that a DataFrame gives the same scores as the equivalent array."""
    X, _ = blobs_data
    array_scores = plot_silhouette_score_vs_cluster_number(X, range(2, 5)).lines[0].get_ydata()
    frame_scores = (
        plot_silhouette_score_vs_cluster_number(pd.DataFrame(X, columns=["a", "b"]), range(2, 5)).lines[0].get_ydata()
    )
    assert list(frame_scores) == pytest.approx(list(array_scores))


def test_plot_silhouette_score_vs_cluster_number_undefined_scores_are_skipped_with_warning(blobs_data):
    """Test that a k whose clustering has fewer than two clusters is left out of the plot with a warning."""
    X, _ = blobs_data

    class CollapsingClustering:
        """Return a single cluster for k=3 and a proper clustering otherwise."""

        def __init__(self, n_clusters):
            self.n_clusters = n_clusters

        def fit_predict(self, data):
            """Return the labels of the fake clustering."""
            if self.n_clusters == 3:
                return np.zeros(len(data), dtype=int)
            return np.arange(len(data)) % self.n_clusters

    with pytest.warns(UserWarning, match=r"undefined for k=\[3\]"):
        ax = plot_silhouette_score_vs_cluster_number(X, range(2, 5), algorithm=CollapsingClustering)

    scores = ax.lines[0].get_ydata()
    assert list(ax.lines[0].get_xdata()) == [2, 3, 4]
    assert np.isnan(scores[1])
    assert not np.isnan(scores[[0, 2]]).any()
    assert [text.get_text() for text in ax.texts] == [f"{scores[0]:.3f}", f"{scores[2]:.3f}"]
    assert "k=3" not in ax.get_legend().get_texts()[0].get_text()


def test_plot_silhouette_score_vs_cluster_number_one_cluster_per_sample_is_undefined():
    """Test that a clustering with as many clusters as samples counts as undefined and is skipped."""
    X = np.array([[0.0, 0.0], [0.1, 0.0], [5.0, 5.0], [5.1, 5.0]])

    class SingletonClustering:
        """Put every sample in its own cluster for k=3 and use a perfect split otherwise."""

        def __init__(self, n_clusters):
            self.n_clusters = n_clusters

        def fit_predict(self, data):
            """Return the labels of the fake clustering."""
            return np.arange(len(data)) if self.n_clusters == 3 else np.array([0, 0, 1, 1])

    with pytest.warns(UserWarning, match=r"undefined for k=\[3\]"):
        ax = plot_silhouette_score_vs_cluster_number(X, [2, 3], algorithm=SingletonClustering)
    assert np.isnan(ax.lines[0].get_ydata()[1])


def test_plot_silhouette_score_vs_cluster_number_all_scores_undefined(blobs_data):
    """Test that ValueError is raised when no k yields a defined silhouette score."""
    X, _ = blobs_data

    class SingleClusterClustering:
        """Return a single cluster for every k."""

        def __init__(self, n_clusters):
            self.n_clusters = n_clusters

        def fit_predict(self, data):
            """Return the labels of the fake clustering."""
            return np.zeros(len(data), dtype=int)

    with pytest.raises(ValueError, match="No silhouette score could be computed for any value in k_range."):
        plot_silhouette_score_vs_cluster_number(X, range(2, 5), algorithm=SingleClusterClustering)


def test_plot_silhouette_score_vs_cluster_number_empty_k_range(blobs_data):
    """Test that an empty k_range raises ValueError."""
    X, _ = blobs_data
    with pytest.raises(ValueError, match="k_range must contain at least one value."):
        plot_silhouette_score_vs_cluster_number(X, [])


@pytest.mark.parametrize(
    "k_range",
    [
        [1, 2, 3],
        [0, 2],
        [-3, 2],
        [2.5, 3],
        [2.0, 3.0],
        ["2", "3"],
        [True, 3],
        [2, 2.0, 3],
        [3, 2, 2.0],
        [2, 3, np.float64(3.0)],
    ],
)
def test_plot_silhouette_score_vs_cluster_number_invalid_k_values(blobs_data, k_range):
    """Test that k values below 2 and non-integer k values raise ValueError."""
    X, _ = blobs_data
    with pytest.raises(ValueError, match="k_range must contain only integers of at least 2."):
        plot_silhouette_score_vs_cluster_number(X, k_range)


@pytest.mark.parametrize("k_range", [[2, 60], [3, 61], range(55, 70)])
def test_plot_silhouette_score_vs_cluster_number_k_not_smaller_than_number_of_samples(blobs_data, k_range):
    """Test that a k that is not smaller than the number of samples raises ValueError."""
    X, _ = blobs_data
    with pytest.raises(ValueError, match="k_range values must be smaller than the number of samples."):
        plot_silhouette_score_vs_cluster_number(X, k_range)


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
