"""Tests for the plots that help choose the number of clusters."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from matplotlib import pyplot as plt
from scipy.spatial.distance import euclidean
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score

from ds_utils.unsupervised.cluster_number import (
    plot_loss_vs_cluster_number,
    plot_silhouette_score_vs_cluster_number,
)

BASELINE_DIR = Path(__file__).parents[1] / "baseline_images" / Path(__file__).parent.name / Path(__file__).stem


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
    mocker.patch("ds_utils.unsupervised.cluster_number.silhouette_score", side_effect=[0.4, 0.7, 0.7, 0.5])
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
    silhouette_mock = mocker.patch("ds_utils.unsupervised.cluster_number.silhouette_score", return_value=0.5)
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
