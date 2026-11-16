"""Tests for the per-sample silhouette analysis."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from matplotlib import pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_samples, silhouette_score

from ds_utils.unsupervised.silhouette import (
    plot_silhouette_analysis,
)

BASELINE_DIR = Path(__file__).parents[1] / "baseline_images" / Path(__file__).parent.name / Path(__file__).stem


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
