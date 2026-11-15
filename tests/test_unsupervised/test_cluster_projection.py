"""Tests for the cluster scatter plots in a reduced-dimension space."""

from pathlib import Path

import numpy as np
import pytest
from matplotlib import pyplot as plt
from plotly import graph_objects as go

from ds_utils.unsupervised.cluster_projection import (
    plot_clusters,
    plot_clusters_plotly,
)
from tests.utils import save_plotly_figure_and_return_matplot

BASELINE_DIR = Path(__file__).parents[1] / "baseline_images" / Path(__file__).parent.name / Path(__file__).stem
RESULT_DIR = Path(__file__).parents[1] / "result_images" / Path(__file__).parent.name / Path(__file__).stem
RESULT_DIR.mkdir(exist_ok=True, parents=True)


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
    mocker.patch("ds_utils.unsupervised.cluster_projection.TSNE.fit_transform", return_value=fixed_2d)
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
    mocker.patch("ds_utils.unsupervised.cluster_projection.UMAP", None)
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
    mocker.patch("ds_utils.unsupervised.cluster_projection.UMAP", mock_umap)

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
