"""Tests for the plot_pca_explained_variance function."""

from pathlib import Path

import pandas as pd
import pytest
from matplotlib import pyplot as plt
from sklearn.datasets import make_classification
from sklearn.preprocessing import MinMaxScaler

from ds_utils.preprocess.visualization import plot_pca_explained_variance

# Directory to store and load pytest-mpl baseline images for this test file
# The path is constructed to mirror the package structure inside the tests folder.
BASELINE_DIR = Path(__file__).parents[1] / "baseline_images" / Path(__file__).parent.name / Path(__file__).stem


@pytest.fixture
def sample_numeric_data():
    """Return a simple numeric DataFrame for PCA."""
    # We use n_redundant=5 to ensure the PCA results in an "elbow" curve,
    # making for a more realistic and visually meaningful test.
    X, _ = make_classification(n_samples=200, n_features=15, n_informative=5, n_redundant=5, random_state=42)
    return pd.DataFrame(X, columns=[f"feature_{i}" for i in range(15)])


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR)
def test_plot_pca_explained_variance_default(sample_numeric_data):
    """Test plot_pca_explained_variance with default parameters."""
    plot_pca_explained_variance(sample_numeric_data)
    return plt.gcf()


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR)
def test_plot_pca_explained_variance_no_scaling(sample_numeric_data):
    """Test plot_pca_explained_variance without scaling."""
    plot_pca_explained_variance(sample_numeric_data, use_scaling=False)
    return plt.gcf()


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR)
def test_plot_pca_explained_variance_custom_scaler(sample_numeric_data):
    """Test plot_pca_explained_variance with a custom scaler."""
    plot_pca_explained_variance(sample_numeric_data, use_scaling=True, scaler=MinMaxScaler())
    return plt.gcf()


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR)
def test_plot_pca_explained_variance_kwargs(sample_numeric_data):
    """Test plot_pca_explained_variance with extra kwargs and specific legend location."""
    ax = plot_pca_explained_variance(sample_numeric_data, legend_loc="upper left", linewidth=2, alpha=0.8)

    # "upper left" is mapped to location code 2 in matplotlib
    assert ax.get_legend() is not None
    assert ax.get_legend()._loc == 2
    return plt.gcf()


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR)
def test_plot_pca_explained_variance_ax_pass_through(sample_numeric_data):
    """Test plot_pca_explained_variance with a passed ax parameter.

    Note: The figsize=(10, 6) here is intentional for testing ax pass-through,
    meaning the baseline image will have different dimensions than the defaults.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title("Custom Ax Title")

    result_ax = plot_pca_explained_variance(sample_numeric_data, ax=ax)

    # Assert that the returned ax is exactly the one we passed in
    assert result_ax is ax
    # The function intentionally overwrites the title, so we assert it changed
    assert result_ax.get_title() == "PCA - Cumulative Explained Variance"

    return fig


def test_plot_pca_explained_variance_pca_kwargs(sample_numeric_data):
    """Test that pca_kwargs are forwarded to PCA correctly."""
    ax = plot_pca_explained_variance(sample_numeric_data, pca_kwargs={"n_components": 5})
    # Index 0 is the cumulative variance line; axhlines are added after in the implementation
    line = ax.get_lines()[0]
    assert len(line.get_xdata()) == 5
    plt.close()


def test_plot_pca_explained_variance_invalid_data():
    """Test plot_pca_explained_variance raises ValueError with non-numeric data."""
    df = pd.DataFrame({"A": [1, 2, 3], "B": ["x", "y", "z"]})
    with pytest.raises(ValueError, match="All columns in X must be numeric."):
        plot_pca_explained_variance(df)
