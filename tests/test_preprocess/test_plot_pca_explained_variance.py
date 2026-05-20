"""Tests for the plot_pca_explained_variance function."""

from pathlib import Path

import pandas as pd
import pytest
from matplotlib import pyplot as plt
from sklearn.datasets import make_classification
from sklearn.preprocessing import MinMaxScaler

from ds_utils.preprocess.visualization import plot_pca_explained_variance

BASELINE_DIR = Path(__file__).parents[1] / "baseline_images" / Path(__file__).parent.name / Path(__file__).stem


@pytest.fixture
def sample_numeric_data():
    """Return a simple numeric DataFrame for PCA."""
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
    plot_pca_explained_variance(sample_numeric_data, legend_loc="upper left", linewidth=2, alpha=0.8)
    return plt.gcf()


def test_plot_pca_explained_variance_invalid_data():
    """Test plot_pca_explained_variance raises ValueError with non-numeric data."""
    df = pd.DataFrame({"A": [1, 2, 3], "B": ["x", "y", "z"]})
    with pytest.raises(ValueError, match="All columns in X must be numeric."):
        plot_pca_explained_variance(df)
