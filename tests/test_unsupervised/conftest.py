"""Configuration and fixtures for the unsupervised tests."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from matplotlib import pyplot as plt
from sklearn.datasets import make_blobs

RESOURCES_DIR = Path(__file__).parents[1] / "resources"


@pytest.fixture
def iris_data():
    """Load and return Iris dataset components for testing."""
    iris_x = pd.read_csv(RESOURCES_DIR / "iris_x_full.csv")
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


@pytest.fixture(autouse=True)
def setup_teardown():
    """Set up and tear down for each test in this module."""
    yield
    plt.cla()
    plt.close(plt.gcf())


@pytest.fixture
def blobs_data():
    """Return three well separated 2D blobs with their generating labels."""
    X, labels = make_blobs(n_samples=60, centers=3, cluster_std=1.0, random_state=42)
    return X, labels
