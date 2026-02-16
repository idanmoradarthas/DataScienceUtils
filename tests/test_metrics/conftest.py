"""Configuration and fixtures for the metrics tests.

This module contains shared fixtures and setup/teardown logic for testing the
metrics functionality in the DataScienceUtils package.
"""

from pathlib import Path
from typing import Dict, Union

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

RESOURCES_DIR = Path(__file__).parent.parent / "resources"


@pytest.fixture
def iris_data() -> Dict[str, np.ndarray]:
    """Load and return iris dataset splits."""
    return {
        key: pd.read_csv(RESOURCES_DIR / f"iris_{key}.csv").values for key in ["x_train", "x_test", "y_train", "y_test"]
    }


@pytest.fixture
def classifiers() -> Dict[str, Union[DecisionTreeClassifier, RandomForestClassifier]]:
    """Create and return classifier instances."""
    return {
        "DecisionTreeClassifier": DecisionTreeClassifier(random_state=0),
        "RandomForestClassifier": RandomForestClassifier(random_state=0, n_estimators=5),
    }


@pytest.fixture(autouse=True)
def setup_teardown():
    """Set up and tear down for each test function in this module."""
    yield
    plt.cla()
    plt.close(plt.gcf())
