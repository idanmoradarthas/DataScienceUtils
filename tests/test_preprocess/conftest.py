"""Pytest fixtures for the preprocess test suite."""

from pathlib import Path

from matplotlib import pyplot as plt
import pandas as pd
import pytest

RESOURCES_PATH = Path(__file__).parent.parent / "resources"


@pytest.fixture
def loan_data():
    """Load and return loan dataset for testing."""
    return pd.read_csv(RESOURCES_PATH.joinpath("loan_final313.csv"), encoding="latin1", parse_dates=["issue_d"]).drop(
        "id", axis=1
    )


@pytest.fixture
def data_1m():
    """Load and return 1M dataset for testing."""
    return pd.read_csv(RESOURCES_PATH.joinpath("data.1M.zip"), compression="zip")


@pytest.fixture
def daily_min_temperatures():
    """Load and return daily minimum temperatures dataset for testing."""
    return pd.read_csv(RESOURCES_PATH.joinpath("daily-min-temperatures.csv"), parse_dates=["Date"])


@pytest.fixture(autouse=True)
def setup_teardown():
    """Set up and tear down for each test function in this module."""
    yield
    plt.cla()
    plt.close(plt.gcf())
