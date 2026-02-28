"""Shared fixtures for the test_xai test package."""

from matplotlib import pyplot as plt
import pytest


@pytest.fixture(autouse=True)
def setup_teardown():
    """Set up and tear down for each test in this module."""
    yield
    plt.cla()
    plt.close(plt.gcf())
