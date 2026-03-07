"""Test configuration and fixtures."""

import matplotlib as mpl
import pytest


@pytest.fixture(autouse=True)
def consistent_font():
    """Set consistent font family for all matplotlib tests.

    This fixture ensures that all tests use DejaVu Sans font,
    which is bundled with matplotlib and available across all platforms.
    This reduces font-rendering variability that could cause test failures.
    """
    original_font = mpl.rcParams["font.family"].copy()
    mpl.rcParams["font.family"] = "DejaVu Sans"  # Bundled with matplotlib
    yield
    mpl.rcParams["font.family"] = original_font
