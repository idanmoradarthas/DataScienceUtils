"""Tests for the package version."""

import ds_utils


def test_version():
    """Test that the package version is correct."""
    assert ds_utils.__version__ == "1.10.0rc3"
