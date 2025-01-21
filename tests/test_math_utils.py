import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray

from ds_utils.math_utils import safe_percentile


@pytest.fixture
def sample_data() -> tuple[pd.Series, NDArray[np.number]]:
    """Fixture providing sample data for testing."""
    series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    array = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    return series, array


def test_valid_series(sample_data):
    """Test percentile calculation with valid pandas Series."""
    series, _ = sample_data
    assert safe_percentile(series, 50) == 3.0
    assert safe_percentile(series, 0) == 1.0
    assert safe_percentile(series, 100) == 5.0
    assert isinstance(safe_percentile(series, 50), np.floating)


def test_valid_array(sample_data):
    """Test percentile calculation with valid numpy array."""
    _, array = sample_data
    assert safe_percentile(array, 50) == 3.0
    assert safe_percentile(array, 0) == 1.0
    assert safe_percentile(array, 100) == 5.0
    assert isinstance(safe_percentile(array, 50), np.floating)


@pytest.mark.parametrize("x, expected",
                         [
                             (pd.Series([1.0, np.nan, 3.0, None, 5.0]), 3.0),
                             (np.array([1.0, np.nan, 3.0, np.nan, 5.0]), 3.0),
                             (pd.Series([np.nan, None, pd.NA]), None),
                             (np.array([np.nan, np.nan, np.nan]), None),
                             (pd.Series([]), None),
                             (np.array([]), None)
                         ],
                         ids=["test_series_with_na", "test_array_with_nan", "test_all_na_series",
                              "test_all_nan_array", "test_empty_series", "test_empty_array"])
def test_with_na(x, expected):
    """Test percentile calculation with pandas Series containing NA values."""
    assert safe_percentile(x, 50) == expected


def test_invalid_percentile_values():
    """Test behavior with invalid percentile values."""
    series = pd.Series([1, 2, 3])
    with pytest.raises(ValueError, match="Percentile must be between 0 and 100"):
        safe_percentile(series, -1)
    with pytest.raises(ValueError, match="Percentile must be between 0 and 100"):
        safe_percentile(series, 101)


def test_invalid_input_types():
    """Test behavior with invalid input types."""
    invalid_inputs = [
        [1, 2, 3],  # list
        (1, 2, 3),  # tuple
        {"a": 1},  # dict
        "123",  # str
        42,  # int
    ]

    for invalid_input in invalid_inputs:
        with pytest.raises(TypeError, match="Input must be pandas Series or numpy array"):
            safe_percentile(invalid_input, 50)


def test_integer_array():
    """Test behavior with integer arrays."""
    array = np.array([1, 2, 3, 4, 5])
    result = safe_percentile(array, 50)
    assert result == 3.0
    assert isinstance(result, np.floating)


def test_mixed_type_series():
    """Test behavior with mixed type Series (should convert to float)."""
    series = pd.Series([1, 2.5, 3, 4.5, 5])
    result = safe_percentile(series, 50)
    assert result == 3.0
    assert isinstance(result, np.floating)
