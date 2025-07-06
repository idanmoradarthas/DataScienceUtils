"""Mathematical utility functions."""
from typing import Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray


def safe_percentile(
    x: Union[pd.Series, NDArray[np.number]], percentile: float
) -> Union[None, np.floating]:
    """Calculate the percentile of an array, handling NA values.

    :param x: Input series or numeric numpy array with potentially nullable values.
    :param percentile: Percentile to calculate, must be between 0 and 100.
    :return: Return None If no valid (non-NA) values are present;
        Otherwise Calculated percentile value.
    :raise ValueError: If percentile is not between 0 and 100.
    :raise TypeError: If input is not pandas Series or numpy array.
    """
    if not isinstance(x, (pd.Series, np.ndarray)):
        raise TypeError("Input must be pandas Series or numpy array")

    if not 0 <= percentile <= 100:
        raise ValueError("Percentile must be between 0 and 100")

    if isinstance(x, np.ndarray):
        valid_values = x[~np.isnan(x)]
    else:
        valid_values = x[x.notna()]

    if len(valid_values) == 0:
        return None

    return np.percentile(valid_values, percentile)
