"""Formatting helpers for preprocess plotting utilities.

This module currently provides a formatter for converting Matplotlib
numeric date values into human-readable datetime strings.
"""

from matplotlib import dates, pyplot as plt


@plt.FuncFormatter
def _convert_numbers_to_dates(x, pos):
    """Convert Matplotlib numeric dates to formatted datetime strings."""
    return dates.num2date(x).strftime("%Y-%m-%d %H:%M")
