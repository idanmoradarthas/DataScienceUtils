##########
Math Utils
##########

This module contains common methods that help calculate statistical values.

***************
Safe Percentile
***************

.. autofunction:: ds_utils.math_utils.safe_percentile

.. highlight:: python

Code Examples
=============
Let's see how to use ``safe_percentile``. First, import the required packages::

    import pandas as pd
    import numpy as np

    from ds_utils.math_utils import safe_percentile

You can use ``safe_percentile`` with a pandas Series::

    series = pd.Series([1, 2, np.nan, 4, 5])
    print(safe_percentile(series, 50))

The code will print ``3.0`` to stdout.

You can also use the method with a NumPy array::

    arr = np.array([1, 2, np.nan, 4, 5])
    print(safe_percentile(arr, 75))

The code will print ``4.5`` to stdout.

The method can also handle a data structure without any valid values::

    empty_series = pd.Series([np.nan, np.nan])
    print(safe_percentile(empty_series, 50) is None)

The code will print ``True`` to stdout.