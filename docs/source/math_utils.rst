##########
Math Utils
##########

The module of math contains common method that help calculate statistical values.

***************
Safe Percentile
***************

.. autofunction:: math_utils::safe_percentile

.. highlight:: python

Code Examples
=============
The following code examples, let's see how to use `safe_percentile`. First, let's import the code::

    import pandas as pd
    import numpy as np

    from ds_utils.math_utils import safe_percentile
You can use `safe_percentile` with a dataframe::

    series = pd.Series([1, 2, np.nan, 4, 5])
    print(safe_percentile(series, 50))
The code will print `3.0` to the stdout. You can also use the method with NDArray::

    arr = np.array([1, 2, np.nan, 4, 5])
    print(safe_percentile(arr, 75))
The code will print `4.5` to the stdout. The method can also handle a data structure without valid values::

    empty_series = pd.Series([np.nan, np.nan])
    print(safe_percentile(empty_series, 50) is None)
the code will print `True` ro the stdout.