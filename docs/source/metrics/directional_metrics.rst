Directional Metrics
===================

Directional metrics are used to evaluate the performance of forecasting models,
especially in time-series and financial contexts where the trend direction is
often more important than the exact magnitude of the change.

Directional Accuracy Score
--------------------------

.. autofunction:: ds_utils.metrics.time_series.directional_accuracy_score

Code Example
~~~~~~~~~~~~

.. code-block:: python

    from ds_utils.metrics.time_series import directional_accuracy_score
    import numpy as np

    # Directional accuracy — time series mode (uses previous value as baseline)
    # true changes from prev: +2, -4, +3, -2
    # pred changes from prev: +1, -5, +4, -1  -> all directions match
    y_true = np.array([100, 102, 98, 101, 99])
    y_pred = np.array([101, 103, 97, 102, 98])
    da = directional_accuracy_score(y_true, y_pred)
    print(f"Directional Accuracy: {da:.2%}")

Output:
Directional Accuracy: 100.00%

Directional Bias Score
----------------------

.. autofunction:: ds_utils.metrics.time_series.directional_bias_score

Code Example
~~~~~~~~~~~~

.. code-block:: python

    from ds_utils.metrics.time_series import directional_bias_score
    import numpy as np

    # Directional bias — detect systematic over/under-prediction
    # All predictions are 0.1 above the true value
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.1, 2.1, 3.1, 4.1, 5.1])
    bias = directional_bias_score(y_true, y_pred)
    print(f"Directional Bias: {bias:.2f}")

Output:
Directional Bias: 1.00
