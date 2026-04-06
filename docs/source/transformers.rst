############
Transformers
############

The ``ds_utils.transformers`` module provides scikit-learn compatible transformers that wrap
or extend preprocessing estimators with ``get_feature_names_out`` (feature names API, SLEP007)
and consistent output dtypes for pipelines and :class:`~sklearn.compose.ColumnTransformer`.

********************************
MultiLabelBinarizerTransformer
********************************

.. autoclass:: ds_utils.transformers.MultiLabelBinarizerTransformer
   :members:

.. highlight:: python

Example
=======
::

    from ds_utils.transformers import MultiLabelBinarizerTransformer
    from sklearn.pipeline import Pipeline

    X = [["sci-fi", "action"], ["romance"], ["action", "comedy"]]
    mlb = MultiLabelBinarizerTransformer()
    X_t = mlb.fit_transform(X)

    names = mlb.get_feature_names_out()

    pipe = Pipeline([("mlb", MultiLabelBinarizerTransformer())])
    pipe.set_output(transform="pandas")
    df = pipe.fit_transform(X)

Both ``X_t`` (as a numpy array) and ``df`` (as a pandas DataFrame) contain the same binarized data.
Their output will be:

+------------+------------+-------------+------------+
|label_action|label_comedy|label_romance|label_sci-fi|
+============+============+=============+============+
|1.0         |0.0         |0.0          |1.0         |
+------------+------------+-------------+------------+
|0.0         |0.0         |1.0          |0.0         |
+------------+------------+-------------+------------+
|1.0         |1.0         |0.0          |0.0         |
+------------+------------+-------------+------------+
