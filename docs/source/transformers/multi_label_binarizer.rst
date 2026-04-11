******************************
MultiLabelBinarizerTransformer
******************************

.. autoclass:: ds_utils.transformers.multi_label_binarizer.MultiLabelBinarizerTransformer
   :members:

.. highlight:: python

Code Examples
=============

The following examples show the three main ways to use ``MultiLabelBinarizerTransformer``.

Direct usage::

    from ds_utils.transformers.multi_label_binarizer import MultiLabelBinarizerTransformer

    X = [["sci-fi", "action"], ["romance"], ["action", "comedy"]]
    mlb = MultiLabelBinarizerTransformer()
    X_t = mlb.fit_transform(X)
    names = mlb.get_feature_names_out()

``X_t`` is a numpy array of shape ``(n_samples, n_classes)``, dtype ``float64``, with columns
corresponding to ``names`` (e.g. ``['label_action', 'label_comedy', 'label_romance', 'label_sci-fi']``).
The output will be:

+------------+------------+-------------+------------+
|label_action|label_comedy|label_romance|label_sci-fi|
+============+============+=============+============+
|1.0         |0.0         |0.0          |1.0         |
+------------+------------+-------------+------------+
|0.0         |0.0         |1.0          |0.0         |
+------------+------------+-------------+------------+
|1.0         |1.0         |0.0          |0.0         |
+------------+------------+-------------+------------+

Pipeline usage with pandas output::

    from ds_utils.transformers.multi_label_binarizer import MultiLabelBinarizerTransformer
    from sklearn.pipeline import Pipeline

    pipe = Pipeline([("mlb", MultiLabelBinarizerTransformer())])
    pipe.set_output(transform="pandas")
    df = pipe.fit_transform(X)

ColumnTransformer usage::

    from ds_utils.transformers.multi_label_binarizer import MultiLabelBinarizerTransformer
    from sklearn.compose import ColumnTransformer
    import pandas as pd

    df = pd.DataFrame({"tags": [["x", "y"], ["z"]], "num": [1.0, 2.0]})
    pre = ColumnTransformer(
        [("mlb", MultiLabelBinarizerTransformer(), ["tags"])],
        remainder="passthrough",
    )
    X_out = pre.fit_transform(df)
