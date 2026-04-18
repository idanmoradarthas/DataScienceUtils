############
Transformers
############

The ``ds_utils.transformers`` package provides scikit-learn compatible transformers that wrap
or extend preprocessing estimators with ``get_feature_names_out`` (feature names API, SLEP007)
and consistent output dtypes for pipelines and :class:`~sklearn.compose.ColumnTransformer`.

.. toctree::
   :maxdepth: 2

   multi_label_binarizer
   sentence_embedding
