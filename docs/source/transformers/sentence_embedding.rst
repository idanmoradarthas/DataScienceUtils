******************************
SentenceEmbeddingTransformer
******************************

.. autoclass:: ds_utils.transformers.sentence_embedding.SentenceEmbeddingTransformer
   :members:

.. highlight:: python

Prerequisites
=============

This transformer requires the optional ``nlp`` dependency group::

    pip install data-science-utils[nlp]

Code Examples
=============

The following examples show how to use ``SentenceEmbeddingTransformer``.

Direct usage::

    from ds_utils.transformers.sentence_embedding import SentenceEmbeddingTransformer

    texts = ["The quick brown fox", "jumps over the lazy dog", "Hello world"]
    embed = SentenceEmbeddingTransformer()
    embeddings = embed.fit_transform(texts)
    names = embed.get_feature_names_out()

``embeddings`` is a numpy array of shape ``(n_samples, embedding_dimension)`` (e.g.
``(3, 384)`` for the default ``sentence-transformers/all-MiniLM-L6-v2`` model).  Feature
names from ``get_feature_names_out()`` follow the pattern ``dim_0``, ``dim_1``, …,
``dim_{n-1}``.  The output will be:

+-----------+-----------+-----------+-----+-----------+
| dim_0     | dim_1     | dim_2     | ... | dim_383   |
+===========+===========+===========+=====+===========+
| -0.0123   |  0.0456   |  0.0789   | ... |  0.0012   |
+-----------+-----------+-----------+-----+-----------+
|  0.0345   | -0.0678   |  0.0901   | ... | -0.0234   |
+-----------+-----------+-----------+-----+-----------+
|  0.0567   |  0.0890   | -0.0123   | ... |  0.0456   |
+-----------+-----------+-----------+-----+-----------+

Pipeline usage with a classifier::

    from ds_utils.transformers.sentence_embedding import SentenceEmbeddingTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import RandomForestClassifier

    pipeline = Pipeline([
        ('embeddings', SentenceEmbeddingTransformer(
            normalize_embeddings=True,
        )),
        ('classifier', RandomForestClassifier()),
    ])
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

Pipeline usage with pandas output::

    from ds_utils.transformers.sentence_embedding import SentenceEmbeddingTransformer
    from sklearn.pipeline import Pipeline

    pipe = Pipeline([("embed", SentenceEmbeddingTransformer())])
    pipe.set_output(transform="pandas")
    df = pipe.fit_transform(["hello", "world"])

``df`` will be a :class:`pandas.DataFrame` with columns ``dim_0``, ``dim_1``, etc.

ColumnTransformer usage::

    from ds_utils.transformers.sentence_embedding import SentenceEmbeddingTransformer
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler
    import pandas as pd

    df = pd.DataFrame({
        "description": ["a product", "another item"],
        "price": [9.99, 19.99],
    })
    pre = ColumnTransformer([
        ("text", SentenceEmbeddingTransformer(), ["description"]),
        ("num", StandardScaler(), ["price"]),
    ])
    X_out = pre.fit_transform(df)

Advanced: using prompts and quantized embeddings::

    from ds_utils.transformers.sentence_embedding import SentenceEmbeddingTransformer

    # Use a prompt for asymmetric search
    embed = SentenceEmbeddingTransformer(
        prompt="search_query: ",
        precision="int8",
        truncate_dim=128,
    )
    embeddings = embed.fit_transform(["How do I reset my password?"])
