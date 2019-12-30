*******
Strings
*******
The module of strings contains methods that help manipulate and process strings in a dataframe.

Append Tags to Frame
====================

.. autofunction:: strings::append_tags_to_frame

.. highlight:: python

Code Example
************
In this example we'll create our own simple dataset and parse it::

    import pandas

    from ds_utils.strings import append_tags_to_frame


    x_train = pandas.DataFrame([{"article_name": "1", "article_tags": "ds,ml,dl"},
                                {"article_name": "2", "article_tags": "ds,ml"}])
    x_test = pandas.DataFrame([{"article_name": "3", "article_tags": "ds,ml,py"}])

    x_train_with_tags, x_test_with_tags = append_tags_to_frame(x_train, x_test, "article_tags", "tag_")

And the following table will be the output for ``x_train_with_tags``:

+------------+------+------+------+
|article_name|tag_ds|tag_ml|tag_dl|
+============+======+======+======+
|1           |1     |1     |1     |
+------------+------+------+------+
|2           |1     |1     |0     |
+------------+------+------+------+

And the following table will be the output for ``x_test_with_tags``:

+------------+------+------+------+
|article_name|tag_ds|tag_ml|tag_dl|
+============+======+======+======+
|3           |1     |1     |0     |
+------------+------+------+------+

Significant Terms
=================

.. autofunction:: strings::significant_terms

Code Example
************
Let's create a simple corpus and extract significant terms from it::

    import pandas

    from ds_utils.strings import significant_terms

    corpus = ['This is the first document.', 'This document is the second document.',
              'And this is the third one.', 'Is this the first document?']
    data_frame = pandas.DataFrame(corpus, columns=["content"])
    subset_data_frame = data_frame[data_frame.index > 1]
    terms = significant_terms(data_frame, subset_data_frame, "content")

And the following table will be the output for ``terms``:

+-----+---+---+----+----+----+-----+--------+------+
|third|one|and|this|the |is  |first|document|second|
+=====+===+===+====+====+====+=====+========+======+
|1.0  |1.0|1.0|0.67|0.67|0.67|0.5  |0.25    |0.0   |
+-----+---+---+----+----+----+-----+--------+------+