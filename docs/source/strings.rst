#######
Strings
#######

The strings module contains methods that help manipulate and process strings in a DataFrame. These functions are particularly useful for tasks involving text analysis, feature engineering, and data preprocessing in data science and machine learning workflows.

********************
Append Tags to Frame
********************

The `append_tags_to_frame` function is designed to extract tags from a specified field in a DataFrame and create new binary columns for each unique tag. This is particularly useful for converting comma-separated tag lists into a one-hot encoded format, which is often required for machine learning models.

.. autofunction:: strings::append_tags_to_frame

.. highlight:: python

Code Example
============
In this example, we'll create a simple dataset and demonstrate how to use the `append_tags_to_frame` function:

``x_train``:

+------------+------------+
|article_name|article_tags|
+============+============+
|1           |ds,ml,dl    |
+------------+------------+
|2           |ds,ml       |
+------------+------------+

``x_test``:

+------------+------------+
|article_name|article_tags|
+============+============+
|3           |ds,ml,py    |
+------------+------------+

Here's how to use the function::

    import pandas as pd
    from ds_utils.strings import append_tags_to_frame

    x_train = pd.DataFrame([
        {"article_name": "1", "article_tags": "ds,ml,dl"},
        {"article_name": "2", "article_tags": "ds,ml"}
    ])
    x_test = pd.DataFrame([
        {"article_name": "3", "article_tags": "ds,ml,py"}
    ])

    x_train_with_tags, x_test_with_tags = append_tags_to_frame(x_train, x_test, "article_tags", "tag_")

The output for ``x_train_with_tags`` will be:

+------------+------+------+------+
|article_name|tag_ds|tag_ml|tag_dl|
+============+======+======+======+
|1           |1     |1     |1     |
+------------+------+------+------+
|2           |1     |1     |0     |
+------------+------+------+------+

And the output for ``x_test_with_tags`` will be:

+------------+------+------+------+
|article_name|tag_ds|tag_ml|tag_dl|
+============+======+======+======+
|3           |1     |1     |0     |
+------------+------+------+------+

*****************
Significant Terms
*****************

The `extract_significant_terms_from_subset` function is used to identify terms that are statistically overrepresented in a subset of documents compared to the full corpus. This can be particularly useful for tasks such as topic modeling, content categorization, or identifying distinctive vocabulary in specific document groups.

.. autofunction:: strings::extract_significant_terms_from_subset

Code Example
============
This example demonstrates how to use the function to extract significant terms from a subset of documents::

    import pandas as pd
    from ds_utils.strings import extract_significant_terms_from_subset

    corpus = [
        'This is the first document.',
        'This document is the second document.',
        'And this is the third one.',
        'Is this the first document?'
    ]
    data_frame = pd.DataFrame(corpus, columns=["content"])
    # Let's differentiate between the last two documents from the full corpus
    subset_data_frame = data_frame[data_frame.index > 1]
    terms = extract_significant_terms_from_subset(data_frame, subset_data_frame, "content")

The output for ``terms`` will be:

+-----+---+---+----+----+----+-----+--------+------+
|third|one|and|this|the |is  |first|document|second|
+=====+===+===+====+====+====+=====+========+======+
|1.0  |1.0|1.0|0.67|0.67|0.67|0.5  |0.25    |0.0   |
+-----+---+---+----+----+----+-----+--------+------+

Explanation of output values:
-----------------------------
The output is a series of terms with their corresponding significance scores. These scores represent how much more frequent a term is in the subset compared to the full corpus. A score of 1.0 indicates that the term appears exclusively in the subset, while lower scores suggest the term is present in both the subset and the full corpus, but more frequently in the subset. Scores closer to 0.0 indicate terms that are not particularly distinctive to the subset.

In this example:

- 'third', 'one', and 'and' have scores of 1.0, meaning they only appear in the subset.
- 'this', 'the', and 'is' have scores of 0.67, indicating they are more common in the subset but also present in the full corpus.
- 'document' has a low score of 0.25, suggesting it's common throughout the corpus and not particularly distinctive to the subset.
- 'second' has a score of 0.0, meaning it doesn't appear in the subset at all.

This function is particularly useful for identifying key terms that characterize specific subsets of your data, which can be valuable for tasks like document classification, content summarization, or exploratory data analysis.