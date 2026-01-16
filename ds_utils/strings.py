"""String manipulation utilities for data science tasks."""

import re
from collections import Counter
from typing import List, Tuple, Optional, Callable, Union

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MultiLabelBinarizer


def _tokenize(text_tags: str) -> List[str]:
    tags = text_tags.split(",")
    tags = [re.sub(r"[^a-zA-Z0-9_$-]", "", x) for x in tags]
    tags = [x.strip() for x in tags]
    tags = [x for x in tags if x]  # More concise than checking length
    return tags


def _normalize_tags(value, tokenizer, lowercase):
    """
    Normalize tag input to a list of strings.

    Handles both string inputs (which need tokenization) and list inputs
    (which are already tokenized).

    :param value: Either a string to tokenize or a list of tags
    :param tokenizer: Tokenizer function to use for string inputs
    :param lowercase: Whether to convert to lowercase
    :return: List of normalized tag strings
    """
    tags = []
    if isinstance(value, str):
        if value:  # non-empty string
            tags = tokenizer(value)
    elif isinstance(value, list):
        tags = value

    # Apply lowercase if requested
    if lowercase:
        tags = [tag.lower() if isinstance(tag, str) else str(tag).lower() for tag in tags]

    return tags


def append_tags_to_frame(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    field_name: str,
    prefix: str = "",
    max_features: Optional[int] = 500,
    min_df: Union[int, float] = 1,
    lowercase: bool = False,
    tokenizer: Optional[Callable[[str], List[str]]] = _tokenize,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Extract tags from a given field and append them to the dataframe.

    :param X_train: Pandas DataFrame with the train features.
    :param X_test: Pandas DataFrame with the test features.
    :param field_name: The feature to parse. The field can contain either comma-separated strings
                       (e.g., "tag1,tag2,tag3") or lists of tags (e.g., ["tag1", "tag2", "tag3"]).
    :param prefix: The prefix for new tag features.
    :param max_features: Maximum number of tag names to consider. Default is 500. This helps limit the number of
                         new columns created, especially useful for datasets with a large number of unique tags.
    :param min_df: When building the tag name set, ignore tags with a document frequency strictly
                   lower than the given threshold. If min_df is a float, the parameter represents a proportion
                   of documents. If integer, it represents absolute counts. Default is 1. This helps filter out
                   rare tags.
    :param lowercase: Convert all characters to lowercase before tokenizing the tag names. Default is False. Set to
                      True if you want case-insensitive tag matching.
    :param tokenizer: Callable to override the string tokenization step while preserving the
                      preprocessing and n-grams generation steps. Default splits by ",", and
                      retains alphanumeric characters with special characters "_", "$", and "-".
    :return: The train and test DataFrames with tags appended.
    :raise KeyError: if one of the frames is missing columns.
    """
    if X_train.empty:
        return pd.DataFrame(), pd.DataFrame()

    x_train_filled = X_train[field_name].fillna("")

    # Tokenize the training data (handles both strings and lists)
    train_tags = x_train_filled.apply(lambda x: _normalize_tags(x, tokenizer, lowercase))

    # Calculate document frequency
    doc_freq = Counter(tag for tags_list in train_tags for tag in set(tags_list))

    # Filter by min_df
    if isinstance(min_df, int):
        tags_to_keep = {tag for tag, freq in doc_freq.items() if freq >= min_df}
    else:  # float
        min_doc_count = min_df * len(X_train)
        tags_to_keep = {tag for tag, freq in doc_freq.items() if freq >= min_doc_count}

    # Select top max_features by frequency
    if max_features is not None:
        # Sort by frequency (descending), then alphabetically for deterministic ordering
        top_tags = sorted(tags_to_keep, key=lambda tag: (-doc_freq[tag], tag))[:max_features]
        tags_to_keep = set(top_tags)

    # Filter the tokenized tags to only include those in tags_to_keep
    train_tags_filtered = train_tags.apply(lambda tags: [tag for tag in tags if tag in tags_to_keep])

    # Use MultiLabelBinarizer to create the binary matrix
    mlb = MultiLabelBinarizer(classes=sorted(list(tags_to_keep)))
    x_train_binarized = mlb.fit_transform(train_tags_filtered)

    # Prepare test data (handles both strings and lists)
    test_tags = X_test[field_name].fillna("").apply(lambda x: _normalize_tags(x, tokenizer, lowercase))
    test_tags_filtered = test_tags.apply(lambda tags: [tag for tag in tags if tag in tags_to_keep])
    x_test_binarized = mlb.transform(test_tags_filtered)

    # Create DataFrames for the binarized tags
    feature_names = [prefix + tag_name for tag_name in mlb.classes_]
    x_train_tags = pd.DataFrame(x_train_binarized, columns=feature_names, index=X_train.index)
    x_test_tags = pd.DataFrame(x_test_binarized, columns=feature_names, index=X_test.index)

    x_train_reduced = X_train.drop(columns=[field_name])
    x_test_reduced = X_test.drop(columns=[field_name])

    return (
        pd.merge(x_train_reduced, x_train_tags, left_index=True, right_index=True, how="left"),
        pd.merge(x_test_reduced, x_test_tags, left_index=True, right_index=True, how="left"),
    )


def extract_significant_terms_from_subset(
    data_frame: pd.DataFrame,
    subset_data_frame: pd.DataFrame,
    field_name: str,
    vectorizer: CountVectorizer = CountVectorizer(encoding="utf-8", lowercase=True, max_features=500),
) -> pd.Series:
    """Return interesting or unusual occurrences of terms in a subset.

    Based on the elasticsearch significant_text aggregation:
    https://www.elastic.co/guide/en/elasticsearch/reference/current/search-aggregations-bucket-significantterms-aggregation.html#_scripted

    :param data_frame: The full dataset.
    :param subset_data_frame: The subset partition data over which the scoring will be calculated.
                              It can be filtered by feature or other boolean criteria.
    :param field_name: The feature to parse.
    :param vectorizer: Text count vectorizer which converts a collection of text to a matrix of token counts.
                       See more info here:
                       https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
    :return: Series of terms with scoring over the subset.

    :author: Eran Hirsch (https://github.com/eranhirs)
    """
    if data_frame.empty:
        return pd.Series()

    count_matrix = vectorizer.fit_transform(data_frame[field_name].dropna())
    matrix_df = pd.DataFrame(count_matrix.toarray(), columns=vectorizer.get_feature_names_out())

    subset_x = vectorizer.transform(subset_data_frame[field_name].dropna())
    subset_matrix_df = pd.DataFrame(subset_x.toarray(), columns=vectorizer.get_feature_names_out())

    subset_freq = subset_matrix_df.sum()
    superset_freq = matrix_df.sum()

    return (subset_freq / (superset_freq - subset_freq + 1)).sort_values(ascending=False)
