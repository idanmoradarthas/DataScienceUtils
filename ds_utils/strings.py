import re
from typing import List, Tuple, Optional, Callable, Union

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


def _tokenize(text_tags: str) -> List[str]:
    tags = text_tags.split(",")
    tags = [re.sub(r"[^a-zA-Z0-9_$-]", "", x) for x in tags]
    tags = [x.strip() for x in tags]
    tags = [x for x in tags if x]  # More concise than checking length
    return tags


def append_tags_to_frame(
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        field_name: str,
        prefix: str = "",
        max_features: Optional[int] = 500,
        min_df: Union[int, float] = 1,
        lowercase: bool = False,
        tokenizer: Optional[Callable[[str], List[str]]] = _tokenize
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract tags from a given field and append them to the dataframe.

    :param X_train: Pandas DataFrame with the train features.
    :param X_test: Pandas DataFrame with the test features.
    :param field_name: The feature to parse.
    :param prefix: The prefix for new tag features.
    :param max_features: Maximum number of tag names to consider. Default is 500. This helps limit the number of
                         new columns created, especially useful for datasets with a large number of unique tags.
    :param min_df: When building the tag name set, ignore tags with a document frequency strictly
                   lower than the given threshold. If float, the parameter represents a proportion
                   of documents. If integer, it represents absolute counts. Default is 1. This helps filter out
                   rare tags.
    :param lowercase: Convert all characters to lowercase before tokenizing the tag names. Default is False. Set to
                      True if you want case-insensitive tag matching.
    :param tokenizer: Callable to override the string tokenization step while preserving the
                      preprocessing and n-grams generation steps. Default splits by ",", and
                      retains alphanumeric characters with special characters "_", "$", and "-".
    :return: The train and test DataFrames with tags appended.
    :raise KeyError: if the one of the frames is missing columns.
    """
    vectorizer = CountVectorizer(binary=True, tokenizer=tokenizer, encoding="utf-8", lowercase=lowercase,
                                 min_df=min_df, max_features=max_features, token_pattern=None)

    if X_train.empty:
        return pd.DataFrame(), pd.DataFrame()

    x_train_filled = X_train[field_name].fillna("")
    x_test_filled = X_test[field_name].fillna("")

    x_train_count_matrix = vectorizer.fit_transform(x_train_filled)
    x_test_count_matrix = vectorizer.transform(x_test_filled)

    feature_names = [prefix + tag_name for tag_name in vectorizer.get_feature_names_out()]

    x_train_tags = pd.DataFrame(x_train_count_matrix.toarray(), columns=feature_names, index=X_train.index)
    x_test_tags = pd.DataFrame(x_test_count_matrix.toarray(), columns=feature_names, index=X_test.index)

    x_train_reduced = X_train.drop(columns=[field_name])
    x_test_reduced = X_test.drop(columns=[field_name])

    return (pd.merge(x_train_reduced, x_train_tags, left_index=True, right_index=True, how="left"),
            pd.merge(x_test_reduced, x_test_tags, left_index=True, right_index=True, how="left"))


def extract_significant_terms_from_subset(
        data_frame: pd.DataFrame,
        subset_data_frame: pd.DataFrame,
        field_name: str,
        vectorizer: CountVectorizer = CountVectorizer(encoding="utf-8",
                                                      lowercase=True,
                                                      max_features=500)
) -> pd.Series:
    """
    Return interesting or unusual occurrences of terms in a subset.

    Based on the elasticsearch significant_text aggregation:
    https://www.elastic.co/guide/en/elasticsearch/reference/current/search-aggregations-bucket-significantterms-aggregation.html#_scripted

    :param data_frame: The full dataset.
    :param subset_data_frame: The subset partition data over which the scoring will be calculated.
                              Can be filtered by feature or other boolean criteria.
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
