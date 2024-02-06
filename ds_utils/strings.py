import re
from typing import List, Tuple, Optional, Callable, Union

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


def _tokenize(text_tags: str) -> List[str]:
    tags = text_tags.split(",")
    tags = [re.sub(r"[^a-zA-Z0-9_$-]", "", x) for x in tags]
    tags = [x.strip() for x in tags]
    tags = [x for x in tags if len(x) > 0]
    return tags


def append_tags_to_frame(X_train: pd.DataFrame, X_test: pd.DataFrame, field_name: str,
                         prefix: Optional[str] = "", max_features: Optional[int] = 500, min_df: Union[int, float] = 1,
                         lowercase=False, tokenizer: Optional[Callable[[str], List[str]]] = _tokenize) -> Tuple[
    pd.DataFrame, pd.DataFrame]:
    """
    Extracts tags from a given field and append them as dataframe.

    :param X_train: Pandas' dataframe with the train features.
    :param X_test: Pandas' dataframe with the test features.
    :param field_name: the feature to parse.
    :param prefix: the given prefix for new tag feature.
    :param max_features: int or None, default=500.
           max tags names to consider.
    :param min_df: float in range [0.0, 1.0] or int, default=1.
           When building the tag name set ignore tags that have a document frequency strictly higher than the given
           threshold (corpus-specific stop words). If a float, the parameter represents a proportion of documents,
           integer absolute counts.
    :param lowercase: boolean, default=False.
           Convert all characters to lowercase before tokenizing the tag names.
    :param tokenizer: callable or None.
           Override the string tokenization step while preserving the preprocessing and n-grams generation steps.
           Default splits by ",", and retain alphanumeric characters with special characters "_", "$" and "-".
    :return: the train and test with tags appended.
    """
    vectorized = CountVectorizer(binary=True, tokenizer=tokenizer, encoding="latin1", lowercase=lowercase,
                                 min_df=min_df, max_features=max_features)
    x_train_count_matrix = vectorized.fit_transform(X_train[field_name].dropna())
    x_train_tags = pd.DataFrame(x_train_count_matrix.toarray(),
                                columns=[prefix + tag_name for tag_name in vectorized.get_feature_names_out()])
    x_train_tags.index = X_train.index

    x_test_count_matrix = vectorized.transform(X_test[field_name].dropna())
    x_test_tags = pd.DataFrame(x_test_count_matrix.toarray(),
                               columns=[prefix + tag_name for tag_name in vectorized.get_feature_names_out()])
    x_test_tags.index = X_test.index

    x_train_reduced = X_train.drop(columns=[field_name])
    x_test_reduced = X_test.drop(columns=[field_name])

    return pd.merge(x_train_reduced, x_train_tags, left_index=True, right_index=True, how="left"), pd.merge(
        x_test_reduced, x_test_tags, left_index=True, right_index=True, how="left")


def extract_significant_terms_from_subset(data_frame: pd.DataFrame, subset_data_frame: pd.DataFrame,
                                          field_name: str,
                                          vectorized: CountVectorizer = CountVectorizer(encoding="latin1",
                                                                                        lowercase=True,
                                                                                        max_features=500)) -> pd.Series:
    """
    Returns interesting or unusual occurrences of terms in a subset.

    Based on the `elasticsearch significant_text aggregation
    <https://www.elastic.co/guide/en/elasticsearch/reference/current/search-aggregations-bucket-significantterms-aggregation.html#_scripted>`_

    :param data_frame: the full data set.
    :param subset_data_frame: the subset partition data, with over it the scoring will be calculated. Can a filter by
           feature or other boolean criteria.
    :param field_name: the feature to parse.
    :param vectorized: text count vectorizer which converts collection of text to a matrix of token counts. See more
                       info `here <https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html>`_ .
    :return: Series of terms with scoring over the subset.

    :author: `Eran Hirsch <https://github.com/eranhirs>`_
    """
    count_matrix = vectorized.fit_transform(data_frame[field_name].dropna())
    matrix_df = pd.DataFrame(count_matrix.toarray(), columns=vectorized.get_feature_names_out())

    subset_x = vectorized.transform(subset_data_frame[field_name].dropna())
    subset_matrix_df = pd.DataFrame(subset_x.toarray(), columns=vectorized.get_feature_names_out())

    subset_freq = subset_matrix_df.sum()
    superset_freq = matrix_df.sum()

    return (subset_freq / (superset_freq - subset_freq + 1)).sort_values(ascending=False)
