import re
from typing import List, Tuple, Optional, Callable, Union

import pandas
from sklearn.feature_extraction.text import CountVectorizer


def _tokenize(text_tags: str) -> List[str]:
    tags = text_tags.split(",")
    tags = [re.sub(r"[^a-zA-Z0-9_$-]", "", x) for x in tags]
    tags = [x.strip() for x in tags]
    tags = [x for x in tags if len(x) > 0]
    return tags


def append_tags_to_frame(X_train: pandas.DataFrame, X_test: pandas.DataFrame, field_name: str,
                         prefix: Optional[str] = "", max_features: Optional[int] = 500, min_df: Union[int, float] = 1,
                         binary: bool = True, lowercase=False,
                         tokenizer: Optional[Callable[[str], List[str]]] = _tokenize) -> Tuple[
    pandas.DataFrame, pandas.DataFrame]:
    """
    Extracts tags from a given field and append them as dataframe.

    :param X_train: Pandas' dataframe with the train features.
    :param X_test: Pandas' dataframe with the test features.
    :param field_name: the feature to parse.
    :param prefix: the given prefix for new tag feature.
    :param max_features: int or None, default=500.
           If not None, build a vocabulary that only consider the top max_features ordered by term frequency across
           the corpus.
           This parameter is ignored if vocabulary is not None.
    :param min_df: float in range [0.0, 1.0] or int, default=1.
           When building the vocabulary ignore terms that have a document frequency strictly higher than the given
           threshold (corpus-specific stop words). If float, the parameter represents a proportion of documents,
           integer absolute counts.
    :param binary: boolean, default=True.
           If True, all non zero counts are set to 1. This is useful for discrete probabilistic models that model
           binary events rather than integer counts.
    :param lowercase: boolean, default=False.
           Convert all characters to lowercase before tokenizing.
    :param tokenizer: callable or None.
           Override the string tokenization step while preserving the preprocessing and n-grams generation steps.
           Default retain alphanumeric characters with special characters "_", "$" and "-".
    :return: the train and test with tags appended.
    """
    vectorizer = CountVectorizer(binary=binary, tokenizer=tokenizer, encoding="latin1", lowercase=lowercase,
                                 min_df=min_df, max_features=max_features)
    x_train_count_matrix = vectorizer.fit_transform(X_train[field_name].dropna())
    x_train_tags = pandas.DataFrame(x_train_count_matrix.toarray(),
                                    columns=[prefix + tag_name for tag_name in vectorizer.get_feature_names()])
    x_train_tags.index = X_train.index

    x_test_count_matrix = vectorizer.transform(X_test[field_name].dropna())
    x_test_tags = pandas.DataFrame(x_test_count_matrix.toarray(),
                                   columns=[prefix + tag_name for tag_name in vectorizer.get_feature_names()])
    x_test_tags.index = X_test.index

    x_train_reduced = X_train.drop(columns=[field_name])
    x_test_reduced = X_test.drop(columns=[field_name])

    return pandas.merge(x_train_reduced, x_train_tags, left_index=True, right_index=True, how="left"), pandas.merge(
        x_test_reduced, x_test_tags, left_index=True, right_index=True, how="left")


def significant_terms(data_frame: pandas.DataFrame, subset_data_frame: pandas.DataFrame, field_name: str,
                      max_features: int = 500, min_df: Union[int, float] = 1, lowercase=False,
                      tokenizer: Optional[Callable[[str], List[str]]] = None,
                      stop_words: Optional[Union[str, List[str]]] = None) -> pandas.Series:
    """
    Returns interesting or unusual occurrences of terms in a subset.

    Based on the `elasticsearch significant_text aggregation
    <https://www.elastic.co/guide/en/elasticsearch/reference/current/search-aggregations-bucket-significantterms-aggregation.html#_scripted>`_

    :param data_frame: the full data set.
    :param subset_data_frame: the subset partition data, with over it the scoring will be calculated. Can a filter by
           feature or other boolean criteria.
    :param field_name: the feature to parse.
    :param max_features: int or None, default=500.
           If not None, build a vocabulary that only consider the top max_features ordered by term frequency across
           the corpus.
    :param min_df: float in range [0.0, 1.0] or int, default=1.
           When building the vocabulary ignore terms that have a document frequency strictly higher than the given
           threshold (corpus-specific stop words). If float, the parameter represents a proportion of documents,
           integer absolute counts. This parameter is ignored if vocabulary is not None.
    :param lowercase: boolean, default=False.
           Convert all characters to lowercase before tokenizing.
    :param tokenizer: callable or None.
           Override the string tokenization step while preserving the preprocessing and n-grams generation steps.
    :param stop_words: string {‘english’}, list, or None (default).
           If ‘english’, a built-in stop word list for English is used. There are several known issues with ‘english’
           and you should consider an alternative (see `Using stop words <https://scikit-learn.org/stable/modules/feature_extraction.html#stop-words>`_).
           If a list, that list is assumed to contain stop words, all of which will be removed from the resulting
           tokens.
           If None, no stop words will be used.
    :return: Series of terms with scoring over the subset.

    :author: `Eran Hirsch <https://github.com/eranhirs>`_
    """

    vectorizer = CountVectorizer(tokenizer=tokenizer, encoding="latin1", lowercase=lowercase, min_df=min_df,
                                 max_features=max_features, stop_words=stop_words)
    count_matrix = vectorizer.fit_transform(data_frame[field_name].dropna())
    matrix_df = pandas.DataFrame(count_matrix.toarray(), columns=vectorizer.get_feature_names())

    subset_X = vectorizer.transform(subset_data_frame[field_name].dropna())
    subset_matrix_df = pandas.DataFrame(subset_X.toarray(), columns=vectorizer.get_feature_names())

    subset_freq = subset_matrix_df.sum()
    superset_freq = matrix_df.sum()

    return (subset_freq / (superset_freq - subset_freq + 1)).sort_values(ascending=False)
