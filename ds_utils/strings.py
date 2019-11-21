import re
from typing import List, Tuple, Optional

import pandas
from sklearn.feature_extraction.text import CountVectorizer


def append_tags_to_frame(x_train: pandas.DataFrame, x_test: pandas.DataFrame, field_name: str,
                         prefix: Optional[str] = "", min_df: float = 1.0, binary: bool = True) -> Tuple[
    pandas.DataFrame, pandas.DataFrame]:
    """
    Extracts tags from a given field and append them as dataframe.

    :param x_train: Pandas' dataframe with the train features.
    :param x_test: Pandas' dataframe with the test features.
    :param field_name: the feature to parse.
    :param prefix: the given prefix for new tag feature.
    :param min_df: ignore terms that have a document frequency strictly lower than the given threshold.
    :param binary: If True, all non zero counts are set to 1.
    :return: the train and test with tags appended.
    """
    vectorizer = CountVectorizer(binary=binary, tokenizer=_tokenize, encoding="latin1", lowercase=False,
                                 min_df=min_df)
    x_train_count_matrix = vectorizer.fit_transform(x_train[field_name])
    x_train_tags = pandas.DataFrame(x_train_count_matrix.A,
                                    columns=[prefix + tag_name for tag_name in vectorizer.get_feature_names()])
    x_train_tags.index = x_train.index

    x_test_count_matrix = vectorizer.transform(x_test[field_name])
    x_test_tags = pandas.DataFrame(x_test_count_matrix.A,
                                   columns=[prefix + tag_name for tag_name in vectorizer.get_feature_names()])
    x_test_tags.index = x_test.index

    x_train_reduced = x_train.drop(columns=[field_name])
    x_test_reduced = x_test.drop(columns=[field_name])

    return pandas.merge(x_train_reduced, x_train_tags, left_index=True, right_index=True, how="left"), pandas.merge(
        x_test_reduced, x_test_tags, left_index=True, right_index=True, how="left")


def _tokenize(text_tags: str) -> List[str]:
    tags = text_tags.split(",")
    tags = [re.sub(r"[^a-zA-Z0-9_$-]", "", x) for x in tags]
    tags = [x.strip() for x in tags]
    tags = [x for x in tags if len(x) > 0]
    return tags
