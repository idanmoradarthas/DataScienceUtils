"""Tests for string manipulation utility functions."""
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_array
from sklearn.feature_extraction.text import CountVectorizer

from ds_utils.strings import append_tags_to_frame, extract_significant_terms_from_subset


@pytest.mark.parametrize(
    ("x_train", "x_test", "expected_train", "expected_test"),
    [
        (
            pd.DataFrame(
                [
                    {"article_name": "1", "article_tags": "ds,ml,dl"},
                    {"article_name": "2", "article_tags": "ds,ml"},
                ]
            ),
            pd.DataFrame([{"article_name": "3", "article_tags": "ds,ml,py"}]),
            pd.DataFrame(
                [
                    {"article_name": "1", "tag_ds": 1, "tag_ml": 1, "tag_dl": 1},
                    {"article_name": "2", "tag_ds": 1, "tag_ml": 1, "tag_dl": 0},
                ],
                columns=["article_name", "tag_dl", "tag_ds", "tag_ml"],
            ),
            pd.DataFrame(
                [{"article_name": "3", "tag_ds": 1, "tag_ml": 1, "tag_dl": 0}],
                columns=["article_name", "tag_dl", "tag_ds", "tag_ml"],
            ),
        ),
        (
            pd.DataFrame(
                [
                    {"article_name": "1", "article_tags": ""},
                    {"article_name": "2", "article_tags": "python"},
                ]
            ),
            pd.DataFrame([{"article_name": "3", "article_tags": "java,python"}]),
            pd.DataFrame(
                [
                    {"article_name": "1", "tag_python": 0},
                    {"article_name": "2", "tag_python": 1},
                ],
                columns=["article_name", "tag_python"],
            ),
            pd.DataFrame(
                [{"article_name": "3", "tag_python": 1}],
                columns=["article_name", "tag_python"],
            ),
        ),
    ],
    ids=["small tags", "regular tags"],
)
def test_append_tags_to_frame(x_train, x_test, expected_train, expected_test):
    """Test appending tags to DataFrame for various scenarios."""
    x_train_with_tags, x_test_with_tags = append_tags_to_frame(
        x_train, x_test, "article_tags", "tag_"
    )
    pd.testing.assert_frame_equal(expected_train, x_train_with_tags, check_like=True)
    pd.testing.assert_frame_equal(expected_test, x_test_with_tags, check_like=True)


@pytest.mark.parametrize(
    ("x_train", "x_test"),
    [
        (
            pd.DataFrame(columns=["article_name", "article_tags"]),
            pd.DataFrame([{"article_name": "3", "article_tags": "ds,ml,py"}]),
        ),
        (
            pd.DataFrame(
                [
                    {"article_name": "1", "article_tags": "ds,ml,dl"},
                    {"article_name": "2", "article_tags": "ds,ml"},
                ]
            ),
            pd.DataFrame(columns=["article_name", "article_tags"]),
        ),
    ],
    ids=["empty_train", "empty_test"],
)
def test_append_tags_to_frame_empty_dataframes(request, x_train, x_test):
    """Test appending tags when one of the input DataFrames is empty."""
    x_train_with_tags, x_test_with_tags = append_tags_to_frame(
        x_train, x_test, "article_tags", "tag_"
    )
    if request.node.callspec.id == "empty_train":
        assert x_train_with_tags.empty
    assert x_test_with_tags.empty


def test_append_tags_to_frame_missing_column():
    """Test append_tags_to_frame raises KeyError if specified field_name is missing."""
    x_train = pd.DataFrame([{"article_name": "1"}])
    x_test = pd.DataFrame([{"article_name": "2"}])
    with pytest.raises(KeyError):
        append_tags_to_frame(x_train, x_test, "article_tags", "tag_")


def test_append_tags_to_frame_custom_tokenizer():
    """Test append_tags_to_frame with a custom tokenizer."""

    def custom_tokenizer(text):
        return text.split("|")

    x_train = pd.DataFrame([{"article_name": "1", "article_tags": "ds|ml|dl"}])
    x_test = pd.DataFrame([{"article_name": "2", "article_tags": "py|ml"}])
    x_train_with_tags, x_test_with_tags = append_tags_to_frame(
        x_train, x_test, "article_tags", "tag_", tokenizer=custom_tokenizer
    )

    expected_train = pd.DataFrame(
        [{"article_name": "1", "tag_ds": 1, "tag_ml": 1, "tag_dl": 1}]
    )
    expected_test = pd.DataFrame(
        [{"article_name": "2", "tag_ds": 0, "tag_ml": 1, "tag_dl": 0}]
    )

    pd.testing.assert_frame_equal(expected_train, x_train_with_tags, check_like=True)
    pd.testing.assert_frame_equal(expected_test, x_test_with_tags, check_like=True)


def test_append_tags_to_frame_with_nan_values():
    """Test append_tags_to_frame when 'article_tags' column contains NaN values."""
    x_train = pd.DataFrame(
        [
            {"article_name": "1", "article_tags": "ds,ml"},
            {"article_name": "2", "article_tags": np.nan},
        ]
    )
    x_test = pd.DataFrame([{"article_name": "3", "article_tags": "py"}])

    x_train_with_tags, x_test_with_tags = append_tags_to_frame(
        x_train, x_test, "article_tags", "tag_"
    )

    expected_train = pd.DataFrame(
        [
            {"article_name": "1", "tag_ds": 1, "tag_ml": 1},
            {"article_name": "2", "tag_ds": 0, "tag_ml": 0},
        ]
    )
    expected_test = pd.DataFrame([{"article_name": "3", "tag_ds": 0, "tag_ml": 0}])

    pd.testing.assert_frame_equal(expected_train, x_train_with_tags, check_like=True)
    pd.testing.assert_frame_equal(expected_test, x_test_with_tags, check_like=True)


@pytest.mark.parametrize(
    ("corpus", "subset_indices", "expected"),
    [
        (
            [
                "This is the first document.",
                "This document is the second document.",
                "And this is the third one.",
                "Is this the first document?",
            ],
            [2, 3],
            pd.Series(
                [
                    1.0,
                    1.0,
                    1.0,
                    0.6666666666666666,
                    0.6666666666666666,
                    0.6666666666666666,
                    0.5,
                    0.25,
                    0.0,
                ],
                index=[
                    "and",
                    "one",
                    "third",
                    "is",
                    "the",
                    "this",
                    "first",
                    "document",
                    "second",
                ],
            ),
        ),
        (
            [
                "Python is great",
                "Java is also good",
                "Python and Java are programming languages",
            ],
            [2],
            pd.Series(
                [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0],
                index=[
                    "and",
                    "are",
                    "languages",
                    "programming",
                    "java",
                    "python",
                    "also",
                    "good",
                    "great",
                    "is",
                ],
            ),
        ),
    ],
    ids=["simple document", "natural documents"],
)
def test_significant_terms(corpus, subset_indices, expected):
    """Test extraction of significant terms from a subset of documents."""
    data_frame = pd.DataFrame(corpus, columns=["content"])
    subset_data_frame = data_frame.iloc[subset_indices]
    terms = extract_significant_terms_from_subset(
        data_frame, subset_data_frame, "content"
    )
    pd.testing.assert_series_equal(expected, terms, check_like=True)


def test_significant_terms_empty_dataframe():
    """Test significant terms extraction with empty DataFrames."""
    data_frame = pd.DataFrame(columns=["content"])
    subset_data_frame = pd.DataFrame(columns=["content"])
    terms = extract_significant_terms_from_subset(
        data_frame, subset_data_frame, "content"
    )
    assert terms.empty


def test_significant_terms_custom_vectorizer(mocker):
    """Test significant terms extraction with a custom CountVectorizer."""
    corpus = ["This is the first document.", "This document is the second document."]
    data_frame = pd.DataFrame(corpus, columns=["content"])
    subset_data_frame = data_frame.iloc[1:]

    custom_vectorizer = mocker.Mock(spec=CountVectorizer)
    fit_transform_result = mocker.Mock(spec=csr_array)
    fit_transform_result.toarray.return_value = np.array([[1, 0], [2, 1]])
    custom_vectorizer.fit_transform.return_value = fit_transform_result
    custom_vectorizer.get_feature_names_out.return_value = np.array(
        ["document", "second"]
    )
    transform_result = mocker.Mock(spec=csr_array)
    transform_result.toarray.return_value = np.array([[2, 1]])
    custom_vectorizer.transform.return_value = transform_result

    terms = extract_significant_terms_from_subset(
        data_frame, subset_data_frame, "content", vectorizer=custom_vectorizer
    )

    expected = pd.Series([1.0, 1.0], index=["document", "second"])
    pd.testing.assert_series_equal(expected, terms, check_like=True)
