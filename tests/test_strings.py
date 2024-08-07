import pandas as pd

from ds_utils.strings import append_tags_to_frame, extract_significant_terms_from_subset


def test_append_tags_to_frame():
    x_train = pd.DataFrame([{"article_name": "1", "article_tags": "ds,ml,dl"},
                            {"article_name": "2", "article_tags": "ds,ml"}])
    x_test = pd.DataFrame([{"article_name": "3", "article_tags": "ds,ml,py"}])

    x_train_expected = pd.DataFrame([{"article_name": "1", "tag_ds": 1, "tag_ml": 1, "tag_dl": 1},
                                     {"article_name": "2", "tag_ds": 1, "tag_ml": 1, "tag_dl": 0}],
                                    columns=["article_name", "tag_dl", "tag_ds", "tag_ml"])
    x_test_expected = pd.DataFrame([{"article_name": "3", "tag_ds": 1, "tag_ml": 1, "tag_dl": 0}],
                                   columns=["article_name", "tag_dl", "tag_ds", "tag_ml"])

    x_train_with_tags, x_test_with_tags = append_tags_to_frame(x_train, x_test, "article_tags", "tag_")
    pd.testing.assert_frame_equal(x_train_expected, x_train_with_tags, check_like=True)
    pd.testing.assert_frame_equal(x_test_expected, x_test_with_tags, check_like=True)


def test_significant_terms():
    corpus = ['This is the first document.', 'This document is the second document.', 'And this is the third one.',
              'Is this the first document?']
    data_frame = pd.DataFrame(corpus, columns=["content"])
    subset_data_frame = data_frame[data_frame.index > 1]
    terms = extract_significant_terms_from_subset(data_frame, subset_data_frame, "content")

    expected = pd.Series(
        [1.0, 1.0, 1.0, 0.6666666666666666, 0.6666666666666666, 0.6666666666666666, 0.5, 0.25, 0.0],
        index=['and', 'one', 'third', 'is', 'the', 'this', 'first', 'document', 'second'])

    pd.testing.assert_series_equal(expected, terms)
