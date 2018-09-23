import pandas

from ds_utils.strings import append_tags_to_frame


def test_append_tags_to_frame():
    x_train = pandas.DataFrame([{"article_name": "1", "article_tags": "ds,ml,dl"},
                                {"article_name": "2", "article_tags": "ds,ml"}])
    x_test = pandas.DataFrame([{"article_name": "3", "article_tags": "ds,ml,py"}])

    x_train_expected = pandas.DataFrame([{"article_name": "1", "tag_ds": 1, "tag_ml": 1, "tag_dl": 1},
                                         {"article_name": "2", "tag_ds": 1, "tag_ml": 1, "tag_dl": 0}])
    x_test_expected = pandas.DataFrame([{"article_name": "3", "tag_ds": 1, "tag_ml": 1, "tag_dl": 0}])

    x_train_with_tags, x_test_with_tags = append_tags_to_frame(x_train, x_test, "article_tags", "tag_", 1, True)
    pandas.testing.assert_frame_equal(x_train_expected, x_train_with_tags)
    pandas.testing.assert_frame_equal(x_test_expected, x_test_with_tags)
