# Strings
Contains functions to help with string operations.
## append_tags_to_frame
Extracts tags from a given field and append them as dataframe.

Input:
* x_train - Pandas' dataframe with the train features.
* x_test - Pandas' dataframe with the test features.
* field_name - the feature to parse.
* prefix - the given prefix for new tag feature.
* min_df - ignore terms that have a document frequency strictly lower than the given threshold.
* binary - If True, all non zero counts are set to 1.

Example:
```python
from ds-utils.strings import append_tags_to_frame

X_train_with_tags, X_test_with_tags = append_tags_to_frame(X_train, X_test, "article_tags", "tag_", 10, True)
```