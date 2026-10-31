# Strings API Reference

Exhaustive parameter lists, function signatures, return specifications, and common mistakes for `ds_utils.strings`.

## Import Paths

```python
from ds_utils.strings import (
    append_tags_to_frame,
    extract_significant_terms_from_subset,
)
```

---

## append_tags_to_frame

Extracts comma-separated tags from a column and appends one-hot encoded binary indicator columns to training and testing DataFrames.

### Signature
```python
append_tags_to_frame(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    field_name: str,
    prefix: str = "",
    max_features: Optional[int] = 500,
    min_df: Union[int, float] = 1,
    lowercase: bool = False,
    sparse: bool = False,
    tokenizer: Optional[Callable[[str], List[str]]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]
```

### Parameters
- `X_train` (*pd.DataFrame*): The training DataFrame used to build the tag vocabulary.
- `X_test` (*pd.DataFrame*): The test DataFrame transformed using the vocabulary derived from `X_train`.
- `field_name` (*str*): The name of the column containing comma-delimited strings of tags.
- `prefix` (*str*, default `""`): Prefix added to all generated binary tag column names.
- `max_features` (*int*, default `500`): Maximum number of top tags to encode.
- `min_df` (*int* or *float*, default `1`): Minimum document frequency for tags.
- `lowercase` (*bool*, default `False`): Whether to lowercase tag values.
- `sparse` (*bool*, default `False`): Whether to return sparse DataFrame structures.

### Returns
- `Tuple[pd.DataFrame, pd.DataFrame]`: `(x_train_encoded, x_test_encoded)` with original columns plus new binary tag indicator columns.

### Split-Safety Guarantees
- Vocabulary extraction is strictly computed from `x_train`.
- Test-only tags present in `x_test` but absent in `x_train` are silently dropped to ensure test rows conform strictly to the feature space established on train.
- Column ordering is identical across both resulting DataFrames.

### Common Mistakes
- Calling `append_tags_to_frame` separately on train and test: Calling on train and test separately results in divergent column schemas if tag distributions differ. Always pass `(x_train, x_test)` simultaneously.
- Concatenating train and test beforehand: Never concatenate splits "to capture all tags"; that leaks test vocabulary into model training.

---

## extract_significant_terms_from_subset

Identifies terms that are statistically over-represented in a specific subset of documents relative to the full corpus using Elasticsearch's significant_text algorithm.

### Signature
```python
extract_significant_terms_from_subset(
    data_frame: pd.DataFrame,
    subset_data_frame: pd.DataFrame,
    field_name: str,
    vectorizer: CountVectorizer = CountVectorizer(max_features=500),
) -> pd.Series
```

### Parameters
- `data_frame` (*pd.DataFrame*): The full corpus representing all documents.
- `subset_data_frame` (*pd.DataFrame*): The subset of interest (e.g. churned users, escalated support tickets).
- `field_name` (*str*): The column name containing freeform text.
- `vectorizer` (*CountVectorizer*, default `CountVectorizer(max_features=500)`): Scikit-learn CountVectorizer instance.

### Returns
- `pd.Series`: Index contains unique terms; values are statistical significance scores in `[0.0, 1.0]`. Access the top distinctive terms using `.sort_values(ascending=False).head(k)`.

### Output Example
| term (index) | score (value) |
|---|---|
| refund | 0.88 |
| crash | 0.74 |
| cancellation | 0.62 |
