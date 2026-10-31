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
    X_train: pandas.DataFrame,
    X_test: pandas.DataFrame,
    field_name: str,
    prefix: str = "",
    max_features: Optional[int] = 500,
    min_df: Union[int, float] = 1,
    lowercase: bool = False,
    sparse: bool = False,
    tokenizer: Optional[Callable[[str], List[str]]] = _tokenize,
) -> Tuple[pandas.DataFrame, pandas.DataFrame]
```

### Parameters
- `X_train` (*pandas.DataFrame*): The training DataFrame used to build the tag vocabulary.
- `X_test` (*pandas.DataFrame*): The test DataFrame transformed using the vocabulary derived from `X_train`.
- `field_name` (*str*): The name of the column containing comma-delimited strings of tags.
- `prefix` (*str*, default `""`): Prefix added to all generated binary tag column names.
- `max_features` (*Optional[int]*, default `500`): Maximum number of top tags to encode. If `None`, all unique tags are encoded.
- `min_df` (*Union[int, float]*, default `1`): Minimum document frequency for tags to be included.
- `lowercase` (*bool*, default `False`): Whether to lowercase tag values prior to tokenization.
- `sparse` (*bool*, default `False`): Whether to return sparse DataFrame structures.
- `tokenizer` (*Optional[Callable]*, default `ds_utils.strings._tokenize`): Tokenizer function. The default `_tokenize` splits strings by comma (`","`), strips punctuation/special characters except alphanumeric, underscores, hyphens, and dollar signs (`[^a-zA-Z0-9_$-]`), strips whitespace, and discards empty tokens.

### Returns
- `Tuple[pandas.DataFrame, pandas.DataFrame]`: `(X_train_encoded, X_test_encoded)` with original columns plus new binary tag indicator columns.

---

## extract_significant_terms_from_subset

Identifies terms that are statistically over-represented in a specific subset of documents relative to the full corpus using Elasticsearch's significant_text heuristic: `subset_freq / (superset_freq - subset_freq + 1)`.

### Signature
```python
extract_significant_terms_from_subset(
    data_frame: pandas.DataFrame,
    subset_data_frame: pandas.DataFrame,
    field_name: str,
    vectorizer: sklearn.feature_extraction.text.CountVectorizer = CountVectorizer(max_features=500),
) -> pandas.Series
```

### Parameters
- `data_frame` (*pandas.DataFrame*): The full corpus representing all documents.
- `subset_data_frame` (*pandas.DataFrame*): The subset of interest (e.g. churned users, escalated support tickets).
- `field_name` (*str*): The column name containing freeform text.
- `vectorizer` (*CountVectorizer*, default `CountVectorizer(max_features=500)`): Scikit-learn CountVectorizer instance used to tokenize and extract candidate terms.

### Returns
- `pandas.Series`: Index contains unique terms; values are statistical significance scores (not bounded by 1.0; higher indicates greater distinctive over-representation in the subset). Access top terms using `.sort_values(ascending=False).head(k)`.
