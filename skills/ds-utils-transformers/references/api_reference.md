# Transformers API Reference

Exhaustive parameter lists, constructor signatures, return specifications, and common mistakes for `ds_utils.transformers`.

## Import Paths

```python
from ds_utils.transformers.multi_label_binarizer import MultiLabelBinarizerTransformer
from ds_utils.transformers.sentence_embedding import SentenceEmbeddingTransformer
```

---

## MultiLabelBinarizerTransformer

Wraps `sklearn.preprocessing.MultiLabelBinarizer` so that it integrates cleanly inside `sklearn.pipeline.Pipeline`, `sklearn.compose.ColumnTransformer`, and supports `set_output(transform="pandas")`.

### Constructor Signature
```python
MultiLabelBinarizerTransformer(
    classes: Optional[Sequence[Any]] = None,
    sparse_output: bool = False,
)
```

### Parameters
- `classes` (*sequence*, optional): Explicit sequence of expected class labels. When provided, fixes the column order and ignores unseen labels during `transform`.
- `sparse_output` (*bool*, default `False`): If `True`, the inner binarizer can use sparse representation; the transformer returns a dense `float64` array.

### Input Data Format
`X` must be an iterable of iterables where each inner iterable represents the labels for a single sample.
- Valid: `[["action", "sci-fi"], ["comedy"]]` or a Pandas Series where each element is a list of strings: `pd.Series([["tag1", "tag2"], ["tag3"]])`.
- Invalid: A flat list of strings `["action", "comedy"]`. Scikit-learn treats each character of the string as a separate label.
- Multi-column DataFrame: If passing a DataFrame, wrap in a `ColumnTransformer` selecting exactly one multi-label column.

### Returns from `transform`
- `np.ndarray`: 2D array of shape `(n_samples, n_classes)` with dtype `float64`.
- If `set_output(transform="pandas")` is enabled on the transformer or pipeline, returns a `pd.DataFrame`.

### Feature Names
`get_feature_names_out(input_features=None)` returns `{prefix}_{sanitized_label}`. If `input_features` is provided (e.g., from a `ColumnTransformer`), its first element is used as the prefix; otherwise defaults to `label`. Column names are sanitized for safe storage in data warehouses and Delta tables.

---

## SentenceEmbeddingTransformer

Wraps `sentence-transformers` models for end-to-end integration into scikit-learn pipelines.

### Prerequisites
Requires the optional NLP dependency group:
```bash
pip install data-science-utils[nlp]
```

### Constructor Signature
```python
SentenceEmbeddingTransformer(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 32,
    show_progress_bar: bool = False,
    normalize_embeddings: bool = False,
    device: Optional[str] = None,
    precision: str = "float32",
    truncate_dim: Optional[int] = None,
    prompt_name: Optional[str] = None,
    prompt: Optional[str] = None,
)
```

### Parameters
- `model_name` (*str*, default `"sentence-transformers/all-MiniLM-L6-v2"`): HuggingFace model repo ID or local directory path.
- `batch_size` (*int*, default `32`): Batch size for inference.
- `show_progress_bar` (*bool*, default `False`): Whether to display a tqdm progress bar during encoding.
- `normalize_embeddings` (*bool*, default `False`): If `True`, L2-normalizes embedding vectors to unit length.
- `device` (*str*, optional): Target compute device (`"cpu"`, `"cuda"`, `"mps"`). `None` auto-detects.
- `precision` (*str*, default `"float32"`): Output floating-point precision.
- `truncate_dim` (*int*, optional): Dimension truncation for Matryoshka models.
- `prompt_name` (*str*, optional): Prompt key defined in the model configuration.
- `prompt` (*str*, optional): Custom prompt string prepended to all input texts.

### Returns from `transform`
- `np.ndarray`: 2D array of shape `(n_samples, embedding_dim)` (e.g. `(n, 384)`).
- Supports `set_output(transform="pandas")` with feature names `["dim_0", "dim_1", ...]`.

### Behavior & Optimizations
- **Lazy Loading**: The model weights are downloaded/loaded into memory on the first call to `.fit()` or `.transform()`.
- **Missing Value Handling**: Any `None` or `NaN` entries in the input text array are automatically replaced with empty strings `""` before encoding to prevent exceptions.
