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

### Signature
```python
MultiLabelBinarizerTransformer(
    *,
    classes: Union[None, np.ndarray, Sequence[Any]] = None,
    sparse_output: bool = False,
) -> None
```

### Parameters
- `classes` (*keyword-only*, *optional*, default `None`): Explicit sequence of expected class labels. When provided, fixes the column order and ignores unseen labels during `transform`.
- `sparse_output` (*keyword-only*, *bool*, default `False`): If `True`, the inner binarizer can use sparse representation; the transformer returns a dense `float64` array.

---

## SentenceEmbeddingTransformer

Wraps `sentence-transformers` models for end-to-end integration into scikit-learn pipelines.

### Prerequisites
Requires the optional NLP dependency group:
```bash
pip install data-science-utils[nlp]
```

### Signature
```python
SentenceEmbeddingTransformer(
    *,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 32,
    show_progress_bar: bool = False,
    normalize_embeddings: bool = False,
    device: Optional[str] = None,
    precision: str = "float32",
    truncate_dim: Optional[int] = None,
    prompt_name: Optional[str] = None,
    prompt: Optional[str] = None,
) -> None
```

### Parameters
- `model_name` (*keyword-only*, *str*, default `"sentence-transformers/all-MiniLM-L6-v2"`): HuggingFace model repo ID or local directory path.
- `batch_size` (*keyword-only*, *int*, default `32`): Batch size for inference.
- `show_progress_bar` (*keyword-only*, *bool*, default `False`): Whether to display a progress bar during encoding.
- `normalize_embeddings` (*keyword-only*, *bool*, default `False`): If `True`, L2-normalizes embedding vectors to unit length.
- `device` (*keyword-only*, *Optional[str]*, default `None`): Target compute device (`"cpu"`, `"cuda"`, `"mps"`).
- `precision` (*keyword-only*, *str*, default `"float32"`): Output floating-point precision.
- `truncate_dim` (*keyword-only*, *Optional[int]*, default `None`): Dimension truncation for Matryoshka models.
- `prompt_name` (*keyword-only*, *Optional[str]*, default `None`): Prompt key defined in the model configuration.
- `prompt` (*keyword-only*, *Optional[str]*, default `None`): Custom prompt string prepended to all input texts.
