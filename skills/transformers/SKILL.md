---
name: ds-utils-transformers
description: >
  Sklearn pipeline transformers for multi-label binarization and sentence embeddings. Use when the user needs MultiLabelBinarizer in a Pipeline, ColumnTransformer, or set_output(transform="pandas"), or when integrating sentence-transformers models into sklearn pipelines for NLP tasks.
license: MIT
metadata:
  author: Idan Morad
  documentation: https://datascienceutils.readthedocs.io/en/stable/
  package: data-science-utils
  repository: https://github.com/idanmoradarthas/DataScienceUtils
---
# Transformers — ds_utils.transformers

Sklearn-compatible wrappers for preprocessing steps that need the feature-names API and stable dtypes.

## Installation

```bash
pip install data-science-utils
# or
conda install -c idanmorad data-science-utils

# For NLP features (SentenceEmbeddingTransformer):
pip install data-science-utils[nlp]
```

## Import

```python
from ds_utils.transformers.multi_label_binarizer import MultiLabelBinarizerTransformer
from ds_utils.transformers.sentence_embedding import SentenceEmbeddingTransformer
```

---

## MultiLabelBinarizerTransformer

Wraps `sklearn.preprocessing.MultiLabelBinarizer` so pipelines get `get_feature_names_out` and dense `float64` matrices.

**1. Direct Object Usage:**

```python
from ds_utils.transformers.multi_label_binarizer import MultiLabelBinarizerTransformer

X = [["sci-fi", "action"], ["romance"], ["action", "comedy"]]
mlb = MultiLabelBinarizerTransformer()
X_t = mlb.fit_transform(X)
feature_names = mlb.get_feature_names_out()
```
**Output (`X_t`): will be a numpy array containing the binarized data, with `feature_names` corresponding to columns.**

| label_action | label_comedy | label_romance | label_sci-fi |
|--------------|--------------|---------------|--------------|
| 1.0          | 0.0          | 0.0           | 1.0          |
| 0.0          | 0.0          | 1.0           | 0.0          |
| 1.0          | 1.0          | 0.0           | 0.0          |

**2. Pipeline Usage with Pandas Output:**

```python
from ds_utils.transformers.multi_label_binarizer import MultiLabelBinarizerTransformer
from sklearn.pipeline import Pipeline

X = [["sci-fi", "action"], ["romance"], ["action", "comedy"]]
pipe = Pipeline([("mlb", MultiLabelBinarizerTransformer())])
pipe.set_output(transform="pandas")
df = pipe.fit_transform(X)
```

**Output (`df`): will be a pandas DataFrame containing the binarized data, with `feature_names` corresponding to columns.**

| label_action | label_comedy | label_romance | label_sci-fi |
|--------------|--------------|---------------|--------------|
| 1.0          | 0.0          | 0.0           | 1.0          |
| 0.0          | 0.0          | 1.0           | 0.0          |
| 1.0          | 1.0          | 0.0           | 0.0          |

**Parameters (constructor):**

- `classes` — optional fixed ordering of labels (passed through to `MultiLabelBinarizer`).
- `sparse_output` — if True, the inner binarizer may use sparse storage; the transformer still returns a dense `float64` array.

**Returns from `transform`:** `numpy.ndarray`, shape `(n_samples, n_classes)`, dtype `float64`.

**Feature names:** `get_feature_names_out(input_features=None)` returns `{prefix}_{sanitized_label}`; default prefix is `label` when `input_features` is omitted, otherwise the first validated input feature name is used as the prefix. Labels are sanitized for safe column names (e.g. Delta tables).

**Common mistakes:**

- Passing a **flat** list of genre strings, e.g. `['sci-fi', 'thriller', 'comedy']`, as the whole `X`. Scikit-learn then treats **each character** as a sample. You need **one iterable of labels per row**: e.g. `[['sci-fi', 'thriller', 'comedy']]` for a single sample, or `[['a', 'b'], ['c']]` for two samples. See the sklearn docs for `MultiLabelBinarizer`.
- Using a **DataFrame with more than one column** for `X`: this transformer expects a single multi-label column (or a 1D array-like of rows).

**Typical workflow**

```python
from ds_utils.transformers.multi_label_binarizer import MultiLabelBinarizerTransformer
from sklearn.compose import ColumnTransformer

# Single column "tags" in a DataFrame
import pandas as pd

df = pd.DataFrame({"tags": [["x", "y"], ["z"]], "num": [1.0, 2.0]})
pre = ColumnTransformer(
    [("mlb", MultiLabelBinarizerTransformer(), ["tags"])],
    remainder="passthrough",
)
X_out = pre.fit_transform(df)
```

---

## SentenceEmbeddingTransformer

Wraps [sentence-transformers](https://sbert.net/) models for use in sklearn pipelines. Produces dense embedding matrices from text inputs with lazy model loading, `None`/`NaN` handling, and `get_feature_names_out` support.

> **Requires:** `pip install data-science-utils[nlp]`

**1. Direct Usage:**

```python
from ds_utils.transformers.sentence_embedding import SentenceEmbeddingTransformer

texts = ["The quick brown fox", "jumps over the lazy dog", "Hello world"]
embed = SentenceEmbeddingTransformer()
embeddings = embed.fit_transform(texts)
feature_names = embed.get_feature_names_out()
```
**Output (`embeddings`): will be a numpy array of shape `(n_samples, embedding_dimension)` (e.g. `(3, 384)` for the default `sentence-transformers/all-MiniLM-L6-v2` model), with `feature_names` corresponding to columns.**

| dim_0    | dim_1    | dim_2    | ... | dim_383  |
|----------|----------|----------|-----|----------|
| -0.0123  |  0.0456  |  0.0789  | ... |  0.0012  |
|  0.0345  | -0.0678  |  0.0901  | ... | -0.0234  |
|  0.0567  |  0.0890  | -0.0123  | ... |  0.0456  |

**2. Pipeline Usage with Classifier:**

```python
from ds_utils.transformers.sentence_embedding import SentenceEmbeddingTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

pipeline = Pipeline([
    ('embeddings', SentenceEmbeddingTransformer(normalize_embeddings=True)),
    ('classifier', RandomForestClassifier()),
])
pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)
```

**3. ColumnTransformer Usage (mixing text and numerical features):**

```python
from ds_utils.transformers.sentence_embedding import SentenceEmbeddingTransformer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
import pandas as pd

df = pd.DataFrame({
    "description": ["a product", "another item"],
    "price": [9.99, 19.99],
})
pre = ColumnTransformer([
    ("text", SentenceEmbeddingTransformer(), ["description"]),
    ("num", StandardScaler(), ["price"]),
])
X_out = pre.fit_transform(df)
```

**Parameters (constructor):**

- `model_name` — name or path of a sentence-transformers model (default: `'sentence-transformers/all-MiniLM-L6-v2'`).
- `batch_size` — batch size for encoding (default: `32`).
- `show_progress_bar` — show progress during encoding (default: `False`).
- `normalize_embeddings` — L2-normalize embeddings to unit length (default: `False`).
- `device` — computation device: `'cpu'`, `'cuda'`, etc. `None` auto-detects (default: `None`).
- `precision` — embedding precision: `'float32'`, `'int8'`, `'uint8'`, `'binary'`, `'ubinary'` (default: `'float32'`).
- `truncate_dim` — truncate embeddings to this dimension, useful for Matryoshka models (default: `None`).
- `prompt_name` — name of a prompt from the model's prompts dictionary (default: `None`).
- `prompt` — raw prompt string to prepend to inputs (default: `None`).

**Returns from `transform`:** `numpy.ndarray`, shape `(n_samples, embedding_dimension)`.

**Feature names:** `get_feature_names_out()` returns `dim_0`, `dim_1`, …, `dim_{n-1}`.

**Lazy loading:** The `SentenceTransformer` model is loaded only when `fit()` is first called. Subsequent `fit()` calls reuse the cached model.

**None/NaN handling:** `None` and `NaN` values in input are automatically replaced with empty strings before encoding.

**Common mistakes:**

- Calling `transform()` before `fit()`: raises `NotFittedError`. Always call `fit()` or `fit_transform()` first.
- Passing a **DataFrame with more than one column**: this transformer expects a single text column.
- Forgetting to install the `nlp` extras: raises `ImportError` with a helpful message.

