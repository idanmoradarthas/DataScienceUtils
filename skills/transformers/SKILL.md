---
name: ds-utils-transformers
description: >
  Sklearn pipeline transformers for multi-label binarization with get_feature_names_out and float64 output. Use when the user needs MultiLabelBinarizer in a Pipeline, ColumnTransformer, or set_output(transform="pandas"), or when feature names must propagate from multi-label columns.
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
```

## Import

```python
from ds_utils.transformers import MultiLabelBinarizerTransformer
```

---

## MultiLabelBinarizerTransformer

Wraps `sklearn.preprocessing.MultiLabelBinarizer` so pipelines get `get_feature_names_out` and dense `float64` matrices.

**1. Direct Object Usage:**

```python
from ds_utils.transformers import MultiLabelBinarizerTransformer

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
from ds_utils.transformers import MultiLabelBinarizerTransformer
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
from ds_utils.transformers import MultiLabelBinarizerTransformer
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
