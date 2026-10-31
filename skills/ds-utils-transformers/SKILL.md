---
name: ds-utils-transformers
description: >
  Scikit-learn pipeline transformers for multi-label binarization and sentence embeddings. Use for integrating MultiLabelBinarizer and sentence-transformers into sklearn Pipeline, ColumnTransformer, and set_output(transform="pandas"). Do NOT use for basic univariate feature plotting or correlation dendrograms.
version: "1.0.0"
license: MIT
metadata:
  author: "Idan Morad <idanmorad.arthas@gmail.com>"
  tags:
    - transformers
    - scikit-learn
    - pipeline
    - sentence-transformers
    - multi-label
  documentation: "https://datascienceutils.readthedocs.io/en/stable/"
  package: "data-science-utils"
  repository: "https://github.com/idanmoradarthas/DataScienceUtils"
---

# ds-utils-transformers

Production-ready, scikit-learn compatible transformers for multi-label lists and dense text embeddings.

## Purpose

`ds_utils.transformers` resolves standard scikit-learn API incompatibilities when working with complex feature types like list-of-tags columns and freeform text. It provides:
1. `MultiLabelBinarizerTransformer`: Enables multi-label tag binarization inside `Pipeline` and `ColumnTransformer` with full `get_feature_names_out()` and `set_output(transform="pandas")` compatibility.
2. `SentenceEmbeddingTransformer`: Encodes text columns into dense semantic vectors using `sentence-transformers` within unified sklearn workflows.

- **When to use**: Building composite feature processing pipelines combining tabular, categorical lists, and unstructured text into a single deployable scikit-learn Pipeline.
- **When NOT to use**: For standalone functional tag appending without pipeline persistence (use `ds-utils-strings`), or exploratory correlation plots (use `ds-utils-preprocess`).

## Prerequisites

- Python 3.10 or higher.
- `data-science-utils` package installed (`pip install data-science-utils`).
- Core dependencies: `numpy`, `pandas`, `scikit-learn`.
- Optional dependency for `SentenceEmbeddingTransformer`: `pip install data-science-utils[nlp]` (provides `sentence-transformers` and `torch`).

## Instructions

### Task-to-Approach Map

| Pipeline Task | Transformer Class | Input Format | Output |
|---|---|---|---|
| Binarize list of tags inside Pipeline / ColumnTransformer | `MultiLabelBinarizerTransformer` | Iterable of iterables (e.g., `[["tagA", "tagB"]]`) | 2D float64 array / DataFrame |
| Encode text columns into semantic embedding vectors | `SentenceEmbeddingTransformer` | 1D iterable of strings | 2D float array (e.g., $N \times 384$) |
| Return Pandas DataFrames from Pipeline stages | `.set_output(transform="pandas")` | Compatible with both transformers | Pandas DataFrame with clean headers |

Detailed constructor options, return types, and parameter defaults are in [references/api_reference.md](references/api_reference.md).

---

### End-to-End Multimodal Pipeline Workflow

A typical production dataset contains mixed data types: numerical columns, a multi-label tag column, and freeform text descriptions.

```text
               Raw Multimodal DataFrame
    (numerical features + list tags + text descriptions)
                         │
                 ColumnTransformer
        ┌────────────────┼────────────────┐
        ▼                ▼                ▼
  [StandardScaler]  [MultiLabel]   [SentenceEmbed]
   Numeric Columns   List of Tags   Text Column
        └────────────────┬────────────────┘
                         ▼
        set_output(transform="pandas")
                         │
                         ▼
            Downstream Estimator
        (e.g., RandomForestClassifier)
```

#### Step 1: Multi-Label Tag Transformation
Scikit-learn's native `MultiLabelBinarizer` lacks the standard estimator signature and fails inside `ColumnTransformer`. `MultiLabelBinarizerTransformer` adheres strictly to the transformer API.
- **Input format**: Must be an iterable of iterables (e.g. `pd.Series([["action", "sci-fi"], ["romance"]])`).
- **Feature naming**: Generates `{column}_{sanitized_label}` safely.

```python
import pandas as pd
from ds_utils.transformers.multi_label_binarizer import MultiLabelBinarizerTransformer

X_tags = [["python", "cloud"], ["sql"], ["cloud", "docker"]]
mlb = MultiLabelBinarizerTransformer()
mlb.set_output(transform="pandas")
df_tags = mlb.fit_transform(X_tags)
print(df_tags.columns.tolist())
# Output: ['label_cloud', 'label_docker', 'label_python', 'label_sql']
```

#### Step 2: Dense Semantic Embeddings (`SentenceEmbeddingTransformer`)
Converts raw text into semantic vectors without manual torch tokenization.
- **Dependency**: Requires `pip install data-science-utils[nlp]`.
- **Robustness**: Handles `None` and `NaN` values by converting them to empty strings.

```python
from ds_utils.transformers.sentence_embedding import SentenceEmbeddingTransformer

texts = ["Production database incident", "Quarterly financial report", "Bug report: UI login failure"]
embedder = SentenceEmbeddingTransformer(normalize_embeddings=True)
embedder.set_output(transform="pandas")
df_embed = embedder.fit_transform(texts)
print(f"Embedding shape: {df_embed.shape}, Features: {df_embed.columns[:3].tolist()}...")
```

#### Step 3: Composite ColumnTransformer & Pipeline
Combine numeric, categorical list, and text features into a single reproducible pipeline.

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Mixed feature DataFrame
df = pd.DataFrame(
    {
        "price": [19.99, 49.99, 99.99, 15.00],
        "categories": [["electronics", "audio"], ["home"], ["electronics", "gaming"], ["home"]],
        "title": [
            "Wireless bluetooth headphones",
            "Ceramic kitchen mug",
            "Mechanical gaming keyboard",
            "Cotton bath towel",
        ],
        "high_seller": [1, 0, 1, 0],
    }
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), ["price"]),
        ("tags", MultiLabelBinarizerTransformer(), "categories"),
        ("text", SentenceEmbeddingTransformer(normalize_embeddings=True), "title"),
    ]
)

pipeline = Pipeline(
    [
        ("features", preprocessor),
        ("model", RandomForestClassifier(random_state=42)),
    ]
)

pipeline.fit(df, df["high_seller"])
preds = pipeline.predict(df)
```
- **What to look for**: Pipeline executes without type errors or dimension mismatches; transforms seamlessly during `fit` and `predict`.
- **Action**: Serialize the entire fitted pipeline using `joblib.dump(pipeline, "model.joblib")`.

---

## Examples

### End-to-End Pipeline with Pandas Output

```python
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from ds_utils.transformers.multi_label_binarizer import MultiLabelBinarizerTransformer

# 1. Dataset with numeric and multi-label columns
data = pd.DataFrame(
    {
        "age": [25, 34, 45],
        "skills": [["python", "data"], ["sql"], ["python", "sql", "git"]],
    }
)

# 2. Build ColumnTransformer with pandas output
preprocessor = ColumnTransformer(
    [
        ("num", StandardScaler(), ["age"]),
        ("tags", MultiLabelBinarizerTransformer(), "skills"),
    ]
)
preprocessor.set_output(transform="pandas")

# 3. Transform and verify columns
processed = preprocessor.fit_transform(data)
print(processed.head())
assert any("python" in c for c in processed.columns)
```

## Limitations

- **Input dimensionality**: Both transformers expect a single column. When used in `ColumnTransformer`, pass a single column string (e.g. `"skills"`) or a single-element list (e.g. `["title"]`).
- **NLP Extras required**: `SentenceEmbeddingTransformer` requires `pip install data-science-utils[nlp]`. It raises an `ImportError` if `sentence-transformers` is missing.
- **Multi-label formatting**: Input to `MultiLabelBinarizerTransformer` must be an iterable of iterables. Passing a flat list of strings will treat each character as a label.

## Troubleshooting

| Error / Symptom | Root Cause | Solution |
|---|---|---|
| `ImportError: SentenceEmbeddingTransformer requires 'sentence-transformers'` | Optional NLP extra is not installed | Run `pip install data-science-utils[nlp]` or `pip install sentence-transformers` |
| `MultiLabelBinarizer` creates one column per letter (e.g., `'a'`, `'b'`) | Passed a flat list of strings `["tag1", "tag2"]` as `X` | Pass a list of lists `[["tag1", "tag2"]]` so each row has its own tag iterable |
| `NotFittedError` when calling `.transform()` | Transformer was not fitted | Ensure `.fit()` or `.fit_transform()` is invoked before calling `.transform()` |
| `ValueError: MultiLabelBinarizerTransformer expects a single column` | Passed multiple columns in `ColumnTransformer` | Target exactly one column: `("tags", MultiLabelBinarizerTransformer(), "col_name")` |
