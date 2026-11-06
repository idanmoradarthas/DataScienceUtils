---
name: ds-utils-transformers
description: >
  Scikit-learn pipeline transformers for multi-label binarization and sentence embeddings. Use for integrating MultiLabelBinarizer and sentence-transformers into sklearn Pipeline, ColumnTransformer, and set_output(transform="pandas"). Do NOT use for basic univariate feature plotting or correlation dendrograms.
version: 1.0.0
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

ds_utils.transformers resolves standard scikit-learn API incompatibilities when working with complex feature types like list-of-tags columns and freeform text. It provides:
1. `MultiLabelBinarizerTransformer`: Enables multi-label tag binarization inside `Pipeline` and `ColumnTransformer` with full `get_feature_names_out()` and `set_output(transform="pandas")` compatibility.
2. `SentenceEmbeddingTransformer`: Encodes text columns into dense semantic vectors using `sentence-transformers` within unified sklearn workflows.

- **When to use**: Building composite feature processing pipelines combining tabular, categorical lists, and unstructured text into a single deployable scikit-learn Pipeline.
- **When NOT to use**: For standalone functional tag appending without pipeline persistence (use ds-utils-strings), or exploratory correlation plots (use ds-utils-preprocess).

## Prerequisites

- Python 3.11 or higher.
- data-science-utils package installed (pip install data-science-utils).
- Core dependencies: numpy, pandas, scikit-learn.
- Optional dependency for `SentenceEmbeddingTransformer`: pip install data-science-utils[nlp] (provides sentence-transformers and torch).

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

#### Inputs you need

The steps below assume:
- A DataFrame `df` containing numeric column(s), a list-of-tags column (or delimited string), a text column, and a target column.
- Your training and testing partitions (`X_train`, `y_train`, `X_test`, `y_test`).

```python
import pandas as pd

# Load your multimodal dataset (replace with your data)
df = pd.read_csv("your_dataset")
# Note: If your tag column is a delimited string (e.g. "python,sql"), convert to lists first:
# df["tags"] = df["tags"].fillna("").str.split(",")
# X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=["target"]), df["target"], ...)
```

#### Step 1: Multi-Label Tag Transformation
Question: how do we encode multi-label tag lists into indicator columns inside a scikit-learn Pipeline?

```python
from ds_utils.transformers.multi_label_binarizer import MultiLabelBinarizerTransformer

# Format illustration only - replace with your column(s)
sample_tags = [["python", "cloud"], ["sql"], ["cloud", "docker"]]

mlb = MultiLabelBinarizerTransformer()
mlb.set_output(transform="pandas")
df_tags = mlb.fit_transform(sample_tags)
print(df_tags.columns.tolist())
```
- **What to look for**: Clean feature column names prefixed with `label_`.
- **Action**: Use `MultiLabelBinarizerTransformer` as a stage inside `ColumnTransformer`. If your raw data contains delimited strings, split them into Python lists via `df[col].str.split(",")` before transforming.
- **Caution**: The transformer expects an iterable of iterables; passing flat strings directly treats whole strings as single labels.

#### Step 2: Dense Semantic Embeddings (`SentenceEmbeddingTransformer`)
Question: how do we transform unstructured text descriptions into dense semantic vectors within a Pipeline?

```python
from ds_utils.transformers.sentence_embedding import SentenceEmbeddingTransformer

# Format illustration only - replace with your column(s)
sample_texts = ["Production database incident", "Quarterly financial report", "Bug report: UI login failure"]

embedder = SentenceEmbeddingTransformer(normalize_embeddings=True)
embedder.set_output(transform="pandas")
df_embed = embedder.fit_transform(sample_texts)
print(f"Embedding shape: {df_embed.shape}")
```
- **What to look for**: A 2D array or DataFrame with 384 embedding dimensions (for the default model).
- **Action**: Pass single text column names to `SentenceEmbeddingTransformer` inside `ColumnTransformer`.
- **Caution**: SentenceEmbeddingTransformer downloads HuggingFace model `sentence-transformers/all-MiniLM-L6-v2` on first use, which requires internet access or a local model path.

#### Step 3: Composite ColumnTransformer & Model Evaluation
Question: does our composite multimodal feature representation improve cross-validated model performance without data leakage?

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from ds_utils.transformers.multi_label_binarizer import MultiLabelBinarizerTransformer

# Configure ColumnTransformer targeting your specific feature columns
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), ["numeric_feature"]),
        ("tags", MultiLabelBinarizerTransformer(), "tag_feature"),
    ]
)
pipeline = Pipeline([("features", preprocessor), ("model", RandomForestClassifier(random_state=42))])

# 1. Fit transformer and verify vocabulary extracted strictly from training data
preprocessor.fit(X_train)
tag_features = [f for f in preprocessor.get_feature_names_out() if f.startswith("tags__")]
print(f"Learned {len(tag_features)} tag features from train split")

# 2. Evaluate with cross-validation on train split
scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="accuracy")
print(f"5-Fold CV Accuracy: {scores.mean():.3f} +/- {scores.std():.3f}")

# 3. Fit on full training set, and evaluate once on held-out test data
pipeline.fit(X_train, y_train)
test_accuracy = pipeline.score(X_test, y_test)
print(f"Held-out Test Accuracy: {test_accuracy:.3f}")
```
- **What to look for**: Cross-validation scores across folds, and confirming `get_feature_names_out()` reflects training vocabulary only.
- **Action**: Compare feature subsets (e.g. drop numeric or tag branches) to measure incremental utility.
- **Caution**: Touch the test set only once after feature selection and hyperparameters are finalized. Unseen tags appearing in test will be ignored with a warning.

---

## Examples

### Runnable demo on a bundled dataset (replace the loading lines with your own data)

```python
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from ds_utils.transformers.multi_label_binarizer import MultiLabelBinarizerTransformer

# Format illustration only - replace with your column(s)
data = pd.DataFrame(
    {
        "tenure": [12, 24, 6, 36, 18, 48, 8, 30],
        "monthly_spend": [45.0, 70.5, 30.0, 95.0, 60.0, 110.0, 40.0, 85.0],
        "services": [
            ["phone", "internet"],
            ["internet"],
            ["phone"],
            ["phone", "internet", "streaming"],
            ["internet", "streaming"],
            ["phone", "internet", "streaming"],
            ["phone"],
            ["internet", "streaming"],
        ],
        "churn": [0, 1, 0, 1, 0, 1, 0, 1],
    }
)

X = data[["tenure", "monthly_spend", "services"]]
y = data["churn"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Build ColumnTransformer with pandas output (runs offline without NLP model downloads)
preprocessor = ColumnTransformer(
    [
        ("num", StandardScaler(), ["tenure", "monthly_spend"]),
        ("tags", MultiLabelBinarizerTransformer(), "services"),
    ]
)
preprocessor.set_output(transform="pandas")

processed_train = preprocessor.fit_transform(X_train)
processed_test = preprocessor.transform(X_test)

feature_names = preprocessor.get_feature_names_out().tolist()
assert any("services" in f for f in feature_names)
```

## Limitations

- **Input dimensionality**: Both transformers expect a single column. When used in `ColumnTransformer`, pass a single column string (e.g. `"skills"`) or a single-element list (e.g. `["title"]`).
- **Unseen tags ignored**: `MultiLabelBinarizerTransformer` ignores tags not seen during fit with a warning (`unknown class(es) [...] will be ignored`); a sample containing only unseen tags becomes all zeros.
- **NLP Extras and model download**: `SentenceEmbeddingTransformer` requires `pip install data-science-utils[nlp]`. It downloads `sentence-transformers/all-MiniLM-L6-v2` on first use, requiring network access or a local model path.
- **String splitting**: `MultiLabelBinarizerTransformer` does not automatically split comma-separated strings (e.g. `"tagA,tagB"` is treated as a single label `label_tagA_tagB`). For delimited strings, split values into lists first or use `append_tags_to_frame` from `ds-utils-strings`.

## Troubleshooting

| Error / Symptom | Root Cause | Solution |
|---|---|---|
| `ImportError: sentence-transformers is required for SentenceEmbeddingTransformer. Install it with:  pip install data-science-utils[nlp]` | Optional NLP extra is not installed | Run `pip install data-science-utils[nlp]` |
| `UserWarning: unknown class(es) [...] will be ignored` / All-zero tag rows during inference | Test data contains tags not present in training data during `fit()` | Expected scikit-learn behavior; unknown tags are dropped and rows with only unseen tags produce zeros |
| Comma-separated tags treated as one feature (e.g., `label_tag1_tag2`) | Passed comma-delimited string without splitting into list | Split string into list `['tag1', 'tag2']` or use `append_tags_to_frame` from `ds-utils-strings` |
| `NotFittedError` when calling `.transform()` | Transformer was not fitted | Ensure `.fit()` or `.fit_transform()` is invoked before calling `.transform()` |
| `ValueError: MultiLabelBinarizerTransformer expects a single column` | Passed multiple columns in `ColumnTransformer` | Target exactly one column: `("tags", MultiLabelBinarizerTransformer(), "col_name")` |
