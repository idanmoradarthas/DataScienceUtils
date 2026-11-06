---
name: ds-utils-strings
description: >
  String preprocessing, split-safe tag encoding, and significant term extraction. Use for converting comma-separated tag columns into model-ready features across train/test splits, or discovering distinctive text terms in document cohorts. Do NOT use for sentence embeddings or general tabular preprocessing.
version: 1.0.0
license: MIT
metadata:
  author: "Idan Morad <idanmorad.arthas@gmail.com>"
  tags:
    - strings
    - text-mining
    - tag-encoding
    - significant-terms
    - feature-engineering
  documentation: "https://datascienceutils.readthedocs.io/en/stable/"
  package: "data-science-utils"
  repository: "https://github.com/idanmoradarthas/DataScienceUtils"
---

# ds-utils-strings

Text analytics, split-safe tag feature binarization, and significant term extraction.

## Purpose

ds_utils.strings provides specialized string operations for feature engineering and text cohort analysis. It converts delimited tag fields into one-hot indicator columns while enforcing strict split-safety, and surfaces statistically overrepresented keywords in text subsets (such as churned customer complaints).

- **When to use**: Encoding comma-separated metadata tags into binary features across train and test sets, or extracting distinctive vocabulary from specific document cohorts.
- **When NOT to use**: For dense semantic sentence embeddings (use ds-utils-transformers), multi-label list pipelines with persistent estimators (use MultiLabelBinarizerTransformer in ds-utils-transformers), or tabular feature distributions (use ds-utils-preprocess).

## Prerequisites

- Python 3.11 or higher.
- data-science-utils package installed (pip install data-science-utils).
- Core dependencies: numpy, pandas, scikit-learn.

## Instructions

### Task-to-Approach Map

| Text Processing Goal | Function | Split-Safety Guarantee | Output |
|---|---|---|---|
| Encode comma-separated tags into binary columns | `append_tags_to_frame(X_train, X_test, field_name)` | **Vocabulary fit on train only** | `Tuple[DataFrame, DataFrame]` |
| Find terms distinguishing a target cohort | `extract_significant_terms_from_subset(data_frame, subset_data_frame, field_name)` | Corpus vs. cohort comparison | Pandas Series of scores |

Full parameter specifications, return schemas, and common mistakes are documented in [references/api_reference.md](references/api_reference.md).

---

### End-to-End Strings Feature Engineering Workflow

```text
              Raw Data with String / Text Columns
                               │
               ┌───────────────┴───────────────┐
               ▼                               ▼
       Categorical Tags                 Freeform Text
               │                               │
       Profile Tag Column               Isolate Sub-Cohort
               │                         (e.g., churned users)
               ▼                               ▼
       Train / Test Split          Significant Term Extraction
               │                   (extract_significant_terms)
               ▼                               │
   Split-Safe Tag Binarization                 ▼
    (append_tags_to_frame)         Targeted Keyword Discovery
               │
               ▼
   Aligned Feature Matrices
```

#### Inputs you need

The steps below assume:
- A DataFrame `df` with a comma-delimited tag column, an optional freeform text column, and an optional cohort indicator flag.
- Your training and testing partitions (`train_df` and `test_df`).

```python
import pandas as pd

# Load your text / tag dataset (replace with your data)
df = pd.read_csv("your_dataset")
# train_df, test_df = train_test_split(df, test_size=0.25, random_state=42)
```

#### Step 1: Profile the tag column before encoding
Question: which tags exist in the data, how common are they, and how messy is the delimiter formatting?

```python
tag_col = "skills"  # replace with your tag column name

# 1. Tag prevalence across rows
tag_counts = df[tag_col].dropna().str.split(",").explode().str.strip().str.lower().value_counts()
print("Top 10 tags:\n", tag_counts.head(10))

# 2. Tag cleanliness and missingness
missing_ratio = df[tag_col].isna().mean()
casing_variants = df[tag_col].dropna().str.split(",").explode().str.strip().nunique()
tags_per_row = df[tag_col].dropna().str.split(",").apply(len).describe()
print(f"Missing ratio: {missing_ratio:.2%}, Unique raw casings: {casing_variants}")
print("Tags per row summary:\n", tags_per_row)
```
- **What to look for**: Discrepancies between lowercase and raw unique counts (casing variants), extreme long tails of one-off tags, and high missingness.
- **Action**: Use the observed tag frequencies to configure `lowercase=True`, cap vocabulary with `max_features` (default 500), and choose a clear column `prefix`.
- **Caution**: Missing values produce all-zero indicator rows without errors; confirm if missing tags should be imputed or handled explicitly.

#### Step 2: Split-Safe Tag Feature Extraction (`append_tags_to_frame`)
Question: how do we binarize multi-label tag strings into numeric columns without leaking test tags into training?

```python
from ds_utils.strings import append_tags_to_frame

train_encoded, test_encoded = append_tags_to_frame(
    train_df,
    test_df,
    field_name=tag_col,
    prefix="tag_",
    max_features=50,
    lowercase=True,
)
print("Engineered train columns:", [c for c in train_encoded.columns if c.startswith("tag_")][:5])
assert list(train_encoded.columns) == list(test_encoded.columns)
```
- **What to look for**: Identical column names and order across train and test sets.
- **Action**: Pass `train_encoded` and `test_encoded` directly to estimators or downstream transformers.
- **Caution**: Never concatenate train and test prior to encoding; `append_tags_to_frame` derives vocabulary strictly from `x_train`.

#### Step 3: Significant Term Extraction in Subsets (`extract_significant_terms_from_subset`)
Question: which words or phrases are statistically over-represented in a specific cohort compared to the full corpus?

```python
from sklearn.feature_extraction.text import CountVectorizer
from ds_utils.strings import extract_significant_terms_from_subset

cohort_flag = "is_churn"  # replace with your cohort column
text_col = "feedback"  # replace with your text column

# Filter cohort and check size first
cohort_df = train_df[train_df[cohort_flag] == 1]
print(f"Cohort size: {len(cohort_df)} out of {len(train_df)} corpus records")

term_scores = extract_significant_terms_from_subset(
    train_df,
    cohort_df,
    field_name=text_col,
    vectorizer=CountVectorizer(stop_words="english"),
)
print("Top distinctive terms in cohort:\n", term_scores.sort_values(ascending=False).head(10))
```
- **What to look for**: High scores indicating terms disproportionately present in the cohort relative to background occurrences.
- **Action**: Investigate top terms for operational root-cause analysis or use them as domain-specific indicator features.
- **Caution**: When cohort terms become model features, extract them strictly on the training partition to avoid target leakage. Read raw counts alongside term scores to ensure terms are not driven by single-document anomalies.

---

## Examples

### Runnable demo on a bundled dataset (replace the loading lines with your own data)

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from ds_utils.strings import append_tags_to_frame, extract_significant_terms_from_subset

# Format illustration only - replace with your column(s)
raw_df = pd.DataFrame(
    {
        "user_id": [101, 102, 103, 104, 105, 106, 107, 108],
        "roles": [
            "engineer,python",
            "analyst,sql",
            "engineer,cloud,aws",
            "analyst,python,sql",
            "manager,agile",
            "engineer,python,docker",
            "manager,finance",
            "engineer,sql,docker",
        ],
        "feedback": [
            "fast pipeline deployment issue resolved",
            "dashboard query is very slow and lagging",
            "cloud storage bucket quota exceeded",
            "report numbers are lagging and inaccurate",
            "team meeting schedule aligned",
            "container deployment failed with timeout",
            "budget allocation approved",
            "database connection pool issue",
        ],
        "flag": [1, 0, 1, 0, 0, 1, 0, 1],
    }
)

train_df, test_df = train_test_split(raw_df, test_size=0.25, random_state=42)

# 1. Profile tags on training split
tag_prevalence = train_df["roles"].dropna().str.split(",").explode().str.strip().str.lower().value_counts()

# 2. Append tags split-safely
train_enc, test_enc = append_tags_to_frame(train_df, test_df, field_name="roles", prefix="role_", lowercase=True)
assert list(train_enc.columns) == list(test_enc.columns)

# 3. Extract distinctive terms for cohort (flag == 1) on train split only
cohort = train_df[train_df["flag"] == 1]
scores = extract_significant_terms_from_subset(
    train_df,
    cohort,
    field_name="feedback",
    vectorizer=CountVectorizer(stop_words="english"),
)
top_terms = scores.sort_values(ascending=False).head(3)
```

## Limitations

- **Feature extraction on training split only**: When terms extracted via `extract_significant_terms_from_subset` are used as downstream model features and the subset is derived from the target variable, they must be computed on the training partition only to prevent data leakage.
- **No persistent transformer instance**: `append_tags_to_frame` is a functional helper that transforms train and test together. It does not return a fitted object to persist for standalone inference on future rows. For an sklearn `Pipeline` / `ColumnTransformer` with `fit` and `transform`, use `MultiLabelBinarizerTransformer` from `ds-utils-transformers`.
- **Delimited strings only**: Expects comma-delimited strings (e.g. `"tag1,tag2"`). For Python lists of strings, use `MultiLabelBinarizerTransformer`.
- **Case sensitivity**: `lowercase=False` by default; distinct casings (e.g. `"Python"` and `"python"`) yield separate columns unless `lowercase=True` is explicitly passed.
- **Vocabulary cap**: `max_features=500` by default silently caps the number of encoded tag columns to the top 500 tags. Pass `max_features=None` to encode all distinct tags.

## Troubleshooting

| Error / Symptom | Root Cause | Solution |
|---|---|---|
| Mismatched columns in train and test | Called `append_tags_to_frame` separately on train and test | Call `append_tags_to_frame(train, test, ...)` together in one call |
| Test vocabulary leaked into model | Concatenated train and test before calling `append_tags_to_frame` | Keep train and test strictly partitioned |
| Silent all-zero indicator rows | Missing (`NaN`, `None`) or non-string values in the tag column | Check `df[col].isna().sum()` and decide whether to fill or drop before encoding; all-zero rows are the expected output for missing tags |
| Common stop words ranked as significant | Default CountVectorizer does not filter English stop words | Pass `vectorizer=CountVectorizer(stop_words='english')` to `extract_significant_terms_from_subset` |
| Need to serialize transformer for production inference or pipeline integration | `append_tags_to_frame` is a functional helper that transforms train and test together without separate fit/transform methods or persistent vocabulary state | Use `MultiLabelBinarizerTransformer` from `ds-utils-transformers` inside an sklearn `Pipeline` / `ColumnTransformer` |
