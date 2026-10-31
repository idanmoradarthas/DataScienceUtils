---
name: ds-utils-strings
description: >
  String preprocessing, split-safe tag encoding, and significant term extraction. Use for converting comma-separated tag columns into model-ready features across train/test splits, or discovering distinctive text terms in document cohorts. Do NOT use for sentence embeddings or general tabular preprocessing.
version: "1.0.0"
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

`ds_utils.strings` provides specialized string operations for feature engineering and text cohort analysis. It converts delimited tag fields into one-hot indicator columns while enforcing strict split-safety, and surfaces statistically overrepresented keywords in text subsets (such as churned customer complaints).

- **When to use**: Encoding comma-separated metadata tags into binary features across train and test sets, or extracting distinctive vocabulary from specific document cohorts.
- **When NOT to use**: For dense semantic sentence embeddings (use `ds-utils-transformers`), multi-label list pipelines with persistent estimators (use `MultiLabelBinarizerTransformer` in `ds-utils-transformers`), or tabular feature distributions (use `ds-utils-preprocess`).

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
       Train / Test Split               Isolate Sub-Cohort
               │                         (e.g., churned users)
               ▼                               ▼
   Split-Safe Tag Binarization     Significant Term Extraction
    (append_tags_to_frame)         (extract_significant_terms)
               │                               │
               ▼                               ▼
   Aligned Feature Matrices        Targeted Keyword Discovery
```

#### Step 1: Split-Safe Tag Feature Extraction (`append_tags_to_frame`)
Raw data often contains comma-delimited multi-tags (e.g. `"python,sql,docker"`).
- **Split-safe by design**: `append_tags_to_frame` accepts `x_train` and `x_test` simultaneously.
- **Leakage prevention**: The tag vocabulary is derived **exclusively from `x_train`**. Unseen tags in `x_test` are dropped, and both resulting DataFrames share identical column sets and order.
- **Mandatory usage rule**: Always pass `(x_train, x_test)` as separate frames. Never concatenate them prior to binarization.

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from ds_utils.strings import append_tags_to_frame

# 1. Sample data with tags
raw_data = pd.DataFrame(
    {
        "user_id": range(6),
        "skills": ["python,sql", "sql,aws", "python,docker", "sql,docker,aws", "kubernetes,aws", "python"],
        "hired": [1, 0, 1, 1, 0, 1],
    }
)

train_df, test_df = train_test_split(raw_data, test_size=0.33, random_state=42)

# 2. Append tags split-safely
train_encoded, test_encoded = append_tags_to_frame(train_df, test_df, field_name="skills", prefix="skill_")
print("Engineered train columns:", [c for c in train_encoded.columns if c.startswith("skill_")])
print("Engineered test columns: ", [c for c in test_encoded.columns if c.startswith("skill_")])
assert list(train_encoded.columns) == list(test_encoded.columns)
```
- **What to look for**: Tag columns are binary 0/1 integers with prefix `skill_`.
- **Action**: Pass `train_encoded` and `test_encoded` directly into standard estimators.

#### Step 2: Significant Term Extraction in Subsets (`extract_significant_terms_from_subset`)
Identify which terms occur unusually frequently in an anomalous or target subset compared to the wider background corpus.

```python
from ds_utils.strings import extract_significant_terms_from_subset

corpus = pd.DataFrame(
    {
        "text": [
            "login button is broken and crashing",
            "smooth checkout experience very happy",
            "app keeps crashing on android device",
            "great product delivery fast",
            "unable to login session expired",
            "perfect service five stars",
        ],
        "is_complaint": [1, 0, 1, 0, 1, 0],
    }
)

complaints_subset = corpus[corpus["is_complaint"] == 1]

# Extract terms that distinguish complaints from the full corpus
term_scores = extract_significant_terms_from_subset(corpus, complaints_subset, field_name="text")
top_distinctive = term_scores.sort_values(ascending=False).head(5)
print("Top distinctive terms in complaints:\n", top_distinctive)
```
- **Interpretation**: Higher scores (0.0 to 1.0) indicate terms strongly over-represented in the subset compared to the general baseline.
- **Action**: Use discovered significant terms as domain-informed keyword features or prioritize them in root-cause investigations.

---

## Examples

### Complete Feature Engineering Script

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from ds_utils.strings import append_tags_to_frame

# 1. Dataset
df = pd.DataFrame(
    {
        "article_id": range(10),
        "topics": [
            "ai,ml",
            "web,css",
            "ml,dl",
            "cloud,aws",
            "web,javascript",
            "ai,dl",
            "cloud,docker",
            "database,sql",
            "database,nosql",
            "ai,robotics",
        ],
        "views": [500, 120, 800, 300, 150, 950, 400, 250, 200, 600],
        "is_viral": [1, 0, 1, 0, 0, 1, 0, 0, 0, 1],
    }
)

train_split, test_split = train_test_split(df, test_size=0.3, random_state=42)

# 2. Tag binarization
train_feat, test_feat = append_tags_to_frame(train_split, test_split, field_name="topics", prefix="topic_")

# 3. Model training on engineered tags
tag_cols = [c for c in train_feat.columns if c.startswith("topic_")]
clf = RandomForestClassifier(random_state=42)
clf.fit(train_feat[tag_cols], train_feat["is_viral"])
predictions = clf.predict(test_feat[tag_cols])
```

## Limitations

- **No persistent transformer instance**: `append_tags_to_frame` is a functional helper that transforms train and test together. It does not return a fitted object to persist for standalone inference on future rows. For an sklearn `Pipeline` / `ColumnTransformer` with `fit` and `transform`, use `MultiLabelBinarizerTransformer` from `ds-utils-transformers`.
- **Delimited strings only**: Expects comma-delimited strings (e.g. `"tag1,tag2"`). For Python lists of strings, use `MultiLabelBinarizerTransformer`.

## Troubleshooting

| Error / Symptom | Root Cause | Solution |
|---|---|---|
| Mismatched columns in train and test | Called `append_tags_to_frame` separately on train and test | Call `append_tags_to_frame(train, test, ...)` together in one call |
| Test vocabulary leaked into model | Concatenated train and test before calling `append_tags_to_frame` | Keep train and test strictly partitioned |
| `TypeError: expected string or bytes-like object` | Tag column contains missing (`NaN`) or non-string values | Impute missing tags with empty string `""` before calling |
| Need to serialize transformer for production inference | `append_tags_to_frame` cannot be pickled as an sklearn estimator | Use `MultiLabelBinarizerTransformer` from `ds-utils-transformers` in an sklearn Pipeline |
