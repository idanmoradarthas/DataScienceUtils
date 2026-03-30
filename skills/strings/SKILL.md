---
name: ds-utils-strings
description: >
  Provides string manipulation and text analysis functions for DataFrames. Use when the user asks to one-hot encode comma-separated tags into boolean columns, or needs to extract statistically significant terms from a subset of text documents in a Python data science project.
metadata:
  author: Idan Morad
  documentation: https://datascienceutils.readthedocs.io/en/stable/
  package: data-science-utils
  repository: https://github.com/idanmoradarthas/DataScienceUtils
---
# Strings — ds_utils.strings

String and tags manipulation tools for feature extraction.

## Installation

```bash
pip install data-science-utils
# or
conda install -c idanmorad data-science-utils
```

## Import

```python
from ds_utils.strings import append_tags_to_frame
from ds_utils.strings import extract_significant_terms_from_subset
```

---

## append_tags_to_frame

Extracts tags from a specified field and creates new binary columns (One-Hot Encoded) for each unique tag.

```python
from ds_utils.strings import append_tags_to_frame

# complete usage example
x_train_tags, x_test_tags = append_tags_to_frame(
    x_train, x_test, "article_tags", prefix="tag_"
)
```

**Parameters:**
- `x_train` — pd.DataFrame, The training dataframe.
- `x_test` — pd.DataFrame, The test dataframe.
- `tag_column` — str, The column containing comma-separated tags.
- `prefix` — str, Prefix for the generated boolean tag columns.

**Returns:** Tuple[pd.DataFrame, pd.DataFrame] representing the one-hot encoded dataset features.

**Output Example:**
If `x_train` has a column `tags` with `"ds,ml"` for `article_1` and `"ml,dl"` for `article_2`:

| article_name | tag_ds | tag_ml | tag_dl |
|--------------|--------|--------|--------|
| article_1    | 1      | 1      | 0      |
| article_2    | 0      | 1      | 1      |

**Common mistakes:**
- `append_tags_to_frame` MUST receive BOTH train AND test together. Calling it separately on each will produce mismatched columns and data leakage.
- Vocabulary is derived from `x_train` only; test-only tags are silently dropped.

---

## extract_significant_terms_from_subset

Identifies terms that are statistically overrepresented in a subset of documents compared to the full corpus.

```python
from ds_utils.strings import extract_significant_terms_from_subset

# complete usage example
subset = df[df['label'] == 'interesting']
terms_scores = extract_significant_terms_from_subset(df, subset, "text_column")
```

**Parameters:**
- `data_frame` — pd.DataFrame, The full corpus representing all documents.
- `subset` — pd.DataFrame, The subset of the dataframe representing specific documents to evaluate against the corpus.
- `field` — str, Target text or document content column.

**Returns:** pd.Series with index as terms and values as scores (0.0 to 1.0).

**Output Example:**
If identifying distinctive terms in the subset documents:

| term     | score |
|----------|-------|
| third    | 1.0   |
| one      | 1.0   |
| and      | 1.0   |
| this     | 0.67  |
| document | 0.25  |

**Common mistakes:**
- Expecting a raw count; it uses the Elasticsearch significant_text algorithm, returning scores from 0–1 per term.

---

## Typical Workflow

```python
import pandas as pd
from ds_utils.strings import append_tags_to_frame

x_train = pd.DataFrame([{"id": 1, "tags": "ds,ml,dl"}])
x_test = pd.DataFrame([{"id": 2, "tags": "ds,ml"}])

x_train_out, x_test_out = append_tags_to_frame(x_train, x_test, "tags", "tag_")
```
