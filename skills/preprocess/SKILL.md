---
name: ds-utils-preprocess
description: >
  Provides data preprocessing visualization and statistical correlation methods. Use when the user asks to visualize feature distributions, wants to plot correlation matrices or dendrograms, explore feature interactions, or needs to calculate mutual information or label-based statistics in a Python data science project using sklearn-compatible models.
license: MIT
metadata:
  author: Idan Morad
  documentation: https://datascienceutils.readthedocs.io/en/stable/
  package: data-science-utils
  repository: https://github.com/idanmoradarthas/DataScienceUtils
---
# Preprocess — ds_utils.preprocess

Utilities for data visualization and robust feature statistics.

## Installation

```bash
pip install data-science-utils
# or
conda install -c idanmorad data-science-utils
```

## Import

```python
from ds_utils.preprocess.visualization import visualize_feature
from ds_utils.preprocess.visualization import visualize_correlations
from ds_utils.preprocess.visualization import plot_correlation_dendrogram
from ds_utils.preprocess.visualization import plot_features_interaction
from ds_utils.preprocess.statistics import get_correlated_features
from ds_utils.preprocess.statistics import extract_statistics_dataframe_per_label
from ds_utils.preprocess.statistics import compute_mutual_information
```

---

## visualize_feature

Visualizes a single feature's distribution based on its data type.

```python
from ds_utils.preprocess.visualization import visualize_feature

# complete usage example
visualize_feature(
    df["feature_name"], 
    remove_na=False, 
    include_outliers=True, 
    outlier_iqr_multiplier=1.5,
    first_day_of_week="Monday", 
    show_counts=True, 
    order=None
)
```

**Parameters:**
- `series` — pd.Series, The feature to visualize.
- `remove_na` — bool, whether to explicitly remove NAs before plotting. (Default: False).
- `include_outliers` — bool, For float features, whether to include outliers in the violin plot. (Default: True).
- `outlier_iqr_multiplier` — float, IQR multiplier for outlier fence computation if `include_outliers=False`. (Default: 1.5).
- `first_day_of_week` — str, Start day for datetime heatmaps (e.g., "Monday" or "Sunday"). (Default: "Monday").
- `show_counts` — bool, Whether to display count labels on categorical bar charts. (Default: True).
- `order` — str or list, Custom sorting for categories (e.g. "count_desc" or ["High", "Low"]). (Default: None).

**Returns:** matplotlib Axes.

**Common mistakes:**
- Always pass a **Series** (e.g., `df["col"]`), never a DataFrame (`df[["col"]]`).
- The function auto-selects the plot by dtype (float→violin, datetime→heatmap, rest→bar). You don't need to specify the plot type.
- High-cardinality features (>10 unique values) will only show the top 10 categories, with the rest grouped into "Other values".

---

## visualize_correlations

Visualizes a correlation matrix as a heatmap.

```python
from ds_utils.preprocess.visualization import visualize_correlations
import matplotlib.pyplot as plt

# complete usage example
visualize_correlations(df.corr())
plt.show()
```

**Parameters:**
- `correlation_matrix` — pd.DataFrame, The correlation matrix.

**Returns:** matplotlib Axes.

**Common mistakes:**
- Always pass the correlation matrix `df.corr()` as the input result, NOT the raw DataFrame `df`.

---

## plot_correlation_dendrogram

Plots a hierarchical dendrogram of a correlation matrix.

```python
from ds_utils.preprocess.visualization import plot_correlation_dendrogram
import matplotlib.pyplot as plt

# complete usage example
plot_correlation_dendrogram(df.corr())
plt.show()
```

**Parameters:**
- `correlation_matrix` — pd.DataFrame, The correlation matrix.

**Returns:** matplotlib Axes.

**Common mistakes:**
- Passing the raw DataFrame instead of `df.corr()`.

---

## plot_features_interaction

Plots the joint distribution and relationship between two features.

```python
from ds_utils.preprocess.visualization import plot_features_interaction

# complete usage example
plot_features_interaction(df, "feature1", "feature2", remove_na=False)
```

**Parameters:**
- `data_frame` — pd.DataFrame, The dataset.
- `feature_1` — str, First feature column name.
- `feature_2` — str, Second feature column name.
- `remove_na` — bool, Whether to remove NA/null values before plotting.
  When `False` (default), missing values are included: shown as rug plots
  on the axes for numeric/datetime plots, or as a separate bar for
  categorical plots.

**Dtype → Chart type matrix:**

| feature_1 ↓  feature_2 → | Numeric | Categorical / Bool | Datetime |
|---|---|---|---|
| **Numeric** | Scatter plot | Box plot | Line plot |
| **Categorical / Bool** | Box plot | Shared histogram | Violin plot |
| **Datetime** | Line plot | Violin plot | Line plot |

**Returns:** matplotlib Axes.

---

## get_correlated_features

Extracts pairs of highly correlated features and their correlation to the target.

```python
from ds_utils.preprocess.statistics import get_correlated_features

# complete usage example
correlations = get_correlated_features(df.corr(), ["f1", "f2"], "target")
```

**Parameters:**
- `correlation_matrix` — pd.DataFrame, The correlation matrix.
- `features` — list, List of feature names to consider.
- `target` — str, The target column name (to report their correlation to the target).

**Returns:** pd.DataFrame with columns `['level_0', 'level_1', 'level_0_level_1_corr', 'level_0_target_corr', 'level_1_target_corr']`.

**Output Example:**
| level_0 | level_1 | level_0_level_1_corr | level_0_target_corr | level_1_target_corr |
|---------|---------|----------------------|---------------------|---------------------|
| featureA| featureB| 0.95                 | 0.51                | 0.48                |

**Common mistakes:**
- `target` must be a column name string, not an array of values.
- `correlation_matrix` must be pre-computed.

---

## extract_statistics_dataframe_per_label

Calculates statistical metrics for a feature grouped by label values.

```python
from ds_utils.preprocess.statistics import extract_statistics_dataframe_per_label

# complete usage example
stats_df = extract_statistics_dataframe_per_label(df, "amount", "category")
```

**Parameters:**
- `df` — pd.DataFrame, The input data.
- `feature_name` — str, the numerical feature to describe.
- `label_name` — str, the categorical label column.

**Returns:** pd.DataFrame of statistics per group, containing count, mean, min, max and percentiles.

**Output Example:**
| category | count | null_count | mean  | min | 1_percentile | 5_percentile | 25_percentile | median | 75_percentile | 95_percentile | 99_percentile | max |
|----------|-------|------------|-------|-----|--------------|--------------|---------------|--------|---------------|---------------|---------------|-----|
| Group_A  | 100   | 0          | 25.5  | 10  | 11.2         | 12.0         | 15.0          | 25.0   | 35.0          | 45.0          | 49.0          | 50  |
| Group_B  | 150   | 2          | 45.0  | 20  | 21.0         | 22.0         | 30.0          | 45.0   | 60.0          | 75.0          | 80.0          | 85  |

---

## compute_mutual_information

Computes mutual information scores between features and a target label.

```python
from ds_utils.preprocess.statistics import compute_mutual_information

# complete usage example
mi_scores = compute_mutual_information(df, ["f1", "f2"], "target", random_state=42)
```

**Parameters:**
- `df` — pd.DataFrame, The data frame.
- `features` — list of str, Features to test.
- `label_col` — str, Target column.
- `random_state` — int, random state for reproducibility.

**Returns:** pd.DataFrame of MI scores sorted descending.

**Output Example:**
| feature_name | mi_score |
|--------------|----------|
| feature1     | 0.245    |
| feature3     | 0.182    |
| feature2     | 0.091    |

---

## Typical Workflow

```python
from ds_utils.preprocess.visualization import visualize_feature, visualize_correlations
from ds_utils.preprocess.statistics import get_correlated_features

visualize_feature(df["age"])
corr = df.corr()
visualize_correlations(corr)
results = get_correlated_features(corr, ["age", "income"], "is_churn")
```
