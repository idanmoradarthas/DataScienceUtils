---
name: ds-utils-preprocess
description: >
  Exploratory data analysis (EDA), feature distribution visualization, and statistical correlation analysis. Use for univariate feature inspection, collinearity detection, dendrograms, PCA explained variance, and mutual information ranking. Do NOT use for model evaluation metrics, NLP embeddings, or tree visualization.
version: "1.0.0"
license: MIT
metadata:
  author: "Idan Morad <idanmorad.arthas@gmail.com>"
  tags:
    - eda
    - preprocessing
    - feature-selection
    - correlation
  documentation: "https://datascienceutils.readthedocs.io/en/stable/"
  package: "data-science-utils"
  repository: "https://github.com/idanmoradarthas/DataScienceUtils"
---

# ds-utils-preprocess

Comprehensive exploratory data analysis (EDA), feature interaction visualization, and leakage-aware feature selection utilities.

## Purpose

`ds_utils.preprocess` accelerates initial dataset exploration and feature selection. It automates type-driven chart selection (violins for continuous floats, heatmaps for temporal features, count bars for categorical variables), visualizes collinearity structures, and computes group statistics and non-linear associations.

- **When to use**: During preliminary EDA, feature hygiene assessment, multicollinearity pruning, and feature relevance ranking before model training.
- **When NOT to use**: For post-modeling error analysis, model metric evaluation (use `ds-utils-metrics`), clustering quality scoring (use `ds-utils-unsupervised`), or text tag encoding (use `ds-utils-strings`).

## Prerequisites

- Python 3.10 or higher.
- `data-science-utils` package installed (`pip install data-science-utils`).
- Core dependencies: `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `scipy`.
- Input data formatted as a Pandas DataFrame or Series.

## Instructions

### Task-to-Approach Map

| I want to... | Function | Split-Aware fitting? | Output Type |
|---|---|---|---|
| Inspect single feature distribution & outliers | `visualize_feature(series)` | Split-agnostic | Matplotlib Axes |
| Check bivariate feature relationships & interactions | `plot_features_interaction(df, f1, f2)` | Split-agnostic | Matplotlib Axes |
| View correlation matrix as heatmap | `visualize_correlations(corr_matrix)` | Fit on Train | Matplotlib Axes |
| Group correlated clusters hierarchically | `plot_correlation_dendrogram(corr_matrix)` | Fit on Train | Matplotlib Axes |
| Identify redundant feature pairs to drop | `get_correlated_features(corr_matrix, features, target)` | **Train split only** | Pandas DataFrame |
| Compare numeric distributions across target classes | `extract_statistics_dataframe_per_label(df, num_col, target)` | **Train split only** | Pandas DataFrame |
| Rank features by non-linear target dependence | `compute_mutual_information(df, features, target)` | **Train split only** | Pandas DataFrame |
| Assess dimensionality reduction potential | `plot_pca_explained_variance(X)` | Fit on Train | Matplotlib Axes |

Full signatures, parameter options, and output table schemas are detailed in [references/api_reference.md](references/api_reference.md).

---

### End-to-End Leakage-Aware EDA Workflow

Data leakage happens when decisions about feature pruning, category grouping, or statistical associations use test-set knowledge. Always partition your dataset **before** running target-aware statistical methods.

```text
                  Raw Dataset
                       │
             ┌─────────┴─────────┐
             ▼                   ▼
      Unlabeled Data      Supervised Data
             │                   │
             │           Train / Test Split  <-- Mandatory boundary
             │                   │
             ▼                   ▼
       Univariate EDA      Target-Aware EDA
     (visualize_feature)  (extract_statistics_dataframe_per_label)
             │                   │
             └─────────┬─────────┘
                       ▼
             Correlation & Structure
       (visualize_correlations, dendrogram)
                       ▼
             Collinearity Pruning
          (get_correlated_features)
                       ▼
             Mutual Information Ranking
          (compute_mutual_information)
                       ▼
             Dimensionality Assessment
          (plot_pca_explained_variance)
```

#### Step 1: Dataset Partitioning (Preventing Data Leakage)
Target-aware metrics (`extract_statistics_dataframe_per_label`, `compute_mutual_information`, and `get_correlated_features`) must be computed strictly on the training partition. Feature pruning choices are frozen on train and applied downstream to test.

```python
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# Synthetic example dataset
np.random.seed(42)
n = 200
X_raw = pd.DataFrame(
    {
        "age": np.random.uniform(20, 70, n),
        "income": np.random.uniform(30000, 120000, n),
        "education": np.random.choice(["HighSchool", "BSc", "MSc", "PhD"], n),
        "signup_date": pd.date_range("2023-01-01", periods=n, freq="D"),
    }
)
X_raw["proxy_income"] = X_raw["income"] * 1.05 + np.random.normal(0, 500, n)
y = (X_raw["income"] > 70000).astype(int)

# 1. Split FIRST
X_train, X_test, y_train, y_test = train_test_split(X_raw, y, test_size=0.25, random_state=42)
train_df = X_train.copy()
train_df["target"] = y_train
```

#### Step 2: Univariate Inspection (`visualize_feature`)
Inspect feature health (data types, missingness, extreme outliers).
- **Plot types**: Floats render as violin plots; datetimes render as day/hour heatmaps; integers and categoricals render as bar charts.
- **Cardinality rule**: If a categorical column has >10 unique values, only the top 10 appear; remaining values are grouped into `"Other values"`.

```python
from ds_utils.preprocess.visualization import visualize_feature
import matplotlib.pyplot as plt

# Float feature with outlier boundaries
ax = visualize_feature(train_df["income"], include_outliers=True, outlier_iqr_multiplier=1.5)
plt.close()

# Categorical feature with custom count ordering
ax_cat = visualize_feature(train_df["education"], show_counts=True, order="count_desc")
plt.close()
```
- **What to look for**: Heavy skew, extreme outliers outside IQR fences, or dominant categories.
- **Action**: Winsorize or transform skewed features; check if `"Other values"` contains high aggregate frequency.

#### Step 3: Target-Aware Statistics (`extract_statistics_dataframe_per_label`)
Evaluate how numeric feature percentiles shift across target classes.

```python
from ds_utils.preprocess.statistics import extract_statistics_dataframe_per_label

stats = extract_statistics_dataframe_per_label(train_df, "income", "target")
print(stats[["count", "mean", "median", "null_count"]])
```
- **Interpretation**: Compare medians and IQR (25th to 75th percentiles) across target groups. Significant separation indicates high predictive utility.
- **Action**: Retain features with distinct class distributions; flag features where null counts correlate with a specific class.

#### Step 4: Redundancy & Multicollinearity Pruning (`get_correlated_features`)
Multicollinearity destabilizes linear models and inflates feature importance variance in trees.
1. Compute the correlation matrix on the training split only: `corr = train_df[numeric_cols].corr()`.
2. Visualize with `visualize_correlations` or `plot_correlation_dendrogram`.
3. Use `get_correlated_features` to identify pairs exceeding a threshold (e.g., 0.85) and compare each feature's correlation to the target.

```python
from ds_utils.preprocess.visualization import visualize_correlations, plot_correlation_dendrogram
from ds_utils.preprocess.statistics import get_correlated_features

num_cols = ["age", "income", "proxy_income"]
train_num = train_df[num_cols + ["target"]]
corr_matrix = train_num.corr()

# Check collinear pairs
collinear_pairs = get_correlated_features(corr_matrix, num_cols, "target", threshold=0.85)
print(collinear_pairs)
```
- **Interpretation**: `level_0_level_1_corr` measures collinearity. `level_0_target_corr` and `level_1_target_corr` indicate relevance to the target.
- **Action**: Between `level_0` and `level_1`, prune the feature with the lower absolute target correlation. Never compute this on the full dataset or test split.

#### Step 5: Non-linear Association Ranking (`compute_mutual_information`)
Linear correlation misses non-monotonic dependencies. Mutual information quantifies total statistical dependency.

```python
from ds_utils.preprocess.statistics import compute_mutual_information

mi_df = compute_mutual_information(train_df, num_cols, "target", random_state=42)
print(mi_df)
```
- **Interpretation**: Features ranked with `mi_score` near 0 share minimal information with the target.
- **Action**: Drop features with near-zero MI that also exhibited low correlation in Step 4.

#### Step 6: Dimensionality Assessment (`plot_pca_explained_variance`)
Determine how many principal components are required to capture 90%+ of the variance in numeric features.

```python
from ds_utils.preprocess.visualization import plot_pca_explained_variance

ax_pca = plot_pca_explained_variance(train_df[num_cols], use_scaling=True)
plt.close()
```
- **Interpretation**: If a small number of components captures almost all variance, dimensionality reduction is viable.
- **Action**: Fit PCA on `X_train` and transform `X_test` if compression is needed.

---

## Examples

### Complete Leakage-Safe Script

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from ds_utils.preprocess.visualization import (
    visualize_feature,
    visualize_correlations,
    plot_pca_explained_variance,
)
from ds_utils.preprocess.statistics import (
    get_correlated_features,
    extract_statistics_dataframe_per_label,
    compute_mutual_information,
)

# 1. Synthetic dataset
np.random.seed(42)
df = pd.DataFrame(
    {
        "tenure": np.random.exponential(12, 100),
        "monthly_charges": np.random.normal(70, 20, 100),
        "total_charges_proxy": np.random.normal(70, 20, 100) * 1.1,
        "churn": np.random.choice([0, 1], 100, p=[0.7, 0.3]),
    }
)

# 2. Split train and test
train, test = train_test_split(df, test_size=0.2, random_state=42)

# 3. Univariate check
ax1 = visualize_feature(train["monthly_charges"])

# 4. Grouped statistics on train
stats = extract_statistics_dataframe_per_label(train, "monthly_charges", "churn")

# 5. Collinearity inspection and pruning on train
num_features = ["tenure", "monthly_charges", "total_charges_proxy"]
corr = train[num_features + ["churn"]].corr()
corr_pairs = get_correlated_features(corr, num_features, "churn", threshold=0.80)

# Drop redundant feature with lower target correlation
to_drop = []
for _, row in corr_pairs.iterrows():
    if abs(row["level_0_target_corr"]) >= abs(row["level_1_target_corr"]):
        to_drop.append(row["level_1"])
    else:
        to_drop.append(row["level_0"])
selected_features = [f for f in num_features if f not in set(to_drop)]

# 6. Mutual information ranking
mi = compute_mutual_information(train, selected_features, "churn", random_state=42)

# 7. Apply selected features downstream to test
X_train_final = train[selected_features]
X_test_final = test[selected_features]
```

## Limitations

- **Univariate only**: `visualize_feature` examines one Series at a time; it cannot color or group by a target label.
- **Pre-computed matrix required**: `visualize_correlations`, `plot_correlation_dendrogram`, and `get_correlated_features` require pre-computed correlation matrices (`df.corr()`), not raw dataframes.
- **Numeric PCA input**: `plot_pca_explained_variance` requires all input columns to be numeric; encode categoricals before passing.

## Troubleshooting

| Error / Symptom | Root Cause | Solution |
|---|---|---|
| `AttributeError: 'DataFrame' object has no attribute 'name'` | Passed DataFrame `df[['col']]` to `visualize_feature` | Pass a 1D Series `df['col']` |
| `KeyError: 'target'` in `get_correlated_features` | Target column was omitted when computing `df.corr()` | Ensure `df[[...features, target]].corr()` includes the target column |
| `ValueError: could not convert string to float` in PCA | DataFrame contains non-numeric / string columns | Filter or encode non-numeric columns before calling `plot_pca_explained_variance` |
| Suboptimal test performance / Overfitting | Leakage from running `get_correlated_features` or MI on full data | Perform train/test split first and calculate statistics exclusively on training data |
