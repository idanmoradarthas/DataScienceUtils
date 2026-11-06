---
name: ds-utils-preprocess
description: >
  Exploratory data analysis (EDA), feature distribution visualization, and statistical correlation analysis. Use for univariate feature inspection, collinearity detection, dendrograms, PCA explained variance, and mutual information ranking. Do NOT use for model evaluation metrics, NLP embeddings, or tree visualization.
version: 1.0.0
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

ds_utils.preprocess accelerates initial dataset exploration and feature selection. It automates type-driven chart selection (violins for continuous floats, heatmaps for temporal features, count bars for categorical variables), visualizes collinearity structures, and computes group statistics and non-linear associations.

- **When to use**: During preliminary EDA, feature hygiene assessment, multicollinearity pruning, and feature relevance ranking before model training.
- **When NOT to use**: For post-modeling error analysis, model metric evaluation (use ds-utils-metrics), clustering quality scoring (use ds-utils-unsupervised), or text tag encoding (use ds-utils-strings).

## Prerequisites

- Python 3.11 or higher.
- data-science-utils package installed (pip install data-science-utils).
- Core dependencies: numpy, pandas, scikit-learn, matplotlib, seaborn, scipy.
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
             ▼                   │
       Bivariate EDA             │
 (plot_features_interaction)     │
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

#### Inputs you need

The steps below assume:
- A DataFrame `df` containing your features and an optional `target` column name string.
- A workflow decision: supervised modeling (partition first to avoid leakage) or unsupervised exploration.

```python
import pandas as pd

# Load your dataset (replace with your data)
df = pd.read_csv("your_dataset")
target = "target_column"  # replace with your target column name
```

#### Step 1: Profile and partition
Question: what are the column types, missingness levels, target balances, and train/test boundaries?

```python
from sklearn.model_selection import train_test_split

print(df.dtypes)
print(df.isna().mean())
print(df[target].value_counts(normalize=True))

# Partition first for supervised workflows
train_df, test_df = train_test_split(df, test_size=0.25, stratify=df[target], random_state=42)
num_cols = train_df.select_dtypes("number").columns.drop(target).tolist()
```
- **What to look for**: High missingness rates, unexpected string types on numeric fields, and class imbalance.
- **Action**: Impute or drop invalid columns; stratify classification splits.
- **Caution**: Computing target-aware statistics before splitting leaks validation/test information into feature choices.

#### Step 2: Univariate Inspection (`visualize_feature`)
Question: what is the distribution shape, scale, and outlier presence for an individual feature?

```python
import matplotlib.pyplot as plt
from ds_utils.preprocess.visualization import visualize_feature

ax = visualize_feature(train_df[num_cols[0]], include_outliers=True, outlier_iqr_multiplier=1.5)
plt.close()
```
- **What to look for**: Heavy skewness, extreme outliers outside IQR fences, or multi-modal distributions.
- **Action**: Winsorize or log-transform heavily skewed features before downstream modeling.
- **Caution**: visualize_feature plots unconditional distributions; it does not stratify by target classes.

#### Step 3: Bivariate Feature Interaction (`plot_features_interaction`)
Question: do two features interact or separate the classes?

```python
from ds_utils.preprocess.visualization import plot_features_interaction

ax_inter = plot_features_interaction(train_df, num_cols[0], num_cols[1])
plt.close()
```
- **What to look for**: Non-linear curves, clustering patterns, or conditional variance shifts between the two features.
- **Action**: Check the interaction chart type determined by the data type matrix documented in the reference file (scatter for numeric pairs, box/violin for numeric-categorical, heatmap for categorical-categorical).
- **Caution**: Interaction plots on high-cardinality categoricals become unreadable; filter to frequent categories first.

#### Step 4: Target-Aware Statistics (`extract_statistics_dataframe_per_label`)
Question: how do numeric feature percentiles and summary statistics shift across target classes?

```python
from ds_utils.preprocess.statistics import extract_statistics_dataframe_per_label

stats = extract_statistics_dataframe_per_label(train_df, num_cols[0], target)
print(stats[["count", "mean", "median", "null_count"]])
```
- **What to look for**: Significant shifts in medians and interquartile ranges across target classes.
- **Action**: Retain features with distinct class separations; investigate any feature whose missingness correlates with a label.
- **Caution**: Run this exclusively on the training split; running across test rows leaks label associations.

#### Step 5: Redundancy & Multicollinearity Pruning (`get_correlated_features`)
Question: which features are redundant with each other, and which one of each pair is more predictive of the target?

```python
from ds_utils.preprocess.visualization import visualize_correlations, plot_correlation_dendrogram
from ds_utils.preprocess.statistics import get_correlated_features

train_num = train_df[num_cols + [target]]
corr_matrix = train_num.corr()

ax_corr = visualize_correlations(corr_matrix)
plt.close()
ax_dendro = plot_correlation_dendrogram(corr_matrix)
plt.close()

collinear_pairs = get_correlated_features(corr_matrix, num_cols, target, threshold=0.85)
```
- **What to look for**: Pairs with high level_0_level_1_corr and differing target correlations.
- **Action**: Between level_0 and level_1, drop the feature with smaller absolute correlation to the target.
- **Caution**: Linear correlation only measures monotonic relationships and misses strong non-linear signals.

#### Step 6: Non-linear Association Ranking (`compute_mutual_information`)
Question: which features share the strongest non-linear mutual dependence with the target?

```python
from ds_utils.preprocess.statistics import compute_mutual_information

mi_df = compute_mutual_information(train_df, num_cols, target, random_state=42)
print(mi_df.head(10))
```
- **What to look for**: Features with high MI scores that might have shown weak linear correlation in Step 5.
- **Action**: Retain non-linearly dependent features for tree ensembles or non-linear models.
- **Caution**: Mutual information estimates can vary with continuous binning; use a fixed random_state.

#### Step 7: Dimensionality Assessment (`plot_pca_explained_variance`)
Question: how many principal components are required to capture 90%+ of the variance in numeric features?

```python
from ds_utils.preprocess.visualization import plot_pca_explained_variance

ax_pca = plot_pca_explained_variance(train_df[num_cols], use_scaling=True)
plt.close()
```
- **What to look for**: The cumulative variance curve crossing 90% or 95%.
- **Action**: Fit PCA on train_df[num_cols] and transform test_df[num_cols] if dimensionality reduction is desired.
- **Caution**: Always set use_scaling=True when features have different units or variances.

---

## Examples

### Runnable demo on a bundled dataset (replace the loading lines with your own data)

```python
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from ds_utils.preprocess.visualization import (
    visualize_feature,
    plot_features_interaction,
    visualize_correlations,
    plot_correlation_dendrogram,
    plot_pca_explained_variance,
)
from ds_utils.preprocess.statistics import (
    extract_statistics_dataframe_per_label,
    get_correlated_features,
    compute_mutual_information,
)

# 1. Load bundled dataset and split
data = load_breast_cancer(as_frame=True)
df = data.frame
target = "target"
train_df, test_df = train_test_split(df, test_size=0.25, stratify=df[target], random_state=42)
num_cols = train_df.select_dtypes("number").columns.drop(target).tolist()

# 2. Univariate & bivariate visualizations
ax_u = visualize_feature(train_df["mean radius"])
plt.close()
ax_bi = plot_features_interaction(train_df, "mean radius", "mean texture")
plt.close()

# 3. Target-aware statistics on train
stats = extract_statistics_dataframe_per_label(train_df, "mean radius", target)

# 4. Correlation matrix, dendrogram, and collinearity pruning (threshold=0.95 gives 15 pairs)
corr = train_df[num_cols + [target]].corr()
ax_c = visualize_correlations(corr)
plt.close()
ax_d = plot_correlation_dendrogram(corr)
plt.close()

pairs = get_correlated_features(corr, num_cols, target, threshold=0.95)
to_drop = set()
for _, row in pairs.iterrows():
    if abs(row["level_0_target_corr"]) >= abs(row["level_1_target_corr"]):
        to_drop.add(row["level_1"])
    else:
        to_drop.add(row["level_0"])
selected_num = [c for c in num_cols if c not in to_drop]

# 5. Mutual information ranking & PCA explained variance
mi_ranking = compute_mutual_information(train_df, selected_num, target, random_state=42)
ax_pca = plot_pca_explained_variance(train_df[selected_num], use_scaling=True)
plt.close()

# 6. Apply selected features downstream to test
X_train_final = train_df[selected_num]
X_test_final = test_df[selected_num]
```

## Limitations

- **Univariate only**: `visualize_feature` examines one Series at a time; it cannot color or group by a target label.
- **Pre-computed matrix required**: `visualize_correlations`, `plot_correlation_dendrogram`, and `get_correlated_features` require pre-computed correlation matrices (`df.corr()`), not raw dataframes.
- **Numeric PCA input**: `plot_pca_explained_variance` requires all input columns to be numeric; encode categoricals before passing.

## Troubleshooting

| Error / Symptom | Root Cause | Solution |
|---|---|---|
| `AttributeError: 'DataFrame' object has no attribute 'unique'` | Passed DataFrame `df[['col']]` to `visualize_feature` | Pass a 1D Series `df['col']` |
| `KeyError: 'target'` in `get_correlated_features` | Target column was omitted when computing `df.corr()` | Ensure `df[[...features, target]].corr()` includes the target column |
| `ValueError: All columns in X must be numeric.` in PCA | DataFrame contains non-numeric or un-encoded categorical columns | Filter or encode non-numeric columns before calling `plot_pca_explained_variance` |
| Suboptimal test performance / Overfitting | Leakage from running `get_correlated_features` or MI on full data | Perform train/test split first and calculate statistics exclusively on training data |
