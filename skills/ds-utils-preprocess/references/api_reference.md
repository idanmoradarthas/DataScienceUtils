# Preprocess API Reference

Exhaustive parameter lists, function signatures, return specifications, and common mistakes for `ds_utils.preprocess`.

## Import Paths

```python
from ds_utils.preprocess.visualization import (
    visualize_feature,
    visualize_correlations,
    plot_correlation_dendrogram,
    plot_features_interaction,
    plot_pca_explained_variance,
)
from ds_utils.preprocess.statistics import (
    get_correlated_features,
    extract_statistics_dataframe_per_label,
    compute_mutual_information,
)
```

---

## visualize_feature

Visualizes a single feature's distribution based on its data type.

### Signature
```python
visualize_feature(
    series: pd.Series,
    remove_na: bool = False,
    include_outliers: bool = True,
    outlier_iqr_multiplier: float = 1.5,
    first_day_of_week: str = "Monday",
    show_counts: bool = True,
    order: Union[str, list, None] = None,
) -> matplotlib.axes.Axes
```

### Parameters
- `series` (*pd.Series*): The feature to visualize.
- `remove_na` (*bool*, default `False`): Whether to explicitly remove NAs before plotting.
- `include_outliers` (*bool*, default `True`): For float features, whether to include outliers in the violin plot.
- `outlier_iqr_multiplier` (*float*, default `1.5`): IQR multiplier for outlier fence computation if `include_outliers=False`.
- `first_day_of_week` (*str*, default `"Monday"`): Start day for datetime heatmaps (e.g., `"Monday"` or `"Sunday"`).
- `show_counts` (*bool*, default `True`): Whether to display count labels on categorical bar charts.
- `order` (*str* or *list*, default `None`): Custom sorting for categories (e.g. `"count_desc"` or `["High", "Low"]`).

### Returns
- `matplotlib.axes.Axes`: The matplotlib Axes containing the plot.

### Common Mistakes
- Passing a DataFrame (`df[["col"]]`) instead of a Series (`df["col"]`).
- Expecting a target overlay: `visualize_feature` is strictly univariate.
- Features with >10 unique values will show the top 10 categories, grouping the rest into `"Other values"`.

---

## visualize_correlations

Visualizes a correlation matrix as an annotated heatmap.

### Signature
```python
visualize_correlations(
    correlation_matrix: pd.DataFrame,
    annot: bool = True,
    fmt: str = ".2f",
    cmap: str = "coolwarm",
    ax: Optional[matplotlib.axes.Axes] = None,
    **kwargs,
) -> matplotlib.axes.Axes
```

### Parameters
- `correlation_matrix` (*pd.DataFrame*): Pre-computed correlation matrix (e.g. `df.corr()`).
- `annot` (*bool*, default `True`): Whether to write the data value in each cell.
- `fmt` (*str*, default `".2f"`): String formatting code to use when adding annotations.
- `cmap` (*str*, default `"coolwarm"`): Colormap name.
- `ax` (*matplotlib.axes.Axes*, optional): Target axes for rendering.

### Returns
- `matplotlib.axes.Axes`: The matplotlib Axes containing the heatmap.

### Common Mistakes
- Passing raw data `df` instead of pre-computed `df.corr()`.

---

## plot_correlation_dendrogram

Plots a hierarchical clustering dendrogram based on a correlation matrix.

### Signature
```python
plot_correlation_dendrogram(
    correlation_matrix: pd.DataFrame,
    ax: Optional[matplotlib.axes.Axes] = None,
    **kwargs,
) -> matplotlib.axes.Axes
```

### Parameters
- `correlation_matrix` (*pd.DataFrame*): Pre-computed correlation matrix.
- `ax` (*matplotlib.axes.Axes*, optional): Target axes for rendering.

### Returns
- `matplotlib.axes.Axes`: The matplotlib Axes containing the dendrogram.

### Common Mistakes
- Passing the raw DataFrame instead of `df.corr()`.

---

## plot_features_interaction

Plots the joint distribution and relationship between two features.

### Signature
```python
plot_features_interaction(
    data_frame: pd.DataFrame,
    feature_1: str,
    feature_2: str,
    remove_na: bool = False,
    ax: Optional[matplotlib.axes.Axes] = None,
) -> matplotlib.axes.Axes
```

### Parameters
- `data_frame` (*pd.DataFrame*): The dataset.
- `feature_1` (*str*): First feature column name.
- `feature_2` (*str*): Second feature column name.
- `remove_na` (*bool*, default `False`): Whether to remove NA/null values before plotting. When `False`, missing values are shown as rug plots or separate bars.
- `ax` (*matplotlib.axes.Axes*, optional): Target axes for rendering.

### Dtype Interaction Matrix
| feature_1 ↓  feature_2 → | Numeric | Categorical / Bool | Datetime |
|---|---|---|---|
| **Numeric** | Scatter plot | Box plot | Line plot |
| **Categorical / Bool** | Box plot | Shared histogram | Violin plot |
| **Datetime** | Line plot | Violin plot | Line plot |

### Returns
- `matplotlib.axes.Axes`: The matplotlib Axes.

---

## plot_pca_explained_variance

Plots the cumulative explained variance ratio of PCA components.

### Signature
```python
plot_pca_explained_variance(
    X: pd.DataFrame,
    use_scaling: bool = True,
    scaler: Optional[Any] = None,
    legend_loc: str = "lower right",
    ax: Optional[matplotlib.axes.Axes] = None,
    pca_kwargs: Optional[dict] = None,
) -> matplotlib.axes.Axes
```

### Parameters
- `X` (*pd.DataFrame*): The dataset containing numerical features.
- `use_scaling` (*bool*, default `True`): Whether to scale data before PCA.
- `scaler` (*TransformerMixin*, optional): Scaler instance (default: `StandardScaler()`).
- `legend_loc` (*str*, default `"lower right"`): Location of the legend.
- `ax` (*matplotlib.axes.Axes*, optional): Target axes.
- `pca_kwargs` (*dict*, optional): Additional kwargs passed to `sklearn.decomposition.PCA`.

### Returns
- `matplotlib.axes.Axes`: The matplotlib Axes.

### Common Mistakes
- Passing DataFrames with non-numeric or un-encoded categorical columns.

---

## get_correlated_features

Extracts pairs of highly correlated features and their correlation to the target column.

### Signature
```python
get_correlated_features(
    correlation_matrix: pd.DataFrame,
    features: List[str],
    target: str,
    threshold: float = 0.95,
) -> pd.DataFrame
```

### Parameters
- `correlation_matrix` (*pd.DataFrame*): Pre-computed correlation matrix.
- `features` (*list of str*): List of feature names to consider.
- `target` (*str*): Target column name in the correlation matrix.
- `threshold` (*float*, default `0.95`): Minimum absolute correlation between two features to report.

### Returns
- `pd.DataFrame`: Columns: `['level_0', 'level_1', 'level_0_level_1_corr', 'level_0_target_corr', 'level_1_target_corr']`.

### Output Example
| level_0 | level_1 | level_0_level_1_corr | level_0_target_corr | level_1_target_corr |
|---------|---------|----------------------|---------------------|---------------------|
| featureA| featureB| 0.95                 | 0.51                | 0.48                |

### Decision Rule
When `featureA` and `featureB` have high collinearity (`level_0_level_1_corr` > threshold), drop the feature with the lower correlation to the target (`min(abs(level_0_target_corr), abs(level_1_target_corr))`).

---

## extract_statistics_dataframe_per_label

Calculates summary statistics for a numerical feature grouped by label values.

### Signature
```python
extract_statistics_dataframe_per_label(
    df: pd.DataFrame,
    feature_name: str,
    label_name: str,
) -> pd.DataFrame
```

### Parameters
- `df` (*pd.DataFrame*): Input dataset.
- `feature_name` (*str*): Numerical feature to describe.
- `label_name` (*str*): Categorical label or target column.

### Returns
- `pd.DataFrame`: Grouped summary table.

### Output Example
| category | count | null_count | mean  | min | 1_percentile | 5_percentile | 25_percentile | median | 75_percentile | 95_percentile | 99_percentile | max |
|----------|-------|------------|-------|-----|--------------|--------------|---------------|--------|---------------|---------------|---------------|-----|
| Group_A  | 100   | 0          | 25.5  | 10  | 11.2         | 12.0         | 15.0          | 25.0   | 35.0          | 45.0          | 49.0          | 50  |
| Group_B  | 150   | 2          | 45.0  | 20  | 21.0         | 22.0         | 30.0          | 45.0   | 60.0          | 75.0          | 80.0          | 85  |

---

## compute_mutual_information

Computes mutual information scores between features and a target label.

### Signature
```python
compute_mutual_information(
    df: pd.DataFrame,
    features: List[str],
    label_col: str,
    random_state: Optional[int] = None,
) -> pd.DataFrame
```

### Parameters
- `df` (*pd.DataFrame*): The dataset.
- `features` (*list of str*): Feature column names to test.
- `label_col` (*str*): Target column name.
- `random_state` (*int*, optional): Random state for reproducibility.

### Returns
- `pd.DataFrame`: Columns `['feature_name', 'mi_score']`, sorted descending by `mi_score`.

### Output Example
| feature_name | mi_score |
|--------------|----------|
| feature1     | 0.245    |
| feature3     | 0.182    |
| feature2     | 0.091    |
