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
    series: pandas.Series,
    remove_na: bool = False,
    *,
    include_outliers: bool = True,
    outlier_iqr_multiplier: float = 1.5,
    first_day_of_week: str = "Monday",
    show_counts: bool = True,
    order: Union[List[str], str, NoneType] = None,
    ax: Optional[matplotlib.axes._axes.Axes] = None,
    **kwargs,
) -> matplotlib.axes._axes.Axes
```

### Parameters
- `series` (*pandas.Series*): The feature to visualize.
- `remove_na` (*bool*, default `False`): Whether to explicitly remove NAs before plotting.
- `include_outliers` (*keyword-only*, *bool*, default `True`): For float features, whether to include outliers in the violin plot.
- `outlier_iqr_multiplier` (*keyword-only*, *float*, default `1.5`): IQR multiplier for outlier fence computation if `include_outliers=False`.
- `first_day_of_week` (*keyword-only*, *str*, default `"Monday"`): Start day for datetime heatmaps (e.g., `"Monday"` or `"Sunday"`).
- `show_counts` (*keyword-only*, *bool*, default `True`): Whether to display count labels on categorical bar charts.
- `order` (*keyword-only*, *str* or *list*, default `None`): Custom sorting for categories (e.g. `"count_desc"` or `["High", "Low"]`).
- `ax` (*keyword-only*, optional, default `None`): Target axes for rendering.
- `**kwargs`: Forwarded to underlying plotting functions.

### Returns
- `matplotlib.axes._axes.Axes`: The matplotlib Axes containing the plot.

---

## visualize_correlations

Visualizes a correlation matrix as an annotated heatmap.

### Signature
```python
visualize_correlations(
    correlation_matrix: pandas.DataFrame,
    *,
    ax: Optional[matplotlib.axes._axes.Axes] = None,
    **kwargs,
) -> matplotlib.axes._axes.Axes
```

### Parameters
- `correlation_matrix` (*pandas.DataFrame*): Pre-computed correlation matrix (e.g. `df.corr()`).
- `ax` (*keyword-only*, optional, default `None`): Target axes for rendering.
- `**kwargs`: Additional keyword arguments passed directly to `seaborn.heatmap` (e.g., `annot=True`, `cmap="coolwarm"`).

### Returns
- `matplotlib.axes._axes.Axes`: The matplotlib Axes containing the heatmap.

---

## plot_correlation_dendrogram

Plots a hierarchical clustering dendrogram based on a correlation matrix.

### Signature
```python
plot_correlation_dendrogram(
    correlation_matrix: pandas.DataFrame,
    cluster_distance_method: Union[str, Callable] = "average",
    *,
    ax: Optional[matplotlib.axes._axes.Axes] = None,
    **kwargs,
) -> matplotlib.axes._axes.Axes
```

### Parameters
- `correlation_matrix` (*pandas.DataFrame*): Pre-computed correlation matrix.
- `cluster_distance_method` (*Union[str, Callable]*, default `"average"`): Linkage method for hierarchical clustering (e.g., `"average"`, `"single"`, `"complete"`).
- `ax` (*keyword-only*, optional, default `None`): Target axes for rendering.
- `**kwargs`: Forwarded to `scipy.cluster.hierarchy.dendrogram`.

### Returns
- `matplotlib.axes._axes.Axes`: The matplotlib Axes containing the dendrogram.

---

## plot_features_interaction

Plots the joint distribution and relationship between two features.

### Signature
```python
plot_features_interaction(
    data: pandas.DataFrame,
    feature_1: str,
    feature_2: str,
    *,
    remove_na: bool = False,
    include_outliers: bool = True,
    outlier_iqr_multiplier: float = 1.5,
    show_ratios: bool = False,
    ax: Optional[matplotlib.axes._axes.Axes] = None,
    **kwargs,
) -> matplotlib.axes._axes.Axes
```

### Parameters
- `data` (*pandas.DataFrame*): The input dataset.
- `feature_1` (*str*): First feature column name.
- `feature_2` (*str*): Second feature column name.
- `remove_na` (*keyword-only*, *bool*, default `False`): Whether to remove NA/null values before plotting. When `False`, missingness patterns are visualized using marginal rug plots or missing category bars.
- `include_outliers` (*keyword-only*, *bool*, default `True`): Whether to include values outside the IQR fence for categorical-vs-numeric violin plots.
- `outlier_iqr_multiplier` (*keyword-only*, *float*, default `1.5`): Multiplier for IQR fence trimming in categorical-vs-numeric plots.
- `show_ratios` (*keyword-only*, *bool*, default `False`): If `True`, display proportions instead of absolute counts when both features are categorical-like.
- `ax` (*keyword-only*, optional, default `None`): Target axes for rendering.
- `**kwargs`: Forwarded to underlying plotting functions (`seaborn.violinplot`, `Axes.scatter`, etc.).

### Dtype Interaction Matrix
Rebuilt from the function docstring and the project documentation:

| feature_1 ↓  feature_2 → | Numeric | Categorical / Bool | Datetime |
|---|---|---|---|
| **Numeric** | Scatter plot | Violin plot by category | Line/scatter over time |
| **Categorical / Bool** | Violin plot by category | Overlaid histograms per category | Violin plot by category |
| **Datetime** | Line/scatter over time | Violin plot by category | Scatter plot (complete cases) |

### Returns
- `matplotlib.axes._axes.Axes`: The matplotlib Axes.

---

## plot_pca_explained_variance

Plots the cumulative explained variance ratio of PCA components.

### Signature
```python
plot_pca_explained_variance(
    X: pandas.DataFrame,
    use_scaling: bool = True,
    scaler: Optional[sklearn.base.TransformerMixin] = None,
    legend_loc: str = "lower right",
    ax: Optional[matplotlib.axes._axes.Axes] = None,
    pca_kwargs: Optional[dict] = None,
    **kwargs,
) -> matplotlib.axes._axes.Axes
```

### Parameters
- `X` (*pandas.DataFrame*): The dataset containing numerical features.
- `use_scaling` (*bool*, default `True`): Whether to scale data before PCA.
- `scaler` (*Optional[TransformerMixin]*, default `None`): Scaler instance (defaults to `StandardScaler()` if `use_scaling=True`).
- `legend_loc` (*str*, default `"lower right"`): Location of the legend.
- `ax` (*keyword-only*, optional, default `None`): Target axes.
- `pca_kwargs` (*Optional[dict]*, default `None`): Additional kwargs passed to `sklearn.decomposition.PCA`.
- `**kwargs`: Forwarded to underlying axes plot calls.

### Returns
- `matplotlib.axes._axes.Axes`: The matplotlib Axes.

---

## get_correlated_features

Extracts pairs of highly correlated features and their correlation to the target column.

### Signature
```python
get_correlated_features(
    correlation_matrix: pandas.DataFrame,
    features: List[str],
    target_feature: str,
    threshold: float = 0.95,
) -> pandas.DataFrame
```

### Parameters
- `correlation_matrix` (*pandas.DataFrame*): Pre-computed correlation matrix.
- `features` (*List[str]*): List of feature names to consider.
- `target_feature` (*str*): Target column name in the correlation matrix.
- `threshold` (*float*, default `0.95`): Minimum absolute correlation between two features to report.

### Returns
- `pandas.DataFrame`: Columns: `['level_0', 'level_1', 'level_0_level_1_corr', 'level_0_target_corr', 'level_1_target_corr']`.

---

## extract_statistics_dataframe_per_label

Calculates summary statistics for a numerical feature grouped by label values.

### Signature
```python
extract_statistics_dataframe_per_label(
    df: pandas.DataFrame,
    feature_name: str,
    label_name: str,
) -> pandas.DataFrame
```

### Parameters
- `df` (*pandas.DataFrame*): Input dataset.
- `feature_name` (*str*): Numerical feature to describe.
- `label_name` (*str*): Categorical label or target column.

### Returns
- `pandas.DataFrame`: Grouped summary table.

---

## compute_mutual_information

Computes mutual information scores between features and a target label.

### Signature
```python
compute_mutual_information(
    df: pandas.DataFrame,
    features: List[str],
    label_col: str,
    *,
    n_neighbors: int = 3,
    random_state: Union[int, numpy.random.mtrand.RandomState, NoneType] = None,
    n_jobs: Optional[int] = None,
    numerical_imputer: sklearn.base.TransformerMixin = SimpleImputer(),
    discrete_imputer: sklearn.base.TransformerMixin = SimpleImputer(strategy="most_frequent"),
    discrete_encoder: sklearn.base.TransformerMixin = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
) -> pandas.DataFrame
```

### Parameters
- `df` (*pandas.DataFrame*): The dataset.
- `features` (*List[str]*): Feature column names to test.
- `label_col` (*str*): Target column name.
- `n_neighbors` (*keyword-only*, *int*, default `3`): Number of neighbors to use for MI estimation.
- `random_state` (*keyword-only*, optional, default `None`): Random state for reproducibility.
- `n_jobs` (*keyword-only*, optional, default `None`): Number of parallel jobs.
- `numerical_imputer` (*keyword-only*, default `SimpleImputer()`): Transformer for numeric null imputation.
- `discrete_imputer` (*keyword-only*, default `SimpleImputer(strategy="most_frequent")`): Transformer for discrete null imputation.
- `discrete_encoder` (*keyword-only*, default `OrdinalEncoder(...)`): Transformer for discrete encoding.

### Returns
- `pandas.DataFrame`: Columns `['feature_name', 'mi_score']`, sorted descending by `mi_score`.
