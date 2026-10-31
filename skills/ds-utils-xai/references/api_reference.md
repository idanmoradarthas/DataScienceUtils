# XAI API Reference

Exhaustive parameter lists, function signatures, return specifications, and common mistakes for `ds_utils.xai`.

## Import Paths

```python
from ds_utils.xai import (
    plot_features_importance,
    draw_dot_data,
    draw_tree,
    generate_decision_paths,
)
```

---

## plot_features_importance

Plots a horizontal/vertical bar chart ranking features by their calculated model importance. Filters out features with zero importance automatically.

### Signature
```python
plot_features_importance(
    feature_names: Union[np.ndarray, List[str]],
    feature_importance: Union[np.ndarray, List[float]],
    *,
    ax: Optional[matplotlib.axes.Axes] = None,
    **kwargs,
) -> matplotlib.axes.Axes
```

### Parameters
- `feature_names`: Array-like of string feature names with shape `(n_features,)`. Must strictly align with the column order supplied during model `.fit()`.
- `feature_importance`: Array-like of importance values with shape `(n_features,)` (e.g. `clf.feature_importances_`).
- `ax`: Optional target matplotlib axes.
- `**kwargs`: Forwarded directly to `matplotlib.axes.Axes.bar()`.

### Returns
- `matplotlib.axes.Axes`: The matplotlib Axes containing the bar plot.

### Common Mistakes
- Passing linear model coefficients (`model.coef_`): Coefficients have signed directional meanings and different scales; this function is tailored for positive `feature_importances_`.
- Mismatched ordering: If `feature_names` does not align with the order of columns in `X_train`, features will be mislabeled.

---

## draw_dot_data & draw_tree

Renders a decision tree diagram into a matplotlib Axes using Graphviz.

### Signatures
```python
draw_dot_data(
    dot_data: str,
    *,
    ax: Optional[matplotlib.axes.Axes] = None,
    **kwargs,
) -> matplotlib.axes.Axes

draw_tree(
    tree: BaseDecisionTree,
    feature_names: Optional[List[str]] = None,
    class_names: Optional[List[str]] = None,
    *,
    ax: Optional[matplotlib.axes.Axes] = None,
    **kwargs,
) -> matplotlib.axes.Axes
```

### Parameters
- `dot_data`: Non-empty Graphviz DOT language string (e.g. from `sklearn.tree.export_graphviz(..., out_file=None)`).
- `tree`: Scikit-learn `DecisionTreeClassifier` or `DecisionTreeRegressor` instance.
- `feature_names`: Optional list of feature names.
- `class_names`: Optional list of class names.
- `ax`: Optional target matplotlib axes.
- `**kwargs`: Forwarded to `matplotlib.axes.Axes.imshow()`.

### Returns
- `matplotlib.axes.Axes`

### Prerequisites
- Requires system Graphviz binaries (`dot`) installed in `PATH` and Python package `pydotplus`.
- Note: `draw_tree` issues a `DeprecationWarning` recommending native `sklearn.tree.plot_tree`. It remains supported for legacy Graphviz rendering pipelines.

---

## generate_decision_paths

Generates Python-syntax `if/else` function code representing the decision logic of a fitted decision tree.

### Signature
```python
generate_decision_paths(
    classifier: BaseDecisionTree,
    feature_names: Optional[List[str]] = None,
    class_names: Optional[List[str]] = None,
    tree_name: Optional[str] = None,
    indent_char: str = "\t",
) -> str
```

### Parameters
- `classifier`: Fitted `DecisionTreeClassifier`.
- `feature_names`: Optional list of feature names.
- `class_names`: Optional list of class names.
- `tree_name`: Name for the generated Python function header.
- `indent_char`: Character used for indentation (default: `"\t"`).

### Returns
- `str`: Executable Python code defining the tree rules.
