# XAI API Reference

Exhaustive parameter lists, function signatures, return specifications, and common mistakes for `ds_utils.xai`.

## Import Paths

```python
from ds_utils.xai import (
    plot_features_importance,
    draw_dot_data,
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

## draw_dot_data

Renders a decision tree diagram from Graphviz DOT data into a matplotlib Axes.

### Signature
```python
draw_dot_data(
    dot_data: str,
    *,
    ax: Optional[matplotlib.axes.Axes] = None,
    **kwargs,
) -> matplotlib.axes.Axes
```

### Parameters
- `dot_data`: Non-empty Graphviz DOT language string (e.g. from `sklearn.tree.export_graphviz(..., out_file=None)`).
- `ax`: Optional target matplotlib axes.
- `**kwargs`: Forwarded to `matplotlib.axes.Axes.imshow()`.

### Returns
- `matplotlib.axes.Axes`

### Prerequisites
- Requires system Graphviz binaries (`dot`) installed in `PATH` and Python package `pydotplus`.

---

## Related scikit-learn tools

These scikit-learn functions complement `draw_dot_data` and work on a single decision tree:
- `plot_tree(decision_tree, feature_names=..., class_names=..., filled=True, ax=...)`: draws the tree with matplotlib only, no Graphviz binary needed.
- `export_text(decision_tree, feature_names=...)`: returns the tree rules as indented text; branches deeper than `max_depth` (default 10) are truncated.
