# Metrics API Reference

Exhaustive parameter lists, function signatures, return specifications, and common mistakes for `ds_utils.metrics`.

## Import Paths

```python
from ds_utils.metrics.confusion_matrix import plot_confusion_matrix
from ds_utils.metrics.learning_curves import plot_metric_growth_per_labeled_instances
from ds_utils.metrics.probability_analysis import (
    visualize_accuracy_grouped_by_probability,
    plot_error_analysis_chart,
)
from ds_utils.metrics.curves import (
    plot_roc_curve_with_thresholds_annotations,
    plot_precision_recall_curve_with_thresholds_annotations,
)
from ds_utils.metrics.error_analysis import generate_error_analysis_report
from ds_utils.metrics.time_series import (
    directional_accuracy_score,
    directional_bias_score,
)
from ds_utils.metrics.regression import (
    regression_auc_score,
    plot_rec_curve_with_annotations,
)
```

---

## plot_confusion_matrix

Computes and plots a confusion matrix along with FPR, FNR, Accuracy, and F1 score.

### Signature
```python
plot_confusion_matrix(
    y_test: Union[np.ndarray, List],
    y_pred: Union[np.ndarray, List],
    labels: List[Any],
    sample_weight: Optional[np.ndarray] = None,
    annot_kws: Optional[Dict] = None,
    cbar: bool = True,
    cbar_kws: Optional[Dict] = None,
    ax: Optional[matplotlib.axes.Axes] = None,
) -> matplotlib.axes.Axes
```

### Parameters
- `y_test`: Ground truth targets.
- `y_pred`: Estimated targets.
- `labels`: Python list of labels used to index the matrix (e.g., `[0, 1]`). Do not pass as a numpy array.
- `sample_weight`: Sample weights array.
- `annot_kws`: Kwargs for annotation text.
- `cbar`: Whether to draw colorbar.
- `cbar_kws`: Kwargs for colorbar.
- `ax`: Target matplotlib axes.

### Returns
- `matplotlib.axes.Axes`

---

## plot_roc_curve_with_thresholds_annotations

Plots interactive ROC curves with threshold hover/annotations using Plotly.

### Signature
```python
plot_roc_curve_with_thresholds_annotations(
    y_true: Union[np.ndarray, List],
    classifiers_names_and_scores_dict: Dict[str, np.ndarray],
    positive_label: Union[int, str] = 1,
    sample_weight: Optional[np.ndarray] = None,
    drop_intermediate: bool = True,
    average: str = "macro",
    max_fpr: Optional[float] = None,
    multi_class: str = "raise",
    mode: str = "lines+markers",
    add_random_classifier_line: bool = True,
    random_classifier_line_kw: Optional[Dict] = None,
) -> plotly.graph_objects.Figure
```

### Parameters
- `y_true`: True binary labels.
- `classifiers_names_and_scores_dict`: Dictionary mapping model name to positive class probabilities: `{name: y_proba[:, positive_class_idx]}`.
- `positive_label`: Label representing positive class.
- `drop_intermediate`: Drop sub-optimal thresholds.
- `mode`: Plotly trace drawing mode (`"lines+markers"`).

### Returns
- `plotly.graph_objects.Figure`: Display with `fig.show()`.

---

## plot_precision_recall_curve_with_thresholds_annotations

Plots interactive Precision-Recall curves with threshold annotations using Plotly.

### Signature
```python
plot_precision_recall_curve_with_thresholds_annotations(
    y_true: Union[np.ndarray, List],
    classifiers_names_and_scores_dict: Dict[str, np.ndarray],
    positive_label: Union[int, str] = 1,
    sample_weight: Optional[np.ndarray] = None,
    drop_intermediate: bool = True,
    mode: str = "lines+markers",
    plot_chance_level: bool = False,
    chance_level_kw: Optional[Dict] = None,
) -> plotly.graph_objects.Figure
```

### Parameters
- `y_true`: True binary labels.
- `classifiers_names_and_scores_dict`: Dictionary `{name: y_proba[:, positive_class_idx]}`.
- `positive_label`: Target positive label.
- `plot_chance_level`: Whether to draw the chance prevalence baseline.

### Returns
- `plotly.graph_objects.Figure`: Display with `fig.show()`.

---

## visualize_accuracy_grouped_by_probability

Visualizes prediction accuracy grouped across probability bins to inspect calibration.

### Signature
```python
visualize_accuracy_grouped_by_probability(
    y_test: Union[np.ndarray, List],
    labeled_class: Union[int, str],
    probabilities: np.ndarray,
    threshold: float = 0.5,
    display_breakdown: bool = False,
    bins: Optional[List[float]] = None,
    ax: Optional[matplotlib.axes.Axes] = None,
) -> matplotlib.axes.Axes
```

### Parameters
- `y_test`: Ground truth targets.
- `labeled_class`: Class to evaluate (e.g., `1`).
- `probabilities`: **1-D** array of classifier probabilities for `labeled_class` (`np.ndarray` of shape `(n_samples,)`). Retrieve via `idx = np.where(clf.classes_ == labeled_class)[0][0]; probabilities = clf.predict_proba(X)[:, idx]`. Passing a 2-D matrix raises `ValueError: Per-column arrays must each be 1-dimensional`.
- `threshold`: Decision threshold.
- `display_breakdown`: Show TP/FP/TN/FN breakdown.
- `bins`: Custom probability bin intervals.

### Returns
- `matplotlib.axes.Axes`

---

## plot_metric_growth_per_labeled_instances

Computes learning curves by evaluating a model trained on incremental dataset fractions.

### Signature
```python
plot_metric_growth_per_labeled_instances(
    x_train: Any,
    y_train: Any,
    x_test: Any,
    y_test: Any,
    classifiers: Dict[str, Any],
    n_samples: Optional[List[int]] = None,
    quantiles: Optional[List[float]] = None,
    metric: Callable = accuracy_score,
    random_state: Optional[int] = None,
    n_jobs: int = 1,
    verbose: int = 0,
    ax: Optional[matplotlib.axes.Axes] = None,
) -> matplotlib.axes.Axes
```

### Parameters
- `classifiers`: Dictionary of `{name: unfitted_estimator}`. Pass unfitted estimators; fitted instances raise errors or lead to misleading curves.

---

## plot_error_analysis_chart

Visualizes the distribution of predicted probabilities split by correct predictions, false positives, and false negatives.

### Signature
```python
plot_error_analysis_chart(
    y_true: Union[np.ndarray, List],
    y_pred: Union[np.ndarray, List],
    y_proba: np.ndarray,
    positive_class: Any,
    classes: Optional[List[Any]] = None,
    ax: Optional[matplotlib.axes.Axes] = None,
    **kwargs,
) -> matplotlib.axes.Axes
```

---

## generate_error_analysis_report

Groups predictions across feature intervals/categories and tabulates error rates.

### Signature
```python
generate_error_analysis_report(
    X: pd.DataFrame,
    y_true: Union[np.ndarray, List],
    y_pred: Union[np.ndarray, List],
    feature_columns: Optional[List[str]] = None,
    bins: int = 10,
    threshold: float = 0.5,
    min_count: int = 1,
    sort_metric: str = "error_rate",
    ascending: bool = False,
) -> pd.DataFrame
```

### Output Schema
| feature | group | count | error_count | error_rate | accuracy |
|---|---|---|---|---|---|
| age | (30.0, 45.0] | 40 | 10 | 0.25 | 0.75 |

---

## directional_accuracy_score & directional_bias_score

Measures trend direction correctness and systematic forecast skew.

### Signatures
```python
directional_accuracy_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    baseline: Optional[np.ndarray] = None,
    sample_weight: Optional[np.ndarray] = None,
    handle_equal: str = "exclude",
) -> float

directional_bias_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
    handle_equal: str = "exclude",
) -> float
```

---

## regression_auc_score & plot_rec_curve_with_annotations

Regression Error Characteristic (REC) curve visualization and Area Over Curve (AOC) metric. Lower AOC is superior.

### Signatures
```python
regression_auc_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
    normalize: bool = True,
) -> float

plot_rec_curve_with_annotations(
    y_true: np.ndarray,
    regressors_names_and_predictions_dict: Dict[str, np.ndarray],
    sample_weight: Optional[np.ndarray] = None,
    normalize_auc: bool = True,
    fig: Optional[plotly.graph_objects.Figure] = None,
    mode: str = "lines+markers",
    show_legend: bool = True,
) -> plotly.graph_objects.Figure
```
