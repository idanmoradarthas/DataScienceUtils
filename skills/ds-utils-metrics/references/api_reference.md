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
    y_test: numpy.ndarray,
    y_pred: numpy.ndarray,
    labels: List[Union[str, int]],
    sample_weight: Optional[List[float]] = None,
    annot_kws: Optional[Dict] = None,
    cbar: bool = True,
    cbar_kws: Optional[Dict] = None,
    **kwargs,
) -> matplotlib.axes._axes.Axes
```

### Parameters
- `y_test`: Ground truth targets (`numpy.ndarray`).
- `y_pred`: Estimated targets (`numpy.ndarray`).
- `labels`: Python list of labels used to index the matrix (`List[Union[str, int]]`, e.g., `[0, 1]`). Do not pass as a numpy array.
- `sample_weight` (*optional*, default `None`): Sample weights list.
- `annot_kws` (*optional*, default `None`): Keyword arguments for text annotations.
- `cbar` (*bool*, default `True`): Whether to draw a colorbar.
- `cbar_kws` (*optional*, default `None`): Keyword arguments for colorbar.
- `**kwargs`: Forwarded directly to `seaborn.heatmap`. Note: `plot_confusion_matrix` does not take an `ax` parameter; passing `ax` raises `TypeError`.

### Returns
- `matplotlib.axes._axes.Axes`

---

## plot_roc_curve_with_thresholds_annotations

Plots interactive ROC curves with threshold hover/annotations using Plotly.

### Signature
```python
plot_roc_curve_with_thresholds_annotations(
    y_true: numpy.ndarray,
    classifiers_names_and_scores_dict: Dict[str, numpy.ndarray],
    *,
    positive_label: Union[int, float, bool, str, NoneType] = None,
    sample_weight: Optional[numpy.ndarray] = None,
    drop_intermediate: bool = True,
    average: Optional[str] = "macro",
    max_fpr: Optional[float] = None,
    multi_class: str = "raise",
    labels: Optional[numpy.ndarray] = None,
    fig: Optional[plotly.graph_objs._figure.Figure] = None,
    mode: Optional[str] = "lines+markers",
    add_random_classifier_line: bool = True,
    random_classifier_line_kw: Optional[Dict] = None,
    show_legend: bool = True,
    **kwargs,
) -> plotly.graph_objs._figure.Figure
```

### Parameters
- `y_true`: True binary or multiclass labels (`numpy.ndarray`).
- `classifiers_names_and_scores_dict`: Dictionary mapping model name to positive class probabilities: `{name: y_proba[:, positive_class_idx]}`.
- `positive_label` (*keyword-only*, default `None`): Label representing positive class. Inferred if `None`.
- `sample_weight` (*keyword-only*, optional, default `None`): Sample weights array.
- `drop_intermediate` (*keyword-only*, default `True`): Drop sub-optimal thresholds.
- `average` (*keyword-only*, optional, default `"macro"`): Multiclass averaging mode.
- `max_fpr` (*keyword-only*, optional, default `None`): Maximum false positive rate for partial AUC.
- `multi_class` (*keyword-only*, default `"raise"`): Multiclass mode handling.
- `labels` (*keyword-only*, optional, default `None`): Class labels for multiclass.
- `fig` (*keyword-only*, optional, default `None`): Existing Plotly figure to draw on.
- `mode` (*keyword-only*, default `"lines+markers"`): Plotly trace drawing mode.
- `add_random_classifier_line` (*keyword-only*, default `True`): Whether to plot diagonal chance baseline.
- `random_classifier_line_kw` (*keyword-only*, optional, default `None`): Styling kwargs for chance baseline.
- `show_legend` (*keyword-only*, default `True`): Whether to show the legend.
- `**kwargs`: Forwarded to underlying figure configuration.

### Returns
- `plotly.graph_objs._figure.Figure`: Display with `fig.show()`.

---

## plot_precision_recall_curve_with_thresholds_annotations

Plots interactive Precision-Recall curves with threshold annotations using Plotly.

### Signature
```python
plot_precision_recall_curve_with_thresholds_annotations(
    y_true: numpy.ndarray,
    classifiers_names_and_scores_dict: Dict[str, numpy.ndarray],
    *,
    positive_label: Union[int, float, bool, str, NoneType] = None,
    sample_weight: Optional[numpy.ndarray] = None,
    drop_intermediate: bool = True,
    fig: Optional[plotly.graph_objs._figure.Figure] = None,
    mode: Optional[str] = "lines+markers",
    plot_chance_level: bool = False,
    chance_level_kw: Optional[Dict] = None,
    show_legend: bool = True,
    **kwargs,
) -> plotly.graph_objs._figure.Figure
```

### Parameters
- `y_true`: True binary labels (`numpy.ndarray`).
- `classifiers_names_and_scores_dict`: Dictionary mapping model name to positive class probabilities: `{name: y_proba[:, positive_class_idx]}`.
- `positive_label` (*keyword-only*, default `None`): Target positive label.
- `sample_weight` (*keyword-only*, optional, default `None`): Sample weights array.
- `drop_intermediate` (*keyword-only*, default `True`): Whether to drop sub-optimal thresholds.
- `fig` (*keyword-only*, optional, default `None`): Existing Plotly figure.
- `mode` (*keyword-only*, default `"lines+markers"`): Plotly trace drawing mode.
- `plot_chance_level` (*keyword-only*, default `False`): Whether to draw the chance prevalence baseline.
- `chance_level_kw` (*keyword-only*, optional, default `None`): Styling kwargs for chance baseline.
- `show_legend` (*keyword-only*, default `True`): Whether to show the legend.
- `**kwargs`: Forwarded to figure configuration.

### Returns
- `plotly.graph_objs._figure.Figure`: Display with `fig.show()`.

---

## visualize_accuracy_grouped_by_probability

Visualizes prediction accuracy grouped across probability bins to inspect calibration.

### Signature
```python
visualize_accuracy_grouped_by_probability(
    y_test: numpy.ndarray,
    labeled_class: Union[str, int],
    probabilities: numpy.ndarray,
    threshold: float = 0.5,
    display_breakdown: bool = False,
    bins: Union[int, Sequence[float], pandas.IntervalIndex, NoneType] = None,
    *,
    ax: Optional[matplotlib.axes._axes.Axes] = None,
    **kwargs,
) -> matplotlib.axes._axes.Axes
```

### Parameters
- `y_test`: Ground truth targets (`numpy.ndarray`).
- `labeled_class`: Class to evaluate (`Union[str, int]`).
- `probabilities`: **1-D** array of classifier probabilities for `labeled_class` (`numpy.ndarray` of shape `(n_samples,)`). Retrieve via `idx = int(np.where(clf.classes_ == labeled_class)[0][0]); probabilities = clf.predict_proba(X)[:, idx]`.
- `threshold` (*float*, default `0.5`): Decision threshold.
- `display_breakdown` (*bool*, default `False`): Show TP/FP/TN/FN breakdown.
- `bins` (*optional*, default `None`): Custom probability bin intervals (defaults to 10 bins from 0 to 1).
- `ax` (*keyword-only*, optional, default `None`): Target matplotlib axes.
- `**kwargs`: Forwarded to plotting function.

### Returns
- `matplotlib.axes._axes.Axes`

---

## plot_metric_growth_per_labeled_instances

Computes learning curves by evaluating classifiers trained on incremental dataset fractions.

### Signature
```python
plot_metric_growth_per_labeled_instances(
    X_train: numpy.ndarray,
    y_train: numpy.ndarray,
    X_test: numpy.ndarray,
    y_test: numpy.ndarray,
    classifiers_dict: Dict[str, sklearn.base.ClassifierMixin],
    n_samples: Optional[List[int]] = None,
    quantiles: Optional[List[float]] = [
        0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.39999999999999997,
        0.44999999999999996, 0.49999999999999994, 0.5499999999999999,
        0.6, 0.65, 0.7, 0.75, 0.7999999999999999, 0.85, 0.9, 0.95, 1.0
    ],
    metric: Callable[[numpy.ndarray, numpy.ndarray], float] = accuracy_score,
    random_state: Union[int, numpy.random.mtrand.RandomState, NoneType] = None,
    n_jobs: Optional[int] = None,
    verbose: int = 0,
    pre_dispatch: Union[int, str, NoneType] = "2*n_jobs",
    *,
    ax: Optional[matplotlib.axes._axes.Axes] = None,
    **kwargs,
) -> matplotlib.axes._axes.Axes
```

### Parameters
- `X_train`: Training feature matrix (`numpy.ndarray`).
- `y_train`: Training labels (`numpy.ndarray`).
- `X_test`: Test feature matrix (`numpy.ndarray`).
- `y_test`: Test labels (`numpy.ndarray`).
- `classifiers_dict`: Dictionary of `{name: unfitted_estimator}`. Pass unfitted estimators; fitted instances raise errors.
- `n_samples` (*optional*, default `None`): Specific number of training samples to evaluate.
- `quantiles` (*optional*): 20 percentage steps from 0.05 to 1.0 used when `n_samples` is None.
- `metric` (default `accuracy_score`): Scikit-learn evaluation metric callable.
- `random_state` (*optional*, default `None`): Random state seed.
- `n_jobs` (*optional*, default `None`): Number of parallel jobs.
- `verbose` (*int*, default `0`): Verbosity level.
- `pre_dispatch` (*default* `"2*n_jobs"`): Number of pre-dispatched jobs.
- `ax` (*keyword-only*, optional, default `None`): Target axes.
- `**kwargs`: Forwarded to plotting.

### Returns
- `matplotlib.axes._axes.Axes`

---

## plot_error_analysis_chart

Visualizes the distribution of predicted probabilities split by correct predictions, false positives, and false negatives.

### Signature
```python
plot_error_analysis_chart(
    y_true: Union[numpy.ndarray, List],
    y_pred: Union[numpy.ndarray, List],
    y_proba: Union[numpy.ndarray, List],
    positive_class: Union[int, str],
    *,
    classes: Optional[List] = None,
    ax: Optional[matplotlib.axes._axes.Axes] = None,
    **kwargs,
) -> matplotlib.axes._axes.Axes
```

### Parameters
- `y_true`: True labels.
- `y_pred`: Predicted labels.
- `y_proba`: Predicted probabilities (1D positive probabilities for binary, 2D matrix for multiclass).
- `positive_class`: Positive class indicator.
- `classes` (*keyword-only*, optional, default `None`): Ordered class labels matching columns of `y_proba` when 2D.
- `ax` (*keyword-only*, optional, default `None`): Target axes.
- `**kwargs`: Forwarded to `seaborn.violinplot`.

### Returns
- `matplotlib.axes._axes.Axes`

---

## generate_error_analysis_report

Groups predictions across feature intervals/categories and tabulates error rates.

### Signature
```python
generate_error_analysis_report(
    X: pandas.DataFrame,
    y_true: numpy.ndarray,
    y_pred: numpy.ndarray,
    feature_columns: Optional[List[str]] = None,
    bins: int = 10,
    threshold: float = 0.5,
    min_count: int = 1,
    sort_metric: str = "error_rate",
    ascending: bool = False,
) -> pandas.DataFrame
```

### Parameters
- `X`: Input features (`pandas.DataFrame`).
- `y_true`: True labels (`numpy.ndarray`).
- `y_pred`: Predicted labels (`numpy.ndarray`).
- `feature_columns` (*optional*, default `None`): Subset of columns to analyze.
- `bins` (*int*, default `10`): Number of bins for numeric features.
- `threshold` (*float*, default `0.5`): Probability threshold.
- `min_count` (*int*, default `1`): Minimum instances per group to report.
- `sort_metric` (*str*, default `"error_rate"`): Metric to sort by.
- `ascending` (*bool*, default `False`): Sort direction.

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
    y_true: numpy.ndarray,
    y_pred: numpy.ndarray,
    baseline: Optional[numpy.ndarray] = None,
    sample_weight: Optional[numpy.ndarray] = None,
    handle_equal: str = "exclude",
) -> float

directional_bias_score(
    y_true: numpy.ndarray,
    y_pred: numpy.ndarray,
    sample_weight: Optional[numpy.ndarray] = None,
    handle_equal: str = "exclude",
) -> float
```

---

## regression_auc_score & plot_rec_curve_with_annotations

Regression Error Characteristic (REC) curve visualization and Area Over Curve (AOC) metric. Lower AOC is superior.

### Signatures
```python
regression_auc_score(
    y_true: numpy.ndarray,
    y_pred: numpy.ndarray,
    sample_weight: Optional[numpy.ndarray] = None,
    normalize: bool = True,
) -> float

plot_rec_curve_with_annotations(
    y_true: numpy.ndarray,
    regressors_names_and_predictions_dict: Dict[str, numpy.ndarray],
    *,
    sample_weight: Optional[numpy.ndarray] = None,
    normalize_auc: bool = True,
    fig: Optional[plotly.graph_objs._figure.Figure] = None,
    mode: Optional[str] = "lines+markers",
    show_legend: bool = True,
    **kwargs,
) -> plotly.graph_objs._figure.Figure
```
