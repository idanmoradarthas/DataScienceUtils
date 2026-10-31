---
name: ds-utils-metrics
description: >
  Machine learning model evaluation, classification diagnostics, threshold tuning, and error slicing. Use for confusion matrices, interactive ROC/PR curves with threshold annotations, probability calibration analysis, error reports, and time-series directional metrics. Do NOT use for data preprocessing or clustering.
version: "1.0.0"
license: MIT
metadata:
  author: "Idan Morad <idanmorad.arthas@gmail.com>"
  tags:
    - metrics
    - evaluation
    - model-diagnostics
    - roc-curve
    - error-analysis
  documentation: "https://datascienceutils.readthedocs.io/en/stable/"
  package: "data-science-utils"
  repository: "https://github.com/idanmoradarthas/DataScienceUtils"
---

# ds-utils-metrics

Comprehensive model evaluation, interactive threshold tuning, probability calibration inspection, and error slicing workflows.

## Purpose

`ds_utils.metrics` provides diagnostic visualizations and error analysis tools for evaluating predictive models. It bridges standard metric calculation with practical decision-making: finding optimal operational thresholds, detecting probability miscalibration, identifying underperforming data slices, and quantifying regression error profiles.

- **When to use**: Model validation, comparing competing candidate classifiers, selecting probability decision thresholds, diagnosing error concentrations across feature sub-populations, or forecasting trend directional accuracy.
- **When NOT to use**: For data exploration prior to training (use `ds-utils-preprocess`), tree interpretability diagrams (use `ds-utils-xai`), or clustering evaluation (use `ds-utils-unsupervised`).

## Instructions

### Task-to-Approach Map

| Evaluation Goal | Function | Engine | Input Requirements |
|---|---|---|---|
| Per-class errors & summary rates (FPR, FNR) | `plot_confusion_matrix` | Matplotlib | `y_true`, `y_pred`, `labels=[0, 1]` list |
| Optimal decision threshold tuning (ROC) | `plot_roc_curve_with_thresholds_annotations` | Plotly | `y_true`, `{name: proba[:, positive_class_idx]}` |
| Precision-recall balance on imbalanced data | `plot_precision_recall_curve_with_thresholds_annotations` | Plotly | `y_true`, `{name: proba[:, positive_class_idx]}` |
| Probability calibration / reliability grouping | `visualize_accuracy_grouped_by_probability` | Matplotlib | `y_true`, `labeled_class`, **1D proba array** |
| Data-size sensitivity & sample efficiency | `plot_metric_growth_per_labeled_instances` | Matplotlib | `X_tr, y_tr, X_te, y_te`, `{name: unfitted_model}` |
| Probability distribution by error type (FP/FN) | `plot_error_analysis_chart` | Matplotlib | `y_true`, `y_pred`, `y_proba` |
| Systematic error slicing across feature cohorts | `generate_error_analysis_report` | Pandas DataFrame | `X`, `y_true`, `y_pred` |
| Regression tolerance error curve & AOC | `plot_rec_curve_with_annotations`, `regression_auc_score` | Plotly & Scalar | `y_true`, `{name: y_pred}` (continuous) |
| Time-series directional change / trend bias | `directional_accuracy_score`, `directional_bias_score` | Scalar | `y_true`, `y_pred` (continuous sequences) |

Exhaustive signatures, parameter defaults, and common pitfalls are in [references/api_reference.md](references/api_reference.md).

---

### End-to-End Evaluation Workflow

Choose the evaluation branch matching your model task:

```text
               Model Predictions Available
                          │
         ┌────────────────┴────────────────┐
         ▼                                 ▼
   Classification                    Regression / Forecasting
         │                                 │
   Confusion Matrix                        Regression AOC & REC Curve
   (plot_confusion_matrix)                 (plot_rec_curve_with_annotations)
         │                                 │
   Threshold Tuning                        Directional Metrics
   (plot_roc / plot_precision_recall)      (directional_accuracy / bias)
         │
   Probability Calibration
   (visualize_accuracy_grouped_by_prob)
         │
   Feature Error Slicing
   (plot_error_analysis_chart & report)
```

#### Classification Branch

##### Step 1: Overall Classification Performance (`plot_confusion_matrix`)
Examine absolute false positives and false negatives alongside accuracy and F1.

```python
import numpy as np
import matplotlib.pyplot as plt
from ds_utils.metrics.confusion_matrix import plot_confusion_matrix

y_true = np.array([0, 1, 0, 1, 1, 0, 1, 0, 1, 0])
y_pred = np.array([0, 1, 0, 0, 1, 0, 1, 1, 1, 0])

ax = plot_confusion_matrix(y_true, y_pred, labels=[0, 1])
plt.close()
```
- **What to look for**: Disproportionate false negative vs false positive rates.
- **Action**: If cost of FP exceeds FN (or vice versa), proceed to threshold tuning in Step 2.

##### Step 2: Interactive Threshold Selection (`plot_roc_curve_with_thresholds_annotations`)
Determine the optimal classification cutoff using annotated Plotly curves.
- **Input requirement**: Pass a dictionary where values are **1D arrays of the positive class probabilities**. Use `clf.classes_` to extract the correct column safely: `pos_idx = int(np.where(clf.classes_ == positive_label)[0][0]); proba = clf.predict_proba(X)[:, pos_idx]`.
- **Return type**: Returns a `plotly.graph_objects.Figure`. Display with `fig.show()` or write using `fig.write_html()`.

```python
from ds_utils.metrics.curves import (
    plot_roc_curve_with_thresholds_annotations,
    plot_precision_recall_curve_with_thresholds_annotations,
)

y_proba_pos = np.array([0.15, 0.85, 0.20, 0.45, 0.90, 0.10, 0.78, 0.65, 0.92, 0.30])
models_dict = {"Model_A": y_proba_pos}

fig_roc = plot_roc_curve_with_thresholds_annotations(y_true, models_dict, positive_label=1)
fig_pr = plot_precision_recall_curve_with_thresholds_annotations(y_true, models_dict, positive_label=1)
```
- **Interpretation**: Hover over curve points to inspect the concrete threshold that achieves the required precision or recall.
- **Action**: Adopt the annotated threshold for decision cutoffs instead of defaulting to 0.5.

##### Step 3: Probability Calibration Inspection (`visualize_accuracy_grouped_by_probability`)
Verify whether predicted probabilities reflect empirical correctness.
- **Input requirement**: Accepts a **1D probability array** corresponding to `labeled_class`. Extract it dynamically using `clf.classes_` (e.g. `col_idx = int(np.where(clf.classes_ == labeled_class)[0][0]); y_proba_1d = clf.predict_proba(X)[:, col_idx]`).

```python
from ds_utils.metrics.probability_analysis import visualize_accuracy_grouped_by_probability

ax_cal = visualize_accuracy_grouped_by_probability(y_true, labeled_class=1, probabilities=y_proba_pos)
plt.close()
```
- **Interpretation**: If samples with predicted probability ~0.80 only demonstrate 50% empirical accuracy, the model is overconfident.
- **Action**: Apply Platt scaling (`CalibratedClassifierCV`) before relying on probabilities in business operations. Note: for an explicit calibration curve plot, use standard `sklearn.calibration.CalibrationDisplay`.

##### Step 4: Slicing Errors Across Cohorts (`generate_error_analysis_report`)
Diagnose which demographic, regional, or numerical segments suffer from heightened error rates.

```python
import pandas as pd
from ds_utils.metrics.error_analysis import generate_error_analysis_report
from ds_utils.metrics.probability_analysis import plot_error_analysis_chart

X_demo = pd.DataFrame(
    {
        "age": [25, 45, 60, 35, 50, 22, 58, 30, 42, 65],
        "channel": ["web", "app", "web", "web", "app", "web", "app", "app", "web", "app"],
    }
)

# 1. Visualize error distribution relative to probabilities
ax_err = plot_error_analysis_chart(y_true, y_pred, y_proba_pos, positive_class=1)
plt.close()

# 2. Generate cohort breakdown
report = generate_error_analysis_report(X_demo, y_true, y_pred, feature_columns=["age", "channel"], bins=3, min_count=1)
print(report.head())
```
- **Interpretation**: Examine slices where `error_rate` is elevated despite adequate sample `count`.
- **Action**: Collect additional training instances for high-error cohorts or engineer targeted features.

---

#### Regression and Forecasting Branch

##### Step 1: REC Tolerance Evaluation (`plot_rec_curve_with_annotations`)
Evaluate regression accuracy across allowable error tolerance thresholds. Lower Area Over Curve (AOC) indicates higher performance.

```python
from ds_utils.metrics.regression import plot_rec_curve_with_annotations, regression_auc_score

y_reg_true = np.array([10.0, 20.5, 30.0, 40.2, 50.0])
y_reg_pred = np.array([10.5, 19.8, 31.0, 42.0, 49.5])

# Standalone normalized AOC (lower is better, 0.0 = perfect)
aoc = regression_auc_score(y_reg_true, y_reg_pred, normalize=True)

# Annotated Plotly REC curve
fig_rec = plot_rec_curve_with_annotations(y_reg_true, {"Regressor_A": y_reg_pred})
```

##### Step 2: Directional Forecast Analysis
In time-series and trading strategies, predicting the direction of change often matters more than absolute mean squared error.

```python
from ds_utils.metrics.time_series import directional_accuracy_score, directional_bias_score

series_true = np.array([100.0, 102.0, 99.0, 103.0, 105.0])
series_pred = np.array([101.0, 103.0, 98.0, 102.5, 106.0])

da = directional_accuracy_score(series_true, series_pred)  # Baseline=None defaults to prior timestep
bias = directional_bias_score(series_true, series_pred)
print(f"Directional Accuracy: {da:.2%}, Directional Bias: {bias:.2f}")
```
- **Interpretation**: `directional_accuracy_score` in [0, 1] measures trend classification. `directional_bias_score` in [-1, +1] quantifies systemic over-prediction (+1) vs under-prediction (-1).

---

## Examples

### Complete Model Diagnostic Script

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from ds_utils.metrics.confusion_matrix import plot_confusion_matrix
from ds_utils.metrics.curves import plot_roc_curve_with_thresholds_annotations
from ds_utils.metrics.probability_analysis import visualize_accuracy_grouped_by_probability
from ds_utils.metrics.error_analysis import generate_error_analysis_report

# 1. Synthetic evaluation data
np.random.seed(42)
X_test = pd.DataFrame({"age": np.random.randint(18, 70, 80), "score": np.random.randn(80)})
y_test = (X_test["score"] + np.random.normal(0, 0.5, 80) > 0).astype(int).values

clf = LogisticRegression()
clf.fit(X_test, y_test)
y_pred = clf.predict(X_test)
y_proba_2d = clf.predict_proba(X_test)

# 2. Confusion matrix
ax_cm = plot_confusion_matrix(y_test, y_pred, labels=[0, 1])
plt.close()

# 3. Interactive ROC using positive class column dynamically determined via clf.classes_
pos_idx = int(np.where(clf.classes_ == 1)[0][0])
fig_roc = plot_roc_curve_with_thresholds_annotations(y_test, {"Logistic": y_proba_2d[:, pos_idx]}, positive_label=1)

# 4. Probability calibration using 1D probabilities for class 1
ax_prob = visualize_accuracy_grouped_by_probability(y_test, 1, y_proba_2d[:, pos_idx])
plt.close()

# 5. Cohort error report
cohort_report = generate_error_analysis_report(X_test, y_test, y_pred, bins=3)
```

## Limitations

- **Engine differences**: Matplotlib functions (`plot_confusion_matrix`, `visualize_accuracy_grouped_by_probability`, `plot_error_analysis_chart`) return `Axes`. Plotly functions (`plot_roc...`, `plot_precision_recall...`, `plot_rec...`) return `Figure` objects requiring `fig.show()` or `fig.write_html()`.
- **Probabilities formatting**: `visualize_accuracy_grouped_by_probability` requires a 1D probability array for the specific `labeled_class`, aligned with `clf.classes_`.
- **Directional minimum samples**: `directional_accuracy_score` with `baseline=None` requires at least 2 samples since the first sample establishes the prior baseline.

## Troubleshooting

| Error / Symptom | Root Cause | Solution |
|---|---|---|
| `TypeError: 'Figure' object is not callable` or plot not showing | Calling `.show()` as a matplotlib method or treating Plotly figure as matplotlib Axes | Use `fig.show()` for Plotly figures (`plot_roc...`, `plot_rec...`) and `plt.show()` for matplotlib |
| `ValueError: Per-column arrays must each be 1-dimensional` in `visualize_accuracy_grouped_by_probability` | Passed full 2D probability matrix instead of 1D column | Select the 1D column matching `labeled_class`: `proba[:, np.where(clf.classes_ == labeled_class)[0][0]]` |
| `KeyError: positive_label` in ROC curves | Passed classifier objects instead of `{name: 1D_array}` | Construct dictionary: `{"model": proba[:, positive_class_idx]}` |
| `ValueError: len(y_true) must be at least 2` | Passed single point to `directional_accuracy_score` with default baseline | Provide at least 2 time steps or supply an explicit `baseline` array |
