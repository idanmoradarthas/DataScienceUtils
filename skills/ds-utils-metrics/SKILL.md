---
name: ds-utils-metrics
description: >
  Machine learning model evaluation, classification diagnostics, threshold tuning, and error slicing. Use for confusion matrices, interactive ROC/PR curves with threshold annotations, probability calibration analysis, error reports, and time-series directional metrics. Do NOT use for data preprocessing or clustering.
version: 1.0.0
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

ds_utils.metrics provides diagnostic visualizations and error analysis tools for evaluating predictive models. It bridges standard metric calculation with practical decision-making: finding optimal operational thresholds, detecting probability miscalibration, identifying underperforming data slices, and quantifying regression error profiles.

- **When to use**: Model validation, comparing competing candidate classifiers, selecting probability decision thresholds, diagnosing error concentrations across feature sub-populations, or forecasting trend directional accuracy.
- **When NOT to use**: For data exploration prior to training (use ds-utils-preprocess), tree interpretability diagrams (use ds-utils-xai), or clustering evaluation (use ds-utils-unsupervised).

## Prerequisites

- Python 3.11 or higher.
- data-science-utils package installed (pip install data-science-utils).
- Core dependencies: numpy, pandas, scikit-learn, matplotlib, seaborn, plotly.
- Trained scikit-learn compatible estimator or prediction arrays (y_true, y_pred, y_proba).

## Instructions

### Task-to-Approach Map

| Evaluation Goal | Function | Engine | Input Requirements |
|---|---|---|---|
| Per-class errors & summary rates (FPR, FNR) | `plot_confusion_matrix` | Matplotlib | `y_true`, `y_pred`, `labels=[0, 1]` list |
| Optimal decision threshold tuning (ROC) | `plot_roc_curve_with_thresholds_annotations` | Plotly | `y_true`, `{name: proba[:, positive_class_idx]}` |
| Precision-recall balance on imbalanced data | `plot_precision_recall_curve_with_thresholds_annotations` | Plotly | `y_true`, `{name: proba[:, positive_class_idx]}` |
| Coarse probability overconfidence check | `visualize_accuracy_grouped_by_probability` | Matplotlib | `y_true`, `labeled_class`, 1D probability array |
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
         │
   Learning Curve Analysis
   (plot_metric_growth_per_labeled_instances)
```

#### Inputs you need

The steps below assume you have partitioned your data and fitted your models:
- **Classification**: a fitted classifier `clf`, test feature matrix or DataFrame `X_test`, ground-truth labels `y_test`, plus training splits `X_train`, `y_train` for learning curves.
- **Regression**: a fitted regressor `reg`, test features `X_test`, ground truth `y_test`.
- **Forecasting**: continuous arrays `y_true_series` and `y_pred_series` ordered chronologically.

```python
import numpy as np
import pandas as pd

# Load your evaluation data (replace with your data)
df = pd.read_csv("your_dataset")
# X_train, X_test, y_train, y_test = train_test_split(...)
# clf = LogisticRegression().fit(X_train, y_train)
```

#### Classification Branch

##### Step 1: Overall Classification Performance (`plot_confusion_matrix`)
Question: where are the classification errors concentrated between false positives and false negatives?

```python
import matplotlib.pyplot as plt
from ds_utils.metrics.confusion_matrix import plot_confusion_matrix

# Predictions on your held-out test data
y_true = y_test
y_pred = clf.predict(X_test)

ax = plot_confusion_matrix(
    y_true, y_pred, labels=[0, 1]
)  # 0/1 assumed - replace with your class labels / positive class
plt.close()
```
- **What to look for**: Disproportionate false negative vs false positive rates.
- **Action**: If cost of FP exceeds FN (or vice versa), proceed to threshold tuning in Step 2.
- **Caution**: Overall accuracy hides severe failure modes on imbalanced classes; evaluate per-class error rates.

##### Step 2: Interactive Threshold Selection (`plot_roc_curve_with_thresholds_annotations`)
Question: which decision threshold optimizes the operational trade-off between sensitivity and precision?

```python
import numpy as np
from ds_utils.metrics.curves import (
    plot_roc_curve_with_thresholds_annotations,
    plot_precision_recall_curve_with_thresholds_annotations,
)

# Extract 1D positive class probabilities safely from your fitted model
pos_idx = int(np.where(clf.classes_ == 1)[0][0])  # 0/1 assumed - replace with your class labels / positive class
y_proba_pos = clf.predict_proba(X_test)[:, pos_idx]
models_dict = {"Model": y_proba_pos}

fig_roc = plot_roc_curve_with_thresholds_annotations(y_true, models_dict, positive_label=1)
fig_pr = plot_precision_recall_curve_with_thresholds_annotations(y_true, models_dict, positive_label=1)
```
- **What to look for**: Hover over curve points to inspect the concrete threshold that achieves the required precision or recall.
- **Action**: Adopt the annotated threshold for decision cutoffs instead of defaulting to 0.5.
- **Caution**: Selecting thresholds on the test set produces overly optimistic results; choose thresholds on validation data or out-of-fold predictions, then evaluate on held-out test data.

##### Step 3: Coarse Probability Overconfidence Check (`visualize_accuracy_grouped_by_probability`)
Question: are predicted probabilities calibrated across confidence bins, or is the model overconfident?

```python
from ds_utils.metrics.probability_analysis import visualize_accuracy_grouped_by_probability

ax_cal = visualize_accuracy_grouped_by_probability(y_true, labeled_class=1, probabilities=y_proba_pos)
plt.close()
```
- **What to look for**: Disproportionate incorrect counts appearing in high-confidence bins (e.g. 0.8-1.0).
- **Action**: If high-confidence bins show significant incorrect proportions, calibrate probabilities using scikit-learn CalibratedClassifierCV.
- **Caution**: This plot displays binned counts as a coarse overconfidence diagnostic, not a continuous reliability curve.

##### Step 4: Slicing Errors Across Cohorts (`generate_error_analysis_report`)
Question: do specific feature subgroups or demographic cohorts suffer from elevated error rates?

```python
from ds_utils.metrics.error_analysis import generate_error_analysis_report
from ds_utils.metrics.probability_analysis import plot_error_analysis_chart

# 1. Visualize error distribution relative to probabilities
ax_err = plot_error_analysis_chart(y_true, y_pred, y_proba_pos, positive_class=1)
plt.close()

# 2. Slice errors across feature columns from your real test DataFrame
feature_cols = list(X_test.columns[:3]) if hasattr(X_test, "columns") else None
report = generate_error_analysis_report(X_test, y_true, y_pred, feature_columns=feature_cols, bins=5, min_count=5)
```
- **What to look for**: Feature groups where error_rate is substantially higher than the global rate despite adequate sample count.
- **Action**: Collect more training instances for underperforming cohorts or engineer targeted subgroup features.
- **Caution**: Slices with small sample counts can show deceptively high error rates by pure chance; filter with min_count.

##### Step 5: Data-Size Sensitivity and Growth Curves (`plot_metric_growth_per_labeled_instances`)
Question: would collecting more labeled training instances meaningfully improve performance?

```python
from sklearn.base import clone
from ds_utils.metrics.learning_curves import plot_metric_growth_per_labeled_instances

# Pass UNFITTED estimator instances in the dictionary
unfitted_models = {"Model": clone(clf)}

n_samples_steps = [int(len(X_train) * frac) for frac in [0.2, 0.4, 0.6, 0.8, 1.0]]
ax_growth = plot_metric_growth_per_labeled_instances(
    np.asarray(X_train), y_train, np.asarray(X_test), y_test, unfitted_models, n_samples=n_samples_steps
)
plt.close()
```
- **What to look for**: Metric score flattening out as sample size increases toward 100%.
- **Action**: If the curve has plateaued, focus on feature engineering or model architecture rather than acquiring more data.
- **Caution**: Pass unfitted estimators (cloned or freshly instantiated).

---

#### Regression and Forecasting Branch

##### Step 1: REC Tolerance Evaluation (`plot_rec_curve_with_annotations`)
Question: what proportion of regression errors fall within acceptable tolerance thresholds?

```python
from ds_utils.metrics.regression import plot_rec_curve_with_annotations, regression_auc_score

y_reg_true = y_test
y_reg_pred = reg.predict(X_test)

aoc = regression_auc_score(y_reg_true, y_reg_pred, normalize=True)
fig_rec = plot_rec_curve_with_annotations(y_reg_true, dict(Regressor=y_reg_pred))
```
- **What to look for**: Area Over Curve (AOC). Lower AOC indicates tighter error bounds and superior regression fit.
- **Action**: Compare competing regressors by their AOC and acceptable operational error tolerances.
- **Caution**: Normalized AOC depends on the maximum absolute error observed; check absolute error percentiles alongside normalized scores.

##### Step 2: Directional Forecast Analysis
Question: does the forecast correctly predict the direction of change, and is there systematic bias?

```python
from ds_utils.metrics.time_series import directional_accuracy_score, directional_bias_score

da = directional_accuracy_score(y_true_series, y_pred_series)
bias = directional_bias_score(y_true_series, y_pred_series)
```
- **What to look for**: Directional accuracy above 50% indicates positive trend prediction skill. Directional bias near 0 indicates neutral forecasting.
- **Action**: If bias is positive or negative, recalibrate baseline level offsets.
- **Caution**: Directional accuracy requires chronologically ordered sequences; shuffling series invalidates the metric.

---

## Examples

### Runnable demo on a bundled dataset (replace the loading lines with your own data)

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from ds_utils.metrics.confusion_matrix import plot_confusion_matrix
from ds_utils.metrics.curves import (
    plot_roc_curve_with_thresholds_annotations,
    plot_precision_recall_curve_with_thresholds_annotations,
)
from ds_utils.metrics.probability_analysis import (
    visualize_accuracy_grouped_by_probability,
    plot_error_analysis_chart,
)
from ds_utils.metrics.error_analysis import generate_error_analysis_report
from ds_utils.metrics.learning_curves import plot_metric_growth_per_labeled_instances
from ds_utils.metrics.regression import plot_rec_curve_with_annotations, regression_auc_score
from ds_utils.metrics.time_series import directional_accuracy_score, directional_bias_score

# --- Classification demo ---
data_clf = load_breast_cancer(as_frame=True)
X_c, y_c = data_clf.data, data_clf.target
X_tr, X_te, y_tr, y_te = train_test_split(X_c, y_c, test_size=0.3, stratify=y_c, random_state=42)

pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42))
pipe.fit(X_tr, y_tr)
y_pred = pipe.predict(X_te)
pos_idx = int(np.where(pipe.classes_ == 1)[0][0])
y_proba_pos = pipe.predict_proba(X_te)[:, pos_idx]

ax_cm = plot_confusion_matrix(y_te.to_numpy(), y_pred, labels=[0, 1])
plt.close()

fig_roc = plot_roc_curve_with_thresholds_annotations(y_te.to_numpy(), {"LogReg": y_proba_pos}, positive_label=1)
fig_pr = plot_precision_recall_curve_with_thresholds_annotations(
    y_te.to_numpy(), {"LogReg": y_proba_pos}, positive_label=1
)
ax_prob = visualize_accuracy_grouped_by_probability(y_te.to_numpy(), 1, y_proba_pos)
plt.close()
ax_err = plot_error_analysis_chart(y_te.to_numpy(), y_pred, y_proba_pos, positive_class=1)
plt.close()

cohort_rep = generate_error_analysis_report(
    X_te, y_te.to_numpy(), y_pred, feature_columns=list(X_te.columns[:3]), bins=3, min_count=5
)

unfitted = {"LogReg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42))}
ax_growth = plot_metric_growth_per_labeled_instances(
    X_tr.to_numpy(), y_tr, X_te.to_numpy(), y_te, unfitted, n_samples=[50, 100, 200, len(X_tr)]
)
plt.close()

# --- Regression demo ---
data_reg = load_diabetes(as_frame=True)
X_r_tr, X_r_te, y_r_tr, y_r_te = train_test_split(data_reg.data, data_reg.target, test_size=0.3, random_state=42)
reg = Ridge().fit(X_r_tr, y_r_tr)
y_reg_pred = reg.predict(X_r_te)
aoc = regression_auc_score(y_r_te.to_numpy(), y_reg_pred)
fig_rec = plot_rec_curve_with_annotations(y_r_te.to_numpy(), {"Ridge": y_reg_pred})

# --- Time-series directional metrics (format illustration only - replace with your column(s)) ---
actual_ts = np.array([100.0, 102.5, 101.0, 104.0, 103.5])
pred_ts = np.array([101.0, 103.0, 100.5, 105.0, 104.0])
da_val = directional_accuracy_score(actual_ts, pred_ts)
db_val = directional_bias_score(actual_ts, pred_ts)
```

## Limitations

- **Threshold selection data**: The classification decision threshold must be tuned on validation data or out-of-fold cross-validation predictions and then evaluated on a held-out test set; picking it on the final test set yields overly optimistic numbers.
- **Engine differences**: Matplotlib functions (`plot_confusion_matrix`, `visualize_accuracy_grouped_by_probability`, `plot_error_analysis_chart`, `plot_metric_growth_per_labeled_instances`) return `Axes`. Plotly functions (`plot_roc...`, `plot_precision_recall...`, `plot_rec...`) return `Figure` objects requiring `fig.show()` or `fig.write_html()`.
- **Probabilities formatting**: `visualize_accuracy_grouped_by_probability` requires a 1D probability array for the specific `labeled_class`, aligned with `clf.classes_`.
- **Learning curve label shape**: Estimators in `classifiers_dict` must be unfitted instances.
- **Directional minimum samples**: `directional_accuracy_score` with `baseline=None` requires at least 2 samples since the first sample establishes the prior baseline.

## Troubleshooting

| Error / Symptom | Root Cause | Solution |
|---|---|---|
| Plot does not appear when running as a script | Plotly figures (`plot_roc...`, `plot_precision_recall...`, `plot_rec...`) require `fig.show()`, while Matplotlib functions return Axes and require `plt.show()` or `plt.savefig(...)` | Call `fig.show()` for Plotly return values and `plt.show()` or `plt.savefig(...)` for Matplotlib Axes |
| `ValueError: Per-column arrays must each be 1-dimensional` in `visualize_accuracy_grouped_by_probability` | Passed full 2D probability matrix instead of 1D column | Select the 1D column matching `labeled_class`: `proba[:, np.where(clf.classes_ == labeled_class)[0][0]]` |
| `AttributeError: 'LogisticRegression' object has no attribute 'shape'` | Passed fitted estimator instance instead of predicted probabilities to ROC/PR functions | Pass dictionary of predicted probability arrays: `{"model": proba[:, pos_idx]}` |
| `ValueError: Shape mismatch: y_true (n,) and y_scores (n, 2) for classifier <name>` | Passed full 2D probability matrix instead of 1D positive-class scores to ROC/PR curves | Extract 1D array of positive-class scores: `proba[:, pos_idx]` |
| `ValueError: Time-series mode (baseline=None) requires at least 2 samples.` | Passed single sample to `directional_accuracy_score` with default baseline | Provide at least 2 time steps or supply an explicit `baseline` array |
