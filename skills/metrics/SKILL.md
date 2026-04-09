---
name: ds-utils-metrics
description: >
  Provides evaluation metrics and visualization charts for machine learning models. Use when the user asks to evaluate a model, wants to plot a confusion matrix, ROC curve, or Precision-Recall curve, needs to analyze learning curves, probability distributions, or error analysis in a Python data science project using sklearn-compatible models.
license: MIT
metadata:
  author: Idan Morad
  documentation: https://datascienceutils.readthedocs.io/en/stable/
  package: data-science-utils
  repository: https://github.com/idanmoradarthas/DataScienceUtils
---
# Metrics — ds_utils.metrics

Visualization methods for evaluating classification model performance.

## Installation

```bash
pip install data-science-utils
# or
conda install -c idanmorad data-science-utils
```

## Import

```python
from ds_utils.metrics.confusion_matrix import plot_confusion_matrix
from ds_utils.metrics.learning_curves import plot_metric_growth_per_labeled_instances
from ds_utils.metrics.probability_analysis import visualize_accuracy_grouped_by_probability
from ds_utils.metrics.curves import plot_roc_curve_with_thresholds_annotations
from ds_utils.metrics.curves import plot_precision_recall_curve_with_thresholds_annotations
from ds_utils.metrics.probability_analysis import plot_error_analysis_chart
from ds_utils.metrics.error_analysis import generate_error_analysis_report
```

---

## plot_confusion_matrix

Computes and plots a confusion matrix, False Positive Rate, False Negative Rate, Accuracy, and F1 score of a classification.

```python
from ds_utils.metrics.confusion_matrix import plot_confusion_matrix
import matplotlib.pyplot as plt

# complete usage example
plot_confusion_matrix(y_test, y_pred, [0, 1])
plt.tight_layout()
plt.show()
```

**Parameters:**
- `y_test` — array-like, Ground truth (correct) target values.
- `y_pred` — array-like, Estimated targets as returned by a classifier.
- `labels` — list, List of labels to index the matrix.
- `sample_weight` — array-like, optional. Sample weights.
- `annot_kws` — dict, optional. Keyword arguments for `ax.text`.
- `cbar` — bool, whether to draw a colorbar. (Default: True).
- `cbar_kws` — dict, optional. Keyword arguments for `figure.colorbar`.

**Returns:** matplotlib Axes.

**Common mistakes:**
- Do NOT pass `labels` as a numpy array; you must use a Python list.

---

## plot_metric_growth_per_labeled_instances

Plots the given metric change with an increasing number of trained instances.

```python
from ds_utils.metrics.learning_curves import plot_metric_growth_per_labeled_instances
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

# complete usage example
plot_metric_growth_per_labeled_instances(
    x_train, y_train, x_test, y_test,
    {
        "Decision Tree": DecisionTreeClassifier(random_state=42)
    }
)
plt.tight_layout()
plt.show()
```

**Parameters:**
- `X_train` — array-like, Training data.
- `y_train` — array-like, Training labels.
- `X_test` — array-like, Test data.
- `y_test` — array-like, Test labels.
- `classifiers` — dict, Dictionary of classifier names and unfitted instances.
- `n_samples` — list of int, optional. Specific numbers of samples to use. (Default: None).
- `quantiles` — list of float. Percentages of data to use if n_samples=None. (Default: 20 steps from 0.05 to 1.0).
- `metric` — callable, Optional. sklearn metric to evaluate. (Default: accuracy_score).
- `random_state` — int, random state for reproducibility.
- `n_jobs` — int, number of parallel jobs to run.
- `verbose` — int, verbosity level.

**Returns:** matplotlib Axes.

**Common mistakes:**
- Passing a pre-fitted estimator instead of an unfitted one.

---

## visualize_accuracy_grouped_by_probability

Visualizes accuracy grouped by probability predictions to evaluate model calibration.

```python
from ds_utils.metrics.probability_analysis import visualize_accuracy_grouped_by_probability
import matplotlib.pyplot as plt

# complete usage example
visualize_accuracy_grouped_by_probability(
    y_test,
    1,
    clf.predict_proba(X_test),
    display_breakdown=False
)
plt.tight_layout()
plt.show()
```

**Parameters:**
- `y_test` — array-like, True labels.
- `labeled_class` — int/str, The specific labeled class to evaluate.
- `probabilities` — array-like, shape (n_samples, n_classes). The full
  probability matrix from `clf.predict_proba(X_test)` — NOT the single
  positive-class column. The function extracts the relevant column
  internally using `labeled_class`.
- `threshold` — float, Probability threshold for classifying the labeled class. (Default: 0.5).
- `display_breakdown` — bool, Whether to display class breakdown (Correct/Incorrect) or True/False Positives/Negatives. (Default: False).
- `bins` — list, Custom probability bins. (Default: 10 bins from 0 to 1).

**Returns:** matplotlib Axes.

---

## plot_roc_curve_with_thresholds_annotations

Plots ROC curves with threshold annotations using Plotly.

```python
from ds_utils.metrics.curves import plot_roc_curve_with_thresholds_annotations

# complete usage example
classifiers_proba = {
    "Decision Tree": clf.predict_proba(X_test)[:, 1]
}
fig = plot_roc_curve_with_thresholds_annotations(
    y_test,
    classifiers_proba,
    positive_label=1
)
fig.show()
```

**Parameters:**
- `y_true` — array, True labels.
- `classifiers_names_and_scores_dict` — dict, Predict proba scores for positive class by classifier name.
- `positive_label` — int/str, The value of the positive class.
- `sample_weight` — array-like, optional. Sample weights.
- `drop_intermediate` — bool, Whether to drop sub-optimal thresholds. (Default: True).
- `average` — str, Averaging mode. (Default: "macro").
- `max_fpr` — float, If provided, limits the x-axis partial AUC.
- `multi_class` — str, Handling of multi-class ROC. (Default: "raise"). 
- `mode` — str, Plotly trace drawing mode. (Default: "lines+markers").
- `add_random_classifier_line` — bool. Plot the naive diagonal. (Default: True).

**Returns:** Plotly Figure.

**Common mistakes:**
- Passing individual classifiers; the function takes a dict of `{name: proba_array}`.
- Passing full `predict_proba` results; dict values MUST be `predict_proba(X)[:, 1]` (positive column only).
- `positive_label` must be a value from `y_true`, not the column index.
- Returning a matplotlib plot; the function returns a **Plotly Figure**, which requires `fig.show()`.

---

## plot_precision_recall_curve_with_thresholds_annotations

Plots Precision-Recall curves with threshold annotations using Plotly.

```python
from ds_utils.metrics.curves import plot_precision_recall_curve_with_thresholds_annotations

# complete usage example
classifiers_proba = {
    "Decision Tree": clf.predict_proba(X_test)[:, 1]
}
fig = plot_precision_recall_curve_with_thresholds_annotations(
    y_test,
    classifiers_proba,
    positive_label=1
)
fig.show()
```

**Parameters:**
- `y_true` — array, True labels.
- `classifiers_names_and_scores_dict` — dict, Predict proba scores for positive class by classifier name.
- `positive_label` — int/str, The value of the positive class.
- `sample_weight` — array-like, optional. Sample weights.
- `drop_intermediate` — bool, Whether to drop sub-optimal thresholds. (Default: True).
- `mode` — str, Plotly trace drawing mode. (Default: "lines+markers").
- `add_random_classifier_line` — bool. Plot the naive diagonal. (Default: False).

**Returns:** Plotly Figure.

**Common mistakes:**
- Passing individual classifiers; the function takes a dict of `{name: proba_array}`.
- Passing full `predict_proba` results; dict values MUST be `predict_proba(X)[:, 1]` (positive column only).
- `positive_label` must be a value from `y_true`, not the column index.
- Returning a matplotlib plot; the function returns a **Plotly Figure**, which requires `fig.show()`.

---

## plot_error_analysis_chart

Automates the creation of an error analysis DataFrame (computing correct, false_positive, false_negative) and visualizes prediction errors relative to predicted probabilities. Supports both binary and multi-class classification using a one-vs-rest scheme.

### Binary classification example

```python
from ds_utils.metrics.probability_analysis import plot_error_analysis_chart
import matplotlib.pyplot as plt

# complete usage example (binary)
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1]  # probability of the positive class

plot_error_analysis_chart(y_test, y_pred, y_proba, positive_class=1)
plt.show()
```

### Multi-class classification example

```python
from ds_utils.metrics.probability_analysis import plot_error_analysis_chart
import matplotlib.pyplot as plt

# complete usage example (multi-class)
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)

plot_error_analysis_chart(
    y_test, y_pred, y_proba,
    positive_class=1,
    classes=clf.classes_.tolist()
)
plt.show()
```

**Parameters:**
- `y_true` — array-like, True labels.
- `y_pred` — array-like, Predicted labels (required).
- `y_proba` — array-like, Predicted probabilities. 1-D for binary, 2-D `(n_samples, n_classes)` for multi-class.
- `positive_class` — The class to treat as positive (used for correct/false_positive/false_negative assignment).
- `classes` — list, optional. Ordered class labels matching columns of `y_proba` when 2-D. If `None`, inferred from `np.unique(y_true)`.
- `ax` — matplotlib Axes, optional. Target axes for rendering.
- `**kwargs` — forwarded to `seaborn.violinplot`.

**Returns:** matplotlib Axes.

**Common mistakes:**
- Forgetting to pass `classes` for multi-class when the class order in `y_proba` does not match `np.unique(y_true)`. Always pass `classes=clf.classes_.tolist()` to be safe.
- For binary classification, pass only the positive class probability column (1-D), not the full 2-D probability matrix, unless you also specify `classes`.
- `y_pred` is required — you must pass pre-computed predictions, not raw probabilities.

---

## generate_error_analysis_report

Provides a tabular error-analysis report that groups predictions by feature values and computes error metrics per group.

```python
import pandas as pd
import numpy as np
from ds_utils.metrics.error_analysis import generate_error_analysis_report

# Setup dummy data with numerical and categorical features
X_test = pd.DataFrame({
    "age": [25, 30, 45, 50, 22, 35, 40, 60],
    "region": ["North", "South", "North", "West", "East", "South", "West", "North"]
})
y_test = np.array([0, 1, 0, 1, 0, 1, 0, 1])
y_pred = np.array([0, 1, 1, 1, 0, 0, 0, 1]) # Errors at index 2 and 5

# complete usage example
report_df = generate_error_analysis_report(
    X_test, y_test, y_pred,
    feature_columns=["age", "region"],
    bins=3,
    min_count=1,
    sort_metric="error_rate",
    ascending=False
)
print(report_df.head())
```

**Parameters:**
- `X` — pandas DataFrame, Feature values.
- `y_true` — array-like, True labels.
- `y_pred` — array-like, Predicted labels.
- `feature_columns` — list, optional. Subset of columns to analyze. If `None`, all columns in `X` are used.
- `bins` — int, default 10. Number of bins for numerical features.
- `threshold` — float, default 0.5. Reserved for future probability-based error definitions.
- `min_count` — int, default 1. Minimum samples per group to include in the report.
- `sort_metric` — str, default "error_rate". Column to sort by.
- `ascending` — bool, default False. Sort direction.

**Returns:** pandas DataFrame.

**Output Example:**
If analyzing features "age" and "region":

| feature | group | count | error_count | error_rate | accuracy |
|---------|-------|-------|-------------|------------|----------|
| age | (34.667, 47.333] | 2 | 1 | 0.50 | 0.50 |
| region | South | 2 | 1 | 0.50 | 0.50 |
| region | North | 3 | 1 | 0.33 | 0.67 |
| age | (21.962, 34.667] | 4 | 1 | 0.25 | 0.75 |
| age | (47.333, 60.0] | 2 | 0 | 0.00 | 1.00 |
| region | East | 1 | 0 | 0.00 | 1.00 |
| region | West | 2 | 0 | 0.00 | 1.00 |

*(Note: Rows with equal error_rate may appear in any order)*

**Common mistakes:**
- Passing columns in `feature_columns` that are not present in `X`.
- Setting `min_count` too high, which may filter out all groups for some features.

---

## directional_accuracy_score

Calculates the proportion of time steps for which a model correctly predicts the direction of
change relative to a baseline. Ideal for time-series forecasting and financial modeling where
trend direction matters more than exact magnitude.

```python
from ds_utils.metrics.time_series import directional_accuracy_score
import numpy as np

# Time series mode (baseline = previous value)
y_true = np.array([100, 102, 98, 101, 99])
y_pred = np.array([101, 103, 97, 102, 98])
da = directional_accuracy_score(y_true, y_pred)
print(f"Directional Accuracy: {da:.2%}")

# Custom baseline
baseline = np.array([100, 100, 100, 100, 100])
da = directional_accuracy_score(y_true, y_pred, baseline=baseline)
```

**Parameters:**

* `y_true` — array-like of shape (n_samples,). True target values.
* `y_pred` — array-like of shape (n_samples,). Predicted target values.
* `baseline` — array-like of shape (n_samples,), optional. Baseline values. If None, uses
  `y_true[i-1]` (time series default). Requires `len(y_true) >= 2`.
* `sample_weight` — array-like, optional. Sample weights.
* `handle_equal` — `{'exclude', 'correct', 'incorrect'}`, default `'exclude'`. How to treat
  samples where `y_true == baseline`.

**Returns:** float in [0, 1]. 1.0 = perfect directional prediction, 0.5 = random baseline.

**Common mistakes:**

* Passing a single-element array with `baseline=None` — raises `ValueError`. You need at least
  2 samples in time-series mode.
* Forgetting that in time-series mode the first sample is dropped (it has no prior value), so a
  5-element input yields 4 evaluated steps.

---

## directional_bias_score

Calculates the systematic tendency of a model to over-predict or under-predict the target values.
Returns a score where 1.0 is complete over-prediction, -1.0 is complete under-prediction, and
0.0 is perfectly balanced.

```python
from ds_utils.metrics.time_series import directional_bias_score
import numpy as np

y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
y_pred = np.array([1.1, 2.1, 3.1, 4.1, 5.1])
bias = directional_bias_score(y_true, y_pred)
print(f"Directional Bias: {bias:.2f}")
```

**Parameters:**

* `y_true` — array-like of shape (n_samples,). True target values.
* `y_pred` — array-like of shape (n_samples,). Predicted target values.
* `sample_weight` — array-like, optional. Sample weights.
* `handle_equal` — `{'exclude', 'neutral'}`, default `'exclude'`. How to treat samples where `y_pred == y_true`.

**Returns:** float in [-1, 1].

**Common mistakes:**

* Using `handle_equal='exclude'` (default) on a dataset where all predictions are exactly correct — raises `ValueError`. Use `'neutral'` if you expect and want to include perfect predictions.

---

## Typical Workflow

```python
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from ds_utils.metrics.confusion_matrix import plot_confusion_matrix
from ds_utils.metrics.learning_curves import plot_metric_growth_per_labeled_instances
from ds_utils.metrics.probability_analysis import visualize_accuracy_grouped_by_probability, plot_error_analysis_chart
from ds_utils.metrics.curves import (
    plot_roc_curve_with_thresholds_annotations,
    plot_precision_recall_curve_with_thresholds_annotations,
)
from ds_utils.metrics.error_analysis import generate_error_analysis_report

clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)
y_pred  = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)

# 1. Confusion matrix
plot_confusion_matrix(y_test, y_pred, labels=[0, 1])
plt.tight_layout()
plt.show()

# 2. Learning curve — how accuracy grows with more training data
plot_metric_growth_per_labeled_instances(
    X_train, y_train, X_test, y_test,
    {"Decision Tree": DecisionTreeClassifier(random_state=42)},
)
plt.tight_layout()
plt.show()

# 3. Accuracy grouped by predicted probability
visualize_accuracy_grouped_by_probability(y_test, 1, y_proba)
plt.tight_layout()
plt.show()

# 4. ROC curve with threshold annotations (Plotly)
fig_roc = plot_roc_curve_with_thresholds_annotations(
    y_test,
    {"Decision Tree": y_proba[:, 1]},
    positive_label=1,
)
fig_roc.show()

# 5. Precision-Recall curve with threshold annotations (Plotly)
fig_pr = plot_precision_recall_curve_with_thresholds_annotations(
    y_test,
    {"Decision Tree": y_proba[:, 1]},
    positive_label=1,
)
fig_pr.show()

# 6. Error analysis chart
y_pred = clf.predict(X_test)
y_proba_pos = clf.predict_proba(X_test)[:, 1]
plot_error_analysis_chart(y_test, y_pred, y_proba_pos, positive_class=1)
plt.tight_layout()
plt.show()

# 7. Error Analysis block report
report_df = generate_error_analysis_report(
    X_test, y_test, y_pred,
    feature_columns=None,
    bins=3
)
print(report_df)

# 8. Directional Metrics (for time-series/forecasting)
from ds_utils.metrics.time_series import directional_accuracy_score, directional_bias_score
da = directional_accuracy_score(y_test, y_pred)
bias = directional_bias_score(y_test, y_pred)
print(f"DA: {da:.2%}, Bias: {bias:.2f}")
```
