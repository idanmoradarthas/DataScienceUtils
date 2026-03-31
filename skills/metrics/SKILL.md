---
name: ds-utils-metrics
description: >
  Provides evaluation metrics and visualization charts for machine learning models. Use when the user asks to evaluate a model, wants to plot a confusion matrix, ROC curve, or Precision-Recall curve, or needs to analyze learning curves or probability distributions in a Python data science project using sklearn-compatible models.
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

## Typical Workflow

```python
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from ds_utils.metrics.confusion_matrix import plot_confusion_matrix
from ds_utils.metrics.learning_curves import plot_metric_growth_per_labeled_instances
from ds_utils.metrics.probability_analysis import visualize_accuracy_grouped_by_probability
from ds_utils.metrics.curves import (
    plot_roc_curve_with_thresholds_annotations,
    plot_precision_recall_curve_with_thresholds_annotations,
)

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
```
