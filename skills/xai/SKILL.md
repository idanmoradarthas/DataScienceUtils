---
name: ds-utils-xai
description: >
  Provides Explainable AI (XAI) tools to interpret machine learning models.
  Use when the user asks to visualize feature importance for tree-based
  models, needs to explain which features drive model decisions, wants to
  render or draw a decision tree diagram, or needs to convert a Graphviz
  DOT string into a matplotlib figure in a Python data science project
  using sklearn-compatible models.
license: MIT
metadata:
  author: Idan Morad
  documentation: https://datascienceutils.readthedocs.io/en/stable/
  package: data-science-utils
  repository: https://github.com/idanmoradarthas/DataScienceUtils
---
# XAI — ds_utils.xai

Explainable AI visualizers and interpretation methods.

## Installation

```bash
pip install data-science-utils
# or
conda install -c idanmorad data-science-utils
```

## Import

```python
from ds_utils.xai import plot_features_importance
from ds_utils.xai import draw_dot_data
from ds_utils.xai import plot_error_analysis_chart
from ds_utils.xai import generate_error_analysis_report
```

---

## plot_features_importance

Plots a feature importance bar chart, ranking features by their calculated impact on the model's decisions.

```python
from ds_utils.xai import plot_features_importance
import matplotlib.pyplot as plt

# complete usage example
plot_features_importance(features_names, clf.feature_importances_)
plt.show()
```

**Parameters:**
- `feature_names` — list, Names of the features used during training.
- `feature_importances` — array-like, Model's feature importances.

**Returns:** matplotlib Axes.

**Common mistakes:**
- The `feature_names` order MUST match the column order used in `.fit()`.
- This function only works with tree-based models that expose `.feature_importances_` (e.g., Decision Tree, Random Forest, GradientBoosting, XGBoost).
- Does NOT work with linear models since `.coef_` implies a different interpretation scale and meaning.

---

## draw_dot_data

Renders a decision tree image from a Graphviz DOT string (for example, DOT produced by `sklearn.tree.export_graphviz`). This is more of a lagacy method, and it is not recommended to use it in new projects. Instead, use sklearn's built in method such as `sklearn.tree.plot_tree`.

```python
from ds_utils.xai import draw_dot_data
from sklearn.tree import export_graphviz
import matplotlib.pyplot as plt

# complete usage example
dot = export_graphviz(clf, feature_names=features, class_names=["no", "yes"], filled=True, rounded=True, out_file=None)
draw_dot_data(dot)
plt.show()
```

**Parameters:**
- `dot_data` - str, Graphviz DOT string to render.
- `ax` - matplotlib Axes, optional. Target axes for rendering.

**Returns:** matplotlib Axes.

**Common mistakes:**
- Passing an empty DOT string. `draw_dot_data` requires a non-empty valid Graphviz DOT payload.
- Passing the estimator object directly; first generate DOT text using `export_graphviz(..., out_file=None)`.

---

## plot_error_analysis_chart

Automates the creation of an error analysis DataFrame (computing correct, false_positive, false_negative) and visualizes prediction errors relative to predicted probabilities. Supports both binary and multi-class classification using a one-vs-rest scheme.

### Binary classification example

```python
from ds_utils.xai import plot_error_analysis_chart
import matplotlib.pyplot as plt

# complete usage example (binary)
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1]  # probability of the positive class

plot_error_analysis_chart(y_test, y_pred, y_proba, positive_class=1)
plt.show()
```

### Multi-class classification example

```python
from ds_utils.xai import plot_error_analysis_chart
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
from ds_utils.xai import generate_error_analysis_report

# complete usage example
report_df = generate_error_analysis_report(
    X_test, y_test, y_pred,
    feature_columns=["age", "gender"],
    bins=5,
    min_count=10,
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
If analyzing features "age" and "income":

| feature | group | count | error_count | error_rate | accuracy |
|---------|-------|-------|-------------|------------|----------|
| age | (18.0, 35.0] | 100 | 10 | 0.10 | 0.90 |
| age | (35.0, 55.0] | 150 | 5 | 0.03 | 0.97 |
| region | North | 120 | 15 | 0.125 | 0.875 |
| income | (2000.0, 5000.0] | 80 | 12 | 0.15 | 0.85 |

**Common mistakes:**
- Passing columns in `feature_columns` that are not present in `X`.
- Setting `min_count` too high, which may filter out all groups for some features.

---

## Typical Workflow

```python
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from ds_utils.xai import plot_features_importance, plot_error_analysis_chart

features = ["age", "income", "credit_score"]
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train[features], y_train)

# 1. Feature importance bar chart
plot_features_importance(features, clf.feature_importances_)
plt.tight_layout()
plt.show()

# 2. Error analysis chart
y_pred = clf.predict(X_test[features])
y_proba = clf.predict_proba(X_test[features])[:, 1]

plot_error_analysis_chart(y_test, y_pred, y_proba, positive_class=1)
plt.tight_layout()
plt.show()
```
