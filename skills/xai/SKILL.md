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

Renders a decision tree image from a Graphviz DOT string (for example, DOT produced by `sklearn.tree.export_graphviz`).

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

## Typical Workflow

```python
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from ds_utils.xai import plot_features_importance, draw_dot_data

features = ["age", "income", "credit_score"]
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train[features], y_train)

# 1. Feature importance bar chart
plot_features_importance(features, clf.feature_importances_)
plt.tight_layout()
plt.show()

# 2. Render the decision tree as a diagram
dot = export_graphviz(
    clf,
    feature_names=features,
    class_names=["no", "yes"],
    filled=True,
    rounded=True,
    out_file=None,          # must be None to get the DOT string back
)
draw_dot_data(dot)
plt.tight_layout()
plt.show()
```
