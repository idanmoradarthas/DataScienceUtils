---
name: ds-utils-xai
description: >
  Provides Explainable AI (XAI) tools to interpret machine learning models. Use when the user asks to visualize feature importance for tree-based models, or needs to explain which features drive model decisions in a Python data science project using sklearn-compatible models.
metadata:
  author: Idan Morad
  version: "1.9.0"
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

## Typical Workflow

```python
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from ds_utils.xai import plot_features_importance
import matplotlib.pyplot as plt

features = ["age", "income", "credit_score"]
clf = DecisionTreeClassifier()
clf.fit(X_train[features], y_train)

plot_features_importance(features, clf.feature_importances_)
plt.show()
```
