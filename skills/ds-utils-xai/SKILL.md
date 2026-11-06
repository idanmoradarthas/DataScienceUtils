---
name: ds-utils-xai
description: >
  Explainable AI (XAI) and model interpretation for tree-based models. Use for plotting feature importance rankings, rendering Graphviz decision tree diagrams, and extracting executable decision paths. Do NOT use for linear models, neural network interpretability, or clustering.
version: 1.0.0
license: MIT
metadata:
  author: "Idan Morad <idanmorad.arthas@gmail.com>"
  tags:
    - xai
    - explainable-ai
    - feature-importance
    - decision-trees
    - model-interpretation
  documentation: "https://datascienceutils.readthedocs.io/en/stable/"
  package: "data-science-utils"
  repository: "https://github.com/idanmoradarthas/DataScienceUtils"
---

# ds-utils-xai

Explainable AI visualizers and interpretation methods for tree-based scikit-learn models.

## Purpose

ds_utils.xai provides tools to inspect and communicate tree-based model decisions. It provides feature importance ranking bar charts, Graphviz DOT rendering into matplotlib axes, and translation of tree split logic into executable Python functions.

- **When to use**: Explaining tree ensembles (Random Forest, Gradient Boosting, LightGBM, XGBoost) and visualizing single decision tree decision rules.
- **When NOT to use**: For linear model coefficient analysis (`coef_`), SHAP/LIME game-theoretic values, neural networks, or unsupervised clustering.

## Prerequisites

- Python 3.11 or higher.
- `data-science-utils` package installed (`pip install data-science-utils`).
- Core dependencies: `numpy`, `scikit-learn`, `matplotlib`, `pydotplus`.
- System Graphviz installation (`apt-get install graphviz` or `brew install graphviz`) for `draw_dot_data` and `draw_tree`.

## Instructions

### Task-to-Approach Map

| Interpretation Goal | Function | Target Model | Output Type |
|---|---|---|---|
| Rank features by global impact | `plot_features_importance` | Any tree model with `.feature_importances_` | Matplotlib Axes |
| Visualize decision tree structure | `draw_dot_data` / `draw_tree` | Single `DecisionTreeClassifier` or `DecisionTreeRegressor` | Matplotlib Axes |
| Convert tree rules into Python code | `generate_decision_paths` | Single `DecisionTreeClassifier` | Python string |

Exhaustive signatures, parameters, and return details are in [references/api_reference.md](references/api_reference.md).

---

### End-to-End Tree Model Interpretation Workflow

```text
              Fitted Model Available
                        │
         ┌──────────────┴──────────────┐
         ▼                             ▼
  Tree Ensemble                  Single Decision Tree
(Random Forest / GBDT)                 │
         │                             ├─────────────────────┐
         ▼                             ▼                     ▼
 Global Feature Importance       Visual Tree Graph      Rule Code Export
 (plot_features_importance)    (draw_dot_data/draw_tree)  (decision_paths)
```

#### Step 1: Global Feature Importance (`plot_features_importance`)
Visualize and rank features by their importance score.
- Features with zero importance are automatically filtered out.
- Features are sorted in descending order of importance.
- The `feature_names` list must strictly match the training column order.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from ds_utils.xai import plot_features_importance

# 1. Fit model
X_train = np.random.randn(100, 4)
y_train = (X_train[:, 0] + X_train[:, 1] > 0).astype(int)
features = ["feat_0", "feat_1", "feat_2", "feat_3"]

rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

# 2. Plot importance ranking
ax_imp = plot_features_importance(features, rf.feature_importances_)
plt.close()
```
- **What to look for**: High concentration of importance on a single feature (potential data leakage or dominant proxy) vs balanced feature utilization.
- **Action**: Verify that top features are valid business signals and not leaked target artifacts. Impurity-based `feature_importances_` are computed on training data and favour high-cardinality or continuous features; cross-check rankings with `sklearn.inspection.permutation_importance` on held-out data.

#### Step 2: Decision Tree Visual Inspection (`draw_dot_data` / `draw_tree`)
Render decision trees directly onto matplotlib canvases.
- **Prerequisite**: Requires Graphviz installed on the system (`apt-get install graphviz` or `brew install graphviz`).
- If Graphviz is unavailable, prefer native `sklearn.tree.plot_tree`.

```python
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from ds_utils.xai import draw_dot_data

dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(X_train, y_train)

dot_data = export_graphviz(
    dt,
    feature_names=features,
    class_names=["class_0", "class_1"],
    filled=True,
    rounded=True,
    out_file=None,
)
ax_tree = draw_dot_data(dot_data)
plt.close()
```
- **What to look for**: Deep paths with few samples, purity of leaf nodes, and sanity-check split thresholds.

#### Step 3: Rule Logic Export (`generate_decision_paths`)
Extract executable Python code from tree rules for auditability or deterministic rule-engine deployment. Note: `generate_decision_paths` issues a `DeprecationWarning` recommending native `sklearn.tree.export_text`. Similarly, `draw_tree` emits a `DeprecationWarning` recommending native `sklearn.tree.plot_tree`.

```python
from ds_utils.xai import generate_decision_paths

python_rules = generate_decision_paths(
    dt, feature_names=features, class_names=["class_0", "class_1"], tree_name="predict_credit"
)
print(python_rules[:150] + "...")
```
- **Action**: Share the generated Python logic with business domain experts for compliance validation.

---

## Examples

### Complete Model Interpretation Script

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from ds_utils.xai import plot_features_importance, generate_decision_paths

# 1. Synthetic dataset
np.random.seed(42)
X = np.random.randn(80, 3)
y = (X[:, 0] > 0.5).astype(int)
feature_names = ["credit_score", "debt_ratio", "income"]

# 2. Train interpretable model
tree = DecisionTreeClassifier(max_depth=2, random_state=42)
tree.fit(X, y)

# 3. Bar chart of feature impact
ax = plot_features_importance(feature_names, tree.feature_importances_)
plt.close()

# 4. Text rule generation
rules = generate_decision_paths(tree, feature_names=feature_names)
assert "def tree(" in rules
```

## Limitations

- **Tree models only**: `plot_features_importance` relies on `.feature_importances_`. It does not support linear models (`coef_`).
- **Graphviz binary dependency**: `draw_dot_data` and `draw_tree` invoke `pydotplus`, which requires Graphviz (`dot` CLI) installed in the operating system environment.
- **Single-tree model scope**: `draw_tree` operates on single decision tree estimators (`DecisionTreeClassifier` or `DecisionTreeRegressor`); `generate_decision_paths` specifically extracts classification rules from `DecisionTreeClassifier`. Neither works on ensemble models like `RandomForestClassifier`.

## Troubleshooting

| Error / Symptom | Root Cause | Solution |
|---|---|---|
| `ValueError: feature_names and feature_importance must have the same length` | Length mismatch between feature list and importance array | Ensure `len(feature_names) == len(clf.feature_importances_)` |
| `ValueError: Failed to create graph from dot data: GraphViz's executables not found` | System Graphviz binary is not installed in PATH | Install Graphviz via system package manager (`apt-get install graphviz` or `brew install graphviz`) or use `sklearn.tree.plot_tree` |
| `AttributeError: 'LogisticRegression' object has no attribute 'feature_importances_'` | Passing non-tree model | Use tree models (e.g. Random Forest, Decision Tree, XGBoost) |
| `ValueError: dot_data must not be empty` | `export_graphviz` produced empty output | Set `out_file=None` in `export_graphviz(...)` to return string |
