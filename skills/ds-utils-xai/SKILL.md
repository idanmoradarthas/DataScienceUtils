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
- **When NOT to use**: For linear model coefficient analysis (coef_), SHAP/LIME game-theoretic values, neural networks, or unsupervised clustering.

## Prerequisites

- Python 3.11 or higher.
- data-science-utils package installed (pip install data-science-utils).
- Core dependencies: numpy, scikit-learn, matplotlib, pydotplus.
- System Graphviz installation (apt-get install graphviz or brew install graphviz) for draw_dot_data and draw_tree.

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
         ┌──────────────────┴──────────────────┐
         ▼                                     ▼
   Tree Ensemble                         Single Decision Tree
(Random Forest / GBDT)                         │
         │                                     │
         ├──────────────────────────┐          │
         ▼                          ▼          │
 Global Importance              Surrogate Tree │
(plot_features_importance)     (explain forest)│
                                    │          │
         ┌──────────────────────────┴──────────┴─────────────┐
         ▼                                                   ▼
 Visual Tree Graph                                   Rule Code Export
(draw_dot_data / draw_tree)                      (generate_decision_paths)
```

#### Inputs you need

The steps below assume:
- Your fitted tree model: an ensemble (e.g. `rf = RandomForestClassifier().fit(X_train, y_train)`) and/or a single decision tree estimator `dt`.
- Your training feature matrix `X_train`, held-out test splits `X_test`, `y_test`.
- Feature column names list: `feature_names = list(X_train.columns)`.
- Class label names from your model: `class_names = [str(c) for c in rf.classes_]`.

```python
import pandas as pd

# Load your dataset (replace with your data)
df = pd.read_csv("your_dataset")
# X_train, X_test, y_train, y_test = train_test_split(...)
# rf = RandomForestClassifier(random_state=42).fit(X_train, y_train)
```

#### Step 1: Global Feature Importance (`plot_features_importance`)
Question: which features have the strongest global contribution in the tree ensemble, and does held-out permutation cross-check corroborate the ranking?

```python
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.inspection import permutation_importance
from ds_utils.xai import plot_features_importance

# 1. Impurity-based feature importance from your fitted forest
feature_names = list(X_train.columns) if hasattr(X_train, "columns") else [f"feat_{i}" for i in range(X_train.shape[1])]
ax_imp = plot_features_importance(feature_names, rf.feature_importances_)
plt.close()

# 2. Held-out cross-check via permutation importance on test split
perm_res = permutation_importance(rf, X_test, y_test, n_repeats=10, random_state=42)
perm_series = pd.Series(perm_res.importances_mean, index=feature_names).sort_values(ascending=False)
print("Top permutation importances on test split:\n", perm_series.head(5))
```
- **What to look for**: Discrepancies between impurity importance and permutation importance on held-out test data.
- **Action**: Verify that top features are valid business signals and not leaked artifacts.
- **Caution**: Impurity importance is computed on training data and favors continuous or high-cardinality features; always cross-check with permutation importance on held-out data.

#### Step 2: Surrogate Shallow Tree & Fidelity Check
Question: does a shallow decision tree accurately approximate the predictions of the complex ensemble on held-out data?

```python
from sklearn.tree import DecisionTreeClassifier

# Fit a shallow surrogate tree on the forest's predictions
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(X_train, rf.predict(X_train))  # explain the forest, not the labels

# Measure surrogate fidelity: agreement with the forest on held-out data
fidelity = (dt.predict(X_test) == rf.predict(X_test)).mean()
print(f"Surrogate tree fidelity to forest on test split: {fidelity:.2%}")
```
- **What to look for**: High prediction agreement between the shallow tree and the forest; fidelity of roughly 90% or more is a common rule of thumb, not a guarantee.
- **Action**: If fidelity is low, increase tree depth or explain regional sub-domains rather than the entire feature space.
- **Caution**: A shallow tree is an authentic explanation of a complex ensemble only when model agreement (fidelity) is high.

#### Step 3: Decision Tree Visual Inspection (`draw_dot_data` / `draw_tree`)
Question: what specific decision logic, split thresholds, and leaf purities govern the tree?

```python
from sklearn.tree import export_graphviz
from ds_utils.xai import draw_dot_data

class_names = [str(c) for c in dt.classes_]
dot_data = export_graphviz(
    dt,
    feature_names=feature_names,
    class_names=class_names,
    filled=True,
    rounded=True,
    out_file=None,
)
ax_tree = draw_dot_data(dot_data)
plt.close()
```
- **What to look for**: Dominant splits near the root, sample sizes per leaf, and node purity.
- **Action**: Share visual decision paths with stakeholders to validate business logic.
- **Caution**: draw_dot_data requires system Graphviz; if unavailable, prefer native scikit-learn plot_tree.

#### Step 4: Rule Logic Export (`generate_decision_paths`)
Question: how can the tree splits be extracted as an auditable, deterministic Python function?

```python
from ds_utils.xai import generate_decision_paths

python_rules = generate_decision_paths(
    dt, feature_names=feature_names, class_names=class_names, tree_name="predict_case"
)
print(python_rules[:150] + "...")
```
- **What to look for**: Nested if/else conditions corresponding to tree branches.
- **Action**: Audit decision thresholds against domain regulations or operational constraints.
- **Caution**: generate_decision_paths operates on single DecisionTreeClassifier instances; it cannot extract paths from tree ensembles. Feature names must be valid Python identifiers; rename columns (spaces, dashes, dots) before fitting the tree, otherwise the exported code does not compile.

---

## Examples

### Runnable demo on a bundled dataset (replace the loading lines with your own data)

```python
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from ds_utils.xai import plot_features_importance, generate_decision_paths

# 1. Load bundled dataset, sanitize feature names to valid identifiers, and partition
cancer = load_breast_cancer(as_frame=True)
X = cancer.data.rename(columns=lambda c: c.replace(" ", "_"))
X_tr, X_te, y_tr, y_te = train_test_split(X, cancer.target, test_size=0.3, random_state=42)
feature_names = list(X.columns)
class_names = [str(name) for name in cancer.target_names]

# 2. Train forest and plot global feature importance
rf = RandomForestClassifier(random_state=42)
rf.fit(X_tr, y_tr)
ax_imp = plot_features_importance(feature_names, rf.feature_importances_)
plt.close()

# 3. Train shallow surrogate tree to explain forest and check fidelity
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(X_tr, rf.predict(X_tr))
fidelity = (dt.predict(X_te) == rf.predict(X_te)).mean()

# 4. Generate auditable Python decision path rules
python_rules = generate_decision_paths(
    dt, feature_names=feature_names, class_names=class_names, tree_name="predict_cancer"
)
compile(python_rules, "<rules>", "exec")
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
| SyntaxError when compiling the output of generate_decision_paths | Feature names contain spaces or punctuation | Rename columns to identifiers before fitting; verify with compile(...) |
