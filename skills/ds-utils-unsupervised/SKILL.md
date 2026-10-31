---
name: ds-utils-unsupervised
description: >
  Unsupervised clustering analysis, optimal cluster count selection, and cluster health diagnostics. Use for elbow loss curves, cluster cardinality, centroid magnitude, and anomaly detection in clusters. Do NOT use for supervised classification metrics, text binarization, or dimensionality reduction.
version: "1.0.0"
license: MIT
metadata:
  author: "Idan Morad <idanmorad.arthas@gmail.com>"
  tags:
    - clustering
    - unsupervised-learning
    - kmeans
    - cluster-cardinality
    - elbow-method
  documentation: "https://datascienceutils.readthedocs.io/en/stable/"
  package: "data-science-utils"
  repository: "https://github.com/idanmoradarthas/DataScienceUtils"
---

# ds-utils-unsupervised

Tools for choosing cluster count and diagnosing clustering quality using cardinality and magnitude analysis.

## Purpose

`ds_utils.unsupervised` provides diagnostics for centroid-based clustering (specifically `sklearn.cluster.KMeans`). It helps data scientists determine the optimal number of clusters $k$ via custom distance loss sweeps, then evaluate cluster compactness and stability using cardinality (number of points) and magnitude (total distance to centroid).

- **When to use**: Selecting $k$ with the elbow method, verifying that clusters are balanced and compact, and spotting anomalous or diffuse clusters.
- **When NOT to use**: For supervised classification/regression metrics (use `ds-utils-metrics`), PCA explained variance (use `ds-utils-preprocess`), or density-based clustering like DBSCAN without centroids.

## Prerequisites

- Python 3.11 or higher.
- `data-science-utils` package installed (`pip install data-science-utils`).
- Core dependencies: `numpy`, `pandas`, `scikit-learn`, `scipy`, `matplotlib`.

## Instructions

### Task-to-Approach Map

| Clustering Diagnostic Task | Function | Required Inputs | Output |
|---|---|---|---|
| Determine optimal cluster count ($k$) | `plot_loss_vs_cluster_number` | `X`, `k_min`, `k_max`, `distance_function` | Matplotlib Axes |
| Check balance / points per cluster | `plot_cluster_cardinality` | `labels_` | Matplotlib Axes |
| Measure compactness per cluster | `plot_cluster_magnitude` | `X`, `labels_`, `cluster_centers_`, `distance_func` | Matplotlib Axes |
| Identify anomalous or diffuse clusters | `plot_magnitude_vs_cardinality` | `X`, `labels_`, `cluster_centers_`, `distance_func` | Matplotlib Axes |

Full signatures, parameter options, and common mistakes are documented in [references/api_reference.md](references/api_reference.md).

---

### End-to-End Clustering Workflow

Follow this sequence to go from unscaled features to a validated clustering model:

```text
               Unlabeled Data Matrix
                         │
           Scale Features (e.g. StandardScaler)
                         │
                         ▼
        Sweep k and Identify Elbow Point
          (plot_loss_vs_cluster_number)
                         │
                         ▼
             Fit Final KMeans(n_clusters=k)
                         │
         ┌───────────────┴───────────────┐
         ▼                               ▼
  Check Balance                   Check Compactness
(plot_cluster_cardinality)      (plot_cluster_magnitude)
         │                               │
         └───────────────┬───────────────┘
                         ▼
           Diagnose Cluster Consistency
          (plot_magnitude_vs_cardinality)
```

#### Step 1: Sweep $k$ and Select Elbow Point (`plot_loss_vs_cluster_number`)
Test a range of cluster counts to observe the diminishing return in distance loss.
- **Input requirement**: `k_min` and `k_max` are required integers without defaults.
- **Distance function**: Must be a **callable** (such as `scipy.spatial.distance.euclidean`), never a string. Pass algorithm parameters (e.g. `random_state`) via the `algorithm_parameters` dict.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import StandardScaler
from ds_utils.unsupervised import plot_loss_vs_cluster_number

# 1. Scale data
np.random.seed(42)
X_raw = np.random.randn(150, 4)
X_scaled = StandardScaler().fit_transform(X_raw)

# 2. Sweep k from 2 to 8
ax_loss = plot_loss_vs_cluster_number(
    X_scaled,
    k_min=2,
    k_max=8,
    distance_function=euclidean,
    algorithm_parameters={"random_state": 42},
)
plt.close()
```
- **What to look for**: The "elbow" point where adding more clusters no longer yields sharp decreases in loss.
- **Action**: Pick the elbow $k$ (e.g., $k=3$) and fit your final model.

#### Step 2: Fit Final Model and Inspect Cardinality (`plot_cluster_cardinality`)
Fit `KMeans` with the chosen $k$ and inspect cluster member counts.

```python
from sklearn.cluster import KMeans
from ds_utils.unsupervised import plot_cluster_cardinality

kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(X_scaled)

ax_card = plot_cluster_cardinality(labels)
plt.close()
```
- **What to look for**: Tiny outlier clusters (cardinality near 1 or 2) or one massive cluster that swallowed the entire dataset.
- **Action**: If one cluster dominates heavily, inspect whether feature scaling was missed or if $k$ should be increased.

#### Step 3: Measure Compactness (`plot_cluster_magnitude`)
Magnitude measures the sum of point-to-centroid distances within each cluster.

```python
from ds_utils.unsupervised import plot_cluster_magnitude

ax_mag = plot_cluster_magnitude(X_scaled, labels, kmeans.cluster_centers_, distance_function=euclidean)
plt.close()
```
- **What to look for**: Disproportionately high magnitude in clusters indicates high internal dispersion / loose clusters.
- **Action**: If a cluster has excessive magnitude relative to others, evaluate whether it contains multiple sub-populations.

#### Step 4: Diagnose Anomalies (`plot_magnitude_vs_cardinality`)
Under uniform density, cluster magnitude should scale roughly linearly with cardinality.

```python
from ds_utils.unsupervised import plot_magnitude_vs_cardinality

ax_joint = plot_magnitude_vs_cardinality(X_scaled, labels, kmeans.cluster_centers_, distance_function=euclidean)
plt.close()
```
- **Interpretation**: Points far above the trendline represent clusters with low cardinality but high magnitude (i.e. dispersed outliers). Points far below are unusually dense cores.
- **Action**: Inspect high-magnitude/low-cardinality clusters for data quality issues or anomalous instances.

---

## Examples

### Complete Clustering Diagnostics Script

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from ds_utils.unsupervised import (
    plot_loss_vs_cluster_number,
    plot_cluster_cardinality,
    plot_cluster_magnitude,
    plot_magnitude_vs_cardinality,
)

# 1. Generate synthetic data
X, _ = make_blobs(n_samples=200, centers=4, n_features=3, random_state=42)
X_norm = StandardScaler().fit_transform(X)

# 2. Find optimal k
plot_loss_vs_cluster_number(X_norm, k_min=2, k_max=6, distance_function=euclidean)
plt.close()

# 3. Fit optimal KMeans
model = KMeans(n_clusters=4, random_state=42)
model.fit(X_norm)

# 4. Evaluate cardinality & magnitude
plot_cluster_cardinality(model.labels_)
plt.close()

plot_cluster_magnitude(X_norm, model.labels_, model.cluster_centers_, euclidean)
plt.close()

plot_magnitude_vs_cardinality(X_norm, model.labels_, model.cluster_centers_, euclidean)
plt.close()
```

## Limitations

- **Centroid-based algorithms only**: All functions expect centroid coordinates (`cluster_centers_`) and discrete partition labels. They do not support soft clustering, DBSCAN, or agglomerative clustering without centroids.
- **Distance function callable**: Distance parameters must be callable functions, not strings.

## Troubleshooting

| Error / Symptom | Root Cause | Solution |
|---|---|---|
| `ValueError: No valid results were obtained. Check your input data and parameters.` (preceded by `Error occurred for k=...: 'str' object is not callable`) | Passed string metric (e.g. `"euclidean"`) instead of callable | Pass callable metric function, e.g., `scipy.spatial.distance.euclidean` |
| `TypeError: missing 2 required positional arguments: 'k_min' and 'k_max'` | Omitted `k_min` or `k_max` | Specify both integers explicitly, e.g. `k_min=2, k_max=10` |
| `AttributeError: 'DBSCAN' object has no attribute 'cluster_centers_'` | Using DBSCAN or model without centroids | Compute centroids manually or use with `sklearn.cluster.KMeans` |
| Plot axes blank or not appearing | Using non-interactive backend without saving | Call `plt.show()` or `plt.savefig("cluster.png")` |
