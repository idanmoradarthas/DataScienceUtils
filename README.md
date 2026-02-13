# Data Science Utils: Frequently Used Methods for Data Science

[![License: MIT](https://img.shields.io/github/license/idanmoradarthas/DataScienceUtils)](https://opensource.org/licenses/MIT)
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/idanmoradarthas/DataScienceUtils)
[![GitHub issues](https://img.shields.io/github/issues/idanmoradarthas/DataScienceUtils)](https://github.com/idanmoradarthas/DataScienceUtils/issues)
[![Documentation Status](https://readthedocs.org/projects/datascienceutils/badge/?version=latest)](https://datascienceutils.readthedocs.io/en/latest/?badge=latest)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/data-science-utils)
![PyPI - Wheel](https://img.shields.io/pypi/wheel/data-science-utils)
[![PyPI version](https://badge.fury.io/py/data-science-utils.svg)](https://badge.fury.io/py/data-science-utils)
[![Anaconda-Server Badge](https://anaconda.org/idanmorad/data-science-utils/badges/version.svg)](https://anaconda.org/idanmorad/data-science-utils)
![Build Status](https://github.com/idanmoradarthas/DataScienceUtils/actions/workflows/test.yml/badge.svg?branch=master)
[![Coverage Status](https://coveralls.io/repos/github/idanmoradarthas/DataScienceUtils/badge.svg?branch=master)](https://coveralls.io/github/idanmoradarthas/DataScienceUtils?branch=master)

Data Science Utils extends the Scikit-Learn API and Matplotlib API to provide simple methods that simplify tasks and
visualizations for data science projects.

# Code Examples and Documentation

This README provides several examples of the package's capabilities. For complete documentation, including more methods and additional examples, please visit:
**[https://datascienceutils.readthedocs.io/en/stable/](https://datascienceutils.readthedocs.io/en/stable/)**

The API of the package is built to work with the Scikit-Learn API and Matplotlib API.

## Metrics

### Plot Confusion Matrix

Computes and plots a confusion matrix, False Positive Rate, False Negative Rate, Accuracy, and F1 score of a
classification.

```python
from ds_utils.metrics import plot_confusion_matrix

plot_confusion_matrix(y_test, y_pred, [0, 1, 2])
```

![multi label classification confusion matrix](https://raw.githubusercontent.com/idanmoradarthas/DataScienceUtils/master/tests/baseline_images/test_metrics/test_plot_confusion_matrix_binary.png)

### Plot Metric Growth per Labeled Instances

Receives train and test sets, and plots the given metric change with an increasing number of trained instances.

```python
from ds_utils.metrics import plot_metric_growth_per_labeled_instances
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

plot_metric_growth_per_labeled_instances(
    x_train, y_train, x_test, y_test,
    {
        "DecisionTreeClassifier": DecisionTreeClassifier(random_state=0),
        "RandomForestClassifier": RandomForestClassifier(random_state=0, n_estimators=5)
    }
)
```

![metric growth per labeled instances with n samples](https://raw.githubusercontent.com/idanmoradarthas/DataScienceUtils/master/tests/baseline_images/test_metrics/test_plot_metric_growth_per_labeled_instances_with_n_samples.png)

### Visualize Accuracy Grouped by Probability

Receives test true labels and classifier probability predictions, divides and classifies the results, and finally
plots a stacked bar chart with the results. [Original code](https://github.com/EthicalML/XAI)

```python
from ds_utils.metrics import visualize_accuracy_grouped_by_probability

visualize_accuracy_grouped_by_probability(
    test["target"],
    1,
    classifier.predict_proba(test[selected_features]),
    display_breakdown=False
)
```

Without breakdown:

![visualize_accuracy_grouped_by_probability](https://raw.githubusercontent.com/idanmoradarthas/DataScienceUtils/master/tests/baseline_images/test_metrics/test_visualize_accuracy_grouped_by_probability_default.png)

With breakdown:

![visualize_accuracy_grouped_by_probability_with_breakdown](https://raw.githubusercontent.com/idanmoradarthas/DataScienceUtils/master/tests/baseline_images/test_metrics/test_visualize_accuracy_grouped_by_probability_with_breakdown.png)

### Receiver Operating Characteristic (ROC) Curve with Probabilities (Thresholds) Annotations

Plot ROC curves with threshold annotations for multiple classifiers, using plotly as a backend.

```python
from ds_utils.metrics import plot_roc_curve_with_thresholds_annotations

classifiers_names_and_scores_dict = {
    "Decision Tree": tree_clf.predict_proba(X_test)[:, 1],
    "Random Forest": rf_clf.predict_proba(X_test)[:, 1],
    "XGBoost": xgb_clf.predict_proba(X_test)[:, 1]
}
fig = plot_roc_curve_with_thresholds_annotations(
    y_true,
    classifiers_names_and_scores_dict,
    positive_label=1
)
fig.show()
```

![plot roc curve with thresholds annotations](https://raw.githubusercontent.com/idanmoradarthas/DataScienceUtils/master/tests/baseline_images/test_metrics/test_plot_roc_curve_with_thresholds_annotations_default.png)

### Precision-Recall Curve with Probabilities (Thresholds) Annotations

Plot Precision-Recall curves with threshold annotations for multiple classifiers, using plotly as a backend.

```python
from ds_utils.metrics import plot_precision_recall_curve_with_thresholds_annotations

classifiers_names_and_scores_dict = {
    "Decision Tree": tree_clf.predict_proba(X_test)[:, 1],
    "Random Forest": rf_clf.predict_proba(X_test)[:, 1],
    "XGBoost": xgb_clf.predict_proba(X_test)[:, 1]
}
fig = plot_precision_recall_curve_with_thresholds_annotations(
    y_true,
    classifiers_names_and_scores_dict,
    positive_label=1
)
fig.show()
```

![plot precision recall curve with thresholds annotations](https://raw.githubusercontent.com/idanmoradarthas/DataScienceUtils/master/tests/baseline_images/test_metrics/test_plot_precision_recall_curve_with_thresholds_annotations_default.png)

## Preprocess

The preprocess module is organized into focused submodules:
- **visualization** - Feature visualization and correlation plots
- **statistics** - Statistical computations and feature analysis

### Visualize Feature

Receives a feature and visualizes its values on a graph:

* If the feature is float, the method plots a violin distribution. You can optionally exclude outliers using the IQR fence method.
* If the feature is datetime, the method plots a 2D heatmap showing day-of-week vs year-week patterns, making weekly and yearly trends immediately visible.
* If the feature is object, categorical, boolean, or integer, the method plots a count plot (histogram). For high-cardinality features (>10 unique values), it shows the top 10 with "Other values". You can customize sorting with `order` (e.g., by count or alphabetically) and toggle count labels with `show_counts`.

When `remove_na=False` (default for `visualize_feature`), missing values are handled as follows:
* **Float**: Missing values are dropped before plotting (violin plots require valid data).
* **Datetime**: Missing values are dropped before plotting.
* **Categorical/Integer/Boolean**: Missing values are counted and shown as a separate category (if present).

```python
from ds_utils.preprocess.visualization import visualize_feature

# Basic usage
visualize_feature(X_train["feature"])

# Handle NA values (removes them before plotting)
visualize_feature(X_train["feature_with_nas"], remove_na=True)

# For float features, you can control outlier handling
visualize_feature(X_train["float_feature"], include_outliers=True)  # Default
visualize_feature(X_train["float_feature"], include_outliers=False, outlier_iqr_multiplier=1.5)

# For datetime features, you can specify the first day of the week
visualize_feature(X_train["datetime_feature"], first_day_of_week="Monday")  # Default
visualize_feature(X_train["datetime_feature"], first_day_of_week="Sunday")

# For categorical/object/boolean/int features, customize order and counts
visualize_feature(X_train["category_feature"], show_counts=True)  # Default, shows count labels on bars
visualize_feature(X_train["category_feature"], show_counts=False)  # Hides count labels
visualize_feature(X_train["category_feature"], order="count_desc")  # Sort by descending count
visualize_feature(X_train["category_feature"], order=["High", "Medium", "Low"])  # Explicit category order
```

| Feature Type      | Plot                                                                                                                                                                             |
|-------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Float             | ![](https://raw.githubusercontent.com/idanmoradarthas/DataScienceUtils/master/tests/baseline_images/test_preprocess/test_visualize_feature_float_datetime_int_float.png)                            |
| Integer           | ![](https://raw.githubusercontent.com/idanmoradarthas/DataScienceUtils/master/tests/baseline_images/test_preprocess/test_visualize_feature_float_datetime_int_int.png)                              |
| Datetime          | ![](https://raw.githubusercontent.com/idanmoradarthas/DataScienceUtils/master/tests/baseline_images/test_preprocess/test_visualize_feature_float_datetime_int_datetime.png)                         |
| Category / Object | ![](https://raw.githubusercontent.com/idanmoradarthas/DataScienceUtils/master/tests/baseline_images/test_preprocess/test_visualize_feature_object_category_more_than_10_categories_show_counts.png) |
| Boolean           | ![](https://raw.githubusercontent.com/idanmoradarthas/DataScienceUtils/master/tests/baseline_images/test_preprocess/test_visualize_feature_bool_show_counts.png)                             |

### Get Correlated Features

Calculate which features are correlated above a threshold and extract a data frame with the correlations and correlation
to the target feature.

```python
from ds_utils.preprocess.statistics import get_correlated_features

correlations = get_correlated_features(correlation_matrix, features, target)
```

| level_0                | level_1                | level_0_level_1_corr | level_0_target_corr | level_1_target_corr |
|------------------------|------------------------|----------------------|---------------------|---------------------|
| income_category_Low    | income_category_Medium | 1.0                  | 0.1182165609358650  | 0.11821656093586504 |
| term\_ 36 months       | term\_ 60 months       | 1.0                  | 0.1182165609358650  | 0.11821656093586504 |
| interest_payments_High | interest_payments_Low  | 1.0                  | 0.1182165609358650  | 0.11821656093586504 |

### Visualize Correlations

Compute pairwise correlation of columns, excluding NA/null values, and visualize it with a heat map.
[Original code](https://seaborn.pydata.org/examples/many_pairwise_correlations.html)

```python
from ds_utils.preprocess.visualization import visualize_correlations

visualize_correlations(correlation_matrix)
```

![visualize features](https://raw.githubusercontent.com/idanmoradarthas/DataScienceUtils/master/tests/baseline_images/test_preprocess/test_visualize_correlations_default.png)

### Plot Correlation Dendrogram

Plot a dendrogram of a correlation matrix. This chart hierarchically displays the most correlated variables by
connecting them in a tree-like structure. The closer to the right that the connection is, the more correlated the
features are. [Original code](https://github.com/EthicalML/XAI)

```python
from ds_utils.preprocess.visualization import plot_correlation_dendrogram

plot_correlation_dendrogram(correlation_matrix)
```

![plot correlation dendrogram](https://raw.githubusercontent.com/idanmoradarthas/DataScienceUtils/master/tests/baseline_images/test_preprocess/test_plot_correlation_dendrogram_default.png)

### Plot Features' Interaction

Plots the joint distribution between two features:

* If both features are either categorical, boolean, or object, the method plots the shared histogram.
* If one feature is either categorical, boolean, or object and the other is numeric, the method plots a boxplot chart.
* If one feature is datetime and the other is numeric or datetime, the method plots a line plot graph.
* If one feature is datetime and the other is either categorical, boolean, or object, the method plots a violin plot (combination of boxplot and kernel density estimate).
* If both features are numeric, the method plots a scatter graph.

When `remove_na=False` (default), missing values are visualized:
* For numeric/datetime plots: Missing values appear as rug plots or markers on the axes boundaries.
* For categorical plots: Missing values are included as a separate category if present.

```python
from ds_utils.preprocess.visualization import plot_features_interaction

plot_features_interaction(data, "feature_1", "feature_2")
```

|                 | Numeric                                                                                                                                                                              | Categorical                                                                                                                                                                           | Boolean                                                                                                                                                                                | Datetime                                                                                                                                                                           |
|-----------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Numeric**     | ![](https://raw.githubusercontent.com/idanmoradarthas/DataScienceUtils/master/tests/baseline_images/test_preprocess/test_plot_relationship_between_features_both_numeric.png)        |                                                                                                                                                                                       |                                                                                                                                                                                        |                                                                                                                                                                                    |
| **Categorical** | ![](https://raw.githubusercontent.com/idanmoradarthas/DataScienceUtils/master/tests/baseline_images/test_preprocess/test_plot_relationship_between_features_numeric_categorical.png) | ![](https://raw.githubusercontent.com/idanmoradarthas/DataScienceUtils/master/tests/baseline_images/test_preprocess/test_plot_relationship_between_features_both_categorical.png)     |                                                                                                                                                                                        |                                                                                                                                                                                    |
| **Boolean**     | ![](https://raw.githubusercontent.com/idanmoradarthas/DataScienceUtils/master/tests/baseline_images/test_preprocess/test_plot_relationship_between_features_numeric_boolean.png)     | ![](https://raw.githubusercontent.com/idanmoradarthas/DataScienceUtils/master/tests/baseline_images/test_preprocess/test_plot_relationship_between_features_categorical_bool.png)     | ![](https://raw.githubusercontent.com/idanmoradarthas/DataScienceUtils/master/tests/baseline_images/test_preprocess/test_plot_relationship_between_features_both_bool.png)             |                                                                                                                                                                                    |
| **Datetime**    | ![](https://raw.githubusercontent.com/idanmoradarthas/DataScienceUtils/master/tests/baseline_images/test_preprocess/test_plot_relationship_between_features_datetime_numeric.png)    | ![](https://raw.githubusercontent.com/idanmoradarthas/DataScienceUtils/master/tests/baseline_images/test_preprocess/test_plot_relationship_between_features_datetime_categorical.png) | ![](https://raw.githubusercontent.com/idanmoradarthas/DataScienceUtils/master/tests/baseline_images/test_preprocess/test_plot_relationship_between_features_datetime_bool_default.png) | ![](https://raw.githubusercontent.com/idanmoradarthas/DataScienceUtils/master/tests/baseline_images/test_preprocess/test_plot_relationship_between_features_datetime_datetime.png) |

### Extract Statistics DataFrame per Label

This method calculates comprehensive statistical metrics for numerical features grouped by label values. Use this when
you want to:

- Analyze how a numerical feature's distribution varies across different categories
- Detect potential patterns or anomalies in feature behavior per group
- Generate detailed statistical summaries for reporting or analysis
- Understand the relationship between features and target variables

```python
from ds_utils.preprocess.statistics import extract_statistics_dataframe_per_label

extract_statistics_dataframe_per_label(
    df=df,
    feature_name='amount',
    label_name='category'
)
```

| category | count | null_count | mean  | min | 1_percentile | 5_percentile | 25_percentile | median | 75_percentile | 95_percentile | 99_percentile | max   |
|----------|-------|------------|-------|-----|--------------|--------------|---------------|--------|---------------|---------------|---------------|-------|
| A        | 2     | 0          | 150.0 | 100 | 100.0        | 100.0        | 100.0         | 150.0  | 200.0         | 200.0         | 200.0         | 200.0 |
| B        | 2     | 0          | 225.0 | 150 | 150.0        | 150.0        | 150.0         | 225.0  | 300.0         | 300.0         | 300.0         | 300.0 |
| C        | 2     | 0          | 212.5 | 175 | 175.0        | 175.0        | 175.0         | 212.5  | 250.0         | 250.0         | 250.0         | 250.0 |

### Compute Mutual Information

This method computes mutual information scores between features and a target label. Mutual information measures the mutual dependence between two variables - higher scores indicate stronger relationships between features and the target label. Features are automatically categorized as numerical or discrete and preprocessed accordingly.

Use this method when you want to:
- Identify which features have the strongest relationship with your target variable
- Perform feature selection based on statistical dependence
- Understand feature importance from an information theory perspective
- Compare the predictive value of different types of features

```python
from ds_utils.preprocess.statistics import compute_mutual_information

# Compute mutual information scores for all features
mi_scores = compute_mutual_information(
    df=df,
    features=['feature1', 'feature2', 'feature3'],
    label_col='target',
    random_state=42
)
```
The result will be a DataFrame sorted by MI score (descending):

| feature_name | mi_score |
|--------------|----------|
| feature1     | 0.245    |
| feature3     | 0.182    |
| feature2     | 0.091    |

## Strings

### Append Tags to Frame

This method extracts tags from a given field and appends them as new columns to the dataframe.

Consider a dataset that looks like this:

``x_train``:

| article_name | article_tags |
|--------------|--------------|
| 1            | ds,ml,dl     |
| 2            | ds,ml        |

``x_test``:

| article_name | article_tags |
|--------------|--------------|
| 3            | ds,ml,py     |

Using this code:

```python
import pandas as pd
from ds_utils.strings import append_tags_to_frame

x_train = pd.DataFrame([{"article_name": "1", "article_tags": "ds,ml,dl"},
                        {"article_name": "2", "article_tags": "ds,ml"}])
x_test = pd.DataFrame([{"article_name": "3", "article_tags": "ds,ml,py"}])

x_train_with_tags, x_test_with_tags = append_tags_to_frame(x_train, x_test, "article_tags", "tag_")
```

The result will be:

``x_train_with_tags``:

| article_name | tag_ds | tag_ml | tag_dl |
|--------------|--------|--------|--------|
| 1            | 1      | 1      | 1      |
| 2            | 1      | 1      | 0      |

``x_test_with_tags``:

| article_name | tag_ds | tag_ml | tag_dl |
|--------------|--------|--------|--------|
| 3            | 1      | 1      | 0      |

### Extract Significant Terms from Subset

This method returns interesting or unusual occurrences of terms in a subset. It is based on the
[elasticsearch significant_text aggregation](https://www.elastic.co/guide/en/elasticsearch/reference/current/search-aggregations-bucket-significantterms-aggregation.html#_scripted).

```python
import pandas as pd
from ds_utils.strings import extract_significant_terms_from_subset

corpus = ['This is the first document.', 'This document is the second document.',
          'And this is the third one.', 'Is this the first document?']
data_frame = pd.DataFrame(corpus, columns=["content"])
# Let's differentiate between the last two documents from the full corpus
subset_data_frame = data_frame[data_frame.index > 1]
terms = extract_significant_terms_from_subset(data_frame, subset_data_frame,
                                              "content")
```

The output for ``terms`` will be the following table:

| third | one | and | this | the  | is   | first | document | second |
|-------|-----|-----|------|------|------|-------|----------|--------|
| 1.0   | 1.0 | 1.0 | 0.67 | 0.67 | 0.67 | 0.5   | 0.25     | 0.0    |

## Unsupervised

### Cluster Cardinality

Cluster cardinality is the number of examples per cluster. This method plots the number of points per cluster as a bar
chart.

```python
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from ds_utils.unsupervised import plot_cluster_cardinality

data = pd.read_csv(path / to / dataset)
estimator = KMeans(n_clusters=8, random_state=42)
estimator.fit(data)

plot_cluster_cardinality(estimator.labels_)

plt.show()
```

![Cluster Cardinality](https://raw.githubusercontent.com/idanmoradarthas/DataScienceUtils/master/tests/baseline_images/test_unsupervised/test_cluster_cardinality.png)

### Plot Cluster Magnitude

Cluster magnitude is the sum of distances from all examples to the centroid of the cluster. This method plots the
Total Point-to-Centroid Distance per cluster as a bar chart.

```python
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial.distance import euclidean
from ds_utils.unsupervised import plot_cluster_magnitude

data = pd.read_csv(path / to / dataset)
estimator = KMeans(n_clusters=8, random_state=42)
estimator.fit(data)

plot_cluster_magnitude(data, estimator.labels_, estimator.cluster_centers_, euclidean)

plt.show()
```

![Plot Cluster Magnitude](https://raw.githubusercontent.com/idanmoradarthas/DataScienceUtils/master/tests/baseline_images/test_unsupervised/test_plot_cluster_magnitude.png)

### Magnitude vs. Cardinality

Higher cluster cardinality tends to result in a higher cluster magnitude, which intuitively makes sense. Clusters
are considered anomalous when cardinality doesn't correlate with magnitude relative to the other clusters. This
method helps find anomalous clusters by plotting magnitude against cardinality as a scatter plot.

```python
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial.distance import euclidean
from ds_utils.unsupervised import plot_magnitude_vs_cardinality

data = pd.read_csv(path / to / dataset)
estimator = KMeans(n_clusters=8, random_state=42)
estimator.fit(data)

plot_magnitude_vs_cardinality(data, estimator.labels_, estimator.cluster_centers_, euclidean)

plt.show()
```

![Magnitude vs. Cardinality](https://raw.githubusercontent.com/idanmoradarthas/DataScienceUtils/master/tests/baseline_images/test_unsupervised/test_plot_magnitude_vs_cardinality.png)

### Optimum Number of Clusters

K-means clustering requires you to decide the number of clusters `k` beforehand. This method runs the KMeans algorithm
and
increases the cluster number at each iteration. The total magnitude or sum of distances is used as the loss metric.

Note: Currently, this method only works with ``sklearn.cluster.KMeans``.

```python
import pandas as pd
from matplotlib import pyplot as plt
from scipy.spatial.distance import euclidean
from ds_utils.unsupervised import plot_loss_vs_cluster_number

data = pd.read_csv(path / to / dataset)

plot_loss_vs_cluster_number(data, 3, 20, euclidean)

plt.show()
```

![Optimum Number of Clusters](https://raw.githubusercontent.com/idanmoradarthas/DataScienceUtils/master/tests/baseline_images/test_unsupervised/test_plot_loss_vs_cluster_number.png)

## XAI (Explainable AI)

## Plot Feature Importance

This method plots feature importance as a bar chart, helping to visualize which features have the most significant
impact on the model's decisions.

```python
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from ds_utils.xai import plot_features_importance

# Load the dataset
data = pd.read_csv(path / to / dataset)
target = data["target"]
features = data.columns.tolist()
features.remove("target")

# Train a decision tree classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(data[features], target)

# Plot feature importance
plot_features_importance(features, clf.feature_importances_)

plt.show()
```

![Plot Features Importance](https://raw.githubusercontent.com/idanmoradarthas/DataScienceUtils/master/tests/baseline_images/test_xai/test_plot_features_importance.png)

This visualization helps in understanding which features are most influential in the model's decision-making process,
providing valuable insights for feature selection and model interpretation.

## Explore More

Excited about what you've seen so far? There's even more to discover! Dive deeper into each module to unlock the full
potential of DataScienceUtils:

* [Metrics](https://datascienceutils.readthedocs.io/en/latest/metrics.html) - Powerful methods for calculating and
  visualizing algorithm performance evaluation. Gain insights into how your models are performing.

* [Preprocess](https://datascienceutils.readthedocs.io/en/latest/preprocess/index.html) - Essential data preprocessing
  techniques to prepare your data for training. Now organized into visualization and statistics submodules for better code organization.

* [Strings](https://datascienceutils.readthedocs.io/en/latest/strings.html) - Efficient methods for manipulating and
  processing strings in dataframes. Handle text data with ease.

* [Unsupervised](https://datascienceutils.readthedocs.io/en/latest/unsupervised.html) - Tools for calculating and
  visualizing the performance of unsupervised models. Understand your clustering and dimensionality reduction results
  better.

* [XAI](https://datascienceutils.readthedocs.io/en/latest/xai.html) - Methods to help explain model decisions, making
  your AI more interpretable and trustworthy.

Each module is designed to streamline your data science workflow, providing you with the tools you need to preprocess
data, train models, evaluate performance, and interpret results. Check out the detailed documentation for each module to
see how DataScienceUtils can enhance your projects!

## Contributing

We're thrilled that you're interested in contributing to Data Science Utils! Your contributions help make this project
better for everyone. Whether you're a seasoned developer or just getting started, there's a place for you here.

### How to Contribute

1. **Find an area to contribute to**: Check out our [issues](https://github.com/idanmoradarthas/DataScienceUtils/issues)
   page for open tasks, or think of a feature you'd like to add.

2. **Fork the repository**: Make your own copy of the project to work on.

3. **Create a branch**: Make your changes in a new git branch.

4. **Make your changes**: Add your improvements or fixes. We appreciate:
    - Bug reports and fixes
    - Feature requests and implementations
    - Documentation improvements
    - Performance optimizations
    - User experience enhancements

5. **Test your changes**: Ensure your code works as expected and doesn't introduce new issues.

6. **Submit a pull request**: Open a PR with a clear title and description of your changes.

### Coding Guidelines

We follow the [Python Software Foundation Code of Conduct](http://www.python.org/psf/codeofconduct/) and
the [Matplotlib Coding Guidelines](https://matplotlib.org/stable/devel/coding_guide.html). Please adhere to
these guidelines in your contributions.

### Getting Help

If you're new to open source or need any help, don't hesitate to ask questions in
the [issues](https://github.com/idanmoradarthas/DataScienceUtils/issues) section or reach out to the
maintainers. We're here to help!

### Code Quality

This project uses [Ruff](https://github.com/astral-sh/ruff) for linting and code formatting.
Contributors are encouraged to use it to ensure code quality and consistency.
You can check your code by running:

```bash
ruff check .
ruff format .
```

These checks are also part of our CI pipeline.

### Why Contribute?

- **Improve your skills**: Gain experience working on a real-world project.
- **Be part of a community**: Connect with other developers and data scientists.
- **Make a difference**: Your contributions will help others in their data science journey.
- **Get recognition**: All contributors are acknowledged in our project.

Remember, no contribution is too small. Whether it's fixing a typo in documentation or adding a major feature, all
contributions are valued and appreciated.

Thank you for helping make Data Science Utils better for everyone!

## Installation Guide

Here are several ways to install the package:

### 1. Install from PyPI (Recommended)

The simplest way to install Data Science Utils and its dependencies is from PyPI using pip, Python's preferred package
installer:

```bash
pip install data-science-utils
```

To upgrade Data Science Utils to the latest version, use:

```bash
pip install -U data-science-utils
```

### 2. Install from Source

If you prefer to install from source, you can clone the repository and install:

```bash
git clone https://github.com/idanmoradarthas/DataScienceUtils.git
cd DataScienceUtils
pip install .
```

Alternatively, you can install directly from GitHub using pip:

```bash
pip install git+https://github.com/idanmoradarthas/DataScienceUtils.git
```

### 3. Install using Anaconda

If you're using Anaconda, you can install using conda:

```bash
conda install idanmorad::data-science-utils
```

### Note on Dependencies

Data Science Utils has several dependencies, including numpy, pandas, matplotlib, plotly and scikit-learn. These will be
automatically installed when you install the package using the methods above.

## Staying Updated

Data Science Utils is an active project that routinely publishes new releases with additional methods and improvements.
We recommend periodically checking for updates to access the latest features and bug fixes.

If you encounter any issues during installation, please check our
GitHub [issues](https://github.com/idanmoradarthas/DataScienceUtils/issues) page or open a new issue for assistance.