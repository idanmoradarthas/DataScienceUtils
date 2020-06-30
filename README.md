# Data Science Utils: Frequently Used Methods for Data Science
[![License: MIT](https://img.shields.io/github/license/idanmoradarthas/DataScienceUtils)](https://opensource.org/licenses/MIT)
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/idanmoradarthas/DataScienceUtils)
[![GitHub issues](https://img.shields.io/github/issues/idanmoradarthas/DataScienceUtils)](https://github.com/idanmoradarthas/DataScienceUtils/issues)
[![Documentation Status](https://readthedocs.org/projects/datascienceutils/badge/?version=latest)](https://datascienceutils.readthedocs.io/en/latest/?badge=latest)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/data-science-utils)
![PyPI - Wheel](https://img.shields.io/pypi/wheel/data-science-utils)
[![PyPI version](https://badge.fury.io/py/data-science-utils.svg)](https://badge.fury.io/py/data-science-utils)
[![Anaconda-Server Badge](https://anaconda.org/idanmorad/data-science-utils/badges/version.svg)](https://anaconda.org/idanmorad/data-science-utils)
[![Build Status](https://travis-ci.org/idanmoradarthas/DataScienceUtils.svg?branch=master)](https://travis-ci.org/idanmoradarthas/DataScienceUtils)
[![Coverage Status](https://coveralls.io/repos/github/idanmoradarthas/DataScienceUtils/badge.svg?branch=master)](https://coveralls.io/github/idanmoradarthas/DataScienceUtils?branch=master)


Data Science Utils extends the Scikit-Learn API and Matplotlib API to provide simple methods that simplify task and 
visualization over data. 

# Code Examples and Documentation
**Let's see some code examples and outputs.** 

**You can read the full documentation with all the code examples from:
[https://datascienceutils.readthedocs.io/en/latest/](https://datascienceutils.readthedocs.io/en/latest/)**

In the documentation you can find more methods and more examples.

The API of the package is build to work with Scikit-Learn API and Matplotlib API. Here are some of capabilities of this
package:

## Metrics
### Plot Confusion Matrix
Computes and plot confusion matrix, False Positive Rate, False Negative Rate, Accuracy and F1 score of a classification.

![multi label classification confusion matrix](https://raw.githubusercontent.com/idanmoradarthas/DataScienceUtils/master/tests/baseline_images/test_metrics/test_print_confusion_matrix.png)

### Plot Metric Growth per Labeled Instances

Receives a train and test sets, and plots given metric change in increasing amount of trained instances.

![metric growth per labeled instances with n samples](https://raw.githubusercontent.com/idanmoradarthas/DataScienceUtils/master/tests/baseline_images/test_metrics/test_plot_metric_growth_per_labeled_instances_with_n_samples.png)

## Preprocess
### Get Correlated Features

Calculate which features correlated above a threshold and extract a data frame with the correlations and correlation to 
the target feature.

|level_0               |level_1               |level_0_level_1_corr|level_0_target_corr|level_1_target_corr|
|----------------------|----------------------|--------------------|-------------------|-------------------|
|income_category_Low   |income_category_Medium|-0.9999999999999999 |-0.1182165609358650|0.11821656093586504|
|term\_ 36 months      |term\_ 60 months      |-1.0                |-0.1182165609358650|0.11821656093586504|
|interest_payments_High|interest_payments_Low |-1.0                |-0.1182165609358650|0.11821656093586504|

## Strings
### Append Tags to Frame

Extracts tags from a given field and append them as dataframe.

A dataset that looks like this:

``x_train``:

|article_name|article_tags|
|------------|------------|
|1           |ds,ml,dl    |
|2           |ds,ml       |

``x_test``:

|article_name|article_tags|
|------------|------------|
|3           |ds,ml,py    |

Using this code:
```python
    import pandas

    from ds_utils.strings import append_tags_to_frame


    x_train = pandas.DataFrame([{"article_name": "1", "article_tags": "ds,ml,dl"},
                                {"article_name": "2", "article_tags": "ds,ml"}])
    x_test = pandas.DataFrame([{"article_name": "3", "article_tags": "ds,ml,py"}])

    x_train_with_tags, x_test_with_tags = append_tags_to_frame(x_train, x_test, "article_tags", "tag_")
```

will be parsed into this:

``x_train_with_tags``:

|article_name|tag_ds|tag_ml|tag_dl|
|------------|------|------|------|
|1           |1     |1     |1     |
|2           |1     |1     |0     |

``x_test_with_tags``:

|article_name|tag_ds|tag_ml|tag_dl|
|------------|------|------|------|
|3           |1     |1     |0     |

### Extract Significant Terms from Subset
Returns interesting or unusual occurrences of terms in a subset. Based on the [elasticsearch significant_text aggregation](https://www.elastic.co/guide/en/elasticsearch/reference/current/search-aggregations-bucket-significantterms-aggregation.html#_scripted).

```python
import pandas

from ds_utils.strings import extract_significant_terms_from_subset

corpus = ['This is the first document.', 'This document is the second document.',
          'And this is the third one.', 'Is this the first document?']
data_frame = pandas.DataFrame(corpus, columns=["content"])
# Let's differentiate between the last two documents from the full corpus
subset_data_frame = data_frame[data_frame.index > 1]
terms = extract_significant_terms_from_subset(data_frame, subset_data_frame, 
                                               "content")

```
And the following table will be the output for ``terms``:

|third|one|and|this|the |is  |first|document|second|
|-----|---|---|----|----|----|-----|--------|------|
|1.0  |1.0|1.0|0.67|0.67|0.67|0.5  |0.25    |0.0   |

## Visualization Aids
### Visualize Feature

Receives a feature and visualize its values on a graph.

|Feature Type      |Plot|
|------------------|----|
|Float             |![](https://raw.githubusercontent.com/idanmoradarthas/DataScienceUtils/master/tests/baseline_images/test_visualization_aids/test_visualize_feature_float.png)|
|Integer           |![](https://raw.githubusercontent.com/idanmoradarthas/DataScienceUtils/master/tests/baseline_images/test_visualization_aids/test_visualize_feature_integer.png)|
|Datetime          |![](https://raw.githubusercontent.com/idanmoradarthas/DataScienceUtils/master/tests/baseline_images/test_visualization_aids/test_visualize_feature_datetime.png)|
|Category / Object |![](https://raw.githubusercontent.com/idanmoradarthas/DataScienceUtils/master/tests/baseline_images/test_visualization_aids/test_visualize_feature_category_more_than_10_categories.png)|
|Boolean           |![](https://raw.githubusercontent.com/idanmoradarthas/DataScienceUtils/master/tests/baseline_images/test_visualization_aids/test_visualize_feature_bool.png)|

### Visualize Correlations
Compute pairwise correlation of columns, excluding NA/null values, and visualize it with heat map.
[Original code](https://seaborn.pydata.org/examples/many_pairwise_correlations.html)

![visualize features](https://raw.githubusercontent.com/idanmoradarthas/DataScienceUtils/master/tests/baseline_images/test_visualization_aids/test_visualize_correlations.png)

### Plot Correlation Dendrogram
Plot dendrogram of a correlation matrix. This consists of a chart that that shows hierarchically the variables that are 
most correlated by the connecting trees. The closer to the right that the connection is, the more correlated the 
features are.

![plot correlation dendrogram](https://raw.githubusercontent.com/idanmoradarthas/DataScienceUtils/master/tests/baseline_images/test_visualization_aids/test_plot_correlation_dendrogram.png)

### Plot Features' Relationship
Plots the joint distribution between two features:
* If both features are either categorical, boolean or object then the method plots the shared histogram.
* If one feature is either categorical, boolean or object and the other is numeric then the method plots a boxplot chart.
* If one feature is datetime and the other is numeric or datetime then the method plots a line plot graph.
* If one feature is datetime and the other is either categorical, boolean or object the method plots a violin plot (combination of boxplot and kernel density estimate).
* If both features are numeric then the method plots scatter graph.

|               | Numeric | Categorical | Boolean | Datetime
|---------------|---------|-------------|---------|---------|
|**Numeric**    |![](https://raw.githubusercontent.com/idanmoradarthas/DataScienceUtils/master/tests/baseline_images/test_visualization_aids/test_plot_relationship_between_features_both_numeric.png)|![](https://raw.githubusercontent.com/idanmoradarthas/DataScienceUtils/master/tests/baseline_images/test_visualization_aids/test_plot_relationship_between_features_numeric_categorical.png)|![](https://raw.githubusercontent.com/idanmoradarthas/DataScienceUtils/master/tests/baseline_images/test_visualization_aids/test_plot_relationship_between_features_numeric_boolean.png)|![](https://raw.githubusercontent.com/idanmoradarthas/DataScienceUtils/master/tests/baseline_images/test_visualization_aids/test_plot_relationship_between_features_datetime_numeric.png)|
|**Categorical**| |![](https://raw.githubusercontent.com/idanmoradarthas/DataScienceUtils/master/tests/baseline_images/test_visualization_aids/test_plot_relationship_between_features_both_categorical.png)|![](https://raw.githubusercontent.com/idanmoradarthas/DataScienceUtils/master/tests/baseline_images/test_visualization_aids/test_plot_relationship_between_features_categorical_bool.png)|![](https://raw.githubusercontent.com/idanmoradarthas/DataScienceUtils/master/tests/baseline_images/test_visualization_aids/test_plot_relationship_between_features_datetime_categorical.png)|
|**Boolean**    | | |![](https://raw.githubusercontent.com/idanmoradarthas/DataScienceUtils/master/tests/baseline_images/test_visualization_aids/test_plot_relationship_between_features_both_bool.png)|![](https://raw.githubusercontent.com/idanmoradarthas/DataScienceUtils/master/tests/baseline_images/test_visualization_aids/test_plot_relationship_between_features_datetime_bool.png)|
|**Datetime**   | | | |![](https://raw.githubusercontent.com/idanmoradarthas/DataScienceUtils/master/tests/baseline_images/test_visualization_aids/test_plot_relationship_between_features_datetime_datetime.png)|

##XAI
### Draw Tree
Receives a decision tree and return a plot graph of the tree for easy interpretation.

![](https://raw.githubusercontent.com/idanmoradarthas/DataScienceUtils/master/tests/baseline_images/test_visualization_aids/test_draw_tree.png)

### Generate Decision Paths
Receives a decision tree and return the underlying decision-rules (or 'decision paths') as text (valid python syntax). 
[Original code](https://stackoverflow.com/questions/20224526/how-to-extract-the-decision-rules-from-scikit-learn-decision-tree)

```python
from sklearn.tree import DecisionTreeClassifier

from ds_utils.xai import generate_decision_paths
    

# Create decision tree classifier object
clf = DecisionTreeClassifier(max_depth=3)

# Train model
clf.fit(x, y)
print(generate_decision_paths(clf, feature_names, target_names.tolist(), 
                              "iris_tree"))
```
The following text will be printed:
```
def iris_tree(petal width (cm), petal length (cm)):
    if petal width (cm) <= 0.8000:
        # return class setosa with probability 0.9804
        return ("setosa", 0.9804)
    else:  # if petal width (cm) > 0.8000
        if petal width (cm) <= 1.7500:
            if petal length (cm) <= 4.9500:
                # return class versicolor with probability 0.9792
                return ("versicolor", 0.9792)
            else:  # if petal length (cm) > 4.9500
                # return class virginica with probability 0.6667
                return ("virginica", 0.6667)
        else:  # if petal width (cm) > 1.7500
            if petal length (cm) <= 4.8500:
                # return class virginica with probability 0.6667
                return ("virginica", 0.6667)
            else:  # if petal length (cm) > 4.8500
                # return class virginica with probability 0.9773
                return ("virginica", 0.9773)
```

**see full example [here](https://datascienceutils.readthedocs.io/en/latest/xai.html#generate-decision-paths).**

Excited?

Read about all the modules here and see more methods and abilities (such as drawing a decision tree and more): 
* [Metrics](https://datascienceutils.readthedocs.io/en/latest/metrics.html) - The module of metrics contains methods that help to calculate and/or visualize evaluation performance of an algorithm.
* [Preprocess](https://datascienceutils.readthedocs.io/en/latest/preprocess.html) - The module of preprocess contains methods that are processes that could be made to data before training.
* [Strings](https://datascienceutils.readthedocs.io/en/latest/strings.html) - The module of strings contains methods that help manipulate and process strings in a dataframe.
* [Visualization Aids](https://datascienceutils.readthedocs.io/en/latest/visualization_aids.html) - The module of visualization aids contains methods that visualize by drawing or printing ML output.
* [XAI](https://datascienceutils.readthedocs.io/en/latest/xai.html) - The module of xai contains methods that help explain a model decisions.

## Contributing
Interested in contributing to Data Science Utils? Great! You're welcome,  and we would love to have you. We follow 
the [Python Software Foundation Code of Conduct](http://www.python.org/psf/codeofconduct/) and 
[Matplotlib Usage Guide](https://matplotlib.org/tutorials/introductory/usage.html#coding-styles).

No matter your level of technical skill, you can be helpful. We appreciate bug reports, user testing, feature 
requests, bug fixes, product enhancements, and documentation improvements.

Thank you for your contributions!

## Find a Bug?
Check if there's already an open [issue](https://github.com/idanmoradarthas/DataScienceUtils/issues) on the topic. If 
needed, file an issue.

## Open Source
Data Science Utils license is [MIT License](https://opensource.org/licenses/MIT). 

## Installing Data Science Utils
Data Science Utils is compatible with Python 3.6 or later. The simplest way to install Data Science Utils and its 
dependencies is from PyPI with pip, Python's preferred package installer:
```bash
pip install data-science-utils
```
Note that this package is an active project and routinely publishes new releases with more methods.  In order to 
upgrade Data Science Utils to the latest version, use pip as follows:
```bash
pip install -U data-science-utils
```
Alternatively you can install from source by cloning the repo and running:
```bash
git clone https://github.com/idanmoradarthas/DataScienceUtils.git
cd DataScienceUtils
python setup.py install
```
Or install using pip from source:
```bash
pip install git+https://github.com/idanmoradarthas/DataScienceUtils.git
```
If you're using Anaconda, you can install using conda:
```bash
conda install -c idanmorad data-science-utils
```