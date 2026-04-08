******************************
Generate Error Analysis Report
******************************

The ``generate_error_analysis_report`` function provides a tabular error-analysis report that groups predictions by feature values and computes error metrics per group. It's particularly useful for identifying specific feature ranges or categories where the model underperforms.

.. autofunction:: ds_utils.metrics.error_analysis.generate_error_analysis_report

Code Example
============

.. highlight:: python

.. code-block:: python

    import pandas as pd
    import numpy as np
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier
    from ds_utils.metrics.error_analysis import generate_error_analysis_report

    # Load dataset and split
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    X["size_category"] = pd.cut(
        X["mean radius"], bins=3, labels=["small", "medium", "large"]
    ).astype(str)
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    # Train a classifier
    clf = DecisionTreeClassifier(random_state=42, max_depth=3)
    clf.fit(X_train[["mean radius", "mean texture"]], y_train)

    y_pred = clf.predict(X_test[["mean radius", "mean texture"]])

    # Generate error analysis report for numerical and categorical features
    report = generate_error_analysis_report(
        X_test, y_test, y_pred,
        feature_columns=["mean radius", "mean texture", "size_category"],
        bins=3,
        sort_metric="error_rate",
        ascending=False
    )
    print(report)

The output will be a pandas DataFrame similar to this:

+---------------+-----------------+-------+-------------+------------+----------+
| feature       | group           | count | error_count | error_rate | accuracy |
+===============+=================+=======+=============+============+==========+
| mean radius   | (16.71, 24.933] | 30    | 3           | 0.100000   | 0.900000 |
+---------------+-----------------+-------+-------------+------------+----------+
| size_category | large           | 15    | 1           | 0.066667   | 0.933333 |
+---------------+-----------------+-------+-------------+------------+----------+
| mean texture  | (25.32, 33.81]  | 17    | 1           | 0.058824   | 0.941176 |
+---------------+-----------------+-------+-------------+------------+----------+
| mean texture  | (16.83, 25.32]  | 78    | 4           | 0.051282   | 0.948718 |
+---------------+-----------------+-------+-------------+------------+----------+
| mean texture  | (8.315, 16.83]  | 48    | 2           | 0.041667   | 0.958333 |
+---------------+-----------------+-------+-------------+------------+----------+
| size_category | medium          | 25    | 1           | 0.040000   | 0.960000 |
+---------------+-----------------+-------+-------------+------------+----------+
| mean radius   | (8.471, 16.71]  | 113   | 4           | 0.035398   | 0.964602 |
+---------------+-----------------+-------+-------------+------------+----------+

*(Note: The size_category rows use raw string values as groups, while numerical features are binned. Rows with equal error_rate may appear in any order. Exact values and bins may vary based on data distribution.)*
