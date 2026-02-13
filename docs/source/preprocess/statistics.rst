##########
Statistics
##########

The statistics submodule contains methods for computing statistical measures on data.

***********************
Get Correlated Features
***********************

This function identifies highly correlated features in your dataset. Use this when you want to:

- Detect multi-collinearity in your feature set
- Simplify your model by removing redundant features
- Understand the relationship between features and the target variable

Insights from this analysis can help in feature selection, reducing overfitting, and improving model interpretability.

.. autofunction:: ds_utils.preprocess.statistics.get_correlated_features

Code Example
============
This example uses a small sample from a dataset available on `Kaggle <https://www.kaggle.com/mrferozi/loan-data-for-dummy-bank>`_, which contains loan data from a dummy bank.

Here's how to use the code::

    import pandas as pd
    from ds_utils.preprocess.statistics import get_correlated_features

    loan_frame = pd.get_dummies(pd.read_csv('path/to/dataset', encoding="latin1", nrows=30))
    target = "loan_condition_cat"
    features = loan_frame.columns.drop(["loan_condition_cat", "issue_d", "application_type"]).tolist()
    correlations = get_correlated_features(loan_frame.corr(), features, target)
    print(correlations)

The following table will be the output:

+----------------------+----------------------+--------------------+-------------------+-------------------+
|level_0               |level_1               |level_0_level_1_corr|level_0_target_corr|level_1_target_corr|
+======================+======================+====================+===================+===================+
|income_category_Low   |income_category_Medium|1.0                 |0.1182165609358650 |0.11821656093586504|
+----------------------+----------------------+--------------------+-------------------+-------------------+
|term\_ 36 months      |term\_ 60 months      |1.0                 |0.1182165609358650 |0.11821656093586504|
+----------------------+----------------------+--------------------+-------------------+-------------------+
|interest_payments_High|interest_payments_Low |1.0                 |0.1182165609358650 |0.11821656093586504|
+----------------------+----------------------+--------------------+-------------------+-------------------+

**************************************
Extract Statistics DataFrame per Label
**************************************
This method calculates comprehensive statistical metrics for numerical features grouped by label values. Use this when you want to:

- Analyze how a numerical feature's distribution varies across different categories
- Detect potential patterns or anomalies in feature behavior per group
- Generate detailed statistical summaries for reporting or analysis
- Understand the relationship between features and target variables

.. autofunction:: ds_utils.preprocess.statistics.extract_statistics_dataframe_per_label

Code Example
============
Here's how to use the method to analyze numerical features across different categories::

    import pandas as pd
    from ds_utils.preprocess.statistics import extract_statistics_dataframe_per_label

    # Load your dataset
    df = pd.DataFrame({
        'amount': [100, 200, 150, 300, 250, 175],
        'category': ['A', 'A', 'B', 'B', 'C', 'C']
    })

    # Calculate statistics for amount grouped by category
    stats = extract_statistics_dataframe_per_label(
        df=df,
        feature_name='amount',
        label_name='category'
    )
    print(stats)

The output will be a DataFrame containing the following statistics for each category:

+----------+-------+-----------+--------+------+-------------+-------------+--------------+--------+--------------+--------------+--------------+-------+
| category | count | null_count| mean   | min  | 1_percentile| 5_percentile| 25_percentile| median | 75_percentile| 95_percentile| 99_percentile| max   |
+==========+=======+===========+========+======+=============+=============+==============+========+==============+==============+==============+=======+
| A        | 2     | 0         | 150.0  | 100  | 100.0       | 100.0       | 100.0        | 150.0  | 200.0        | 200.0        | 200.0        | 200.0 |
+----------+-------+-----------+--------+------+-------------+-------------+--------------+--------+--------------+--------------+--------------+-------+
| B        | 2     | 0         | 225.0  | 150  | 150.0       | 150.0       | 150.0        | 225.0  | 300.0        | 300.0        | 300.0        | 300.0 |
+----------+-------+-----------+--------+------+-------------+-------------+--------------+--------+--------------+--------------+--------------+-------+
| C        | 2     | 0         | 212.5  | 175  | 175.0       | 175.0       | 175.0        | 212.5  | 250.0        | 250.0        | 250.0        | 250.0 |
+----------+-------+-----------+--------+------+-------------+-------------+--------------+--------+--------------+--------------+--------------+-------+

This comprehensive set of statistics helps in understanding the distribution of numerical features across different categories, which can be valuable for:

- Identifying outliers within specific groups
- Understanding data skewness per category
- Detecting potential data quality issues
- Making informed decisions about feature engineering strategies

**************************
Compute Mutual Information
**************************

This method computes the mutual information between each feature and a specified target variable. Mutual information measures the dependency between two variables, with a higher value indicating a stronger relationship.

Use this when you want to:

* Identify the most informative features for a classification task.
* Perform feature selection based on the relevance of each feature to the target.
* Gain insight into the underlying relationships within your data, which can guide feature engineering.

.. autofunction:: ds_utils.preprocess.statistics.compute_mutual_information

Code Example
============
This example uses a sample DataFrame to demonstrate the calculation of mutual information.

Here's how to use the code::

    import pandas as pd
    from ds_utils.preprocess.statistics import compute_mutual_information

    sample_df = pd.DataFrame({
        "value": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        "category": ["A", "A", "A", "B", "B", "B", "C", "C", "C", "C"],
        "text_col": ["x", "y", "z", "x", "y", "z", "x", "y", "z", "x"],
    })
    target = "category"
    mutual_information_df = compute_mutual_information(sample_df, target)
    print(mutual_information_df)

The following table will be the output:

+----------+--------------------+
| feature  | mutual_information |
+==========+====================+
| value    | 0.046              |
+----------+--------------------+
| text_col | 0.941              |
+----------+--------------------+
