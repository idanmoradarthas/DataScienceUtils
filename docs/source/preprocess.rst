**********
Preprocess
**********
The module of preprocess contains methods that are processes that could be made to data before training.

Get Correlated Features
=======================

.. autofunction:: preprocess::get_correlated_features

.. highlight:: python

Code Example
************
The example uses a small sample from of a dataset from
`kaggle <https://www.kaggle.com/mrferozi/loan-data-for-dummy-bank>`_, which a dummy bank provides loans.

Let's see how to use the code::

    import pandas
    from ds_utils.preprocess import get_correlated_features


    loan_frame = pandas.read_csv(path/to/dataset, encoding="latin1", nrows=30)
    target = "loan_condition_cat"
    features = train.columns.drop("loan_condition_cat", "issue_d", "application_type").tolist()
    correlations = get_correlated_features(pandas.get_dummies(loan_frame), features, target)
    print(correlations)

The following table will be the output:

+----------------------+----------------------+--------------------+-------------------+-------------------+
|level_0               |level_1               |level_0_level_1_corr|level_0_target_corr|level_1_target_corr|
+======================+======================+====================+===================+===================+
|income_category_Low   |income_category_Medium|-0.9999999999999999 |-0.1182165609358650|0.11821656093586504|
+----------------------+----------------------+--------------------+-------------------+-------------------+
|term\_ 36 months      |term\_ 60 months      |-1.0                |-0.1182165609358650|0.11821656093586504|
+----------------------+----------------------+--------------------+-------------------+-------------------+
|interest_payments_High|interest_payments_Low |-1.0                |-0.1182165609358650|0.11821656093586504|
+----------------------+----------------------+--------------------+-------------------+-------------------+