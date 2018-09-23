# Pre-Process
Contains common functions for pre-process before fitting a model.
## get_correlated_features
Calculate which features correlated above a threshold and extract a data frame with the correlations and correlation to 
the target feature.

Input:
* data_frame - the data frame.
* features - list of features names.
* target_feature - name of target feature.
* threshold - the threshold (default 0.95).

Example:
```python
import pandas
from ds_utils.preprocess import get_correlated_features


data_frame = pandas.read_csv(".../file/location.csv", enconding="latin1")
features = data_frame.columns.drop("loan_condition_cat").tolist()
get_correlated_features(data_frame, features, "loan_condition_cat", 0.95)
```

Output:

|level_0|level_1|level_0_level_1_corr|level_0_target_corr|level_1_target_corr|
|:------|:------|:-------------------|:------------------|:------------------|
|income_category_Low|income_category_Medium|-1.0|-0.119|0.119|
|term_ 36 months|term_ 60 months|-1.0|-0.119|0.119|
|interest_payments_High|interest_payments_Low|-1.0|-0.119|0.119|