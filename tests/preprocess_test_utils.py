import pandas as pd
import numpy as np

def assert_series_called_with(mock_calls, expected_series, percentile):
    """Helper function to check if a pandas Series was called with specific values."""
    for args, _ in mock_calls:
        series, p = args
        if (p == percentile and
                isinstance(series, pd.Series) and
                series.equals(expected_series)):
            return True
    return False

# Mock DataFrames for test_plot_relationship_between_features
MOCK_DATA_1M_DF = pd.DataFrame({
    'x1': np.random.rand(10),  # numeric
    'x4': np.random.rand(10),  # numeric
    'x5': np.random.rand(10),  # numeric
    'x7': ['A', 'B', 'A', 'C', 'B', 'A', 'A', 'C', 'B', 'C'],  # categorical
    'x10': ['X', 'Y', 'X', 'Y', 'X', 'Z', 'Z', 'Y', 'X', 'Z'],  # categorical
    'x12': [0, 1, 1, 0, 0, 1, 0, 1, 1, 0]  # boolean-like
})

MOCK_LOAN_DATA_DF = pd.DataFrame({
    'issue_d': pd.to_datetime(['2023-01-01', '2023-02-01', '2023-01-15', '2023-03-01', '2023-02-10']),
    'home_ownership': ['RENT', 'MORTGAGE', 'OWN', 'RENT', 'MORTGAGE'],
    'loan_condition_cat': [0, 1, 1, 0, 1]
})

MOCK_DAILY_MIN_TEMP_DF = pd.DataFrame({
    'Date': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05']),
    'Temp': [10.0, 12.5, 11.0, 9.5, 13.0]
})

# Mock DataFrame for test_plot_correlation_dendrogram
MOCK_DENDROGRAM_DF = pd.DataFrame({
    'x1': np.random.rand(20), 'x2': np.random.rand(20), 'x3': np.random.rand(20),
    'x4': np.random.rand(20), 'x5': np.random.rand(20), 'x6': np.random.rand(20),
    'x7': np.random.choice(['A', 'B', 'C'], 20),
    'x8': np.random.choice([True, False], 20),
    'x9': np.random.randn(20),
    'x10': np.random.choice(['X', 'Y', 'Z'], 20),
    'x11': np.random.randint(0, 2, 20),
    'x12': np.random.randint(0, 2, 20)
})

# Mock DataFrames for test_get_correlated_features
FEATURE_NAMES_FOR_CORR_TEST = ['income_category_Low', 'income_category_Medium',
                               'term_ 36 months', 'term_ 60 months',
                               'interest_payments_High', 'interest_payments_Low',
                               'loan_condition_cat', 'other_feature_a', 'other_feature_b']
# Using a fixed seed for reproducibility
_rng = np.random.default_rng(42)
_corr_data = _rng.random((len(FEATURE_NAMES_FOR_CORR_TEST), len(FEATURE_NAMES_FOR_CORR_TEST)))
np.fill_diagonal(_corr_data, 1.0)
_corr_data = (_corr_data + _corr_data.T) / 2
np.fill_diagonal(_corr_data, 1.0)

MOCK_CORR_DATA_FOR_GET_FEATURES = pd.DataFrame(_corr_data, index=FEATURE_NAMES_FOR_CORR_TEST, columns=FEATURE_NAMES_FOR_CORR_TEST)
MOCK_CORR_DATA_FOR_GET_FEATURES.loc['income_category_Low', 'income_category_Medium'] = 1.0
MOCK_CORR_DATA_FOR_GET_FEATURES.loc['income_category_Medium', 'income_category_Low'] = 1.0
MOCK_CORR_DATA_FOR_GET_FEATURES.loc['term_ 36 months', 'term_ 60 months'] = 1.0
MOCK_CORR_DATA_FOR_GET_FEATURES.loc['term_ 60 months', 'term_ 36 months'] = 1.0
MOCK_CORR_DATA_FOR_GET_FEATURES.loc['interest_payments_High', 'interest_payments_Low'] = 1.0
MOCK_CORR_DATA_FOR_GET_FEATURES.loc['interest_payments_Low', 'interest_payments_High'] = 1.0
MOCK_CORR_DATA_FOR_GET_FEATURES.loc['income_category_Low', 'loan_condition_cat'] = 0.11821656093586508
MOCK_CORR_DATA_FOR_GET_FEATURES.loc['income_category_Medium', 'loan_condition_cat'] = 0.11821656093586504
MOCK_CORR_DATA_FOR_GET_FEATURES.loc['term_ 36 months', 'loan_condition_cat'] = 0.223606797749979
MOCK_CORR_DATA_FOR_GET_FEATURES.loc['term_ 60 months', 'loan_condition_cat'] = 0.223606797749979
MOCK_CORR_DATA_FOR_GET_FEATURES.loc['interest_payments_High', 'loan_condition_cat'] = 0.13363062095621223
MOCK_CORR_DATA_FOR_GET_FEATURES.loc['interest_payments_Low', 'loan_condition_cat'] = 0.13363062095621223
_other_cols_for_corr_test = ['other_feature_a', 'other_feature_b']
for _col in _other_cols_for_corr_test:
    MOCK_CORR_DATA_FOR_GET_FEATURES.loc[_col, 'loan_condition_cat'] = _rng.random() * 0.05
    MOCK_CORR_DATA_FOR_GET_FEATURES.loc['loan_condition_cat', _col] = MOCK_CORR_DATA_FOR_GET_FEATURES.loc[_col, 'loan_condition_cat']
    for _other_col_inner in _other_cols_for_corr_test:
        if _col != _other_col_inner:
             MOCK_CORR_DATA_FOR_GET_FEATURES.loc[_col, _other_col_inner] = _rng.random() * 0.5
             MOCK_CORR_DATA_FOR_GET_FEATURES.loc[_other_col_inner, _col] = MOCK_CORR_DATA_FOR_GET_FEATURES.loc[_col, _other_col_inner]

COLS_FOR_EMPTY_TEST = ["Clothing ID", "Age", "Title", "Review Text", "Rating", "Recommended IND",
                       "Positive Feedback Count", "Division Name", "Department Name", "Class Name"]
_empty_corr_data = _rng.random((len(COLS_FOR_EMPTY_TEST), len(COLS_FOR_EMPTY_TEST)))
MOCK_EMPTY_CORR_DF = pd.DataFrame(_empty_corr_data, index=COLS_FOR_EMPTY_TEST, columns=COLS_FOR_EMPTY_TEST)
for _col in MOCK_EMPTY_CORR_DF.columns:
    if _col != "Class Name":
         MOCK_EMPTY_CORR_DF[_col] = MOCK_EMPTY_CORR_DF[_col] * 0.1
    MOCK_EMPTY_CORR_DF.loc[_col,_col] = 1.0
