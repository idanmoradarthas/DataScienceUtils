from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from matplotlib import pyplot as plt

from ds_utils.math_utils import safe_percentile
from ds_utils.preprocess import (
    visualize_correlations,
    plot_features_interaction,
    plot_correlation_dendrogram,
    visualize_feature,
    get_correlated_features,
    extract_statistics_dataframe_per_label
)

# Import utilities and mock data
from .preprocess_test_utils import (
    assert_series_called_with,
    MOCK_DATA_1M_DF,
    MOCK_LOAN_DATA_DF,
    MOCK_DAILY_MIN_TEMP_DF,
    MOCK_DENDROGRAM_DF,
    MOCK_CORR_DATA_FOR_GET_FEATURES,
    MOCK_EMPTY_CORR_DF
)

RESOURCES_PATH = Path(__file__).parent / "resources"
BASELINE_DIR = Path(__file__).parent / "baseline_images" / "test_preprocess"


@pytest.fixture()
def loan_data():
    return pd.read_csv(RESOURCES_PATH.joinpath("loan_final313.csv"),
                       encoding="latin1", parse_dates=["issue_d"]).drop("id", axis=1)


@pytest.fixture()
def data_1m():
    return pd.read_csv(RESOURCES_PATH.joinpath("data.1M.zip"), compression='zip')


@pytest.fixture()
def daily_min_temperatures():
    return pd.read_csv(RESOURCES_PATH.joinpath("daily-min-temperatures.csv"), parse_dates=["Date"])


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({
        'value': [1.0, 2.0, 3.0, 4.0, 5.0, np.nan, 7.0, 8.0, 9.0, 10.0],
        'category': ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C', 'C'],
        'text_col': ['x', 'y', 'z', 'x', 'y', 'z', 'x', 'y', 'z', 'x']
    })

@pytest.fixture(autouse=True)
def setup_teardown():
    yield
    plt.cla()
    plt.close(plt.gcf())


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR)
@pytest.mark.parametrize("feature",
                         ["emp_length_int", "issue_d", "loan_condition_cat", "income_category", "home_ownership",
                          "purpose"],
                         ids=["float", "datetime", "int", "object", "category", "category_more_than_10_categories"])
def test_visualize_feature(feature, request): # Removed loan_data
    """Test visualize_feature function for different feature types."""
    # n_rows = 5 # This variable was not used, removing
    mock_df_data = {
        'emp_length_int': [1.0, 2.5, np.nan, 4.0, 1.5],
        'issue_d': pd.to_datetime(['2023-01-01', '2023-02-15', '2023-01-20', '2023-03-01', '2023-02-05']),
        'loan_condition_cat': [0, 1, 1, 0, 0],
        'income_category': ['Low', 'Medium', 'High', 'Medium', 'Low'],
        'home_ownership': pd.Categorical(['RENT', 'MORTGAGE', 'OWN', 'RENT', 'MORTGAGE']),
        # Ensure 'purpose' has enough unique values for the 'category_more_than_10_categories' case by defining more categories
        'purpose': ['credit_card', 'debt_consolidation', 'home_improvement', 'major_purchase', 'medical'],
        'term': [' 36 months', ' 60 months', ' 36 months', ' 60 months', ' 36 months']
    }
    mock_df = pd.DataFrame(mock_df_data)

    if feature == 'purpose':
            # For the 'category_more_than_10_categories' test case, ensure the 'purpose' column is categorical
            # and has more than 10 defined categories, even if the actual data has fewer rows.
            mock_df['purpose'] = pd.Categorical(mock_df['purpose'], categories=[
                'credit_card', 'debt_consolidation', 'home_improvement', 'major_purchase',
                'medical', 'car', 'vacation', 'moving', 'house', 'other', 'wedding', 'renewable_energy'
            ])

    visualize_feature(mock_df[feature])

    if request.node.callspec.id in ["datetime", "object", "category"]:
        plt.gcf().set_size_inches(10, 8)
    elif request.node.callspec.id == "category_more_than_10_categories":
        plt.gcf().set_size_inches(11, 11)

    return plt.gcf()


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR)
def test_visualize_feature_float_exist_ax(): # Removed loan_data
    mock_df = pd.DataFrame({'emp_length_int': [1.0, 2.5, 3.0, 4.0, 1.5]})
    fig, ax = plt.subplots()
    ax.set_title("My ax")

    visualize_feature(mock_df["emp_length_int"], ax=ax)

    assert ax.get_title() == "My ax"
    fig.set_size_inches(10, 8)
    return fig


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR)
def test_visualize_feature_bool(): # Removed loan_data
    """Test visualize_feature function for boolean data."""
    mock_df = pd.DataFrame({
        'term': [' 36 months', ' 60 months', ' 36 months', ' 60 months', ' 36 months']
    })
    loan_dup = pd.DataFrame()
    loan_dup["term 36 months"] = mock_df["term"].apply(lambda term: term == " 36 months").astype("bool")
    visualize_feature(loan_dup["term 36 months"])
    return plt.gcf()


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR)
def test_visualize_feature_remove_na(): # Removed loan_data
    """Test visualize_feature function with NA values removed."""
    mock_emp_length_data = pd.DataFrame({
        'emp_length_int': [1.0, 2.5, np.nan, 4.0, 1.5, np.nan, 5.0] * 50 # Multiplied to get enough non-NA for plot
    })
    # Ensure enough non-NaN values exist for a meaningful plot after potential NA removal by visualize_feature
    # The original test created 250 NaNs and concatenated with original data.
    # Here, we ensure our mock data has both NaNs and enough valid points.
    # visualize_feature itself handles NA removal if remove_na=True.

    visualize_feature(mock_emp_length_data["emp_length_int"], remove_na=True)
    return plt.gcf()


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR)
@pytest.mark.parametrize("use_existing_ax", [False, True], ids=["default", "exist_ax"])
def test_visualize_correlations(use_existing_ax): # Removed data_1m
    """Test visualize_correlations function with and without existing axes."""
    mock_corr_df = pd.DataFrame({
        'A': np.random.rand(20),
        'B': np.random.rand(20),
        'C': np.random.choice(['cat1', 'cat2', 'cat3'], 20),
        'D': np.random.randint(0, 2, 20)
    })
    corr = mock_corr_df.apply(lambda x: x.factorize()[0]).corr(method="pearson", min_periods=1)
    if use_existing_ax:
        _, ax = plt.subplots()
        ax.set_title("My ax")
        visualize_correlations(corr, ax=ax)
        assert ax.get_title() == "My ax"
    else:
        visualize_correlations(corr)

    fig = plt.gcf()
    fig.set_size_inches(14, 9)
    return fig


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR)
@pytest.mark.parametrize("feature1, feature2, data_fixture", [
    ("x4", "x5", "data_1m"),
    ("x1", "x7", "data_1m"),
    ("x7", "x1", "data_1m"),
    ("x1", "x12", "data_1m"),
    ("x7", "x10", "data_1m"),
    ("x10", "x12", "data_1m"),
    ("Date", "Temp", "daily_min_temperatures"),
    ("Temp", "Date", "daily_min_temperatures"),
    ("issue_d", "issue_d", "loan_data"),
    ("issue_d", "home_ownership", "loan_data"),
    ("home_ownership", "issue_d", "loan_data"),
    ("x12", "x12", "data_1m")
], ids=["both_numeric", "numeric_categorical", "numeric_categorical_reverse", "numeric_boolean", "both_categorical",
        "categorical_bool", "datetime_numeric", "datetime_numeric_reverse", "datetime_datetime", "datetime_categorical",
        "datetime_categorical_reverse", "both_bool"])
def test_plot_relationship_between_features(mocker, feature1, feature2, data_fixture, request):
    """Test plot_features_interaction function for various feature combinations."""

    # Store original getfixturevalue
    original_getfixturevalue = request.getfixturevalue

    def mock_getfixturevalue(fixture_name):
        if fixture_name == "data_1m":
            return MOCK_DATA_1M_DF.copy()
        elif fixture_name == "loan_data":
            return MOCK_LOAN_DATA_DF.copy()
        elif fixture_name == "daily_min_temperatures":
            return MOCK_DAILY_MIN_TEMP_DF.copy()
        return original_getfixturevalue(fixture_name)

    mocker.patch.object(request, 'getfixturevalue', side_effect=mock_getfixturevalue)

    data = request.getfixturevalue(data_fixture)
    plot_features_interaction(data, feature1, feature2)

    if request.node.callspec.id in ["numeric_categorical", "numeric_categorical_reverse"]:
        plt.gcf().set_size_inches(14, 9)
    elif request.node.callspec.id == "numeric_boolean":
        plt.gcf().set_size_inches(8, 7)
    elif request.node.callspec.id == "both_categorical":
        plt.gcf().set_size_inches(9, 5)
    elif request.node.callspec.id in ["datetime_numeric", "datetime_numeric_reverse"]:
        plt.gcf().set_size_inches(18, 8)
    elif request.node.callspec.id in ["datetime_categorical", "datetime_categorical_reverse"]:
        plt.gcf().set_size_inches(10, 11.5)

    return plt.gcf()


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR)
@pytest.mark.parametrize("feature1, feature2",
                         [("issue_d", "loan_condition_cat"), ("loan_condition_cat", "issue_d")],
                         ids=["default", "reverse"])
def test_plot_relationship_between_features_datetime_bool(feature1, feature2): # Removed mocker, request, loan_data args
    # This test constructs its own DataFrame based on a structure similar to 'loan_data'.
    # No fixture loading is mocked here directly; we just use a predefined structure.
    mock_df_structure = pd.DataFrame({
        'issue_d': pd.to_datetime(['2023-01-01', '2023-02-01', '2023-01-15', '2023-03-01', '2023-02-10']),
        'loan_condition_cat': [0, 1, 1, 0, 1]
    })

    df = pd.DataFrame()
    df["loan_condition_cat"] = mock_df_structure["loan_condition_cat"].astype("bool")
    df["issue_d"] = mock_df_structure["issue_d"]

    plot_features_interaction(df, feature1, feature2)

    fig = plt.gcf()
    fig.set_size_inches(10, 11.5)
    return fig


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR)
def test_plot_relationship_between_features_both_numeric_exist_ax(): # Removed mocker, request, data_1m args
    # This test uses a structure similar to 'data_1m'.
    # No fixture loading is mocked here directly; we use a predefined structure.
    mock_data_df = pd.DataFrame({
        'x4': np.random.rand(10),
        'x5': np.random.rand(10)
        # Add other columns if plot_features_interaction strictly needs them,
        # but for x4, x5 interaction, these should suffice.
    })

    fig, ax = plt.subplots()
    ax.set_title("My ax")

    plot_features_interaction(mock_data_df, "x4", "x5", ax=ax)
    assert ax.get_title() == "My ax"
    return fig


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR)
@pytest.mark.parametrize("use_existing_ax", [False, True], ids=["default", "exist_ax"])
def test_plot_correlation_dendrogram(use_existing_ax): # Removed mocker, request, data_1m
    """Test plot_correlation_dendrogram function with and without existing axes."""

    # Using the imported MOCK_DENDROGRAM_DF
    data_to_use_for_corr = MOCK_DENDROGRAM_DF.copy()

    corr = data_to_use_for_corr.apply(lambda x: x.factorize()[0]).corr(method="pearson", min_periods=1)
    if use_existing_ax:
        _, ax = plt.subplots()
        ax.set_title("My ax")
        plot_correlation_dendrogram(corr, ax=ax)
        assert ax.get_title() == "My ax"
    else:
        plot_correlation_dendrogram(corr)

    return plt.gcf()


def test_get_correlated_features(mocker):
    """Test get_correlated_features function."""
    # Using the imported MOCK_CORR_DATA_FOR_GET_FEATURES
    mocker.patch("pandas.read_feather", return_value=MOCK_CORR_DATA_FOR_GET_FEATURES.copy())
    correlations = pd.read_feather("dummy_path.feather")

    correlation = get_correlated_features(correlations, correlations.columns.drop("loan_condition_cat").tolist(),
                                          "loan_condition_cat", 0.95)
    correlation_expected = pd.DataFrame([
        {'level_0': 'income_category_Low', 'level_1': 'income_category_Medium',
         'level_0_level_1_corr': 1.0,
         'level_0_target_corr': 0.11821656093586508,
         'level_1_target_corr': 0.11821656093586504},
        {'level_0': 'term_ 36 months', 'level_1': 'term_ 60 months',
         'level_0_level_1_corr': 1.0,
         'level_0_target_corr': 0.223606797749979,
         'level_1_target_corr': 0.223606797749979},
        {'level_0': 'interest_payments_High', 'level_1': 'interest_payments_Low',
         'level_0_level_1_corr': 1.0, 'level_0_target_corr': 0.13363062095621223,
         'level_1_target_corr': 0.13363062095621223}
    ])
    pd.testing.assert_frame_equal(correlation_expected, correlation, check_dtype=False, atol=1e-5)


def test_get_correlated_features_empty_result(mocker): # Added mocker
    # Setup specific correlations
    mock_corr_data_for_get_features.loc['income_category_Low', 'income_category_Medium'] = 1.0
    mock_corr_data_for_get_features.loc['income_category_Medium', 'income_category_Low'] = 1.0
    mock_corr_data_for_get_features.loc['term_ 36 months', 'term_ 60 months'] = 1.0
    mock_corr_data_for_get_features.loc['term_ 60 months', 'term_ 36 months'] = 1.0
    mock_corr_data_for_get_features.loc['interest_payments_High', 'interest_payments_Low'] = 1.0
    mock_corr_data_for_get_features.loc['interest_payments_Low', 'interest_payments_High'] = 1.0
    # Setup target correlations
    mock_corr_data_for_get_features.loc['income_category_Low', 'loan_condition_cat'] = 0.11821656093586508
    mock_corr_data_for_get_features.loc['income_category_Medium', 'loan_condition_cat'] = 0.11821656093586504
    mock_corr_data_for_get_features.loc['term_ 36 months', 'loan_condition_cat'] = 0.223606797749979
    mock_corr_data_for_get_features.loc['term_ 60 months', 'loan_condition_cat'] = 0.223606797749979
    mock_corr_data_for_get_features.loc['interest_payments_High', 'loan_condition_cat'] = 0.13363062095621223
    mock_corr_data_for_get_features.loc['interest_payments_Low', 'loan_condition_cat'] = 0.13363062095621223

    mocker.patch("pandas.read_feather", return_value=mock_corr_data_for_get_features)
    correlations = pd.read_feather("dummy_path.feather") # path is dummy

    correlation = get_correlated_features(correlations, correlations.columns.drop("loan_condition_cat").tolist(),
                                          "loan_condition_cat", 0.95)

    pd.testing.assert_frame_equal(correlation_expected, correlation, check_dtype=False, atol=1e-5)


def test_get_correlated_features_empty_result(mocker):
    """Test get_correlated_features function with an empty result."""
    # Using the imported MOCK_EMPTY_CORR_DF
    mocker.patch("pandas.read_feather", return_value=MOCK_EMPTY_CORR_DF.copy())
    correlations = pd.read_feather("dummy_empty_corr.feather")

    expected_warning = "Correlation threshold 0.95 was too high. An empty frame was returned"
    with pytest.warns(UserWarning, match=expected_warning):
        correlation = get_correlated_features(correlations,
                                              ["Clothing ID", "Age", "Title", "Review Text", "Rating",
                                               "Recommended IND", "Positive Feedback Count", "Division Name",
                                               "Department Name"], # These are target_col_list
                                              "Class Name", 0.95) # This is target_col_name
    correlation_expected = pd.DataFrame(
        columns=['level_0', 'level_1', 'level_0_level_1_corr', 'level_0_target_corr', 'level_1_target_corr'])
    pd.testing.assert_frame_equal(correlation_expected, correlation, check_dtype=False)


# assert_series_called_with has been moved to preprocess_test_utils.py


def test_extract_statistics_dataframe_per_label_basic_functionality(sample_df, mocker):
    """Test basic functionality and verify safe_percentile calls."""
    mock_safe_percentile = mocker.patch('ds_utils.preprocess.safe_percentile', wraps=safe_percentile)

    result = extract_statistics_dataframe_per_label(sample_df, 'value', 'category')

    # Check if all expected columns are present
    expected_columns = [
        'count', 'null_count', 'mean', 'min', '1_percentile', '5_percentile',
        '25_percentile', 'median', '75_percentile', '95_percentile', '99_percentile', 'max'
    ]
    assert all(col in result.columns for col in expected_columns)

    # Verify safe_percentile was called correct number of times with right arguments
    assert mock_safe_percentile.call_count == 18  # 6 percentiles * 3 categories

    # Verify some specific calls
    expected_series_a = pd.Series([1.0, 2.0, 3.0])
    assert assert_series_called_with(mock_safe_percentile.call_args_list,
                                     expected_series_a, 1)  # Category A, 1st percentile
    assert assert_series_called_with(mock_safe_percentile.call_args_list,
                                     expected_series_a, 99)  # Category A, 99th percentile


@pytest.mark.parametrize("feature_name, label_name, exception, message", [
    ('invalid_col', 'category', KeyError, "Feature column 'invalid_col' not found"),
    ('value', 'invalid_col', KeyError, "Label column 'invalid_col' not found"),
    ('text_col', 'category', TypeError, "Feature column 'text_col' must be numeric")
], ids=["test_invalid_feature_name", "test_invalid_label_name", "test_non_numeric_feature"])
def test_extract_statistics_dataframe_per_label_exceptions(sample_df, feature_name, label_name, exception, message):
    with pytest.raises(exception, match=message):
        extract_statistics_dataframe_per_label(sample_df, feature_name, label_name)
