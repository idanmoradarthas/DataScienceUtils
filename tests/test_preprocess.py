"""Tests for data preprocessing functions and visualizations."""

from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import pytest

from ds_utils.math_utils import safe_percentile
from ds_utils.preprocess import (
    compute_mutual_information,
    extract_statistics_dataframe_per_label,
    get_correlated_features,
    plot_correlation_dendrogram,
    plot_features_interaction,
    visualize_correlations,
    visualize_feature,
)

RESOURCES_PATH = Path(__file__).parent / "resources"
BASELINE_DIR = Path(__file__).parent / "baseline_images" / "test_preprocess"


@pytest.fixture
def loan_data():
    """Load and return loan dataset for testing."""
    return pd.read_csv(RESOURCES_PATH.joinpath("loan_final313.csv"), encoding="latin1", parse_dates=["issue_d"]).drop(
        "id", axis=1
    )


@pytest.fixture
def data_1m():
    """Load and return 1M dataset for testing."""
    return pd.read_csv(RESOURCES_PATH.joinpath("data.1M.zip"), compression="zip")


@pytest.fixture
def daily_min_temperatures():
    """Load and return daily minimum temperatures dataset for testing."""
    return pd.read_csv(RESOURCES_PATH.joinpath("daily-min-temperatures.csv"), parse_dates=["Date"])


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Create a sample DataFrame for testing."""
    return pd.DataFrame(
        {
            "value": [1.0, 2.0, 3.0, 4.0, 5.0, np.nan, 7.0, 8.0, 9.0, 10.0],
            "category": ["A", "A", "A", "B", "B", "B", "C", "C", "C", "C"],
            "text_col": ["x", "y", "z", "x", "y", "z", "x", "y", "z", "x"],
        }
    )


@pytest.fixture(autouse=True)
def setup_teardown():
    """Set up and tear down for each test function in this module."""
    yield
    plt.cla()
    plt.close(plt.gcf())


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR)
@pytest.mark.parametrize(
    "feature",
    ["emp_length_int", "issue_d", "loan_condition_cat"],
    ids=["float", "datetime", "int"],
)
def test_visualize_feature_float_datetime_int(loan_data, feature, request):
    """Test visualize_feature function for float, datetime and int features."""
    visualize_feature(loan_data[feature])

    if request.node.callspec.id in ["datetime"]:
        plt.gcf().set_size_inches(10, 11)
    return plt.gcf()


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR)
@pytest.mark.parametrize(
    ("feature", "show_counts"),
    [
        ("income_category", True),
        ("home_ownership", True),
        ("purpose", True),
        ("income_category", False),
        ("home_ownership", False),
        ("purpose", False),
    ],
    ids=[
        "object_show_counts",
        "category_show_counts",
        "category_more_than_10_categories_show_counts",
        "object_no_show_counts",
        "category_no_show_counts",
        "category_more_than_10_categories_no_show_counts",
    ],
)
def test_visualize_feature_object(loan_data, feature, show_counts, request):
    """Test visualize_feature function for object and category features."""
    if request.node.callspec.id in ["object_show_counts", "object_no_show_counts"]:
        visualize_feature(loan_data[feature], order=["Low", "Medium", "High"], show_counts=show_counts)
    else:
        visualize_feature(loan_data[feature], show_counts=show_counts)

    if request.node.callspec.id in [
        "object_show_counts",
        "category_show_counts",
        "object_no_show_counts",
        "category_no_show_counts",
    ]:
        plt.gcf().set_size_inches(10, 9)
    elif request.node.callspec.id in [
        "category_more_than_10_categories_show_counts",
        "category_more_than_10_categories_no_show_counts",
    ]:
        plt.gcf().set_size_inches(11, 14)
    return plt.gcf()


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR)
def test_visualize_feature_float_exist_ax(loan_data):
    """Test visualize_feature with a float feature on an existing Axes."""
    fig, ax = plt.subplots()
    ax.set_title("My ax")

    visualize_feature(loan_data["emp_length_int"], ax=ax)

    assert ax.get_title() == "My ax"
    fig.set_size_inches(10, 8)
    return fig


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR)
def test_visualize_feature_float_exclude_outliers(loan_data):
    """Test visualize_feature function with outliers excluded."""
    visualize_feature(loan_data["emp_length_int"], include_outliers=False, outlier_iqr_multiplier=0.01)
    return plt.gcf()


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR)
@pytest.mark.parametrize("show_counts", [True, False], ids=["show_counts", "no_show_counts"])
def test_visualize_feature_bool(loan_data, show_counts):
    """Test visualize_feature function for boolean data."""
    loan_dup = pd.DataFrame()
    loan_dup["term 36 months"] = loan_data["term"].apply(lambda term: term == " 36 months").astype("bool")
    visualize_feature(loan_dup["term 36 months"], order=["True", "False"], show_counts=show_counts)
    return plt.gcf()


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR)
def test_visualize_feature_remove_na(loan_data):
    """Test visualize_feature function with NA values removed."""
    loan_data_dup = pd.concat(
        [loan_data[["emp_length_int"]], pd.DataFrame([np.nan] * 250, columns=["emp_length_int"])], ignore_index=True
    ).sample(frac=1, random_state=0)

    visualize_feature(loan_data_dup["emp_length_int"], remove_na=True)
    return plt.gcf()


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR)
def test_visualize_feature_datetime_heatmap_sunday_start(loan_data):
    """Test visualize_feature with datetime data starting week on Sunday."""
    visualize_feature(loan_data["issue_d"], first_day_of_week="Sunday")
    plt.gcf().set_size_inches(10, 8)
    return plt.gcf()


def test_visualize_feature_datetime_invalid_first_day():
    """Test visualize_feature with invalid first_day_of_week parameter."""
    dates = pd.date_range("2023-01-01", "2023-12-31", freq="D")
    series = pd.Series(dates, name="test_dates")

    with pytest.raises(ValueError, match="first_day_of_week must be one of"):
        visualize_feature(series, first_day_of_week="InvalidDay")


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR)
def test_visualize_feature_datetime_missing_days_adds_columns():
    """Ensure heatmap adds missing weekday columns when some days are absent in data."""
    dates_mon = pd.date_range("2024-01-01", periods=4, freq="W-MON")
    dates_tue = pd.date_range("2024-01-02", periods=4, freq="W-TUE")
    combined = np.concatenate([dates_mon.values, dates_tue.values])
    series = pd.Series(pd.to_datetime(combined), name="test_dates").sort_values().reset_index(drop=True)

    visualize_feature(series, first_day_of_week="Monday")
    plt.gcf().set_size_inches(10, 8)
    return plt.gcf()


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR)
@pytest.mark.parametrize("order", ["count_desc", "count_asc", "alpha_asc", "alpha_desc"])
def test_visualize_feature_object_order(loan_data, order):
    """Test visualize_feature function with different order parameters."""
    visualize_feature(loan_data["purpose"], order=order)
    plt.gcf().set_size_inches(11, 14)
    return plt.gcf()


def test_visualize_feature_object_order_error():
    """Test visualize_feature function with an invalid order parameter."""
    with pytest.raises(ValueError, match="Invalid order string: 'invalid_order'. Must be one of: "):
        visualize_feature(pd.Series(["A", "B", "C"]), order="invalid_order")


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR)
@pytest.mark.parametrize(
    "order", [["High", "Low", "Medium"], ["Medium", "High", "Low"]], ids=["high_low_medium", "medium_high_low"]
)
def test_visualize_feature_object_order_list(loan_data, order):
    """Test visualize_feature function with a list of order parameters."""
    visualize_feature(loan_data["income_category"], order=order)
    plt.gcf().set_size_inches(11, 14)
    return plt.gcf()


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR)
def test_visualize_feature_object_null_values(loan_data):
    """Test visualize_feature function with null values in an object feature."""
    series = pd.concat(
        [loan_data["income_category"], pd.Series([np.nan] * 500, name="income_category")], ignore_index=True
    )
    visualize_feature(series, remove_na=False)
    plt.gcf().set_size_inches(8, 7)
    return plt.gcf()


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR)
@pytest.mark.parametrize("use_existing_ax", [False, True], ids=["default", "exist_ax"])
def test_visualize_correlations(data_1m, use_existing_ax):
    """Test visualize_correlations function with and without existing axes."""
    corr = data_1m.apply(lambda x: x.factorize()[0]).corr(method="pearson", min_periods=1)
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
@pytest.mark.parametrize(
    ("feature1", "feature2", "data_fixture"),
    [
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
    ],
    ids=[
        "both_numeric",
        "numeric_categorical",
        "numeric_categorical_reverse",
        "numeric_boolean",
        "both_categorical",
        "categorical_bool",
        "datetime_numeric",
        "datetime_numeric_reverse",
        "datetime_datetime",
        "datetime_categorical",
        "datetime_categorical_reverse",
    ],
)
def test_plot_relationship_between_features(feature1, feature2, data_fixture, request):
    """Test plot_features_interaction function for various feature combinations."""
    data = request.getfixturevalue(data_fixture)
    plot_features_interaction(data, feature1, feature2)

    if request.node.callspec.id in ["numeric_categorical", "numeric_categorical_reverse"]:
        plt.gcf().set_size_inches(14, 9)
    elif request.node.callspec.id == "numeric_boolean":
        plt.gcf().set_size_inches(8, 7)
    elif request.node.callspec.id == "both_categorical":
        plt.gcf().set_size_inches(12, 5)
    elif request.node.callspec.id in ["datetime_numeric", "datetime_numeric_reverse"]:
        plt.gcf().set_size_inches(18, 8)
    elif request.node.callspec.id in ["datetime_categorical", "datetime_categorical_reverse"]:
        plt.gcf().set_size_inches(10, 11.5)

    return plt.gcf()


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR)
def test_plot_relationship_between_features_both_bool(loan_data):
    """Test interaction plot for two boolean features."""
    data = pd.DataFrame()
    data["is_home_ownership_rent"] = loan_data["home_ownership"] == "RENT"
    data["is_low_interest_payments"] = loan_data["interest_payments"] == "Low"
    plot_features_interaction(data, "is_home_ownership_rent", "is_low_interest_payments")
    return plt.gcf()


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR)
@pytest.mark.parametrize(
    ("feature1", "feature2"),
    [("issue_d", "loan_condition_cat"), ("loan_condition_cat", "issue_d")],
    ids=["default", "reverse"],
)
def test_plot_relationship_between_features_datetime_bool(loan_data, feature1, feature2):
    """Test interaction plot between a datetime and a boolean feature."""
    df = pd.DataFrame()
    df["loan_condition_cat"] = loan_data["loan_condition_cat"].astype("bool")
    df["issue_d"] = loan_data["issue_d"]

    plot_features_interaction(df, feature1, feature2)

    fig = plt.gcf()
    fig.set_size_inches(10, 11.5)
    return fig


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR)
def test_plot_features_interaction_numeric_numeric_missingness_rugs(data_1m):
    """Numeric vs numeric: show rug markers when either axis has missing values."""
    df = data_1m.copy()

    # Create many missing values spread across the dataset so rug markers
    # are visible across the full plotted range.
    n = len(df)
    # Choose evenly spaced indices (avoid endpoints to preserve plenty of complete cases)
    x5_missing_idx = np.linspace(10, n - 11, 80, dtype=int)
    x4_missing_idx = np.linspace(20, n - 21, 80, dtype=int)

    # Ensure disjoint sets so each missingness type is clearly visible
    x4_missing_idx = np.setdiff1d(x4_missing_idx, x5_missing_idx)

    # - x4 present, x5 missing
    df.loc[df.index[x5_missing_idx], "x5"] = np.nan
    # - x5 present, x4 missing
    df.loc[df.index[x4_missing_idx], "x4"] = np.nan

    # remove_na defaults to False; omit it to avoid redundant explicit defaults.
    plot_features_interaction(df, "x4", "x5")
    plt.gcf().set_size_inches(10, 8)
    return plt.gcf()


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR)
def test_plot_features_interaction_datetime_numeric_missingness_no_complete_data(daily_min_temperatures, monkeypatch):
    """Datetime vs numeric: handle missing-on-either-side even when there are no complete pairs."""
    fixed_now = pd.Timestamp("2024-01-15 00:00:00")
    monkeypatch.setattr(pd.Timestamp, "now", classmethod(lambda cls: fixed_now))

    df = daily_min_temperatures.copy()

    # Make sure there are NO complete cases, but we still have:
    # - datetime present, numeric missing (missing numeric markers)
    # - numeric present, datetime missing (missing datetime rug)
    df.loc[df.index[:40], "Temp"] = np.nan  # Date present, Temp missing
    df.loc[df.index[40:80], "Date"] = pd.NaT  # Temp present, Date missing

    # Force numeric values (where present) to be equal so y_min == y_max branch triggers
    df.loc[df["Date"].isna(), "Temp"] = 7.0

    # remove_na defaults to False; omit it to avoid redundant explicit defaults.
    plot_features_interaction(df, "Date", "Temp")
    plt.gcf().set_size_inches(12, 6)
    return plt.gcf()


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR)
@pytest.mark.parametrize(
    "case",
    ["with_complete_and_both_missing_rugs", "no_complete_fallback_now"],
    ids=["with_complete_and_both_missing_rugs", "no_complete_fallback_now"],
)
def test_plot_features_interaction_datetime_datetime_missingness(daily_min_temperatures, monkeypatch, case):
    """Datetime vs datetime: visualize missing values on both axes (including no-complete-cases fallback)."""
    df = daily_min_temperatures.copy()
    df["Date2"] = df["Date"] + pd.Timedelta(days=1)

    if case == "with_complete_and_both_missing_rugs":
        # Ensure complete cases exist plus missingness in each datetime column.
        df.loc[df.index[:25], "Date2"] = pd.NaT  # feature_2 missing
        df.loc[df.index[100:150], "Date"] = pd.NaT  # feature_1 missing
        # remove_na defaults to False; omit it to avoid redundant explicit defaults.
        plot_features_interaction(df, "Date", "Date2")
        plt.gcf().set_size_inches(12, 6)
        return plt.gcf()

    if case == "no_complete_fallback_now":
        fixed_now = pd.Timestamp("2024-02-01 00:00:00")
        monkeypatch.setattr(pd.Timestamp, "now", classmethod(lambda cls: fixed_now))

        # Ensure there are NO complete cases, but we still show both missingness types:
        # - Date2 missing while Date present  -> rug on x-axis
        # - Date missing while Date2 present  -> rug on y-axis
        n = len(df)
        idx = np.linspace(0, n - 1, 260, dtype=int)
        df = df.iloc[idx].copy()

        # Preserve a source of non-missing Date2 values even after we set Date to NaT.
        date2_full = df["Date2"].copy()

        # Start by making Date2 missing everywhere (so no complete pairs).
        df.loc[:, "Date2"] = pd.NaT

        # For a subset of rows, make Date missing but keep Date2 present.
        date_missing_pos = np.linspace(5, len(df) - 6, 120, dtype=int)
        df.loc[df.index[date_missing_pos], "Date"] = pd.NaT
        df.loc[df.index[date_missing_pos], "Date2"] = date2_full.loc[df.index[date_missing_pos]]

        # remove_na defaults to False; omit it to avoid redundant explicit defaults.
        ax = plot_features_interaction(df, "Date", "Date2")
        ax.set_title("No complete date pairs; missing shown as rugs")
        plt.gcf().set_size_inches(12, 6)
        return plt.gcf()

    raise AssertionError(f"Unknown case: {case}")


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR)
@pytest.mark.parametrize(
    "case",
    ["categorical_categorical_missingness", "categorical_numeric_missingness"],
    ids=["categorical_categorical_missingness", "categorical_numeric_missingness"],
)
def test_plot_features_interaction_categorical_missingness(data_1m, case):
    """Categorical interactions: include missing category and/or missing numeric values in the plot."""
    df = data_1m.copy()

    if case == "categorical_categorical_missingness":
        df.loc[df.index[:15], "x7"] = np.nan
        df.loc[df.index[15:30], "x10"] = np.nan
        # remove_na defaults to False; omit it to avoid redundant explicit defaults.
        plot_features_interaction(df, "x7", "x10")
        plt.gcf().set_size_inches(12, 5)
        return plt.gcf()

    if case == "categorical_numeric_missingness":
        np.random.seed(0)  # jitter in rug plot uses np.random.uniform
        df.loc[df.index[:20], "x7"] = np.nan
        df.loc[df.index[20:40], "x1"] = np.nan
        # remove_na defaults to False; omit it to avoid redundant explicit defaults.
        plot_features_interaction(df, "x7", "x1")
        plt.gcf().set_size_inches(14, 7)
        return plt.gcf()

    raise AssertionError(f"Unknown case: {case}")


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR)
@pytest.mark.parametrize(
    "case",
    ["some_missing_datetime", "all_datetime_missing_fallback_now"],
    ids=["some_missing_datetime", "all_datetime_missing_fallback_now"],
)
def test_plot_features_interaction_categorical_datetime_missingness(loan_data, monkeypatch, case):
    """Categorical vs datetime: show missing category and missing datetimes (including all-missing fallback)."""
    df = loan_data[["home_ownership", "issue_d"]].copy()

    if case == "some_missing_datetime":
        df.loc[df.index[:250], "home_ownership"] = np.nan
        df.loc[df.index[250:500], "issue_d"] = pd.NaT
        # remove_na defaults to False; omit it to avoid redundant explicit defaults.
        plot_features_interaction(df, "home_ownership", "issue_d")
        plt.gcf().set_size_inches(10, 11.5)
        return plt.gcf()

    if case == "all_datetime_missing_fallback_now":
        fixed_now = pd.Timestamp("2024-03-01 00:00:00")
        monkeypatch.setattr(pd.Timestamp, "now", classmethod(lambda cls: fixed_now))
        df.loc[:, "issue_d"] = pd.NaT
        # remove_na defaults to False; omit it to avoid redundant explicit defaults.
        plot_features_interaction(df, "home_ownership", "issue_d")
        plt.gcf().set_size_inches(10, 11.5)
        return plt.gcf()

    raise AssertionError(f"Unknown case: {case}")


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR)
def test_plot_features_interaction_remove_na_true_drops_rows(data_1m):
    """remove_na=True: drop rows where either feature is missing before plotting."""
    df = data_1m.copy()
    df.loc[df.index[:25], "x4"] = np.nan
    df.loc[df.index[25:50], "x5"] = np.nan

    plot_features_interaction(df, "x4", "x5", remove_na=True)
    plt.gcf().set_size_inches(10, 8)
    return plt.gcf()


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR)
def test_plot_relationship_between_features_both_numeric_exist_ax(data_1m):
    """Test interaction plot for two numeric features on an existing Axes."""
    fig, ax = plt.subplots()
    ax.set_title("My ax")

    plot_features_interaction(data_1m, "x4", "x5", ax=ax)
    assert ax.get_title() == "My ax"
    return fig


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR)
def test_plot_relashionship_between_features_numeric_categorical_without_outliers(data_1m):
    """Test interaction plot for two numeric features without outliers."""
    plot_features_interaction(data_1m, "x1", "x7", include_outliers=False, outlier_iqr_multiplier=0.01)
    plt.gcf().set_size_inches(14, 9)
    return plt.gcf()


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR)
@pytest.mark.parametrize(("feature1", "feature2"), [("x7", "x10"), ("x10", "x12")], ids=["both", "bool"])
def test_plot_features_interaction_show_ratios_categorical(feature1, feature2, data_1m):
    """Test plotting categorical features interactions with ratios."""
    plot_features_interaction(data_1m, feature1, feature2, show_ratios=True)
    return plt.gcf()


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR)
def test_plot_features_interaction_show_ratios_bool(loan_data):
    """Test plotting boolean features interactions with ratios."""
    data = pd.DataFrame()
    data["is_home_ownership_rent"] = loan_data["home_ownership"] == "RENT"
    data["is_low_interest_payments"] = loan_data["interest_payments"] == "Low"
    plot_features_interaction(data, "is_home_ownership_rent", "is_low_interest_payments", show_ratios=True)
    return plt.gcf()


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR)
@pytest.mark.parametrize("use_existing_ax", [False, True], ids=["default", "exist_ax"])
def test_plot_correlation_dendrogram(data_1m, use_existing_ax):
    """Test plot_correlation_dendrogram function with and without existing axes."""
    corr = data_1m.apply(lambda x: x.factorize()[0]).corr(method="pearson", min_periods=1)
    if use_existing_ax:
        _, ax = plt.subplots()
        ax.set_title("My ax")
        plot_correlation_dendrogram(corr, ax=ax)
        assert ax.get_title() == "My ax"
    else:
        plot_correlation_dendrogram(corr)

    return plt.gcf()


def test_get_correlated_features():
    """Test get_correlated_features function."""
    correlations = pd.read_feather(RESOURCES_PATH.joinpath("loan_final313_small_corr.feather"))
    correlation = get_correlated_features(
        correlations, correlations.columns.drop("loan_condition_cat").tolist(), "loan_condition_cat", 0.95
    )
    correlation_expected = pd.DataFrame(
        [
            {
                "level_0": "income_category_Low",
                "level_1": "income_category_Medium",
                "level_0_level_1_corr": 1.0,
                "level_0_target_corr": 0.11821656093586508,
                "level_1_target_corr": 0.11821656093586504,
            },
            {
                "level_0": "term_ 36 months",
                "level_1": "term_ 60 months",
                "level_0_level_1_corr": 1.0,
                "level_0_target_corr": 0.223606797749979,
                "level_1_target_corr": 0.223606797749979,
            },
            {
                "level_0": "interest_payments_High",
                "level_1": "interest_payments_Low",
                "level_0_level_1_corr": 1.0,
                "level_0_target_corr": 0.13363062095621223,
                "level_1_target_corr": 0.13363062095621223,
            },
        ]
    )
    pd.testing.assert_frame_equal(correlation_expected, correlation)


def test_get_correlated_features_empty_result():
    """Test get_correlated_features function with an empty result."""
    correlations = pd.read_feather(RESOURCES_PATH.joinpath("clothing_classification_train_corr.feather"))
    expected_warning = "Correlation threshold 0.95 was too high. An empty frame was returned"
    with pytest.warns(UserWarning, match=expected_warning):
        correlation = get_correlated_features(
            correlations,
            [
                "Clothing ID",
                "Age",
                "Title",
                "Review Text",
                "Rating",
                "Recommended IND",
                "Positive Feedback Count",
                "Division Name",
                "Department Name",
            ],
            "Class Name",
            0.95,
        )
    correlation_expected = pd.DataFrame(
        columns=["level_0", "level_1", "level_0_level_1_corr", "level_0_target_corr", "level_1_target_corr"]
    )
    pd.testing.assert_frame_equal(correlation_expected, correlation)


def assert_series_called_with(mock_calls, expected_series, percentile):
    """Check if a pandas' Series was called with specific values."""
    for args, _ in mock_calls:
        series, p = args
        if p == percentile and isinstance(series, pd.Series) and series.equals(expected_series):
            return True
    return False


def test_extract_statistics_dataframe_per_label_basic_functionality(sample_df, mocker):
    """Test basic functionality and verify safe_percentile calls."""
    mock_safe_percentile = mocker.patch("ds_utils.preprocess.safe_percentile", wraps=safe_percentile)

    result = extract_statistics_dataframe_per_label(sample_df, "value", "category")

    # Check if all expected columns are present
    expected_columns = [
        "count",
        "null_count",
        "mean",
        "min",
        "1_percentile",
        "5_percentile",
        "25_percentile",
        "median",
        "75_percentile",
        "95_percentile",
        "99_percentile",
        "max",
    ]
    assert all(col in result.columns for col in expected_columns)

    # Verify safe_percentile was called correct number of times with right arguments
    assert mock_safe_percentile.call_count == 18  # 6 percentiles * 3 categories

    # Verify some specific calls
    expected_series_a = pd.Series([1.0, 2.0, 3.0])
    assert assert_series_called_with(
        mock_safe_percentile.call_args_list, expected_series_a, 1
    )  # Category A, 1st percentile
    assert assert_series_called_with(
        mock_safe_percentile.call_args_list, expected_series_a, 99
    )  # Category A, 99th percentile


@pytest.mark.parametrize(
    ("feature_name", "label_name", "exception", "message"),
    [
        ("invalid_col", "category", KeyError, "Feature column 'invalid_col' not found"),
        ("value", "invalid_col", KeyError, "Label column 'invalid_col' not found"),
        ("text_col", "category", TypeError, "Feature column 'text_col' must be numeric"),
    ],
    ids=["test_invalid_feature_name", "test_invalid_label_name", "test_non_numeric_feature"],
)
def test_extract_statistics_dataframe_per_label_exceptions(sample_df, feature_name, label_name, exception, message):
    """Test exceptions for extract_statistics_dataframe_per_label."""
    with pytest.raises(exception, match=message):
        extract_statistics_dataframe_per_label(sample_df, feature_name, label_name)


def test_compute_mutual_information(data_1m):
    """Test basic functionality for compute_mutual_information."""
    df = data_1m.copy()
    features = df.columns.tolist()
    rng = np.random.default_rng(seed=42)
    df["target"] = rng.choice(["class_1", "class_2", "class_3"], size=len(df))

    expected = pd.DataFrame(
        [
            ["x3", 0.001766052134277718],
            ["x2", 0.001479947096839851],
            ["x1", 0.0009494384943877776],
            ["x8", 0.00036423417047570794],
            ["x4", 0.00014870988232429383],
            ["x5", 0.00013023297539671574],
            ["x7", 9.545793450264087e-05],
            ["x10", 3.387523951139948e-06],
            ["x12", 7.467128754767849e-07],
            ["x6", 0.0],
            ["x9", 0.0],
            ["x11", 0.0],
        ],
        columns=["feature_name", "mi_score"],
    )
    results = compute_mutual_information(df, features, "target", random_state=42)
    pd.testing.assert_frame_equal(expected, results)


def test_compute_mutual_information_empty_features_list():
    """Test compute_mutual_information with empty features list."""
    df = pd.DataFrame({"feature1": [1, 2, 3, 4, 5], "target": [0, 1, 0, 1, 0]})
    with pytest.raises(ValueError, match="features list cannot be empty"):
        compute_mutual_information(df, [], "target")


def test_compute_mutual_information_missing_label_column():
    """Test compute_mutual_information with missing label column."""
    df = pd.DataFrame({"num_high_corr": [1, 2, 3, 4, 5], "target": [0, 1, 0, 1, 0]})
    with pytest.raises(KeyError, match="Label column 'nonexistent' not found"):
        compute_mutual_information(df, ["num_high_corr"], "nonexistent")


def test_compute_mutual_information_missing_feature_columns():
    """Test compute_mutual_information with missing feature columns."""
    df = pd.DataFrame({"num_high_corr": [1, 2, 3, 4, 5], "target": [0, 1, 0, 1, 0]})
    with pytest.raises(KeyError, match="Features not found in DataFrame: \\['nonexistent1', 'nonexistent2'\\]"):
        compute_mutual_information(df, ["num_high_corr", "nonexistent1", "nonexistent2"], "target")


def test_compute_mutual_information_all_null_target():
    """Test compute_mutual_information when target column has only null values."""
    df = pd.DataFrame({"feature1": [1, 2, 3, 4, 5], "target": [np.nan, np.nan, np.nan, np.nan, np.nan]})

    with pytest.raises(ValueError, match="Label column 'target' contains only null values"):
        compute_mutual_information(df, ["feature1"], "target")


@pytest.mark.parametrize(
    ("valid_feature_name", "valid_feature_data", "missing_feature_name", "missing_dtype"),
    [
        ("numerical_feature", [1, 2, 3, 4, 5], "missing_numerical", float),
        ("categorical_feature", ["A", "B", "A", "B", "A"], "missing_categorical", object),
        ("boolean_feature", [True, False, True, False, True], "missing_boolean", "boolean"),
    ],
    ids=["numerical", "categorical", "boolean"],
)
def test_compute_mutual_information_fully_missing_feature(
    valid_feature_name, valid_feature_data, missing_feature_name, missing_dtype
):
    """Test that fully missing features are handled, get a score of 0, and raise a warning."""
    df = pd.DataFrame(
        {
            valid_feature_name: valid_feature_data,
            missing_feature_name: [np.nan] * 5,
            "target": [0, 1, 0, 1, 0],
        }
    )
    df[missing_feature_name] = df[missing_feature_name].astype(missing_dtype)
    features = [valid_feature_name, missing_feature_name]

    expected_warning = f"Features \\['{missing_feature_name}'\\] contain only null values and will be ignored."
    with pytest.warns(UserWarning, match=expected_warning):
        mi_scores = compute_mutual_information(df, features, "target", random_state=42)

    assert missing_feature_name in mi_scores["feature_name"].values
    assert mi_scores.loc[mi_scores["feature_name"] == missing_feature_name, "mi_score"].iloc[0] == 0.0


def test_compute_mutual_information_all_features_fully_missing():
    """Test compute_mutual_information when ALL features contain only null values.

    This tests the edge case where all features are missing, which should return
    a DataFrame with all features having MI score of 0, sorted by feature_name.
    """
    df = pd.DataFrame(
        {
            "missing_feature1": [np.nan, np.nan, np.nan, np.nan, np.nan],
            "missing_feature2": [np.nan, np.nan, np.nan, np.nan, np.nan],
            "missing_feature3": [np.nan, np.nan, np.nan, np.nan, np.nan],
            "target": [0, 1, 0, 1, 0],
        }
    )

    features = ["missing_feature1", "missing_feature2", "missing_feature3"]

    expected_warning = (
        r"Features \['missing_feature1', 'missing_feature2', 'missing_feature3'\] "
        r"contain only null values and will be ignored."
    )

    with pytest.warns(UserWarning, match=expected_warning):
        mi_scores = compute_mutual_information(df, features, "target", random_state=42)

    # Verify the exact DataFrame structure (checks length, values, and ordering)
    expected_df = pd.DataFrame(
        {
            "feature_name": ["missing_feature1", "missing_feature2", "missing_feature3"],
            "mi_score": [0.0, 0.0, 0.0],
        }
    )
    pd.testing.assert_frame_equal(mi_scores, expected_df)


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR)
def test_plot_features_interaction_datetime_numeric_defaults(daily_min_temperatures, monkeypatch):
    """Test datetime vs numeric with NO complete data to trigger default limits."""
    # Mock current time for deterministic default x-limits
    fixed_now = pd.Timestamp("2024-01-15 00:00:00")
    monkeypatch.setattr(pd.Timestamp, "now", classmethod(lambda cls: fixed_now))

    # Take a small slice and ensure NO complete cases exist
    df = daily_min_temperatures.head(20).copy()

    # Half have missing numeric (Date present, Temp missing)
    df.loc[df.index[:10], "Temp"] = np.nan

    # Half have missing date (Temp present, Date missing)
    df.loc[df.index[10:], "Date"] = pd.NaT

    # Force numeric values to be constant to trigger y_min == y_max logic (line 520)
    df.loc[df["Date"].isna(), "Temp"] = 7.0

    # This should trigger lines 516-524
    plot_features_interaction(df, "Date", "Temp")
    plt.gcf().set_size_inches(12, 6)
    return plt.gcf()


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR)
@pytest.mark.parametrize("scenario", ["f2_missing", "f1_missing"])
def test_plot_features_interaction_datetime_datetime_defaults(daily_min_temperatures, monkeypatch, scenario):
    """Test datetime vs datetime with NO complete data and ONE feature fully missing."""
    fixed_now = pd.Timestamp("2024-02-01 00:00:00")
    monkeypatch.setattr(pd.Timestamp, "now", classmethod(lambda cls: fixed_now))

    df = daily_min_temperatures.head(20).copy()
    df["Date2"] = df["Date"] + pd.Timedelta(days=1)

    if scenario == "f2_missing":
        # Feature 2 (Date2) is FULLY missing. Feature 1 (Date) is present.
        # No complete cases.
        df["Date2"] = pd.NaT
        # This triggers lines 594-599 (default y-limits)
        plot_features_interaction(df, "Date", "Date2")

    else:  # f1_missing
        # Feature 1 (Date) is FULLY missing. Feature 2 (Date2) is present.
        # No complete cases.
        df["Date"] = pd.NaT
        # This triggers lines 633-638 (default x-limits)
        plot_features_interaction(df, "Date", "Date2")

    plt.gcf().set_size_inches(12, 9)
    return plt.gcf()


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR)
@pytest.mark.parametrize(
    ("feature_1", "feature_2"),
    [
        ("home_ownership", "purpose"),
        ("home_ownership", "emp_length_int"),
    ],
    ids=["cat_cat", "cat_num"],
)
def test_plot_features_interaction_categorical_missing_f1_scenarios(loan_data, feature_1, feature_2):
    """Test interactions where the first categorical feature has missing values."""
    df = loan_data[[feature_1, feature_2]].head(50).copy()

    # Introduce missing values in Feature 1
    df.iloc[0:10, 0] = np.nan

    plot_features_interaction(df, feature_1, feature_2)
    if feature_2 == "purpose":
        plt.gcf().set_size_inches(11, 18)
    else:
        plt.gcf().set_size_inches(11, 12)
    return plt.gcf()


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR)
@pytest.mark.parametrize(
    ("feature_1", "feature_2", "missing_val_2"),
    [
        ("home_ownership", "purpose", np.nan),
        ("home_ownership", "issue_d", pd.NaT),
    ],
    ids=["cat_cat", "cat_datetime"],
)
def test_plot_features_interaction_remove_na_scenarios(loan_data, feature_1, feature_2, missing_val_2):
    """Test interactions with remove_na=True."""
    df = loan_data[[feature_1, feature_2]].head(50).copy()

    # Introduce missing values
    df.iloc[0:5, 0] = np.nan
    df.iloc[5:10, 1] = missing_val_2

    plot_features_interaction(df, feature_1, feature_2, remove_na=True)
    if feature_2 == "purpose":
        plt.gcf().set_size_inches(10, 18)
    else:
        plt.gcf().set_size_inches(10, 13)
    return plt.gcf()


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR)
def test_plot_features_interaction_categorical_datetime_missing_logic(loan_data):
    """Test cat vs datetime with missing datetime values (coverage for missing logic)."""
    df = loan_data[["home_ownership", "issue_d"]].head(50).copy()

    # Introduce missing values in Datetime (Feature 2)
    # This ensures 'has_missing_datetime' is True
    df.iloc[0:10, 1] = pd.NaT

    # Defaults to remove_na=False
    plot_features_interaction(df, "home_ownership", "issue_d")
    plt.gcf().set_size_inches(10, 13)
    return plt.gcf()


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR)
def test_plot_features_interaction_datetime_numeric_single_date(daily_min_temperatures, monkeypatch):
    """Test datetime vs numeric with NO complete data and SINGLE unique date in 'missing numeric' set."""
    fixed_now = pd.Timestamp("2024-01-15 00:00:00")
    monkeypatch.setattr(pd.Timestamp, "now", classmethod(lambda cls: fixed_now))

    # Take a small slice
    df = daily_min_temperatures.head(20).copy()

    # 1. No complete data
    # 2. 'missing numeric' (Date present, Temp missing) should have only ONE unique date
    # 3. 'missing datetime' (Temp present, Date missing) can be empty or not, let's keep it empty for simplicity

    df["Temp"] = np.nan  # All numeric missing
    # Set all dates to be the same to trigger x_min == x_max logic
    single_date = pd.Timestamp("2024-01-01")
    df["Date"] = single_date

    # This should trigger lines 510-511 (x_min == x_max)
    plot_features_interaction(df, "Date", "Temp")
    plt.gcf().set_size_inches(12, 6)
    return plt.gcf()


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR)
@pytest.mark.parametrize("scenario", ["missing_f2_single", "missing_f1_single"])
def test_plot_features_interaction_datetime_datetime_single_value(daily_min_temperatures, monkeypatch, scenario):
    """Test datetime vs datetime with NO complete data and SINGLE unique value in missing set."""
    fixed_now = pd.Timestamp("2024-03-01 00:00:00")
    monkeypatch.setattr(pd.Timestamp, "now", classmethod(lambda cls: fixed_now))

    df = daily_min_temperatures.head(20).copy()
    df["Date2"] = df["Date"]

    if scenario == "missing_f2_single":
        # Feature 2 (Date2) missing, Feature 1 (Date) present.
        # Feature 1 has only ONE unique value.
        df["Date2"] = pd.NaT
        df["Date"] = pd.Timestamp("2024-01-01")
        # Triggers lines 634-635 (y_min == y_max for rug on bottom)
        plot_features_interaction(df, "Date", "Date2")

    else:  # missing_f1_single
        # Feature 1 (Date) missing, Feature 2 (Date2) present.
        # Feature 2 has only ONE unique value.
        df["Date"] = pd.NaT
        df["Date2"] = pd.Timestamp("2024-01-01")
        # Triggers lines 673-674 (x_min == x_max for rug on left)
        plot_features_interaction(df, "Date", "Date2")

    plt.gcf().set_size_inches(12, 6)
    return plt.gcf()


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR)
@pytest.mark.parametrize("scenario", ["missing_f2_avail_f2_single", "missing_f1_avail_f1_single"])
def test_plot_features_interaction_datetime_datetime_disjoint_single_value(daily_min_temperatures, scenario):
    """Test datetime vs datetime with NO complete data, but 'available' data for limits is single-valued.

    This targets lines 621-622 and 660-661 in preprocess.py.
    """
    df = pd.DataFrame()
    ts1 = pd.Timestamp("2024-01-01")
    ts2 = pd.Timestamp("2024-02-01")

    if scenario == "missing_f2_avail_f2_single":
        # We want to trigger lines 621-622.
        # Condition:
        # 1. missing_f2 > 0 (Rows with Date1 present, Date2 missing)
        # 2. complete_data == 0
        # 3. available_f2 > 0 (Rows with Date2 present) and len(unique) == 1

        # Row 1: Date1 present, Date2 missing (Satisfies 1)
        df.loc[0, "Date"] = ts1
        df.loc[0, "Date2"] = pd.NaT

        # Row 2: Date1 missing, Date2 present (Satisfies 3)
        df.loc[1, "Date"] = pd.NaT
        df.loc[1, "Date2"] = ts2

        plot_features_interaction(df, "Date", "Date2")

    else:  # missing_f1_avail_f1_single
        # We want to trigger lines 660-661.
        # Condition:
        # 1. missing_f1 > 0 (Rows with Date2 present, Date1 missing)
        # 2. complete_data == 0
        # 3. available_f1 > 0 (Rows with Date1 present) and len(unique) == 1

        # Row 1: Date2 present, Date1 missing (Satisfies 1)
        df.loc[0, "Date"] = pd.NaT
        df.loc[0, "Date2"] = ts2

        # Row 2: Date2 missing, Date1 present (Satisfies 3)
        df.loc[1, "Date"] = ts1
        df.loc[1, "Date2"] = pd.NaT

        plot_features_interaction(df, "Date", "Date2")

    plt.gcf().set_size_inches(12, 6)
    return plt.gcf()
