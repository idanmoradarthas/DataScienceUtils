from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from matplotlib import pyplot as plt

from ds_utils.preprocess.visualization import visualize_feature, visualize_correlations, plot_features_interaction, \
    plot_correlation_dendrogram
from tests.test_preprocess.conftest import setup_teardown, loan_data, data_1m, daily_min_temperatures

BASELINE_DIR = Path(__file__).parent.parent / "baseline_images" / "test_preprocess" / "test_visualization"


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
def test_plot_features_interaction_datetime_numeric_no_dates_at_all(daily_min_temperatures, monkeypatch):
    """Test datetime vs numeric with NO complete data and NO valid dates at all.

    This targets lines 540-542 in preprocess.py:
    if len(complete_data) == 0 and len(missing_numeric) == 0:
        ... default x-limits ...
    """
    fixed_now = pd.Timestamp("2024-06-01 00:00:00")
    monkeypatch.setattr(pd.Timestamp, "now", classmethod(lambda cls: fixed_now))

    # Take a small slice
    df = daily_min_temperatures.head(20).copy()

    # Create scenario:
    # 1. Date is missing everywhere.
    # 2. Temp is present (at least some rows).
    df["Date"] = pd.NaT
    # Keep Temp values as is (numeric present)

    # removing NA is False by default.
    # missing_datetime > 0 (all rows)
    # complete_data == 0
    # missing_numeric == 0 (no rows have valid Date and missing Temp, because NO rows have valid Date)

    plot_features_interaction(df, "Date", "Temp")
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
