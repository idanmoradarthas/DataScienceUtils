from pathlib import Path

import matplotlib

matplotlib.use('agg')
import numpy
import pandas
from matplotlib import pyplot

from ds_utils.visualization_aids import visualize_correlations, \
    plot_features_interaction, plot_correlation_dendrogram, visualize_feature
from tests.utils import compare_images_from_paths

data_1M = pandas.read_csv(Path(__file__).parents[0].joinpath("resources").joinpath("data.1M.zip"), compression='zip')
loan_data = pandas.read_csv(Path(__file__).parents[0].joinpath("resources").joinpath("loan_final313.csv"),
                            encoding="latin1", parse_dates=["issue_d"]).drop("id", axis=1)

Path(__file__).parents[0].absolute().joinpath("result_images").mkdir(exist_ok=True)
Path(__file__).parents[0].absolute().joinpath("result_images").joinpath("test_visualization_aids").mkdir(
    exist_ok=True)


def test_visualize_feature_float():
    visualize_feature(loan_data["emp_length_int"])

    result_path = Path(__file__).parents[0].absolute().joinpath("result_images").joinpath(
        "test_visualization_aids").joinpath("test_visualize_feature_float.png")
    pyplot.savefig(str(result_path))

    baseline_path = Path(__file__).parents[0].absolute().joinpath("baseline_images").joinpath(
        "test_visualization_aids").joinpath("test_visualize_feature_float.png")
    pyplot.cla()
    pyplot.close(pyplot.gcf())
    compare_images_from_paths(str(baseline_path), str(result_path))


def test_visualize_feature_float_exist_ax():
    pyplot.figure()
    ax = pyplot.gca()

    ax.set_title("My ax")

    visualize_feature(loan_data["emp_length_int"], ax=ax)

    result_path = Path(__file__).parents[0].absolute().joinpath("result_images").joinpath(
        "test_visualization_aids").joinpath("test_visualize_feature_float_exist_ax.png")
    pyplot.gcf().set_size_inches(10, 8)
    pyplot.savefig(str(result_path))

    baseline_path = Path(__file__).parents[0].absolute().joinpath("baseline_images").joinpath(
        "test_visualization_aids").joinpath("test_visualize_feature_float_exist_ax.png")
    pyplot.cla()
    pyplot.close(pyplot.gcf())
    compare_images_from_paths(str(baseline_path), str(result_path))


def test_visualize_feature_datetime():
    visualize_feature(loan_data["issue_d"])

    result_path = Path(__file__).parents[0].absolute().joinpath("result_images").joinpath(
        "test_visualization_aids").joinpath("test_visualize_feature_datetime.png")
    pyplot.gcf().set_size_inches(10, 8)
    pyplot.savefig(str(result_path))

    baseline_path = Path(__file__).parents[0].absolute().joinpath("baseline_images").joinpath(
        "test_visualization_aids").joinpath("test_visualize_feature_datetime.png")
    pyplot.cla()
    pyplot.close(pyplot.gcf())
    compare_images_from_paths(str(baseline_path), str(result_path))


def test_visualize_feature_int():
    visualize_feature(loan_data["loan_condition_cat"])

    result_path = Path(__file__).parents[0].absolute().joinpath("result_images").joinpath(
        "test_visualization_aids").joinpath("test_visualize_feature_int.png")
    pyplot.savefig(str(result_path))

    baseline_path = Path(__file__).parents[0].absolute().joinpath("baseline_images").joinpath(
        "test_visualization_aids").joinpath("test_visualize_feature_int.png")
    pyplot.cla()
    pyplot.close(pyplot.gcf())
    compare_images_from_paths(str(baseline_path), str(result_path))


def test_visualize_feature_object():
    visualize_feature(loan_data["income_category"])

    result_path = Path(__file__).parents[0].absolute().joinpath("result_images").joinpath(
        "test_visualization_aids").joinpath("test_visualize_feature_object.png")
    pyplot.gcf().set_size_inches(10, 8)
    pyplot.savefig(str(result_path))

    baseline_path = Path(__file__).parents[0].absolute().joinpath("baseline_images").joinpath(
        "test_visualization_aids").joinpath("test_visualize_feature_object.png")
    pyplot.cla()
    pyplot.close(pyplot.gcf())
    compare_images_from_paths(str(baseline_path), str(result_path))


def test_visualize_feature_category():
    visualize_feature(loan_data["home_ownership"].astype("category"))

    result_path = Path(__file__).parents[0].absolute().joinpath("result_images").joinpath(
        "test_visualization_aids").joinpath("test_visualize_feature_category.png")
    pyplot.gcf().set_size_inches(10, 8)
    pyplot.savefig(str(result_path))

    baseline_path = Path(__file__).parents[0].absolute().joinpath("baseline_images").joinpath(
        "test_visualization_aids").joinpath("test_visualize_feature_category.png")
    pyplot.cla()
    pyplot.close(pyplot.gcf())
    compare_images_from_paths(str(baseline_path), str(result_path))


def test_visualize_feature_category_more_than_10_categories():
    visualize_feature(loan_data["purpose"].astype("category"))

    result_path = Path(__file__).parents[0].absolute().joinpath("result_images").joinpath(
        "test_visualization_aids").joinpath("test_visualize_feature_category_more_than_10_categories.png")
    pyplot.gcf().set_size_inches(11, 11)
    pyplot.savefig(str(result_path))

    baseline_path = Path(__file__).parents[0].absolute().joinpath("baseline_images").joinpath(
        "test_visualization_aids").joinpath("test_visualize_feature_category_more_than_10_categories.png")
    pyplot.cla()
    pyplot.close(pyplot.gcf())
    compare_images_from_paths(str(baseline_path), str(result_path))


def test_visualize_feature_bool():
    loan_dup = pandas.DataFrame()
    loan_dup["term 36 months"] = loan_data["term"].apply(lambda term: True if term == " 36 months" else False).astype(
        "bool")
    visualize_feature(loan_dup["term 36 months"])

    result_path = Path(__file__).parents[0].absolute().joinpath("result_images").joinpath(
        "test_visualization_aids").joinpath("test_visualize_feature_bool.png")
    pyplot.savefig(str(result_path))

    baseline_path = Path(__file__).parents[0].absolute().joinpath("baseline_images").joinpath(
        "test_visualization_aids").joinpath("test_visualize_feature_bool.png")
    pyplot.cla()
    pyplot.close(pyplot.gcf())
    compare_images_from_paths(str(baseline_path), str(result_path))


def test_visualize_feature_remove_na():
    loan_data_dup = pandas.DataFrame()
    loan_data_dup["emp_length_int"] = loan_data["emp_length_int"]
    loan_data_dup = pandas.concat(
        [loan_data_dup,
         pandas.DataFrame([numpy.nan] * 250, columns=["emp_length_int"])],
        ignore_index=True).sample(frac=1, random_state=0)

    assert loan_data_dup["emp_length_int"].isna().sum() == 250

    visualize_feature(loan_data_dup["emp_length_int"], remove_na=True)

    result_path = Path(__file__).parents[0].absolute().joinpath("result_images").joinpath(
        "test_visualization_aids").joinpath("test_visualize_feature_remove_na.png")
    pyplot.savefig(str(result_path))

    baseline_path = Path(__file__).parents[0].absolute().joinpath("baseline_images").joinpath(
        "test_visualization_aids").joinpath("test_visualize_feature_remove_na.png")
    pyplot.cla()
    pyplot.close(pyplot.gcf())
    compare_images_from_paths(str(baseline_path), str(result_path))


def test_visualize_correlations():
    visualize_correlations(data_1M)

    result_path = Path(__file__).parents[0].absolute().joinpath("result_images").joinpath(
        "test_visualization_aids").joinpath("test_visualize_correlations.png")
    pyplot.gcf().set_size_inches(14, 9)
    pyplot.savefig(str(result_path))

    baseline_path = Path(__file__).parents[0].absolute().joinpath("baseline_images").joinpath(
        "test_visualization_aids").joinpath("test_visualize_correlations.png")
    pyplot.cla()
    pyplot.close(pyplot.gcf())
    compare_images_from_paths(str(baseline_path), str(result_path))


def test_visualize_correlations_exist_ax():
    pyplot.figure()
    ax = pyplot.gca()

    ax.set_title("My ax")

    visualize_correlations(data_1M, ax=ax)

    result_path = Path(__file__).parents[0].absolute().joinpath("result_images").joinpath(
        "test_visualization_aids").joinpath("test_visualize_correlations_exist_ax.png")
    pyplot.gcf().set_size_inches(14, 9)
    pyplot.savefig(str(result_path))

    baseline_path = Path(__file__).parents[0].absolute().joinpath("baseline_images").joinpath(
        "test_visualization_aids").joinpath("test_visualize_correlations_exist_ax.png")
    pyplot.cla()
    pyplot.close(pyplot.gcf())
    compare_images_from_paths(str(baseline_path), str(result_path))


def test_plot_relationship_between_features_both_numeric():
    plot_features_interaction("x4", "x5", data_1M)

    result_path = Path(__file__).parents[0].absolute().joinpath("result_images").joinpath(
        "test_visualization_aids").joinpath("test_plot_relationship_between_features_both_numeric.png")
    pyplot.savefig(str(result_path))

    baseline_path = Path(__file__).parents[0].absolute().joinpath("baseline_images").joinpath(
        "test_visualization_aids").joinpath("test_plot_relationship_between_features_both_numeric.png")
    pyplot.cla()
    pyplot.close(pyplot.gcf())
    compare_images_from_paths(str(baseline_path), str(result_path))


def test_plot_relationship_between_features_both_numeric_exist_ax():
    pyplot.figure()
    ax = pyplot.gca()

    ax.set_title("My ax")

    plot_features_interaction("x4", "x5", data_1M, ax=ax)

    result_path = Path(__file__).parents[0].absolute().joinpath("result_images").joinpath(
        "test_visualization_aids").joinpath("test_plot_relationship_between_features_both_numeric_exist_ax.png")
    pyplot.savefig(str(result_path))

    baseline_path = Path(__file__).parents[0].absolute().joinpath("baseline_images").joinpath(
        "test_visualization_aids").joinpath("test_plot_relationship_between_features_both_numeric_exist_ax.png")
    pyplot.cla()
    pyplot.close(pyplot.gcf())
    compare_images_from_paths(str(baseline_path), str(result_path))


def test_plot_relationship_between_features_numeric_categorical():
    plot_features_interaction("x1", "x7", data_1M)

    result_path = Path(__file__).parents[0].absolute().joinpath("result_images").joinpath(
        "test_visualization_aids").joinpath("test_plot_relationship_between_features_numeric_categorical.png")
    pyplot.gcf().set_size_inches(14, 9)
    pyplot.savefig(str(result_path))

    baseline_path = Path(__file__).parents[0].absolute().joinpath("baseline_images").joinpath(
        "test_visualization_aids").joinpath("test_plot_relationship_between_features_numeric_categorical.png")
    pyplot.cla()
    pyplot.close(pyplot.gcf())
    compare_images_from_paths(str(baseline_path), str(result_path))


def test_plot_relationship_between_features_numeric_categorical_reverse():
    plot_features_interaction("x7", "x1", data_1M)

    result_path = Path(__file__).parents[0].absolute().joinpath("result_images").joinpath(
        "test_visualization_aids").joinpath("test_plot_relationship_between_features_numeric_categorical_reverse.png")
    pyplot.gcf().set_size_inches(14, 9)
    pyplot.savefig(str(result_path))

    baseline_path = Path(__file__).parents[0].absolute().joinpath("baseline_images").joinpath(
        "test_visualization_aids").joinpath("test_plot_relationship_between_features_numeric_categorical_reverse.png")
    pyplot.cla()
    pyplot.close(pyplot.gcf())
    compare_images_from_paths(str(baseline_path), str(result_path))


def test_plot_relationship_between_features_numeric_boolean():
    plot_features_interaction("x1", "x12", data_1M)

    result_path = Path(__file__).parents[0].absolute().joinpath("result_images").joinpath(
        "test_visualization_aids").joinpath("test_plot_relationship_between_features_numeric_boolean.png")
    pyplot.gcf().set_size_inches(8, 7)
    pyplot.savefig(str(result_path))

    baseline_path = Path(__file__).parents[0].absolute().joinpath("baseline_images").joinpath(
        "test_visualization_aids").joinpath("test_plot_relationship_between_features_numeric_boolean.png")
    pyplot.cla()
    pyplot.close(pyplot.gcf())
    compare_images_from_paths(str(baseline_path), str(result_path))


def test_plot_relationship_between_features_both_categorical():
    plot_features_interaction("x7", "x10", data_1M)

    result_path = Path(__file__).parents[0].absolute().joinpath("result_images").joinpath(
        "test_visualization_aids").joinpath("test_plot_relationship_between_features_both_categorical.png")
    pyplot.gcf().set_size_inches(9, 5)
    pyplot.savefig(str(result_path))

    baseline_path = Path(__file__).parents[0].absolute().joinpath("baseline_images").joinpath(
        "test_visualization_aids").joinpath("test_plot_relationship_between_features_both_categorical.png")
    pyplot.cla()
    pyplot.close(pyplot.gcf())
    compare_images_from_paths(str(baseline_path), str(result_path))


def test_plot_relationship_between_features_categorical_bool():
    plot_features_interaction("x10", "x12", data_1M)

    result_path = Path(__file__).parents[0].absolute().joinpath("result_images").joinpath(
        "test_visualization_aids").joinpath("test_plot_relationship_between_features_categorical_bool.png")
    pyplot.savefig(str(result_path))

    baseline_path = Path(__file__).parents[0].absolute().joinpath("baseline_images").joinpath(
        "test_visualization_aids").joinpath("test_plot_relationship_between_features_categorical_bool.png")
    pyplot.cla()
    pyplot.close(pyplot.gcf())
    compare_images_from_paths(str(baseline_path), str(result_path))


def test_plot_relationship_between_features_datetime_numeric():
    daily_min_temperatures = pandas.read_csv(
        Path(__file__).parents[0].joinpath("resources").joinpath("daily-min-temperatures.csv"), parse_dates=["Date"])

    plot_features_interaction("Date", "Temp", daily_min_temperatures)

    result_path = Path(__file__).parents[0].absolute().joinpath("result_images").joinpath(
        "test_visualization_aids").joinpath("test_plot_relationship_between_features_datetime_numeric.png")
    pyplot.gcf().set_size_inches(18, 8)
    pyplot.savefig(str(result_path))

    baseline_path = Path(__file__).parents[0].absolute().joinpath("baseline_images").joinpath(
        "test_visualization_aids").joinpath("test_plot_relationship_between_features_datetime_numeric.png")
    pyplot.cla()
    pyplot.close(pyplot.gcf())
    compare_images_from_paths(str(baseline_path), str(result_path))


def test_plot_relationship_between_features_datetime_numeric_2():
    daily_min_temperatures = pandas.read_csv(
        Path(__file__).parents[0].joinpath("resources").joinpath("daily-min-temperatures.csv"), parse_dates=["Date"])

    plot_features_interaction("Temp", "Date", daily_min_temperatures)

    result_path = Path(__file__).parents[0].absolute().joinpath("result_images").joinpath(
        "test_visualization_aids").joinpath("test_plot_relationship_between_features_datetime_numeric_2.png")
    pyplot.gcf().set_size_inches(18, 8)
    pyplot.savefig(str(result_path))

    baseline_path = Path(__file__).parents[0].absolute().joinpath("baseline_images").joinpath(
        "test_visualization_aids").joinpath("test_plot_relationship_between_features_datetime_numeric_2.png")
    pyplot.cla()
    pyplot.close(pyplot.gcf())
    compare_images_from_paths(str(baseline_path), str(result_path))


def test_plot_relationship_between_features_datetime_datetime():
    plot_features_interaction("issue_d", "issue_d", loan_data)

    result_path = Path(__file__).parents[0].absolute().joinpath("result_images").joinpath(
        "test_visualization_aids").joinpath("test_plot_relationship_between_features_datetime_datetime.png")
    pyplot.savefig(str(result_path))

    baseline_path = Path(__file__).parents[0].absolute().joinpath("baseline_images").joinpath(
        "test_visualization_aids").joinpath("test_plot_relationship_between_features_datetime_datetime.png")
    pyplot.cla()
    pyplot.close(pyplot.gcf())
    compare_images_from_paths(str(baseline_path), str(result_path))


def test_plot_relationship_between_features_datetime_categorical():
    plot_features_interaction("issue_d", "home_ownership", loan_data)

    result_path = Path(__file__).parents[0].absolute().joinpath("result_images").joinpath(
        "test_visualization_aids").joinpath("test_plot_relationship_between_features_datetime_categorical.png")
    pyplot.gcf().set_size_inches(10, 11.5)
    pyplot.savefig(str(result_path))

    baseline_path = Path(__file__).parents[0].absolute().joinpath("baseline_images").joinpath(
        "test_visualization_aids").joinpath("test_plot_relationship_between_features_datetime_categorical.png")
    pyplot.cla()
    pyplot.close(pyplot.gcf())
    compare_images_from_paths(str(baseline_path), str(result_path))


def test_plot_relationship_between_features_datetime_categorical_2():
    plot_features_interaction("home_ownership", "issue_d", loan_data)

    result_path = Path(__file__).parents[0].absolute().joinpath("result_images").joinpath(
        "test_visualization_aids").joinpath("test_plot_relationship_between_features_datetime_categorical_2.png")
    pyplot.gcf().set_size_inches(10, 11.5)
    pyplot.savefig(str(result_path))

    baseline_path = Path(__file__).parents[0].absolute().joinpath("baseline_images").joinpath(
        "test_visualization_aids").joinpath("test_plot_relationship_between_features_datetime_categorical_2.png")
    pyplot.cla()
    pyplot.close(pyplot.gcf())
    compare_images_from_paths(str(baseline_path), str(result_path))


def test_plot_relationship_between_features_datetime_bool():
    df = pandas.DataFrame()
    df["loan_condition_cat"] = loan_data["loan_condition_cat"].astype("bool")
    df["issue_d"] = loan_data["issue_d"]
    plot_features_interaction("issue_d", "loan_condition_cat", df)

    result_path = Path(__file__).parents[0].absolute().joinpath("result_images").joinpath(
        "test_visualization_aids").joinpath("test_plot_relationship_between_features_datetime_bool.png")
    pyplot.gcf().set_size_inches(10, 11.5)
    pyplot.savefig(str(result_path))

    baseline_path = Path(__file__).parents[0].absolute().joinpath("baseline_images").joinpath(
        "test_visualization_aids").joinpath("test_plot_relationship_between_features_datetime_bool.png")
    pyplot.cla()
    pyplot.close(pyplot.gcf())
    compare_images_from_paths(str(baseline_path), str(result_path))


def test_plot_relationship_between_features_datetime_bool_2():
    df = pandas.DataFrame()
    df["loan_condition_cat"] = loan_data["loan_condition_cat"].astype("bool")
    df["issue_d"] = loan_data["issue_d"]
    plot_features_interaction("loan_condition_cat", "issue_d", df)

    result_path = Path(__file__).parents[0].absolute().joinpath("result_images").joinpath(
        "test_visualization_aids").joinpath("test_plot_relationship_between_features_datetime_bool_2.png")
    pyplot.gcf().set_size_inches(10, 11.5)
    pyplot.savefig(str(result_path))

    baseline_path = Path(__file__).parents[0].absolute().joinpath("baseline_images").joinpath(
        "test_visualization_aids").joinpath("test_plot_relationship_between_features_datetime_bool_2.png")
    pyplot.cla()
    pyplot.close(pyplot.gcf())
    compare_images_from_paths(str(baseline_path), str(result_path))


def test_plot_relationship_between_features_both_bool():
    plot_features_interaction("x12", "x12", data_1M)

    result_path = Path(__file__).parents[0].absolute().joinpath("result_images").joinpath(
        "test_visualization_aids").joinpath("test_plot_relationship_between_features_both_bool.png")
    pyplot.savefig(str(result_path))

    baseline_path = Path(__file__).parents[0].absolute().joinpath("baseline_images").joinpath(
        "test_visualization_aids").joinpath("test_plot_relationship_between_features_both_bool.png")
    pyplot.cla()
    pyplot.close(pyplot.gcf())
    compare_images_from_paths(str(baseline_path), str(result_path))


def test_plot_correlation_dendrogram():
    plot_correlation_dendrogram(data_1M)

    result_path = Path(__file__).parents[0].absolute().joinpath("result_images").joinpath(
        "test_visualization_aids").joinpath("test_plot_correlation_dendrogram.png")
    pyplot.savefig(str(result_path))

    baseline_path = Path(__file__).parents[0].absolute().joinpath("baseline_images").joinpath(
        "test_visualization_aids").joinpath("test_plot_correlation_dendrogram.png")
    pyplot.cla()
    pyplot.close(pyplot.gcf())
    compare_images_from_paths(str(baseline_path), str(result_path))


def test_plot_correlation_dendrogram_exist_ax():
    pyplot.figure()
    ax = pyplot.gca()

    ax.set_title("My ax")

    plot_correlation_dendrogram(data_1M, ax=ax)

    result_path = Path(__file__).parents[0].absolute().joinpath("result_images").joinpath(
        "test_visualization_aids").joinpath("test_plot_correlation_dendrogram_exist_ax.png")
    pyplot.savefig(str(result_path))

    baseline_path = Path(__file__).parents[0].absolute().joinpath("baseline_images").joinpath(
        "test_visualization_aids").joinpath("test_plot_correlation_dendrogram_exist_ax.png")
    pyplot.cla()
    pyplot.close(pyplot.gcf())
    compare_images_from_paths(str(baseline_path), str(result_path))
