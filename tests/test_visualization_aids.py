from pathlib import Path

import matplotlib
import pytest

matplotlib.use('agg')
import numpy
import pandas
from matplotlib import pyplot
from sklearn import datasets

from ds_utils.visualization_aids import visualize_features, visualize_correlations, \
    plot_features_interaction
from tests.utils import compare_images_from_paths

iris = datasets.load_iris()
x = iris.data
y = iris.target

data_1M = pandas.read_csv(Path(__file__).parents[0].joinpath("resources").joinpath("data.1M.zip"), compression='zip')
loan_data = pandas.read_csv(Path(__file__).parents[0].joinpath("resources").joinpath("loan_final313.csv"),
                            encoding="latin1", parse_dates=["issue_d"]).drop("id", axis=1)

Path(__file__).parents[0].absolute().joinpath("result_images").mkdir(exist_ok=True)
Path(__file__).parents[0].absolute().joinpath("result_images").joinpath("test_visualization_aids").mkdir(
    exist_ok=True)


def test_visualize_features():
    visualize_features(loan_data)

    result_path = Path(__file__).parents[0].absolute().joinpath("result_images").joinpath(
        "test_visualization_aids").joinpath("test_visualize_features.png")
    pyplot.gcf().set_size_inches(20, 30)
    pyplot.savefig(str(result_path))

    baseline_path = Path(__file__).parents[0].absolute().joinpath("baseline_images").joinpath(
        "test_visualization_aids").joinpath("test_visualize_features.png")
    pyplot.cla()
    pyplot.close(pyplot.gcf())
    compare_images_from_paths(str(baseline_path), str(result_path))


def test_visualize_features_list_of_features():
    frame = pandas.DataFrame(x, columns=iris.feature_names)
    visualize_features(frame, iris.feature_names[:2])

    result_path = Path(__file__).parents[0].absolute().joinpath("result_images").joinpath(
        "test_visualization_aids").joinpath("test_visualize_features_list_of_features.png")
    pyplot.savefig(str(result_path))

    baseline_path = Path(__file__).parents[0].absolute().joinpath("baseline_images").joinpath(
        "test_visualization_aids").joinpath("test_visualize_features_list_of_features.png")
    pyplot.cla()
    pyplot.close(pyplot.gcf())
    compare_images_from_paths(str(baseline_path), str(result_path))


def test_visualize_features_remove_na():
    loan_data_dup = loan_data.sample(1000, random_state=0)
    loan_data_dup = pandas.concat(
        [loan_data_dup,
         pandas.DataFrame([[numpy.nan] * len(loan_data_dup.columns)] * 250, columns=loan_data_dup.columns)],
        ignore_index=True).sample(frac=1, random_state=0)

    visualize_features(loan_data_dup, remove_na=True)

    result_path = Path(__file__).parents[0].absolute().joinpath("result_images").joinpath(
        "test_visualization_aids").joinpath("test_visualize_features_remove_na.png")
    pyplot.gcf().set_size_inches(20, 30)
    pyplot.savefig(str(result_path))

    baseline_path = Path(__file__).parents[0].absolute().joinpath("baseline_images").joinpath(
        "test_visualization_aids").joinpath("test_visualize_features_remove_na.png")
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


@pytest.mark.skip()
def test_loop_plot_features_relationship_example():
    figure, axes = pyplot.subplots(6, 2)
    axes = axes.flatten()
    figure.set_size_inches(16, 25)

    feature_1 = "x1"
    other_features = ["x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12"]

    for i in range(0, len(other_features)):
        axes[i].set_title(f"{feature_1} vs. {other_features[i]}")
        plot_features_interaction(feature_1, other_features[i], data_1M, ax=axes[i])

    figure.delaxes(axes[11])
    figure.subplots_adjust(hspace=0.7)

    result_path = Path(__file__).parents[0].absolute().joinpath("baseline_images").joinpath(
        "test_visualization_aids").joinpath("loop_plot_features_relationship_example.png")
    pyplot.savefig(str(result_path))
    pyplot.cla()
    pyplot.close(pyplot.gcf())


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
