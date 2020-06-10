from pathlib import Path

import matplotlib

matplotlib.use('agg')
import numpy
import pandas
from matplotlib import pyplot
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier

from ds_utils.visualization_aids import draw_tree, visualize_features, draw_dot_data, visualize_correlations
from tests.utils import compare_images_paths

iris = datasets.load_iris()
x = iris.data
y = iris.target

Path(__file__).parents[0].absolute().joinpath("result_images").mkdir(exist_ok=True)
Path(__file__).parents[0].absolute().joinpath("result_images").joinpath("test_visualization_aids").mkdir(
    exist_ok=True)


def test_draw_tree():
    # Create decision tree classifier object
    clf = DecisionTreeClassifier(random_state=0)

    # Train model
    clf.fit(x, y)

    draw_tree(clf, iris.feature_names, iris.target_names)
    result_path = Path(__file__).parents[0].absolute().joinpath("result_images").joinpath(
        "test_visualization_aids").joinpath("test_draw_tree.png")
    pyplot.savefig(str(result_path))

    baseline_path = Path(__file__).parents[0].absolute().joinpath("baseline_images").joinpath(
        "test_visualization_aids").joinpath("test_draw_tree.png")
    compare_images_paths(str(baseline_path), str(result_path))


def test_draw_tree_exists_ax():
    # Create decision tree classifier object
    clf = DecisionTreeClassifier(random_state=0)

    # Train model
    clf.fit(x, y)

    pyplot.figure()
    ax = pyplot.gca()

    ax.set_title("My ax")

    draw_tree(clf, iris.feature_names, iris.target_names, ax=ax)

    result_path = Path(__file__).parents[0].absolute().joinpath("result_images").joinpath(
        "test_visualization_aids").joinpath("test_draw_tree_exists_ax.png")
    pyplot.savefig(str(result_path))

    baseline_path = Path(__file__).parents[0].absolute().joinpath("baseline_images").joinpath(
        "test_visualization_aids").joinpath("test_draw_tree_exists_ax.png")
    compare_images_paths(str(baseline_path), str(result_path))


def test_draw_dot_data():
    dot_data = "digraph D{\n" \
               "\tA [shape=diamond]\n" \
               "\tB [shape=box]\n" \
               "\tC [shape=circle]\n" \
               "\n" \
               "\tA -> B [style=dashed, color=grey]\n" \
               "\tA -> C [color=\"black:invis:black\"]\n" \
               "\tA -> D [penwidth=5, arrowhead=none]\n" \
               "\n" \
               "}"

    draw_dot_data(dot_data)

    result_path = Path(__file__).parents[0].absolute().joinpath("result_images").joinpath(
        "test_visualization_aids").joinpath("test_draw_dot_data.png")
    pyplot.savefig(str(result_path))

    baseline_path = Path(__file__).parents[0].absolute().joinpath("baseline_images").joinpath(
        "test_visualization_aids").joinpath("test_draw_dot_data.png")
    compare_images_paths(str(baseline_path), str(result_path))


def test_draw_dot_data_exist_ax():
    dot_data = "digraph D{\n" \
               "\tA [shape=diamond]\n" \
               "\tB [shape=box]\n" \
               "\tC [shape=circle]\n" \
               "\n" \
               "\tA -> B [style=dashed, color=grey]\n" \
               "\tA -> C [color=\"black:invis:black\"]\n" \
               "\tA -> D [penwidth=5, arrowhead=none]\n" \
               "\n" \
               "}"

    pyplot.figure()
    ax = pyplot.gca()

    ax.set_title("My ax")

    draw_dot_data(dot_data, ax=ax)

    result_path = Path(__file__).parents[0].absolute().joinpath("result_images").joinpath(
        "test_visualization_aids").joinpath("test_draw_dot_data_exist_ax.png")
    pyplot.savefig(str(result_path))

    baseline_path = Path(__file__).parents[0].absolute().joinpath("baseline_images").joinpath(
        "test_visualization_aids").joinpath("test_draw_dot_data_exist_ax.png")
    compare_images_paths(str(baseline_path), str(result_path))


def test_visualize_features():
    file_path = Path(__file__).parents[0].joinpath("resources").joinpath("loan_final313.csv")
    loan_frame = pandas.read_csv(file_path, encoding="latin1", parse_dates=["issue_d"])
    loan_frame = loan_frame.drop("id", axis=1)

    visualize_features(loan_frame)

    result_path = Path(__file__).parents[0].absolute().joinpath("result_images").joinpath(
        "test_visualization_aids").joinpath("test_visualize_features.png")
    pyplot.gcf().set_size_inches(20, 30)
    pyplot.savefig(str(result_path))

    baseline_path = Path(__file__).parents[0].absolute().joinpath("baseline_images").joinpath(
        "test_visualization_aids").joinpath("test_visualize_features.png")
    compare_images_paths(str(baseline_path), str(result_path))


def test_visualize_features_list_of_features():
    frame = pandas.DataFrame(x, columns=iris.feature_names)
    visualize_features(frame, iris.feature_names[:2])

    result_path = Path(__file__).parents[0].absolute().joinpath("result_images").joinpath(
        "test_visualization_aids").joinpath("test_visualize_features_list_of_features.png")
    pyplot.savefig(str(result_path))

    baseline_path = Path(__file__).parents[0].absolute().joinpath("baseline_images").joinpath(
        "test_visualization_aids").joinpath("test_visualize_features_list_of_features.png")
    compare_images_paths(str(baseline_path), str(result_path))


def test_visualize_features_remove_na():
    file_path = Path(__file__).parents[0].joinpath("resources").joinpath("loan_final313.csv")
    loan_frame = pandas.read_csv(file_path, encoding="latin1", parse_dates=["issue_d"])
    loan_frame = loan_frame.drop("id", axis=1)
    loan_frame = loan_frame.sample(1000, random_state=0)
    loan_frame = pandas.concat(
        [loan_frame, pandas.DataFrame([[numpy.nan] * len(loan_frame.columns)] * 250, columns=loan_frame.columns)],
        ignore_index=True).sample(frac=1, random_state=0)

    visualize_features(loan_frame, remove_na=True)

    result_path = Path(__file__).parents[0].absolute().joinpath("result_images").joinpath(
        "test_visualization_aids").joinpath("test_visualize_features_remove_na.png")
    pyplot.gcf().set_size_inches(20, 30)
    pyplot.savefig(str(result_path))

    baseline_path = Path(__file__).parents[0].absolute().joinpath("baseline_images").joinpath(
        "test_visualization_aids").joinpath("test_visualize_features_remove_na.png")
    compare_images_paths(str(baseline_path), str(result_path))


def test_visualize_correlations():
    file_path = Path(__file__).parents[0].joinpath("resources").joinpath("data.1M.zip")
    data = pandas.read_csv(file_path, compression='zip', index_col=0)
    data_minimal = data[['days_since_first', 'positives', 'scan_date', 'times_submitted', 'unique_sources',
                         'additional_info.exiftool.CodeSize', 'additional_info.exiftool.FileType',
                         'additional_info.exiftool.InitializedDataSize',
                         'additional_info.exiftool.UninitializedDataSize',
                         'additional_info.exiftool.PEType', 'size', 'network_activity']].copy()
    data_minimal["network_activity"] = data_minimal["network_activity"].astype("bool")
    data_minimal = data_minimal.rename(
        {"days_since_first": "x1", "positives": "x2", "scan_date": "x3", "times_submitted": "x4",
         "unique_sources": "x5", "additional_info.exiftool.CodeSize": "x6",
         "additional_info.exiftool.FileType": "x7",
         "additional_info.exiftool.InitializedDataSize": "x8",
         "additional_info.exiftool.UninitializedDataSize": "x9",
         "additional_info.exiftool.PEType": "x10", "size": "x11",
         "network_activity": "x12"}, axis="columns")

    visualize_correlations(data_minimal)

    result_path = Path(__file__).parents[0].absolute().joinpath("result_images").joinpath(
        "test_visualization_aids").joinpath("test_visualize_correlations.png")
    pyplot.gcf().set_size_inches(14, 9)
    pyplot.savefig(str(result_path))

    baseline_path = Path(__file__).parents[0].absolute().joinpath("baseline_images").joinpath(
        "test_visualization_aids").joinpath("test_visualize_correlations.png")
    compare_images_paths(str(baseline_path), str(result_path))


def test_visualize_correlations_exist_ax():
    file_path = Path(__file__).parents[0].joinpath("resources").joinpath("data.1M.zip")
    data = pandas.read_csv(file_path, compression='zip', index_col=0)
    data_minimal = data[['days_since_first', 'positives', 'scan_date', 'times_submitted', 'unique_sources',
                         'additional_info.exiftool.CodeSize', 'additional_info.exiftool.FileType',
                         'additional_info.exiftool.InitializedDataSize',
                         'additional_info.exiftool.UninitializedDataSize',
                         'additional_info.exiftool.PEType', 'size', 'network_activity']].copy()
    data_minimal["network_activity"] = data_minimal["network_activity"].astype("bool")
    data_minimal = data_minimal.rename(
        {"days_since_first": "x1", "positives": "x2", "scan_date": "x3", "times_submitted": "x4",
         "unique_sources": "x5", "additional_info.exiftool.CodeSize": "x6",
         "additional_info.exiftool.FileType": "x7",
         "additional_info.exiftool.InitializedDataSize": "x8",
         "additional_info.exiftool.UninitializedDataSize": "x9",
         "additional_info.exiftool.PEType": "x10", "size": "x11",
         "network_activity": "x12"}, axis="columns")

    pyplot.figure()
    ax = pyplot.gca()

    ax.set_title("My ax")

    visualize_correlations(data_minimal, ax=ax)

    result_path = Path(__file__).parents[0].absolute().joinpath("result_images").joinpath(
        "test_visualization_aids").joinpath("test_visualize_correlations_exist_ax.png")
    pyplot.gcf().set_size_inches(14, 9)
    pyplot.savefig(str(result_path))

    baseline_path = Path(__file__).parents[0].absolute().joinpath("baseline_images").joinpath(
        "test_visualization_aids").joinpath("test_visualize_correlations_exist_ax.png")
    compare_images_paths(str(baseline_path), str(result_path))
