from pathlib import Path

import pandas
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier

from ds_utils.visualization_aids import draw_tree, visualize_features
from tests.utils import compare_images_paths


def test_draw_tree():
    iris = datasets.load_iris()
    x = iris.data
    y = iris.target

    # Create decision tree classifier object
    clf = DecisionTreeClassifier(random_state=0)

    # Train model
    clf.fit(x, y)

    plot = draw_tree(clf, iris.feature_names, iris.target_names)

    Path(__file__).parents[0].absolute().joinpath("result_images").mkdir(exist_ok=True)
    Path(__file__).parents[0].absolute().joinpath("result_images").joinpath("test_visualization_aids").mkdir(
        exist_ok=True)
    result_path = Path(__file__).parents[0].absolute().joinpath("result_images").joinpath(
        "test_visualization_aids").joinpath("test_draw_tree.png")
    plot.savefig(str(result_path))

    baseline_path = Path(__file__).parents[0].absolute().joinpath("baseline_images").joinpath(
        "test_visualization_aids").joinpath("test_draw_tree.png")
    compare_images_paths(str(baseline_path), str(result_path))


def test_visualize_features():
    file_path = Path(__file__).parents[0].joinpath("resources").joinpath("loan_final313.csv")
    loan_frame = pandas.read_csv(file_path, encoding="latin1", parse_dates=["issue_d"])
    loan_frame = loan_frame.drop("id", axis=1)

    plot = visualize_features(loan_frame)
    Path(__file__).parents[0].absolute().joinpath("result_images").mkdir(exist_ok=True)
    Path(__file__).parents[0].absolute().joinpath("result_images").joinpath("test_visualization_aids").mkdir(
        exist_ok=True)
    result_path = Path(__file__).parents[0].absolute().joinpath("result_images").joinpath(
        "test_visualization_aids").joinpath("test_visualize_features.png")
    plot.savefig(str(result_path))

    baseline_path = Path(__file__).parents[0].absolute().joinpath("baseline_images").joinpath(
        "test_visualization_aids").joinpath("test_visualize_features.png")
    compare_images_paths(str(baseline_path), str(result_path))
