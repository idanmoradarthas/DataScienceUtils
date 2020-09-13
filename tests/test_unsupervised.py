from pathlib import Path

import pandas
from matplotlib import pyplot
from sklearn.cluster import KMeans

from ds_utils.unsupervised import plot_cluster_cardinality
from tests.utils import compare_images_from_paths

iris_x = pandas.read_csv(Path(__file__).parents[0].joinpath("resources").joinpath("iris_x_full.csv"))


def test_cluster_cardinality():
    estimator = KMeans(n_clusters=8)
    estimator.fit(iris_x)

    plot_cluster_cardinality(estimator.labels_)

    result_path = Path(__file__).parents[0].absolute().joinpath("result_images").joinpath(
        "test_unsupervised").joinpath("test_cluster_cardinality.png")
    pyplot.savefig(str(result_path))

    baseline_path = Path(__file__).parents[0].absolute().joinpath("baseline_images").joinpath(
        "test_unsupervised").joinpath("test_cluster_cardinality.png")
    pyplot.cla()
    pyplot.close(pyplot.gcf())
    compare_images_from_paths(str(baseline_path), str(result_path))
