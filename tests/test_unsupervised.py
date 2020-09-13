from pathlib import Path

import numpy
import pandas
from matplotlib import pyplot
from scipy.spatial.distance import euclidean

from ds_utils.unsupervised import plot_cluster_cardinality, plot_cluster_magnitude
from tests.utils import compare_images_from_paths

iris_x = pandas.read_csv(Path(__file__).parents[0].joinpath("resources").joinpath("iris_x_full.csv"))
labels = numpy.asarray(
    [1, 6, 6, 6, 1, 1, 6, 1, 6, 6, 1, 6, 6, 6, 1, 1, 1, 1, 1, 1, 1, 1, 6, 1, 6, 6, 1, 1, 1, 6, 6, 1, 1, 1, 6,
     6, 1, 1, 6, 1, 1, 6, 6, 1, 1, 6, 1, 6, 1, 6, 0, 0, 0, 3, 0, 3, 0, 5, 0, 3, 5, 3, 3, 0, 3, 0, 3, 3, 4, 3,
     4, 3, 4, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 4, 3, 0, 0, 0, 3, 3, 3, 0, 3, 5, 3, 3, 3, 0, 5, 3, 7, 4, 2, 7, 7,
     2, 3, 2, 7, 2, 7, 4, 7, 4, 4, 7, 7, 2, 2, 4, 7, 4, 2, 4, 7, 2, 4, 4, 7, 2, 2, 2, 7, 4, 4, 2, 7, 7, 4, 7,
     7, 7, 4, 7, 7, 7, 4, 7, 7, 4])
cluster_centers = numpy.asarray([
    [6.442105263157895223e+00, 2.978947368421052566e+00, 4.594736842105263008e+00, 1.431578947368421062e+00],
    [5.242857142857142883e+00, 3.667857142857142705e+00, 1.499999999999999556e+00, 2.821428571428574728e-01],
    [7.474999999999999645e+00, 3.125000000000000000e+00, 6.299999999999999822e+00, 2.049999999999999822e+00],
    [5.620833333333333570e+00, 2.691666666666666430e+00, 4.075000000000000178e+00, 1.262499999999999956e+00],
    [6.036842105263158231e+00, 2.705263157894736814e+00, 5.000000000000000000e+00, 1.778947368421052611e+00],
    [5.000000000000000000e+00, 2.299999999999999822e+00, 3.274999999999999911e+00, 1.024999999999999911e+00],
    [4.704545454545455030e+00, 3.122727272727272574e+00, 1.413636363636363136e+00, 2.000000000000001776e-01],
    [6.568181818181818343e+00, 3.086363636363636420e+00, 5.536363636363637042e+00, 2.163636363636363580e+00]])


def test_cluster_cardinality():
    plot_cluster_cardinality(numpy.asarray(labels))

    result_path = Path(__file__).parents[0].absolute().joinpath("result_images").joinpath(
        "test_unsupervised").joinpath("test_cluster_cardinality.png")
    pyplot.savefig(str(result_path))

    baseline_path = Path(__file__).parents[0].absolute().joinpath("baseline_images").joinpath(
        "test_unsupervised").joinpath("test_cluster_cardinality.png")
    pyplot.cla()
    pyplot.close(pyplot.gcf())
    compare_images_from_paths(str(baseline_path), str(result_path))


def test_cluster_cardinality_exist_ax():
    pyplot.figure()
    ax = pyplot.gca()

    ax.set_title("My ax")

    plot_cluster_cardinality(numpy.asarray(labels), ax=ax)

    result_path = Path(__file__).parents[0].absolute().joinpath("result_images").joinpath(
        "test_unsupervised").joinpath("test_cluster_cardinality_exist_ax.png")
    pyplot.savefig(str(result_path))

    baseline_path = Path(__file__).parents[0].absolute().joinpath("baseline_images").joinpath(
        "test_unsupervised").joinpath("test_cluster_cardinality_exist_ax.png")
    pyplot.cla()
    pyplot.close(pyplot.gcf())
    compare_images_from_paths(str(baseline_path), str(result_path))


def test_plot_cluster_magnitude():
    plot_cluster_magnitude(iris_x, labels, cluster_centers, euclidean)

    result_path = Path(__file__).parents[0].absolute().joinpath("result_images").joinpath(
        "test_unsupervised").joinpath("test_plot_cluster_magnitude.png")
    pyplot.savefig(str(result_path))

    baseline_path = Path(__file__).parents[0].absolute().joinpath("baseline_images").joinpath(
        "test_unsupervised").joinpath("test_plot_cluster_magnitude.png")
    pyplot.cla()
    pyplot.close(pyplot.gcf())
    compare_images_from_paths(str(baseline_path), str(result_path))


def test_plot_cluster_magnitude_exist_ax():
    pyplot.figure()
    ax = pyplot.gca()

    ax.set_title("My ax")

    plot_cluster_magnitude(iris_x, labels, cluster_centers, euclidean, ax=ax)

    result_path = Path(__file__).parents[0].absolute().joinpath("result_images").joinpath(
        "test_unsupervised").joinpath("test_plot_cluster_magnitude_exist_ax.png")
    pyplot.savefig(str(result_path))

    baseline_path = Path(__file__).parents[0].absolute().joinpath("baseline_images").joinpath(
        "test_unsupervised").joinpath("test_plot_cluster_magnitude_exist_ax.png")
    pyplot.cla()
    pyplot.close(pyplot.gcf())
    compare_images_from_paths(str(baseline_path), str(result_path))
