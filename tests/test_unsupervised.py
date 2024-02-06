from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.spatial.distance import euclidean

from ds_utils.unsupervised import plot_cluster_cardinality, plot_cluster_magnitude, plot_magnitude_vs_cardinality, \
    plot_loss_vs_cluster_number
from tests.utils import compare_images_from_paths

iris_x = pd.read_csv(Path(__file__).parents[0].joinpath("resources").joinpath("iris_x_full.csv"))
labels = np.asarray(
    [1, 6, 6, 6, 1, 1, 6, 1, 6, 6, 1, 6, 6, 6, 1, 1, 1, 1, 1, 1, 1, 1, 6, 1, 6, 6, 1, 1, 1, 6, 6, 1, 1, 1, 6,
     6, 1, 1, 6, 1, 1, 6, 6, 1, 1, 6, 1, 6, 1, 6, 0, 0, 0, 3, 0, 3, 0, 5, 0, 3, 5, 3, 3, 0, 3, 0, 3, 3, 4, 3,
     4, 3, 4, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 4, 3, 0, 0, 0, 3, 3, 3, 0, 3, 5, 3, 3, 3, 0, 5, 3, 7, 4, 2, 7, 7,
     2, 3, 2, 7, 2, 7, 4, 7, 4, 4, 7, 7, 2, 2, 4, 7, 4, 2, 4, 7, 2, 4, 4, 7, 2, 2, 2, 7, 4, 4, 2, 7, 7, 4, 7,
     7, 7, 4, 7, 7, 7, 4, 7, 7, 4])
cluster_centers = np.asarray([
    [6.442105263157895223e+00, 2.978947368421052566e+00, 4.594736842105263008e+00, 1.431578947368421062e+00],
    [5.242857142857142883e+00, 3.667857142857142705e+00, 1.499999999999999556e+00, 2.821428571428574728e-01],
    [7.474999999999999645e+00, 3.125000000000000000e+00, 6.299999999999999822e+00, 2.049999999999999822e+00],
    [5.620833333333333570e+00, 2.691666666666666430e+00, 4.075000000000000178e+00, 1.262499999999999956e+00],
    [6.036842105263158231e+00, 2.705263157894736814e+00, 5.000000000000000000e+00, 1.778947368421052611e+00],
    [5.000000000000000000e+00, 2.299999999999999822e+00, 3.274999999999999911e+00, 1.024999999999999911e+00],
    [4.704545454545455030e+00, 3.122727272727272574e+00, 1.413636363636363136e+00, 2.000000000000001776e-01],
    [6.568181818181818343e+00, 3.086363636363636420e+00, 5.536363636363637042e+00, 2.163636363636363580e+00]])

Path(__file__).parents[0].absolute().joinpath("result_images").mkdir(exist_ok=True)
Path(__file__).parents[0].absolute().joinpath("result_images").joinpath("test_unsupervised").mkdir(exist_ok=True)


def test_cluster_cardinality():
    plot_cluster_cardinality(np.asarray(labels))

    result_path = Path(__file__).parents[0].absolute().joinpath("result_images").joinpath(
        "test_unsupervised").joinpath("test_cluster_cardinality.png")
    plt.savefig(str(result_path))

    baseline_path = Path(__file__).parents[0].absolute().joinpath("baseline_images").joinpath(
        "test_unsupervised").joinpath("test_cluster_cardinality.png")
    plt.cla()
    plt.close(plt.gcf())
    compare_images_from_paths(str(baseline_path), str(result_path))


def test_cluster_cardinality_exist_ax():
    plt.figure()
    ax = plt.gca()

    ax.set_title("My ax")

    plot_cluster_cardinality(np.asarray(labels), ax=ax)

    result_path = Path(__file__).parents[0].absolute().joinpath("result_images").joinpath(
        "test_unsupervised").joinpath("test_cluster_cardinality_exist_ax.png")
    plt.savefig(str(result_path))

    baseline_path = Path(__file__).parents[0].absolute().joinpath("baseline_images").joinpath(
        "test_unsupervised").joinpath("test_cluster_cardinality_exist_ax.png")
    plt.cla()
    plt.close(plt.gcf())
    compare_images_from_paths(str(baseline_path), str(result_path))


def test_plot_cluster_magnitude():
    plot_cluster_magnitude(iris_x.values, labels, cluster_centers, euclidean)

    result_path = Path(__file__).parents[0].absolute().joinpath("result_images").joinpath(
        "test_unsupervised").joinpath("test_plot_cluster_magnitude.png")
    plt.savefig(str(result_path))

    baseline_path = Path(__file__).parents[0].absolute().joinpath("baseline_images").joinpath(
        "test_unsupervised").joinpath("test_plot_cluster_magnitude.png")
    plt.cla()
    plt.close(plt.gcf())
    compare_images_from_paths(str(baseline_path), str(result_path))


def test_plot_cluster_magnitude_exist_ax():
    plt.figure()
    ax = plt.gca()

    ax.set_title("My ax")

    plot_cluster_magnitude(iris_x.values, labels, cluster_centers, euclidean, ax=ax)

    result_path = Path(__file__).parents[0].absolute().joinpath("result_images").joinpath(
        "test_unsupervised").joinpath("test_plot_cluster_magnitude_exist_ax.png")
    plt.savefig(str(result_path))

    baseline_path = Path(__file__).parents[0].absolute().joinpath("baseline_images").joinpath(
        "test_unsupervised").joinpath("test_plot_cluster_magnitude_exist_ax.png")
    plt.cla()
    plt.close(plt.gcf())
    compare_images_from_paths(str(baseline_path), str(result_path))


def test_plot_magnitude_vs_cardinality():
    plot_magnitude_vs_cardinality(iris_x.values, labels, cluster_centers, euclidean)

    result_path = Path(__file__).parents[0].absolute().joinpath("result_images").joinpath(
        "test_unsupervised").joinpath("test_plot_magnitude_vs_cardinality.png")
    plt.savefig(str(result_path))

    baseline_path = Path(__file__).parents[0].absolute().joinpath("baseline_images").joinpath(
        "test_unsupervised").joinpath("test_plot_magnitude_vs_cardinality.png")
    plt.cla()
    plt.close(plt.gcf())
    compare_images_from_paths(str(baseline_path), str(result_path))


def test_plot_magnitude_vs_cardinality_exist_ax():
    plt.figure()
    ax = plt.gca()

    ax.set_title("My ax")

    plot_magnitude_vs_cardinality(iris_x.values, labels, cluster_centers, euclidean, ax=ax)

    result_path = Path(__file__).parents[0].absolute().joinpath("result_images").joinpath(
        "test_unsupervised").joinpath("test_plot_magnitude_vs_cardinality_exist_ax.png")
    plt.savefig(str(result_path))

    baseline_path = Path(__file__).parents[0].absolute().joinpath("baseline_images").joinpath(
        "test_unsupervised").joinpath("test_plot_magnitude_vs_cardinality_exist_ax.png")
    plt.cla()
    plt.close(plt.gcf())
    compare_images_from_paths(str(baseline_path), str(result_path))


def test_plot_loss_vs_cluster_number():
    plot_loss_vs_cluster_number(iris_x.values, 3, 20, euclidean)

    result_path = Path(__file__).parents[0].absolute().joinpath("result_images").joinpath(
        "test_unsupervised").joinpath("test_plot_loss_vs_cluster_number.png")
    plt.savefig(str(result_path))

    baseline_path = Path(__file__).parents[0].absolute().joinpath("baseline_images").joinpath(
        "test_unsupervised").joinpath("test_plot_loss_vs_cluster_number.png")
    plt.cla()
    plt.close(plt.gcf())
    compare_images_from_paths(str(baseline_path), str(result_path))


def test_plot_loss_vs_cluster_number_exist_ax():
    plt.figure()
    ax = plt.gca()

    ax.set_facecolor('tab:red')

    plot_loss_vs_cluster_number(iris_x.values, 3, 20, euclidean, ax=ax)

    result_path = Path(__file__).parents[0].absolute().joinpath("result_images").joinpath(
        "test_unsupervised").joinpath("test_plot_loss_vs_cluster_number_exist_ax.png")
    plt.savefig(str(result_path))

    baseline_path = Path(__file__).parents[0].absolute().joinpath("baseline_images").joinpath(
        "test_unsupervised").joinpath("test_plot_loss_vs_cluster_number_exist_ax.png")
    plt.cla()
    plt.close(plt.gcf())
    compare_images_from_paths(str(baseline_path), str(result_path))


def test_plot_loss_vs_cluster_number_given_parameters():
    plot_loss_vs_cluster_number(iris_x.values, 3, 20, euclidean,
                                algorithm_parameters={"random_state": 42, "algorithm": "full"})

    result_path = Path(__file__).parents[0].absolute().joinpath("result_images").joinpath(
        "test_unsupervised").joinpath("test_plot_loss_vs_cluster_number_given_parameters.png")
    plt.savefig(str(result_path))

    baseline_path = Path(__file__).parents[0].absolute().joinpath("baseline_images").joinpath(
        "test_unsupervised").joinpath("test_plot_loss_vs_cluster_number_given_parameters.png")
    plt.cla()
    plt.close(plt.gcf())
    compare_images_from_paths(str(baseline_path), str(result_path))
