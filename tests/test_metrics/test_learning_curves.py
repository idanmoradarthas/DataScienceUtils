"""Test the learning curves metrics."""

from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np
from numpy.random import RandomState
import pandas as pd
import pytest

from ds_utils.metrics.learning_curves import plot_metric_growth_per_labeled_instances

BASELINE_DIR = Path(__file__).parents[1] / "baseline_images" / Path(__file__).parent.name / Path(__file__).stem


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR)
@pytest.mark.parametrize(
    ("n_samples", "quantiles", "random_state", "use_dummies"),
    [
        (None, np.linspace(0.05, 1, 20).tolist(), 42, False),
        (None, np.linspace(0.05, 1, 20).tolist(), 42, True),  # This is the y_shape_n_outputs case
        (list(range(10, 100, 10)), None, 42, False),
        (None, np.linspace(0.05, 1, 20).tolist(), 1, False),
        (None, np.linspace(0.05, 1, 20).tolist(), RandomState(5), False),
    ],
    ids=["no_n_samples", "y_shape_n_outputs", "with_n_samples", "given_random_state_int", "given_random_state"],
)
def test_plot_metric_growth_per_labeled_instances(
    iris_data, classifiers, n_samples, quantiles, random_state, use_dummies
):
    """Test plotting metric growth with various samples, quantiles, random states."""
    if use_dummies:
        y_train = pd.get_dummies(pd.DataFrame(iris_data["y_train"]).astype(str))
        y_test = pd.get_dummies(pd.DataFrame(iris_data["y_test"]).astype(str))
    else:
        y_train, y_test = iris_data["y_train"], iris_data["y_test"]

    ax = plot_metric_growth_per_labeled_instances(
        iris_data["x_train"],
        y_train,
        iris_data["x_test"],
        y_test,
        classifiers,
        n_samples=n_samples,
        quantiles=quantiles,
        random_state=random_state,
    )

    # Assert that the number of lines in the plot matches the number of classifiers
    assert len(ax.lines) == len(classifiers)

    # Assert that the x-axis label is correct
    assert ax.get_xlabel() == "Number of training samples"

    # Assert that the y-axis label is correct
    assert ax.get_ylabel() == "Metric score"

    return plt.gcf()


def test_plot_metric_growth_per_labeled_instances_no_n_samples_no_quantiles(iris_data, classifiers):
    """Test ValueError if both n_samples and quantiles are None."""
    with pytest.raises(ValueError, match="n_samples must be specified if quantiles is None"):
        plot_metric_growth_per_labeled_instances(
            iris_data["x_train"],
            iris_data["y_train"],
            iris_data["x_test"],
            iris_data["y_test"],
            classifiers,
            n_samples=None,
            quantiles=None,
        )


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR)
def test_plot_metric_growth_per_labeled_instances_exists_ax(iris_data, classifiers):
    """Test plotting metric growth on an existing Axes object."""
    fig, ax = plt.subplots()
    ax.set_title("My ax")
    plot_metric_growth_per_labeled_instances(
        iris_data["x_train"],
        iris_data["y_train"],
        iris_data["x_test"],
        iris_data["y_test"],
        classifiers,
        ax=ax,
        random_state=42,
    )

    assert ax.get_title() == "My ax"

    return fig


def test_plot_metric_growth_per_labeled_instances_verbose(iris_data, classifiers, capsys):
    """Test verbose output of plot_metric_growth_per_labeled_instances."""
    plot_metric_growth_per_labeled_instances(
        iris_data["x_train"], iris_data["y_train"], iris_data["x_test"], iris_data["y_test"], classifiers, verbose=1
    )
    captured = capsys.readouterr().out
    expected = (
        "Fitting classifier DecisionTreeClassifier for 20 times\nFitting classifier RandomForestClassifier"
        " for 20 times\n"
    )
    assert captured == expected
