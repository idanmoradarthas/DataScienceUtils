"""Test the probability analysis metrics visualization."""

from pathlib import Path

from matplotlib import pyplot as plt
import pandas as pd
import pytest

from ds_utils.metrics.probability_analysis import visualize_accuracy_grouped_by_probability

BASELINE_DIR = Path(__file__).parent.parent / "baseline_images" / "test_metrics" / "test_probability_analysis"


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR)
@pytest.mark.parametrize(
    ("display_breakdown", "bins", "threshold"),
    [(False, None, 0.5), (True, None, 0.5), (False, [0, 0.3, 0.5, 0.8, 1], 0.5), (False, None, 0.3)],
    ids=["default", "with_breakdown", "custom_bins", "custom_threshold"],
)
def test_visualize_accuracy_grouped_by_probability(display_breakdown, bins, threshold):
    """Test visualizing accuracy grouped by probability with different parameters."""
    class_with_probabilities = pd.read_csv(Path(__file__).parent.joinpath("resources", "class_with_probabilities.csv"))
    ax = visualize_accuracy_grouped_by_probability(
        class_with_probabilities["loan_condition_cat"],
        1,
        class_with_probabilities["probabilities"],
        display_breakdown=display_breakdown,
        bins=bins,
        threshold=threshold,
    )

    # Assert that the x-axis label is correct
    assert ax.get_xlabel() == "Probability Range"

    # Assert that the y-axis label is correct
    assert ax.get_ylabel() == "Count"

    # Assert that the title is correct
    assert ax.get_title() == "Accuracy Distribution for 1 Class"

    figure = plt.gcf()
    figure.set_size_inches(10, 8)
    return figure


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR)
def test_visualize_accuracy_grouped_by_probability_exists_ax():
    """Test visualizing accuracy grouped by probability on an existing Axes object."""
    fig, ax = plt.subplots()
    ax.set_title("My ax")

    class_with_probabilities = pd.read_csv(Path(__file__).parent.joinpath("resources", "class_with_probabilities.csv"))
    visualize_accuracy_grouped_by_probability(
        class_with_probabilities["loan_condition_cat"], 1, class_with_probabilities["probabilities"], ax=ax
    )

    assert ax.get_title() == "My ax"

    figure = plt.gcf()
    figure.set_size_inches(10, 8)
    return figure
