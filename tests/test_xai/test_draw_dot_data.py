"""Tests for the draw_dot_data function in ds_utils.xai."""

from pathlib import Path

from matplotlib import pyplot as plt
import pytest

from ds_utils.xai import draw_dot_data

BASELINE_DIR = Path(__file__).parent.parent / "baseline_images" / "test_xai" / "test_draw_dot_data"


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR, tolerance=11)
def test_draw_dot_data():
    """Test drawing a graph from DOT data."""
    dot_data = (
        "digraph D{\n"
        "\tA [shape=diamond]\n"
        "\tB [shape=box]\n"
        "\tC [shape=circle]\n"
        "\n"
        "\tA -> B [style=dashed, color=grey]\n"
        '\tA -> C [color="black:invis:black"]\n'
        "\tA -> D [penwidth=5, arrowhead=none]\n"
        "\n"
        "}"
    )

    draw_dot_data(dot_data)
    return plt.gcf()


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR, tolerance=11)
def test_draw_dot_data_exist_ax():
    """Test drawing a graph from DOT data on an existing Axes object."""
    dot_data = (
        "digraph D{\n"
        "\tA [shape=diamond]\n"
        "\tB [shape=box]\n"
        "\tC [shape=circle]\n"
        "\n"
        "\tA -> B [style=dashed, color=grey]\n"
        '\tA -> C [color="black:invis:black"]\n'
        "\tA -> D [penwidth=5, arrowhead=none]\n"
        "\n"
        "}"
    )

    fig, ax = plt.subplots()
    ax.set_title("My ax")

    draw_dot_data(dot_data, ax=ax)
    assert ax.get_title() == "My ax"
    return fig


def test_draw_dot_data_empty_input():
    """Test draw_dot_data raises ValueError for empty input string."""
    with pytest.raises(ValueError, match="dot_data must not be empty"):
        draw_dot_data("")


def test_draw_dot_data_invalid_input(mocker):
    """Test draw_dot_data raises ValueError for invalid DOT data."""
    mock_pydotplus = mocker.patch("ds_utils.xai.pydotplus")
    mock_pydotplus.graph_from_dot_data.side_effect = Exception("Invalid dot data")

    with pytest.raises(ValueError, match="Failed to create graph from dot data"):
        draw_dot_data("invalid dot data")
