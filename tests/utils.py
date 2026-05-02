"""Testing utility functions."""

from pathlib import Path

from matplotlib import pyplot as plt
from plotly import graph_objects as go


def save_plotly_figure_and_return_matplot(fig: go.Figure, path_to_save: Path) -> plt.Figure:
    """Save plotly figure and convert to a matplotlib figure for comparison."""
    fig.write_image(str(path_to_save))
    img = plt.imread(path_to_save)
    figure, ax = plt.subplots()
    ax.imshow(img)
    ax.axis("off")
    return figure
