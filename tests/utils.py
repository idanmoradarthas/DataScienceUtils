from pathlib import Path
from typing import Union, Optional

import matplotlib
import plotly
from matplotlib import pyplot as plt
from matplotlib.testing.compare import compare_images

TOLERANCE = 50


def compare_images_from_paths(first: str, second: str) -> None:
    results = compare_images(first, second, TOLERANCE)
    if results is not None:  # the images compare favorably
        assert False


def save_result_figure(
        figure: Union[matplotlib.figure.Figure, plotly.graph_objects.Figure],
        output_path: Path,
        convert_to_matplotlib_figure: bool = False
) -> Optional[matplotlib.figure.Figure]:
    """
    Saves a given figure (either Matplotlib or Plotly) to the specified file path.

    If the figure is a Plotly figure, it will be saved using Plotly's `write_image` method.
    If the figure is a Matplotlib figure, it will be saved using Matplotlib's `savefig` method.

    Optionally, if `convert_to_matplotlib_figure` is set to True and the input is a Plotly figure,
    the function reads the saved image and returns a Matplotlib figure displaying the image.

    :param figure: The figure to save, either a Matplotlib or Plotly figure.
    :param output_path: Path where the figure will be saved.
    :param convert_to_matplotlib_figure: Flag to convert the saved figure to a Matplotlib figure and return it. 
                                         This is useful for uniform display or further Matplotlib manipulation.
    :return: A Matplotlib figure object if `convert_to_matplotlib_figure` is True, else None.
    """
    if isinstance(figure, plotly.graph_objects.Figure):
        figure.write_image(str(output_path))
    else:
        figure.savefig(str(output_path))

    if convert_to_matplotlib_figure:
        img = plt.imread(output_path)
        matplotlib_figure, ax = plt.subplots()
        ax.imshow(img)
        ax.axis("off")  # Hide the axes for a clean comparison
        return matplotlib_figure
