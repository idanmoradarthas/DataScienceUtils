import os
import warnings
from io import StringIO, BytesIO
from typing import Optional, List, Union

import numpy as np
import pydotplus
from matplotlib import pyplot as plt, axes, image
from sklearn.tree import BaseDecisionTree
from sklearn.tree import _tree as sklearn_tree, export_graphviz


def _recurse(
        node: int,
        depth: int,
        tree: sklearn_tree.Tree,
        feature_name: List[str],
        class_names: Optional[List[str]],
        output: StringIO,
        indent_char: str
):
    indent = indent_char * depth
    if tree.feature[node] != sklearn_tree.TREE_UNDEFINED:
        name = feature_name[node]
        threshold = tree.threshold[node]
        output.write(f"{indent}if {name} <= {threshold:.4f}:{os.linesep}")
        _recurse(tree.children_left[node], depth + 1, tree, feature_name, class_names, output, indent_char)
        output.write(f"{indent}else:  # if {name} > {threshold:.4f}{os.linesep}")
        _recurse(tree.children_right[node], depth + 1, tree, feature_name, class_names, output, indent_char)
    else:
        values = tree.value[node][0]
        index = int(np.argmax(values))
        prob_array = values / np.sum(values)
        if np.max(prob_array) >= 1:
            prob_array = values / (np.sum(values) + 1)
        class_name = class_names[index] if class_names else f"class_{index}"
        output.write(
            f"{indent}# return class {class_name} with probability {prob_array[index]:.4f}{os.linesep}")
        output.write(f"{indent}return (\"{class_name}\", {prob_array[index]:.4f}){os.linesep}")


def generate_decision_paths(
        classifier: BaseDecisionTree,
        feature_names: Optional[List[str]] = None,
        class_names: Optional[List[str]] = None,
        tree_name: Optional[str] = None,
        indent_char: str = "\t"
) -> str:
    """
    Generate decision rules (or 'decision paths') as text (valid Python syntax) from a decision tree.
    `Original code <https://stackoverflow.com/questions/20224526/how-to-extract-the-decision-rules-from-scikit-learn-decision-tree>`_

    :param classifier: Decision tree classifier
    :param feature_names: List of feature names
    :param class_names: List of class names or labels
    :param tree_name: Name of the tree (function signature)
    :param indent_char: Character used for indentation
    :return: Textual representation of the decision paths of the tree
    """
    warnings.warn("This module is deprecated. Use sklearn.tree.export_text instead", DeprecationWarning, stacklevel=2)

    tree = classifier.tree_
    feature_names = feature_names or [f"feature_{i}" for i in range(classifier.n_features_in_)]
    required_features = [feature_names[i] if i != sklearn_tree.TREE_UNDEFINED else "undefined!" for i in tree.feature]
    tree_name = tree_name or "tree"

    output = StringIO()
    signature_vars = list(dict.fromkeys(f for f in required_features if f != 'undefined!'))
    output.write(f"def {tree_name}({', '.join(signature_vars)}):{os.linesep}")

    _recurse(0, 1, tree, required_features, class_names, output, indent_char)

    result = output.getvalue()
    output.close()
    return result


def draw_tree(
        tree: BaseDecisionTree,
        feature_names: Optional[List[str]] = None,
        class_names: Optional[List[str]] = None,
        *,
        ax: Optional[axes.Axes] = None,
        **kwargs
) -> axes.Axes:
    """
    Plot a graph of the decision tree for easy interpretation.

    :param tree: Decision tree
    :param feature_names: List of feature names
    :param class_names: List of class names or labels
    :param ax: Axes object to draw the plot onto, otherwise uses the current Axes
    :param kwargs: Additional keyword arguments passed to matplotlib.axes.Axes.imshow()
    :return: Axes object with the plot drawn onto it
    """
    warnings.warn("This module is deprecated. Use sklearn.tree.plot_tree instead", DeprecationWarning, stacklevel=2)
    dot_data = export_graphviz(tree, feature_names=feature_names, out_file=None, filled=True, rounded=True,
                               special_characters=True, class_names=class_names)
    return draw_dot_data(dot_data, ax=ax, **kwargs)


def draw_dot_data(
        dot_data: str,
        *,
        ax: Optional[axes.Axes] = None,
        **kwargs
) -> axes.Axes:
    """
    Plot a graph from Graphviz's Dot language string.

    :param dot_data: Graphviz's Dot language string. Use sklearn.tree.export_graphviz to generate the dot data string.
    :param ax: Axes object to draw the plot onto, otherwise uses the current Axes
    :param kwargs: Additional keyword arguments passed to matplotlib.axes.Axes.imshow()
    :return: Axes object with the plot drawn onto it
    :raises ValueError: If the dot_data is empty or invalid
    """
    if not dot_data:
        raise ValueError("dot_data must not be empty")

    if ax is None:
        _, ax = plt.subplots()

    try:
        sio = BytesIO()
        graph = pydotplus.graph_from_dot_data(dot_data)
        sio.write(graph.create_png())
        sio.seek(0)
        img = image.imread(sio, format="png")
        ax.imshow(img, **kwargs)
        ax.set_axis_off()
    except Exception as e:
        raise ValueError(f"Failed to create graph from dot data: {str(e)}")

    return ax


def plot_features_importance(
        feature_names: Union[np.ndarray, List[str]],
        feature_importance: Union[np.ndarray, List[float]],
        *,
        ax: Optional[axes.Axes] = None,
        **kwargs
) -> axes.Axes:
    """
    Plot feature importance as a bar chart.

    :param feature_names: Numpy array or list of feature names of shape (n_features,)
    :param feature_importance:  Numpy array or list of feature importance values of shape (n_features,)
    :param ax: Axes object to draw the plot onto, otherwise uses the current Axes
    :param kwargs: Additional keyword arguments passed to matplotlib.axes.Axes.bar()
    :return: Axes object with the plot drawn onto it
    :raises ValueError: If feature_names and feature_importance have different lengths or invalid input
    """
    # Convert inputs to numpy arrays if they aren't already
    names = np.asarray(feature_names)
    importance = np.asarray(feature_importance)

    # Validate input
    if names.ndim != 1 or importance.ndim != 1:
        raise ValueError("feature_names and feature_importance must be 1-dimensional")

    if len(feature_names) != len(feature_importance):
        raise ValueError("feature_names and feature_importance must have the same length")

    if ax is None:
        _, ax = plt.subplots()

    # Filter out zero importance features
    mask = importance > 0
    filtered_names = names[mask]
    filtered_importance = importance[mask]

    # Sort features by importance in descending order
    sorted_indices = np.argsort(filtered_importance)[::-1]
    sorted_names = filtered_names[sorted_indices]
    sorted_importance = filtered_importance[sorted_indices]

    ax.bar(sorted_names, sorted_importance, **kwargs)
    ax.set_ylabel('Importance')
    if not ax.get_title():
        ax.set_title('Feature Importance')
    plt.xticks(rotation=90)

    return ax
