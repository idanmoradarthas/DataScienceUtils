import os
import warnings
from io import StringIO, BytesIO
from typing import Optional, List

import numpy as np
import pydotplus
from matplotlib import axes, image
from matplotlib import pyplot as plt
from sklearn.tree import BaseDecisionTree
from sklearn.tree import _tree as sklearn_tree, export_graphviz


def generate_decision_paths(classifier: BaseDecisionTree, feature_names: Optional[List[str]] = None,
                            class_names: Optional[List[str]] = None, tree_name: Optional[str] = None,
                            indent_char: str = "\t") -> str:
    """
    Receives a decision tree and return the underlying decision-rules (or 'decision paths') as text (valid python
    syntax). `Original code <https://stackoverflow.com/questions/20224526/how-to-extract-the-decision-rules-from-scikit-learn-decision-tree>`_

    :param classifier: decision tree.
    :param feature_names: the features' names.
    :param class_names: the classes' names or labels.
    :param tree_name: the name of the tree (function signature).
    :param indent_char: the character used for indentation.
    :return: textual representation of the decision paths of the tree.
    """
    tree = classifier.tree_
    if feature_names:
        required_features = [feature_names[i] if i != sklearn_tree.TREE_UNDEFINED else "undefined!" for i in
                             tree.feature]
    else:
        required_features = [f"feature_{i}" if i != sklearn_tree.TREE_UNDEFINED else "undefined!" for i in tree.feature]
    if not tree_name:
        tree_name = "tree"
    output = StringIO()
    signature_vars = list()
    [signature_vars.append(feature) for feature in required_features if
     (feature not in signature_vars) and (feature != 'undefined!')]
    output.write(
        f"def {tree_name}({', '.join(signature_vars)}):{os.linesep}")

    _recurse(0, 1, tree, required_features, class_names, output, indent_char)
    ans = output.getvalue()
    output.close()
    return ans


def _recurse(node, depth, tree, feature_name, class_names, output, indent_char):
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
        if class_names:
            class_name = class_names[index]
        else:
            class_name = f"class_{index}"
        output.write(
            f"{indent}# return class {class_name} with probability {prob_array[index]:.4f}{os.linesep}")
        output.write(f"{indent}return (\"{class_name}\", {prob_array[index]:.4f}){os.linesep}")


def draw_tree(tree: BaseDecisionTree, feature_names: Optional[List[str]] = None,
              class_names: Optional[List[str]] = None, *, ax: Optional[axes.Axes] = None,
              **kwargs) -> axes.Axes:
    """
    Receives a decision tree and return a plot graph of the tree for easy interpretation.

    :param tree: decision tree.
    :param feature_names: the features' names.
    :param class_names: the classes' names or labels.
    :param ax: Axes object to draw the plot onto, otherwise uses the current Axes.
    :param kwargs: other keyword arguments

                   All other keyword arguments are passed to ``matplotlib.axes.Axes.pcolormesh()``.
    :return: Returns the Axes object with the plot drawn onto it.
    """
    warnings.warn("This module is deprecated. Use sklearn.tree.plot_tree instead", DeprecationWarning, stacklevel=2)
    return draw_dot_data(export_graphviz(tree, feature_names=feature_names, out_file=None, filled=True, rounded=True,
                                         special_characters=True, class_names=class_names), ax=ax, **kwargs)


def draw_dot_data(dot_data: str, *, ax: Optional[axes.Axes] = None, **kwargs) -> axes.Axes:
    """
    Receives a Graphiz's Dot language string and return a plot graph of the data.

    :param dot_data: Graphiz's Dot language string.
    :param ax: Axes object to draw the plot onto, otherwise uses the current Axes.
    :param kwargs: other keyword arguments

                   All other keyword arguments are passed to ``matplotlib.axes.Axes.pcolormesh()``.
    :return: Returns the Axes object with the plot drawn onto it.
    """
    if ax is None:
        plt.figure()
        ax = plt.gca()
    sio = BytesIO()
    graph = pydotplus.graph_from_dot_data(dot_data)
    sio.write(graph.create_png())
    sio.seek(0)
    img = image.imread(sio, format="png")
    ax.imshow(img, **kwargs)
    ax.set_axis_off()
    return ax


def plot_features_importance(feature_names: List[str], feature_importance: List[float], *,
                             ax: Optional[axes.Axes] = None, **kwargs) -> axes.Axes:
    """
    plot feature importance as a bar chart.

    :param feature_names: strings list of feature names
    :param feature_importance: float list of feature importance
    :param ax: Axes object to draw the plot onto, otherwise uses the current Axes.
    :param kwargs: other keyword arguments

                   All other keyword arguments are passed to ``matplotlib.axes.Axes.pcolormesh()``.
    :return: Returns the Axes object with the plot drawn onto it.
    """
    if ax is None:
        plt.figure()
        ax = plt.gca()

    names = np.array(feature_names)
    importance = np.array(feature_importance)
    non_zero_importance = np.nonzero(importance)

    ax.bar(names[non_zero_importance], importance[non_zero_importance], **kwargs)
    plt.xticks(rotation=90)

    return ax
