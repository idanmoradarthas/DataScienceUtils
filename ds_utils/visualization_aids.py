import os
from io import BytesIO, StringIO
from typing import Optional, List

import numpy
import pandas
import pydotplus
import seaborn
from matplotlib import axes, pyplot, image
from sklearn.tree import _tree as sklearn_tree
from sklearn.tree import export_graphviz

try:
    from sklearn.tree import BaseDecisionTree
except ImportError:
    from sklearn.tree.tree import BaseDecisionTree


def draw_tree(tree: BaseDecisionTree, feature_names: Optional[List[str]] = None,
              class_names: Optional[List[str]] = None, *, ax: Optional[axes.Axes] = None,
              **kwargs) -> axes.Axes:
    """
    Receives a decision tree and return a plot graph of the tree for easy interpretation.

    :param tree: decision tree.
    :param feature_names: the features names.
    :param class_names: the classes names or labels.
    :param ax: Axes object to draw the plot onto, otherwise uses the current Axes.
    :param kwargs: other keyword arguments
                    All other keyword arguments are passed to ``ax.pcolormesh``.
    :return: Returns the Axes object with the plot drawn onto it.
    """
    return draw_dot_data(export_graphviz(tree, feature_names=feature_names, out_file=None, filled=True, rounded=True,
                                         special_characters=True, class_names=class_names), ax=ax, **kwargs)


def draw_dot_data(dot_data: str, *, ax: Optional[axes.Axes] = None, **kwargs) -> axes.Axes:
    """
    Receives a Graphiz's Dot language string and return a plot graph of the data.

    :param dot_data: Graphiz's Dot language string.
    :param ax: Axes object to draw the plot onto, otherwise uses the current Axes.
    :param kwargs: other keyword arguments
                    All other keyword arguments are passed to ``ax.pcolormesh``.
    :return: Returns the Axes object with the plot drawn onto it.
    """
    if ax is None:
        pyplot.figure()
        ax = pyplot.gca()
    sio = BytesIO()
    graph = pydotplus.graph_from_dot_data(dot_data)
    sio.write(graph.create_png())
    sio.seek(0)
    img = image.imread(sio, format="png")
    ax.imshow(img, **kwargs)
    ax.set_axis_off()
    return ax


def visualize_features(frame: pandas.DataFrame, features: Optional[List[str]] = None, num_columns: int = 2,
                       remove_na: bool = False) -> axes.Axes:
    """
    Receives a data frame and visualize the features values on graphs.

    :param frame: the data frame.
    :param features: list of feature to visualize.
    :param num_columns: number of columns in the grid.
    :param remove_na: True to ignore NA values when plotting; False otherwise.
    :return: Returns the Axes object with the plot drawn onto it.
    """
    if not features:
        features = frame.columns

    if len(features) <= num_columns:
        nrows = 1
    else:
        nrows = int(len(features) / num_columns)
        if len(features) % num_columns > 0:
            nrows += 1
    figure, subplots = pyplot.subplots(nrows=nrows, ncols=num_columns)
    subplots = subplots.flatten()

    i = 0

    for feature in features:
        feature_series = frame[feature]
        frame_reduced = frame
        if remove_na:
            feature_series = feature_series.dropna()
            frame_reduced = frame.dropna(subset=[feature])
        if str(feature_series.dtype).startswith("float"):
            plot = seaborn.distplot(feature_series, ax=subplots[i])
        elif str(feature_series.dtype).startswith("datetime"):
            plot = frame_reduced.groupby(feature).size().plot(ax=subplots[i])
        else:
            plot = seaborn.countplot(feature_series, ax=subplots[i])
        plot.set_title(f"{feature} ({feature_series.dtype})")
        plot.set_xlabel("")

        pyplot.setp(subplots[i].get_xticklabels(), rotation=45)
        i += 1

    if i < len(subplots):
        for j in range(i, len(subplots)):
            figure.delaxes(subplots[j])
    pyplot.subplots_adjust(hspace=0.5)
    return subplots


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
        index = int(numpy.argmax(values))
        prob_array = values / numpy.sum(values)
        if numpy.max(prob_array) >= 1:
            prob_array = values / (numpy.sum(values) + 1)
        if class_names:
            class_name = class_names[index]
        else:
            class_name = f"class_{index}"
        output.write(
            f"{indent}# return class {class_name} with probability {prob_array[index]:.4f}{os.linesep}")
        output.write(f"{indent}return (\"{class_name}\", {prob_array[index]:.4f}){os.linesep}")


def generate_decision_paths(classifier: BaseDecisionTree, feature_names: Optional[List[str]] = None,
                            class_names: Optional[List[str]] = None, tree_name: Optional[str] = None,
                            indent_char: str = "\t") -> str:
    """
    Receives a decision tree and return the underlying decision-rules (or 'decision paths') as text (valid python
    syntax). `Original code <https://stackoverflow.com/questions/20224526/how-to-extract-the-decision-rules-from-scikit-learn-decision-tree>`_

    :param classifier: decision tree.
    :param feature_names: the features names.
    :param class_names: the classes names or labels.
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
    for feature in required_features:
        if (feature not in signature_vars) and (feature != 'undefined!'):
            signature_vars.append(feature)
    output.write(
        f"def {tree_name}({', '.join(signature_vars)}):{os.linesep}")

    _recurse(0, 1, tree, required_features, class_names, output, indent_char)
    ans = output.getvalue()
    output.close()
    return ans
