import os
from io import BytesIO, StringIO
from typing import Optional, List

import numpy
import pandas
import pydotplus
import seaborn
import sklearn.tree
from matplotlib import pyplot, image
from sklearn.tree import _tree as sklearn_tree
from sklearn.tree import export_graphviz


def draw_tree(tree: sklearn.tree.BaseDecisionTree, feature_names: Optional[List[str]] = None,
              class_names: Optional[List[str]] = None) -> pyplot.Figure:
    """
    Receives a decision tree and return a plot graph of the tree for easy interpretation.

    :param tree: decision tree.
    :param feature_names: the features names.
    :param class_names: the classes names or labels.
    :return: matplotlib Figure.
    """
    figure = pyplot.figure()
    sio = BytesIO()
    graph = pydotplus.graph_from_dot_data(
        export_graphviz(tree, feature_names=feature_names, out_file=None, filled=True, rounded=True,
                        special_characters=True, class_names=class_names))
    sio.write(graph.create_png())
    sio.seek(0)
    img = image.imread(sio, format="png")
    pyplot.imshow(img)
    pyplot.gca().set_axis_off()
    return figure


def visualize_features(frame: pandas.DataFrame) -> pyplot.Figure:
    """
    Receives a data frame and visualize the features values on graphs.

    :param frame: the data frame.
    :return: matplotlib Figure.
    """
    features = frame.columns
    figure, axes = pyplot.subplots(nrows=int(len(features) / 2) + 1, ncols=2, figsize=(20, 30))
    axes = axes.flatten()
    i = 0

    for feature in features:
        if frame[feature].dtype == "float64":
            plot = seaborn.distplot(frame[feature], ax=axes[i])
        elif frame[feature].dtype == "datetime64[ns]":
            plot = frame.groupby(feature).size().plot(ax=axes[i])
        else:
            plot = seaborn.countplot(frame[feature], ax=axes[i])
        plot.set_title(f"{feature} ({frame[feature].dtype})")
        plot.set_xlabel("")

        pyplot.setp(axes[i].get_xticklabels(), rotation=45)
        i += 1

    if len(features) % 2 == 1:
        figure.delaxes(axes[i])
    pyplot.subplots_adjust(hspace=0.5)
    return figure


def _recurse(node, depth, tree, feature_name, class_names, output):
    indent = "  " * depth
    if tree.feature[node] != sklearn_tree.TREE_UNDEFINED:
        name = feature_name[node]
        threshold = tree.threshold[node]
        output.write(f"{indent}if {name} <= {threshold:.4f}:{os.linesep}")
        _recurse(tree.children_left[node], depth + 1, tree, feature_name, class_names, output)
        output.write(f"{indent}else:  # if {name} > {threshold:.4f}{os.linesep}")
        _recurse(tree.children_right[node], depth + 1, tree, feature_name, class_names, output)
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


def print_decision_paths(classifier: sklearn.tree.BaseDecisionTree, feature_names: Optional[List[str]] = None,
                         class_names: Optional[List[str]] = None, tree_name: Optional[str] = None) -> str:
    """
    Receives a decision tree and return the underlying decision-rules (or 'decision paths') as text (valid python
    syntax). Original code: https://stackoverflow.com/questions/20224526/how-to-extract-the-decision-rules-from-scikit-learn-decision-tree

    :param classifier: decision tree.
    :param feature_names: the features names.
    :param class_names: the classes names or labels.
    :param tree_name: the name of the tree (function signature)
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

    _recurse(0, 1, tree, required_features, class_names, output)
    ans = output.getvalue()
    output.close()
    return ans
