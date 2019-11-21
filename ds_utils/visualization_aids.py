from io import BytesIO
from typing import Optional, List

import pandas
import pydotplus
import seaborn
import sklearn.tree.tree
from matplotlib import pyplot, image
from sklearn.tree import export_graphviz


def draw_tree(tree: sklearn.tree.tree.BaseDecisionTree, features_names: Optional[List[str]],
              class_names: Optional[List[str]]) -> pyplot.Figure:
    """
    Receives a decision tree and return a plot graph of the tree for easy interpretation.

    :param tree: decision tree.
    :param features_names: the features names.
    :param  class_names: the classes names or labels.
    :return: matplotlib Figure.
    """
    figure = pyplot.figure()
    sio = BytesIO()
    graph = pydotplus.graph_from_dot_data(
        export_graphviz(tree, feature_names=features_names, out_file=None, filled=True, rounded=True,
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
