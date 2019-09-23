from io import StringIO
from typing import Optional, List

import pandas
import pydotplus
import seaborn
import sklearn
from IPython.display import Image
from matplotlib import pyplot
from sklearn.tree import export_graphviz


def draw_tree(tree: sklearn.tree.tree.BaseDecisionTree, features_names: Optional[List[str]]) -> Image:
    """
    This method using graphviz draw a given tree.
    :param tree: decision tree.
    :param features_names: the features names.
    :return: Ipython image of the built tree.
    """
    dot_data = StringIO()
    export_graphviz(tree, feature_names=features_names, out_file=dot_data, filled=True, rounded=True,
                    special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    return Image(graph.create_png())


def visualize_features(frame: pandas.DataFrame) -> None:
    """
    Visualize features values on graphs.
    :param frame: the data frame.
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
    pyplot.show()
