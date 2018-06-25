from typing import Optional, List

import pydotplus
import sklearn
from IPython.display import Image
from sklearn.externals.six import StringIO
from sklearn.tree import export_graphviz


def draw_tree(tree: sklearn.tree.tree.BaseDecisionTree, features_names: Optional[List[str]]) -> Image:
    dot_data = StringIO()
    export_graphviz(tree, feature_names=features_names, out_file=dot_data, filled=True, rounded=True,
                    special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    return Image(graph.create_png())
