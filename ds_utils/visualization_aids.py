import os
from io import BytesIO, StringIO
from typing import Optional, List, Callable, Dict, Union

import numpy
import pandas
import pydotplus
import seaborn
import sklearn.tree
from joblib import Parallel, delayed
from matplotlib import axes, pyplot, image
from numpy.random.mtrand import RandomState
from sklearn import clone
from sklearn.tree import _tree as sklearn_tree
from sklearn.tree import export_graphviz
from sklearn.utils import shuffle


def draw_tree(tree: sklearn.tree.tree.BaseDecisionTree, feature_names: Optional[List[str]] = None,
              class_names: Optional[List[str]] = None, *, ax: Optional[axes.Axes] = None,
              **kwargs) -> axes.Axes:
    """
    Receives a decision tree and return a plot graph of the tree for easy interpretation.

    :param tree: decision tree.
    :param feature_names: the features names.
    :param class_names: the classes names or labels.
    :param ax: Axes object to draw the plot onto, otherwise uses the current Axes.
    :return: Returns the Axes object with the plot drawn onto it.
    """
    if ax is None:
        pyplot.figure()
        ax = pyplot.gca()
    sio = BytesIO()
    graph = pydotplus.graph_from_dot_data(
        export_graphviz(tree, feature_names=feature_names, out_file=None, filled=True, rounded=True,
                        special_characters=True, class_names=class_names))
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


def print_decision_paths(classifier: sklearn.tree.tree.BaseDecisionTree, feature_names: Optional[List[str]] = None,
                         class_names: Optional[List[str]] = None, tree_name: Optional[str] = None,
                         indent_char: str = "\t") -> str:
    """
    Receives a decision tree and return the underlying decision-rules (or 'decision paths') as text (valid python
    syntax). Original code: https://stackoverflow.com/questions/20224526/how-to-extract-the-decision-rules-from-scikit-learn-decision-tree

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


def _perform_data_partition_and_evaluation(X_train, y_train, X_test, y_test, classifier, metric):
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    return metric(y_test, y_pred)


def plot_metric_growth_per_labeled_instances(X_train: numpy.ndarray, y_train: numpy.ndarray, X_test: numpy.ndarray,
                                             y_test: numpy.ndarray,
                                             classifiers_dict: Dict[str, sklearn.base.ClassifierMixin],
                                             n_samples: Optional[List[int]] = None,
                                             quantiles: Optional[List[float]] = numpy.linspace(0.05, 1, 20).tolist(),
                                             metric: Callable[[numpy.ndarray,
                                                               numpy.ndarray], float] = sklearn.metrics.accuracy_score,
                                             random_state: Optional[Union[int, RandomState]] = None,
                                             n_jobs: Optional[int] = None, verbose: int = 0,
                                             pre_dispatch: Optional[Union[int, str]] = "2*n_jobs", *,
                                             ax: Optional[axes.Axes] = None, **kwargs) -> axes.Axes:
    """
    Receives a train and test sets, and plots given metric change in increasing amount of trained instances.

    :param X_train: {array-like or sparse matrix} of shape (n_samples, n_features)
                    The training input samples.

    :param y_train: 1d array-like, or label indicator array / sparse matrix
                    The target values (class labels) as integers or strings.

    :param X_test: {array-like or sparse matrix} of shape (n_samples, n_features)
                    The test or evaluation input samples.

    :param y_test: 1d array-like, or label indicator array / sparse matrix
                    Predicted labels, as returned by a classifier.

    :param classifiers_dict: mapping from classifier name to classifier object.

    :param n_samples: List of numbers of samples for training batches, optional (default=None).

    :param quantiles: List of percentages of samples for training batches, optional (default=[0.05, 0.1, 0.15, 0.2,
                        0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1].
                        Used when n_samples=None.

    :param metric: sklearn.metrics api function which receives y_true and y_pred and returns float.

    :param random_state: int, RandomState instance or None, optional (default=None)
        The seed of the pseudo random number generator to use when shuffling
        the data.  If int, random_state is the seed used by the random number
        generator; If RandomState instance, random_state is the random number
        generator; If None, the random number generator is the RandomState
        instance initiated with seed zero.

    :param n_jobs: int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    :param verbose: integer. Controls the verbosity: the higher, the more messages.

    :param pre_dispatch:  int, or string, optional
        Controls the number of jobs that get dispatched during parallel
        execution. Reducing this number can be useful to avoid an
        explosion of memory consumption when more jobs get dispatched
        than CPUs can process. This parameter can be:
            - None, in which case all the jobs are immediately
              created and spawned. Use this for lightweight and
              fast-running jobs, to avoid delays due to on-demand
              spawning of the jobs
            - An int, giving the exact number of total jobs that are
              spawned
            - A string, giving an expression as a function of n_jobs,
              as in '2*n_jobs'

    :param ax: Axes object to draw the plot onto, otherwise uses the current Axes.

    :return: Returns the Axes object with the plot drawn onto it.
    """
    if ax is None:
        pyplot.figure()
        ax = pyplot.gca()

    if random_state is None:
        random_state = RandomState(seed=0)
    elif not isinstance(random_state, RandomState):
        random_state = RandomState(seed=random_state)

    if n_samples is None:
        if quantiles is not None:
            n_samples = [int(quantile * X_train.shape[0]) for quantile in quantiles]
        else:
            raise ValueError("n_samples must be specified if quantiles is None")

    parallel = Parallel(n_jobs=n_jobs, verbose=verbose, pre_dispatch=pre_dispatch)

    classifiers_dict_results = dict()
    samples_list = [shuffle(X_train, y_train, random_state=random_state, n_samples=n_sample) for n_sample in n_samples]

    with parallel:
        for classifier_name, classifier in classifiers_dict.items():
            if verbose > 0:
                print(f"Fitting classifier {classifier_name} for {len(n_samples)} times")
            scores = parallel(delayed(_perform_data_partition_and_evaluation)(x_train_part, y_train_part,
                                                                              X_test, y_test, clone(classifier),
                                                                              metric) for
                              x_train_part, y_train_part in samples_list)
            classifiers_dict_results.update({classifier_name: scores})

    for classifier_name, scores in classifiers_dict_results.items():
        ax.plot(n_samples, scores, label=classifier_name, **kwargs)

    ax.legend(loc="lower right", **kwargs)

    return ax
