"""Module for plotting learning curves and analyzing metric growth.

This module provides functions to visualize the performance of machine learning models
as a function of the training dataset size. It includes tools for plotting metric growth
across varying numbers of labeled instances.
"""

from typing import Callable, Dict, List, Optional, Union

from joblib import Parallel, delayed
from matplotlib import axes, pyplot as plt
import numpy as np
from numpy.random import RandomState
from sklearn import clone
from sklearn.base import ClassifierMixin
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle


def plot_metric_growth_per_labeled_instances(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    classifiers_dict: Dict[str, ClassifierMixin],
    n_samples: Optional[List[int]] = None,
    quantiles: Optional[List[float]] = np.linspace(0.05, 1, 20).tolist(),
    metric: Callable[[np.ndarray, np.ndarray], float] = accuracy_score,
    random_state: Optional[Union[int, RandomState]] = None,
    n_jobs: Optional[int] = None,
    verbose: int = 0,
    pre_dispatch: Optional[Union[int, str]] = "2*n_jobs",
    *,
    ax: Optional[axes.Axes] = None,
    **kwargs,
) -> axes.Axes:
    """Plot learning curves showing metric performance vs. training set size.

    Receives train and test sets, and plots the change in the given metric with increasing numbers of trained
    instances.

    :param X_train: array-like or sparse matrix of shape (n_samples, n_features). The training input samples.
    :param y_train: array-like of shape (n_samples,). The target values (class labels) as integers or strings.
    :param X_test: array-like or sparse matrix of shape (n_samples, n_features). The test or evaluation input samples.
    :param y_test: array-like of shape (n_samples,). The true labels for X_test.
    :param classifiers_dict: mapping from classifier name to a classifier object.
    :param n_samples: List of numbers of samples for training batches, optional (default=None).
    :param quantiles: List of sample percentages for training batches, optional (default=[0.05, 0.1, ..., 0.95, 1]).
                      Used when n_samples=None.
    :param metric: sklearn.metrics API function which receives y_true and y_pred and returns float.
    :param random_state: int, RandomState instance or None, optional (default=None).
                         Controls the shuffling applied to the data before applying the split.
    :param n_jobs: int or None, optional (default=None). The number of jobs to run in parallel.
    :param verbose: int, optional (default=0). Controls the verbosity when fitting and predicting.
    :param pre_dispatch: int or string, optional. Controls the number of jobs that get dispatched during parallel
                         execution.
    :param ax: matplotlib Axes object, optional. The axes to plot on.
    :param kwargs: additional keyword arguments to be passed to the plot function.
    :return: The Axes object with the plot drawn onto it.

    :raises ValueError: If both n_samples and quantiles are None.
    """
    if ax is None:
        _, ax = plt.subplots()

    random_state = RandomState(random_state) if not isinstance(random_state, RandomState) else random_state

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
            scores = parallel(
                delayed(_perform_data_partition_and_evaluation)(
                    x_train_part, y_train_part, X_test, y_test, clone(classifier), metric
                )
                for x_train_part, y_train_part in samples_list
            )
            classifiers_dict_results[classifier_name] = scores

    for classifier_name, scores in classifiers_dict_results.items():
        ax.plot(n_samples, scores, label=classifier_name, **kwargs)

    ax.legend(loc="lower right", **kwargs)
    ax.set_xlabel("Number of training samples")
    ax.set_ylabel("Metric score")

    return ax


def _perform_data_partition_and_evaluation(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    classifier: ClassifierMixin,
    metric: Callable[[np.ndarray, np.ndarray], float],
) -> float:
    if y_train.shape[1] == 1:
        classifier.fit(X_train, y_train.ravel())
    else:
        classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    return metric(y_test, y_pred)
