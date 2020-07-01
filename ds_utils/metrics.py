from typing import Union, List, Optional, Callable, Dict

import numpy
import pandas
import seaborn
import sklearn
from joblib import Parallel, delayed
from matplotlib import pyplot, axes
from numpy.random.mtrand import RandomState
from sklearn import clone
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.utils import shuffle


def plot_confusion_matrix(y_test: numpy.ndarray, y_pred: numpy.ndarray, labels: List[Union[str, int]],
                          sample_weight: Optional[List[float]] = None, annot_kws=None, cbar=True, cbar_kws=None,
                          **kwargs) -> axes.Axes:
    """
    Computes and plot confusion matrix, False Positive Rate, False Negative Rate, Accuracy and F1 score of a
    classification.

    :param y_test: array, shape = [n_samples]. Ground truth (correct) target values.
    :param y_pred: array, shape = [n_samples]. Estimated targets as returned by a classifier.
    :param labels: array, shape = [n_classes]. List of labels to index the matrix. This may be used to reorder or
                    select a subset of labels.
    :param sample_weight: array-like of shape = [n_samples], optional

                            Sample weights.
    :param annot_kws: dict of key, value mappings, optional

                        Keyword arguments for ``ax.text``.
    :param cbar: boolean, optional

                Whether to draw a colorbar.
    :param cbar_kws: dict of key, value mappings, optional

                    Keyword arguments for ``figure.colorbar``
    :param kwargs: other keyword arguments

                   All other keyword arguments are passed to ``matplotlib.axes.Axes.pcolormesh()``.
    :return: Returns the Axes object with the matrix drawn onto it.
    """
    if len(labels) < 2:
        raise ValueError("Number of labels must be greater than 1")

    cnf_matrix = confusion_matrix(y_test, y_pred, labels, sample_weight)
    if len(labels) == 2:
        tn, fp, fn, tp = cnf_matrix.ravel()
        npv, ppv, tnr, tpr = _calc_precision_recall(fn, fp, tn, tp)

        table = numpy.array([[tn, fp, tnr], [fn, tp, tpr], [npv, ppv, numpy.NaN]], dtype=numpy.float)
        df = pandas.DataFrame(table, columns=[f"{labels[0]} - Predicted", f"{labels[1]} - Predicted", "Recall"],
                              index=[f"{labels[0]} - Actual", f"{labels[1]} - Actual", "Precision"])
    else:
        fp = (cnf_matrix.sum(axis=0) - numpy.diag(cnf_matrix)).astype(float)
        fn = (cnf_matrix.sum(axis=1) - numpy.diag(cnf_matrix)).astype(float)
        tp = (numpy.diag(cnf_matrix)).astype(float)
        tn = (cnf_matrix.sum() - (fp + fn + tp)).astype(float)
        _, ppv, tnr, tpr = _calc_precision_recall(fn, fp, tn, tp)
        df = pandas.DataFrame(cnf_matrix, columns=[f"{label} - Predicted" for label in labels],
                              index=[f"{label} - Actual" for label in labels])
        df["Recall"] = tpr
        df = df.append(
            pandas.DataFrame([ppv], columns=[f"{label} - Predicted" for label in labels], index=["Precision"]),
            sort=False)

    figure, subplots = pyplot.subplots(nrows=3, ncols=1, gridspec_kw={'height_ratios': [1, 8, 1]})
    subplots = subplots.flatten()

    subplots[0].set_axis_off()
    if len(labels) == 2:
        subplots[0].text(0, 0.85, f"False Positive Rate: {1 - tnr:.4f}")
        subplots[0].text(0, 0.35, f"False Negative Rate: {1 - tpr:.4f}")
    else:
        subplots[0].text(0, 0.85, f"False Positive Rate: {numpy.array2string(1 - tnr, precision=2, separator=',')}")
        subplots[0].text(0, 0.35, f"False Negative Rate: {numpy.array2string(1 - tpr, precision=2, separator=',')}")
    subplots[0].text(0, -0.5, "Confusion Matrix:")

    seaborn.heatmap(df, annot=True, fmt=".3f", ax=subplots[1], annot_kws=annot_kws, cbar=cbar, cbar_kws=cbar_kws,
                    **kwargs)

    subplots[2].set_axis_off()
    subplots[2].text(0, 0.15, f"Accuracy: {accuracy_score(y_test, y_pred, sample_weight=sample_weight):.4f}")
    if len(labels) == 2:
        f_score = f1_score(y_test, y_pred, labels, labels[1], "binary", sample_weight)
    else:
        f_score = f1_score(y_test, y_pred, labels, average="micro", sample_weight=sample_weight)
    subplots[2].text(0, -0.5, f"F1 Score: {f_score:.4f}")
    return subplots


def _calc_precision_recall(fn, fp, tn, tp):
    tpr = (tp / (tp + fn))
    tnr = (tn / (tn + fp))
    npv = (tn / (tn + fn))
    ppv = (tp / (tp + fp))
    return npv, ppv, tnr, tpr


def plot_metric_growth_per_labeled_instances(X_train: numpy.ndarray, y_train: numpy.ndarray, X_test: numpy.ndarray,
                                             y_test: numpy.ndarray,
                                             classifiers_dict: Dict[str, sklearn.base.ClassifierMixin],
                                             n_samples: Optional[List[int]] = None,
                                             quantiles: Optional[List[float]] = numpy.linspace(0.05, 1, 20).tolist(),
                                             metric: Callable[[numpy.ndarray,
                                                               numpy.ndarray], float] = accuracy_score,
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

        The seed of the pseudo random number generator to use when shuffling the data.

        * If int, random_state is the seed used by the random number generator;
        * If RandomState instance, random_state is the random number generator;
        * If None, the random number generator is the RandomState instance initiated with seed zero.

    :param n_jobs: int or None, optional (default=None)

        Number of jobs to run in parallel.

        * ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        * ``-1`` means using all processors.

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

    :param kwargs: other keyword arguments
                    All other keyword arguments are passed to ``ax.pcolormesh``.

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


def _perform_data_partition_and_evaluation(X_train, y_train, X_test, y_test, classifier, metric):
    if y_train.shape[1] == 1:
        classifier.fit(X_train, y_train.values.ravel())
    else:
        classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    return metric(y_test, y_pred)
