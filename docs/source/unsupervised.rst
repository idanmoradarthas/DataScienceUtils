############
Unsupervised
############

The module of unsupervised contains methods that calculate and/or visualize evaluation performance of an unsupervised
model.

Mostly inspired by the Interpet Results of Cluster in Google's Machine Learning Crash Course. See more information
`here <https://developers.google.com/machine-learning/clustering/interpret>`_

*******************
Cluster Cardinality
*******************

.. autofunction:: unsupervised::plot_cluster_cardinality

.. highlight:: python

In following examples we are going to use the iris dataset from scikit-learn. so firstly let's import it::

    from sklearn import datasets


    iris = datasets.load_iris()
    x = iris.data

We'll create a simple K-Means algorithm with k=8 and plot how many point goes to each cluster::

    from matplotlib import pyplot
    from sklearn.cluster import KMeans

    from ds_utils.unsupervised import plot_cluster_cardinality


    estimator = KMeans(n_clusters=8)
    estimator.fit(x)

    plot_cluster_cardinality(estimator.labels_)

    pyplot.show()

And the following image will be shown:

.. image:: ../../tests/baseline_images/test_unsupervised/test_cluster_cardinality.png
    :align: center
    :alt: Cluster Cardinality