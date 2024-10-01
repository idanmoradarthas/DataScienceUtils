############
Unsupervised
############

The unsupervised module contains methods for calculating and visualizing evaluation performance of unsupervised
models. These tools are primarily inspired by the "Interpret Results of Cluster" section in Google's Machine Learning
Crash Course. For more information, see `here <https://developers.google.com/machine-learning/clustering/interpret>`_.

************************
Plot Cluster Cardinality
************************

The plot_cluster_cardinality function visualizes the number of points in each cluster, which can help identify imbalanced clusters or outliers.

.. autofunction:: unsupervised::plot_cluster_cardinality

.. highlight:: python

In the following example, we'll use the iris dataset from scikit-learn and create a simple K-Means algorithm with k=8 to plot how many points go to each cluster::

    from sklearn import datasets
    from matplotlib import pyplot as plt
    from sklearn.cluster import KMeans
    from ds_utils.unsupervised import plot_cluster_cardinality

    # Load the iris dataset
    iris = datasets.load_iris()
    X = iris.data

    # Create and fit the K-Means model
    estimator = KMeans(n_clusters=8, random_state=42)
    estimator.fit(X)

    # Plot the cluster cardinality
    plot_cluster_cardinality(estimator.labels_)
    plt.show()

And the following image will be shown:

.. image:: ../../tests/baseline_images/test_unsupervised/test_cluster_cardinality.png
    :align: center
    :alt: Cluster Cardinality

**********************
Plot Cluster Magnitude
**********************

The plot_cluster_magnitude function visualizes the total point-to-centroid distance for each cluster, which can help
identify compact or dispersed clusters.

.. autofunction:: unsupervised::plot_cluster_magnitude

Here's an example of how to use the plot_cluster_magnitude function::

    from matplotlib import pyplot as plt
    from sklearn.cluster import KMeans
    from scipy.spatial.distance import euclidean
    from ds_utils.unsupervised import plot_cluster_magnitude

    # Create and fit the K-Means model
    estimator = KMeans(n_clusters=8, random_state=42)
    estimator.fit(X)

    #Plot the cluster magnitude
    plot_cluster_magnitude(X, estimator.labels_, estimator.cluster_centers_, euclidean)
    plt.show()

And the following image will be shown:

.. image:: ../../tests/baseline_images/test_unsupervised/test_plot_cluster_magnitude.png
    :align: center
    :alt: Plot Cluster Magnitude

*************************
Magnitude vs. Cardinality
*************************

The plot_magnitude_vs_cardinality function creates a scatter plot of cluster magnitude against cardinality, which can help identify anomalous clusters.

.. autofunction:: unsupervised::plot_magnitude_vs_cardinality

Here's how to use the plot_magnitude_vs_cardinality function::

    from matplotlib import pyplot as plt
    from sklearn.cluster import KMeans
    from scipy.spatial.distance import euclidean
    from ds_utils.unsupervised import plot_magnitude_vs_cardinality

    # Create and fit the K-Means model
    estimator = KMeans(n_clusters=8, random_state=42)
    estimator.fit(X)

    # Plot magnitude vs. cardinality
    plot_magnitude_vs_cardinality(X, estimator.labels_, estimator.cluster_centers_, euclidean)
    plt.show()

And the following image will be shown:

.. image:: ../../tests/baseline_images/test_unsupervised/test_plot_magnitude_vs_cardinality.png
    :align: center
    :alt: Magnitude vs. Cardinality

**************************
Optimum Number of Clusters
**************************
The plot_loss_vs_cluster_number function helps determine the optimal number of clusters by plotting the total magnitude (sum of distances) as loss against the number of clusters.

.. autofunction:: unsupervised::plot_loss_vs_cluster_number

Here's an example of how to use the plot_loss_vs_cluster_number function::

    from matplotlib import pyplot as plt
    from scipy.spatial.distance import euclidean
    from ds_utils.unsupervised import plot_loss_vs_cluster_number

    # Plot loss vs. number of clusters
    plot_loss_vs_cluster_number(X, 3, 20, euclidean)
    plt.show()

And the following image will be shown:

.. image:: ../../tests/baseline_images/test_unsupervised/test_plot_loss_vs_cluster_number.png
    :align: center
    :alt: Optimum Number of Clusters
