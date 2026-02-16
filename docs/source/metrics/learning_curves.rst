
****************************************
Plot Metric Growth per Labeled Instances
****************************************

.. autofunction:: ds_utils.metrics.learning_curves.plot_metric_growth_per_labeled_instances

Code Example
============
In this example, we'll divide the data into train and test sets, decide on which classifiers we want to measure, and plot
the results::

    from matplotlib import pyplot as plt
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier

    from ds_utils.metrics.learning_curves import plot_metric_growth_per_labeled_instances

    # Load and prepare the data
    features = IRIS.data
    labels = IRIS.target

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=.3, random_state=0)

    # Define classifiers to compare
    classifiers = {
        "DecisionTreeClassifier": DecisionTreeClassifier(random_state=0),
        "RandomForestClassifier": RandomForestClassifier(random_state=0, n_estimators=5)
    }

    # Plot metric growth for different amounts of training data
    plot_metric_growth_per_labeled_instances(X_train, y_train, X_test, y_test, classifiers)
    plt.show()

And the following image will be shown:

.. image:: ../../../tests/baseline_images/test_metrics/test_learning_curves/test_plot_metric_growth_per_labeled_instances_with_n_samples.png
    :align: center
    :alt: Plot of Metric Growth per Labeled Instances
