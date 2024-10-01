####################
XAI (Explainable AI)
####################
The module of xai contains methods that help explain a model decisions.

In order for this module to work properly, Graphiz must be installed. In Linux based operating systems use::

    sudo apt-get install graphviz

In Windows based operating systems use::

    choco install graphviz

In macOS operating systems use::

    brew install graphviz

Or using conda::

    conda install graphviz

For more information see `here <https://graphviz.gitlab.io/download>`_.

*********
Draw Tree
*********
.. deprecated:: 1.6.4
    Use `sklearn.tree.plot_tree <https://scikit-learn.org/stable/modules/generated/sklearn.tree.plot_tree.html>`_ instead

.. autofunction:: xai::draw_tree

.. highlight:: python

Code Example
============
In following examples we are going to use the iris dataset from scikit-learn. so firstly let's import it::

    from sklearn import datasets


    iris = datasets.load_iris()
    x = iris.data
    y = iris.target

We'll create a simple decision tree classifier and plot it::

    from matplotlib import pyplot as plt
    from sklearn.tree import DecisionTreeClassifier

    from ds_utils.xai import draw_tree


    # Create decision tree classifier object
    clf = DecisionTreeClassifier(random_state=0)

    # Train model
    clf.fit(x, y)

    draw_tree(clf, iris.feature_names, iris.target_names)
    plt.show()

And the following image will be shown:

.. image:: ../../tests/baseline_images/test_xai/test_draw_tree.png
    :align: center
    :alt: Decision Tree Visualization

*************
Draw Dot Data
*************

.. autofunction:: xai::draw_dot_data

Code Example
============
We'll create a simple diagram and plot it::

    from matplotlib import pyplot as plt

    from ds_utils.xai import draw_dot_data


    dot_data = "digraph D{\n" \
               "\tA [shape=diamond]\n" \
               "\tB [shape=box]\n" \
               "\tC [shape=circle]\n" \
               "\n" \
               "\tA -> B [style=dashed, color=grey]\n" \
               "\tA -> C [color=\"black:invis:black\"]\n" \
               "\tA -> D [penwidth=5, arrowhead=none]\n" \
               "\n" \
               "}"

    draw_dot_data(dot_data)
    plt.show()

And the following image will be shown:

.. image:: ../../tests/baseline_images/test_xai/test_draw_dot_data.png
    :align: center
    :alt: Diagram Visualization


***********************
Generate Decision Paths
***********************
.. deprecated:: 1.7.4
    Use `sklearn.tree.export_text <https://scikit-learn.org/stable/modules/generated/sklearn.tree.export_text.html>`_ instead

.. autofunction:: xai::generate_decision_paths

Code Example
============
We'll create a simple decision tree classifier and print it::

    from sklearn.tree import DecisionTreeClassifier

    from ds_utils.xai import generate_decision_paths


    # Create decision tree classifier object
    clf = DecisionTreeClassifier(random_state=0, max_depth=3)

    # Train model
    clf.fit(x, y)
    print(generate_decision_paths(clf, iris.feature_names, iris.target_names.tolist(),
                         "iris_tree"))

.. highlight:: none

The following text will be printed::

    def iris_tree(petal width (cm), petal length (cm)):
        if petal width (cm) <= 0.8000:
            # return class setosa with probability 0.9804
            return ("setosa", 0.9804)
        else:  # if petal width (cm) > 0.8000
            if petal width (cm) <= 1.7500:
                if petal length (cm) <= 4.9500:
                    # return class versicolor with probability 0.9792
                    return ("versicolor", 0.9792)
                else:  # if petal length (cm) > 4.9500
                    # return class virginica with probability 0.6667
                    return ("virginica", 0.6667)
            else:  # if petal width (cm) > 1.7500
                if petal length (cm) <= 4.8500:
                    # return class virginica with probability 0.6667
                    return ("virginica", 0.6667)
                else:  # if petal length (cm) > 4.8500
                    # return class virginica with probability 0.9773
                    return ("virginica", 0.9773)


*************************
Plot Features` Importance
*************************

.. autofunction:: xai::plot_features_importance

Code Example
============

.. highlight:: python

For this example I created a dummy data set. You can find the data at the resources directory in the packages tests folder.

Let's see how to use the code::

    import pandas as pd

    from matplotlib import pyplot as plt
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.tree import DecisionTreeClassifier

    from ds_utils.xai import plot_features_importance


    data_1M = pd.read_csv(path/to/dataset)
    target = data_1M["x12"]
    categorical_features = ["x7", "x10"]
    for i in range(0, len(categorical_features)):
        enc = OneHotEncoder(sparse=False, handle_unknown="ignore")
        enc_out = enc.fit_transform(data_1M[[categorical_features[i]]])
        for j in range(0, len(enc.categories_[0])):
            data_1M[categorical_features[i] + "_" + enc.categories_[0][j]] = enc_out[:, j]
    features = data_1M.columns.to_list()
    features.remove("x12")
    features.remove("x7")
    features.remove("x10")

    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(data_1M[features], target)
    plot_features_importance(features, clf.feature_importances_)

    plt.show()

And the following image will be shown:

.. image:: ../../tests/baseline_images/test_xai/test_plot_features_importance.png
    :align: center
    :alt: Plot Features Importance