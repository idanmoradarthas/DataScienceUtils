###
XAI
###
The module of xai contains methods that help explain a model decisions.

***********************
Generate Decision Paths
***********************

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

