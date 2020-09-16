import os
from pathlib import Path

import pandas
from matplotlib import pyplot
from sklearn.tree import DecisionTreeClassifier

from ds_utils.xai import generate_decision_paths, draw_tree, draw_dot_data, plot_features_importance
from tests.utils import compare_images_from_paths

iris_x = pandas.read_csv(Path(__file__).parents[0].joinpath("resources").joinpath("iris_x_full.csv"))
iris_y = pandas.read_csv(Path(__file__).parents[0].joinpath("resources").joinpath("iris_y_full.csv"))
data_1M = pandas.read_csv(Path(__file__).parents[0].joinpath("resources").joinpath("data.1M.zip"), compression='zip')

Path(__file__).parents[0].absolute().joinpath("result_images").mkdir(exist_ok=True)
Path(__file__).parents[0].absolute().joinpath("result_images").joinpath("test_xai").mkdir(exist_ok=True)


def test_print_decision_paths():
    # Create decision tree classifier object
    clf = DecisionTreeClassifier(random_state=0, max_depth=3)

    # Train model
    clf.fit(iris_x, iris_y)

    result = generate_decision_paths(clf,
                                     ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'],
                                     ['setosa', 'versicolor', 'virginica'], "iris_tree", "  ")

    expected = 'def iris_tree(petal width (cm), petal length (cm)):' + os.linesep + \
               '  if petal width (cm) <= 0.8000:' + os.linesep + \
               '    # return class setosa with probability 0.9804' + os.linesep + \
               '    return ("setosa", 0.9804)' + os.linesep + \
               '  else:  # if petal width (cm) > 0.8000' + os.linesep + \
               '    if petal width (cm) <= 1.7500:' + os.linesep + \
               '      if petal length (cm) <= 4.9500:' + os.linesep + \
               '        # return class versicolor with probability 0.9792' + os.linesep + \
               '        return ("versicolor", 0.9792)' + os.linesep + \
               '      else:  # if petal length (cm) > 4.9500' + os.linesep + \
               '        # return class virginica with probability 0.6667' + os.linesep + \
               '        return ("virginica", 0.6667)' + os.linesep + \
               '    else:  # if petal width (cm) > 1.7500' + os.linesep + \
               '      if petal length (cm) <= 4.8500:' + os.linesep + \
               '        # return class virginica with probability 0.6667' + os.linesep + \
               '        return ("virginica", 0.6667)' + os.linesep + \
               '      else:  # if petal length (cm) > 4.8500' + os.linesep + \
               '        # return class virginica with probability 0.9773' + os.linesep + \
               '        return ("virginica", 0.9773)' + os.linesep

    assert result == expected


def test_print_decision_paths_no_tree_name():
    # Create decision tree classifier object
    clf = DecisionTreeClassifier(random_state=0, max_depth=3)

    # Train model
    clf.fit(iris_x, iris_y)

    result = generate_decision_paths(clf,
                                     ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'],
                                     ['setosa', 'versicolor', 'virginica'], indent_char="  ")

    expected = 'def tree(petal width (cm), petal length (cm)):' + os.linesep + \
               '  if petal width (cm) <= 0.8000:' + os.linesep + \
               '    # return class setosa with probability 0.9804' + os.linesep + \
               '    return ("setosa", 0.9804)' + os.linesep + \
               '  else:  # if petal width (cm) > 0.8000' + os.linesep + \
               '    if petal width (cm) <= 1.7500:' + os.linesep + \
               '      if petal length (cm) <= 4.9500:' + os.linesep + \
               '        # return class versicolor with probability 0.9792' + os.linesep + \
               '        return ("versicolor", 0.9792)' + os.linesep + \
               '      else:  # if petal length (cm) > 4.9500' + os.linesep + \
               '        # return class virginica with probability 0.6667' + os.linesep + \
               '        return ("virginica", 0.6667)' + os.linesep + \
               '    else:  # if petal width (cm) > 1.7500' + os.linesep + \
               '      if petal length (cm) <= 4.8500:' + os.linesep + \
               '        # return class virginica with probability 0.6667' + os.linesep + \
               '        return ("virginica", 0.6667)' + os.linesep + \
               '      else:  # if petal length (cm) > 4.8500' + os.linesep + \
               '        # return class virginica with probability 0.9773' + os.linesep + \
               '        return ("virginica", 0.9773)' + os.linesep

    assert result == expected


def test_print_decision_paths_no_feature_names():
    # Create decision tree classifier object
    clf = DecisionTreeClassifier(random_state=0, max_depth=3)

    # Train model
    clf.fit(iris_x, iris_y)

    result = generate_decision_paths(clf, None, ['setosa', 'versicolor', 'virginica'], "iris_tree", "  ")

    expected = 'def iris_tree(feature_3, feature_2):' + os.linesep + \
               '  if feature_3 <= 0.8000:' + os.linesep + \
               '    # return class setosa with probability 0.9804' + os.linesep + \
               '    return ("setosa", 0.9804)' + os.linesep + \
               '  else:  # if feature_3 > 0.8000' + os.linesep + \
               '    if feature_3 <= 1.7500:' + os.linesep + \
               '      if feature_2 <= 4.9500:' + os.linesep + \
               '        # return class versicolor with probability 0.9792' + os.linesep + \
               '        return ("versicolor", 0.9792)' + os.linesep + \
               '      else:  # if feature_2 > 4.9500' + os.linesep + \
               '        # return class virginica with probability 0.6667' + os.linesep + \
               '        return ("virginica", 0.6667)' + os.linesep + \
               '    else:  # if feature_3 > 1.7500' + os.linesep + \
               '      if feature_2 <= 4.8500:' + os.linesep + \
               '        # return class virginica with probability 0.6667' + os.linesep + \
               '        return ("virginica", 0.6667)' + os.linesep + \
               '      else:  # if feature_2 > 4.8500' + os.linesep + \
               '        # return class virginica with probability 0.9773' + os.linesep + \
               '        return ("virginica", 0.9773)' + os.linesep

    assert result == expected


def test_print_decision_paths_no_class_names():
    # Create decision tree classifier object
    clf = DecisionTreeClassifier(random_state=0, max_depth=3)

    # Train model
    clf.fit(iris_x, iris_y)

    result = generate_decision_paths(clf,
                                     ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'],
                                     None, "iris_tree", "  ")

    expected = 'def iris_tree(petal width (cm), petal length (cm)):' + os.linesep + \
               '  if petal width (cm) <= 0.8000:' + os.linesep + \
               '    # return class class_0 with probability 0.9804' + os.linesep + \
               '    return ("class_0", 0.9804)' + os.linesep + \
               '  else:  # if petal width (cm) > 0.8000' + os.linesep + \
               '    if petal width (cm) <= 1.7500:' + os.linesep + \
               '      if petal length (cm) <= 4.9500:' + os.linesep + \
               '        # return class class_1 with probability 0.9792' + os.linesep + \
               '        return ("class_1", 0.9792)' + os.linesep + \
               '      else:  # if petal length (cm) > 4.9500' + os.linesep + \
               '        # return class class_2 with probability 0.6667' + os.linesep + \
               '        return ("class_2", 0.6667)' + os.linesep + \
               '    else:  # if petal width (cm) > 1.7500' + os.linesep + \
               '      if petal length (cm) <= 4.8500:' + os.linesep + \
               '        # return class class_2 with probability 0.6667' + os.linesep + \
               '        return ("class_2", 0.6667)' + os.linesep + \
               '      else:  # if petal length (cm) > 4.8500' + os.linesep + \
               '        # return class class_2 with probability 0.9773' + os.linesep + \
               '        return ("class_2", 0.9773)' + os.linesep

    assert result == expected


def test_draw_tree():
    # Create decision tree classifier object
    clf = DecisionTreeClassifier(random_state=0)

    # Train model
    clf.fit(iris_x, iris_y)

    draw_tree(clf, ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'],
              ['setosa', 'versicolor', 'virginica'])
    result_path = Path(__file__).parents[0].absolute().joinpath("result_images").joinpath(
        "test_visualization_aids").joinpath("test_draw_tree.png")
    pyplot.savefig(str(result_path))

    baseline_path = Path(__file__).parents[0].absolute().joinpath("baseline_images").joinpath(
        "test_visualization_aids").joinpath("test_draw_tree.png")
    pyplot.cla()
    pyplot.close(pyplot.gcf())
    compare_images_from_paths(str(baseline_path), str(result_path))


def test_draw_tree_exists_ax():
    # Create decision tree classifier object
    clf = DecisionTreeClassifier(random_state=0)

    # Train model
    clf.fit(iris_x, iris_y)

    pyplot.figure()
    ax = pyplot.gca()

    ax.set_title("My ax")

    draw_tree(clf, ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'],
              ['setosa', 'versicolor', 'virginica'], ax=ax)

    result_path = Path(__file__).parents[0].absolute().joinpath("result_images").joinpath(
        "test_visualization_aids").joinpath("test_draw_tree_exists_ax.png")
    pyplot.savefig(str(result_path))

    baseline_path = Path(__file__).parents[0].absolute().joinpath("baseline_images").joinpath(
        "test_visualization_aids").joinpath("test_draw_tree_exists_ax.png")
    pyplot.cla()
    pyplot.close(pyplot.gcf())
    compare_images_from_paths(str(baseline_path), str(result_path))


def test_draw_dot_data():
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

    result_path = Path(__file__).parents[0].absolute().joinpath("result_images").joinpath(
        "test_visualization_aids").joinpath("test_draw_dot_data.png")
    pyplot.savefig(str(result_path))

    baseline_path = Path(__file__).parents[0].absolute().joinpath("baseline_images").joinpath(
        "test_visualization_aids").joinpath("test_draw_dot_data.png")
    pyplot.cla()
    pyplot.close(pyplot.gcf())
    compare_images_from_paths(str(baseline_path), str(result_path))


def test_draw_dot_data_exist_ax():
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

    pyplot.figure()
    ax = pyplot.gca()

    ax.set_title("My ax")

    draw_dot_data(dot_data, ax=ax)

    result_path = Path(__file__).parents[0].absolute().joinpath("result_images").joinpath(
        "test_visualization_aids").joinpath("test_draw_dot_data_exist_ax.png")
    pyplot.savefig(str(result_path))

    baseline_path = Path(__file__).parents[0].absolute().joinpath("baseline_images").joinpath(
        "test_visualization_aids").joinpath("test_draw_dot_data_exist_ax.png")
    pyplot.cla()
    pyplot.close(pyplot.gcf())
    compare_images_from_paths(str(baseline_path), str(result_path))


def test_plot_features_importance():
    importance = [0.047304175084187376, 0.011129476233187116, 0.01289095487553893, 0.015528563988219685,
                  0.010904371085026893, 0.03088871541039015, 0.03642466650007851, 0.005758395681352302,
                  0.3270373201417306, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3.2562763221434323e-08, 0, 0,
                  1.3846860504742983e-05, 0, 0, 0, 0, 0, 0, 7.31917344936033e-07, 3.689321224943123e-05, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0.5019726768895573, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0.00010917955786877644, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    features = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x8', 'x9', 'x11', 'x7_value_0', 'x7_value_1', 'x7_value_10',
                'x7_value_11', 'x7_value_12', 'x7_value_13', 'x7_value_14', 'x7_value_15', 'x7_value_16', 'x7_value_17',
                'x7_value_18', 'x7_value_19', 'x7_value_2', 'x7_value_20', 'x7_value_21', 'x7_value_22', 'x7_value_23',
                'x7_value_24', 'x7_value_25', 'x7_value_26', 'x7_value_27', 'x7_value_28', 'x7_value_29', 'x7_value_3',
                'x7_value_30', 'x7_value_31', 'x7_value_32', 'x7_value_33', 'x7_value_34', 'x7_value_35', 'x7_value_36',
                'x7_value_37', 'x7_value_38', 'x7_value_39', 'x7_value_4', 'x7_value_40', 'x7_value_41', 'x7_value_42',
                'x7_value_43', 'x7_value_44', 'x7_value_45', 'x7_value_46', 'x7_value_47', 'x7_value_48', 'x7_value_49',
                'x7_value_5', 'x7_value_50', 'x7_value_51', 'x7_value_52', 'x7_value_53', 'x7_value_54', 'x7_value_55',
                'x7_value_56', 'x7_value_57', 'x7_value_58', 'x7_value_59', 'x7_value_6', 'x7_value_60', 'x7_value_61',
                'x7_value_62', 'x7_value_63', 'x7_value_64', 'x7_value_65', 'x7_value_66', 'x7_value_67', 'x7_value_68',
                'x7_value_69', 'x7_value_7', 'x7_value_70', 'x7_value_71', 'x7_value_8', 'x7_value_9', 'x10_value_0',
                'x10_value_1', 'x10_value_2']
    plot_features_importance(features, importance)

    result_path = Path(__file__).parents[0].absolute().joinpath("result_images").joinpath(
        "test_xai").joinpath("test_plot_features_importance.png")
    pyplot.gcf().set_size_inches(17, 10)
    pyplot.savefig(str(result_path))

    baseline_path = Path(__file__).parents[0].absolute().joinpath("baseline_images").joinpath(
        "test_xai").joinpath("test_plot_features_importance.png")
    pyplot.cla()
    pyplot.close(pyplot.gcf())
    compare_images_from_paths(str(baseline_path), str(result_path))


def test_plot_features_importance_exists_ax():
    pyplot.figure()
    ax = pyplot.gca()

    ax.set_title("My ax")
    importance = [0.047304175084187376, 0.011129476233187116, 0.01289095487553893, 0.015528563988219685,
                  0.010904371085026893, 0.03088871541039015, 0.03642466650007851, 0.005758395681352302,
                  0.3270373201417306, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3.2562763221434323e-08, 0, 0,
                  1.3846860504742983e-05, 0, 0, 0, 0, 0, 0, 7.31917344936033e-07, 3.689321224943123e-05, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0.5019726768895573, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0.00010917955786877644, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    features = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x8', 'x9', 'x11', 'x7_value_0', 'x7_value_1', 'x7_value_10',
                'x7_value_11', 'x7_value_12', 'x7_value_13', 'x7_value_14', 'x7_value_15', 'x7_value_16', 'x7_value_17',
                'x7_value_18', 'x7_value_19', 'x7_value_2', 'x7_value_20', 'x7_value_21', 'x7_value_22', 'x7_value_23',
                'x7_value_24', 'x7_value_25', 'x7_value_26', 'x7_value_27', 'x7_value_28', 'x7_value_29', 'x7_value_3',
                'x7_value_30', 'x7_value_31', 'x7_value_32', 'x7_value_33', 'x7_value_34', 'x7_value_35', 'x7_value_36',
                'x7_value_37', 'x7_value_38', 'x7_value_39', 'x7_value_4', 'x7_value_40', 'x7_value_41', 'x7_value_42',
                'x7_value_43', 'x7_value_44', 'x7_value_45', 'x7_value_46', 'x7_value_47', 'x7_value_48', 'x7_value_49',
                'x7_value_5', 'x7_value_50', 'x7_value_51', 'x7_value_52', 'x7_value_53', 'x7_value_54', 'x7_value_55',
                'x7_value_56', 'x7_value_57', 'x7_value_58', 'x7_value_59', 'x7_value_6', 'x7_value_60', 'x7_value_61',
                'x7_value_62', 'x7_value_63', 'x7_value_64', 'x7_value_65', 'x7_value_66', 'x7_value_67', 'x7_value_68',
                'x7_value_69', 'x7_value_7', 'x7_value_70', 'x7_value_71', 'x7_value_8', 'x7_value_9', 'x10_value_0',
                'x10_value_1', 'x10_value_2']
    plot_features_importance(features, importance, ax=ax)

    result_path = Path(__file__).parents[0].absolute().joinpath("result_images").joinpath(
        "test_xai").joinpath("test_plot_features_importance_exists_ax.png")
    pyplot.gcf().set_size_inches(17, 10)
    pyplot.savefig(str(result_path))

    baseline_path = Path(__file__).parents[0].absolute().joinpath("baseline_images").joinpath(
        "test_xai").joinpath("test_plot_features_importance_exists_ax.png")
    pyplot.cla()
    pyplot.close(pyplot.gcf())
    compare_images_from_paths(str(baseline_path), str(result_path))
