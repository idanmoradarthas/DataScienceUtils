import json
from pathlib import Path
from typing import Dict, Union, Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
from matplotlib import pyplot as plt
from numpy.random.mtrand import RandomState
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

# Import mock classifiers from the new utility file
from .metrics_test_utils import MockClassifier, AnotherMockClassifier

from ds_utils.metrics import (
    plot_confusion_matrix,
    plot_metric_growth_per_labeled_instances,
    visualize_accuracy_grouped_by_probability,
    plot_roc_curve_with_thresholds_annotations, plot_precision_recall_curve_with_thresholds_annotations
)

BASELINE_DIR = Path(__file__).parent / "baseline_images" / "test_metrics"
RESULT_DIR = Path(__file__).parent / "result_images" / "test_metrics"
RESOURCES_DIR = Path(__file__).parent / "resources"


@pytest.fixture
def iris_data() -> Dict[str, np.ndarray]:
    """Load and return iris dataset splits."""
    return {
        key: pd.read_csv(RESOURCES_DIR / f"iris_{key}.csv").values
        for key in ["x_train", "x_test", "y_train", "y_test"]
    }

# The classifiers fixture has been removed as it was unused.
# MockClassifier and AnotherMockClassifier classes have been moved to metrics_test_utils.py


@pytest.fixture
def plotly_models_dict() -> Dict[str, Any]:
    """Load plotly models data from JSON file."""
    with (RESOURCES_DIR / "plotly_models.json").open("r") as file:
        return json.load(file)


@pytest.fixture(autouse=True)
def setup_teardown():
    """Setup and teardown for tests."""
    RESULT_DIR.mkdir(exist_ok=True, parents=True)
    yield
    plt.close("all")  # Close all figures instead of just current


def save_plotly_figure_and_return_matplot(fig: go.Figure, path_to_save: Path) -> plt.Figure:
    """Save plotly figure and convert to matplotlib figure for comparison."""
    fig.write_image(str(path_to_save))
    img = plt.imread(path_to_save)
    figure, ax = plt.subplots()
    ax.imshow(img)
    ax.axis("off")
    return figure


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR)
@pytest.mark.parametrize("custom_y_test, custom_y_pred, labels", [
    ("1 1 1 1 1 0 1 0 1 0 0 1 0 0 1 0 0 0 1 0 1 0 0 1 1 1 1 1 0 0 0 1 0 1 0 1 0 0 0 0 1 1 1 1 0 1 1 0 1 0",
     "0 1 1 1 1 0 0 0 1 0 0 0 0 1 1 0 1 1 0 0 1 0 0 1 1 0 1 1 0 0 0 1 0 1 0 1 1 0 1 0 1 1 1 1 0 1 1 1 1 1",
     [1, 0]),
    ("1 0 1 1 0 0 0 0 2 2 1 1 1 2 2 0 1 0 0 1 1 2 2 2 2 1 1 0 1 1 0 0 2 0 1 1 0 2 1 2 2 1 2 1 0 0 0 1 0 2 1 0 "
     "1 2 2 2 1 1 2 2 1 2 1 0 1 1 2 0 0 2 0 2 1 2 0",
     "0 0 2 2 2 0 1 0 1 2 2 2 2 2 2 0 2 1 2 2 0 2 2 2 1 1 2 0 1 2 0 2 2 0 2 2 2 2 2 2 2 0 2 1 0 0 1 1 1 0 1 1 2 "
     "0 1 2 0 0 0 2 2 2 2 0 0 2 2 1 0 2 0 0 2 0 2",
     [0, 1, 2])
], ids=["binary", "multiclass"])
def test_plot_confusion_matrix(custom_y_test, custom_y_pred, labels):
    y_test = np.fromstring(custom_y_test, dtype=int, sep=' ')
    y_pred = np.fromstring(custom_y_pred, dtype=int, sep=' ')

    ax = plot_confusion_matrix(y_test, y_pred, labels)

    # Assert that the confusion matrix is correctly calculated
    cm = ax[1].get_children()[0].get_array().data[:len(labels), :len(labels)]
    np.testing.assert_array_equal(cm,
                                  np.array([[np.sum((y_test == i) & (y_pred == j)) for j in labels] for i in labels]))

    # Assert that the accuracy and F1 score are correctly calculated
    accuracy = float(ax[2].texts[0].get_text().split(': ')[1])
    assert accuracy == np.mean(y_test == y_pred)

    return plt.gcf()


def test_plot_confusion_matrix_mocked_metrics(mocker):
    y_test_sample = np.array([0, 1, 0, 1, 0, 1])
    y_pred_sample = np.array([0, 1, 1, 1, 0, 0])
    labels_sample = [0, 1]

    # Define mocked return values
    mock_cm_array = np.array([[2, 1], [1, 2]])  # Example: TN=2, FP=1, FN=1, TP=2
    mock_accuracy = 0.6667 # (2+2)/(2+1+1+2) = 4/6
    mock_f1 = 0.6667 # Example F1

    # Mock the scikit-learn functions within ds_utils.metrics module
    mock_sk_confusion_matrix = mocker.patch("ds_utils.metrics.confusion_matrix", return_value=mock_cm_array)
    mock_sk_accuracy_score = mocker.patch("ds_utils.metrics.accuracy_score", return_value=mock_accuracy)
    mock_sk_f1_score = mocker.patch("ds_utils.metrics.f1_score", return_value=mock_f1)

    # Call the function that uses the mocked metrics
    ax_subplots = plot_confusion_matrix(y_test_sample, y_pred_sample, labels_sample)

    # Assertions:
    # 1. Check if mocked functions were called correctly
    mock_sk_confusion_matrix.assert_called_once_with(y_test_sample, y_pred_sample, labels=labels_sample, sample_weight=None)
    mock_sk_accuracy_score.assert_called_once_with(y_test_sample, y_pred_sample, sample_weight=None)
    # For f1_score, the call depends on whether it's binary or multiclass for `pos_label` and `average`
    # Assuming binary for this example based on labels_sample = [0, 1]
    mock_sk_f1_score.assert_called_once_with(y_test_sample, y_pred_sample, labels=labels_sample, pos_label=labels_sample[1], average="binary", sample_weight=None)

    # 2. Check if plot displays mocked values
    # Heatmap data (ax_subplots[1] is the heatmap axis)
    # The confusion matrix itself is plotted. The df derived from it for heatmap might have recall/precision cols/rows.
    # The raw cnf_matrix is used for the heatmap cells directly if we consider only the cells for labels.
    # The df created internally has labels as "0 - Predicted", "1 - Predicted" etc.
    # The heatmap function `sns.heatmap` receives `df`.
    # The actual plotted data (colors) comes from `df.iloc[0:len(labels), 0:len(labels)]` effectively.
    # However, the test `test_plot_confusion_matrix` checks `ax[1].get_children()[0].get_array().data`
    # which seems to be the direct way to get the plotted numerical values for the cells.
    # Let's assume the mock_cm_array is what should appear in the core cells of the heatmap.
    # The `plot_confusion_matrix` internal logic might add Recall/Precision rows/cols to the DataFrame it plots.
    # The `_create_binary_confusion_matrix` and `_create_multiclass_confusion_matrix` build a DataFrame `df`.
    # For binary, `df` has 3x3 shape (values, recall | precision, nan).
    # The actual heatmap part will be the top-left 2x2 from this df.
    # df.iloc[0:2, 0:2] should correspond to mock_cm_array
    # Let's check the text annotations if they are simpler or use the direct cell values.
    # The test `test_plot_confusion_matrix` checks `ax[1].get_children()[0].get_array().data[:len(labels), :len(labels)]`
    # This is what we should assert against.

    # For binary case, the df is 3x3. The heatmap is on this.
    # df.iloc[0,0] = TN, df.iloc[0,1] = FP
    # df.iloc[1,0] = FN, df.iloc[1,1] = TP
    # So, mock_cm_array[0,0] = TN, mock_cm_array[0,1] = FP etc.
    # The df structure is:
    #           Pred 0 | Pred 1 | Recall
    # Actual 0 | TN     | FP     | TNR
    # Actual 1 | FN     | TP     | TPR
    # Precision| NPV    | PPV    | NaN
    # The heatmap is on this entire df. So, the data array from heatmap will be this 3x3.
    # We are interested in the 2x2 confusion matrix part.
    heatmap_data = ax_subplots[1].get_children()[0].get_array().data
    # The mock_cm_array is [[TN, FP], [FN, TP]].
    # So, heatmap_data[0,0] should be mock_cm_array[0,0] (TN)
    # heatmap_data[0,1] should be mock_cm_array[0,1] (FP)
    # heatmap_data[1,0] should be mock_cm_array[1,0] (FN)
    # heatmap_data[1,1] should be mock_cm_array[1,1] (TP)
    np.testing.assert_array_equal(heatmap_data[:2, :2], mock_cm_array)


    # Accuracy text (ax_subplots[2] is the text axis)
    assert ax_subplots[2].texts[0].get_text() == f"Accuracy: {mock_accuracy:.4f}"
    # F1 score text
    assert ax_subplots[2].texts[1].get_text() == f"F1 Score: {mock_f1:.4f}"

    # No need to return plt.gcf() as this is not an mpl_image_compare test by default


def test_print_confusion_matrix_exception():
    with pytest.raises(ValueError):
        plot_confusion_matrix(np.array([]), np.array([]), [])


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR)
@pytest.mark.parametrize("n_samples, quantiles, random_state", [
    (None, np.linspace(0.05, 1, 20).tolist(), 42), # Kept original y_shape_n_outputs logic target
    (list(range(10, 100, 10)), None, 42),
    (None, np.linspace(0.05, 1, 20).tolist(), 1),
    (None, np.linspace(0.05, 1, 20).tolist(), RandomState(5))
], ids=["y_shape_n_outputs_mocked", "with_n_samples_mocked", "given_random_state_int_mocked", "given_random_state_mocked"])
def test_plot_metric_growth_per_labeled_instances(iris_data, n_samples, quantiles, random_state,
                                                  request):
    mock_classifiers = {
        "MockClassifier": MockClassifier(),
        "AnotherMockClassifier": AnotherMockClassifier()
    }

    if request.node.callspec.id == "y_shape_n_outputs_mocked":
        # This specific parameterization was originally for testing y_shape with n_outputs > 1
        # We keep the y data transformation for this case, but use mock classifiers.
        y_train = pd.get_dummies(pd.DataFrame(iris_data["y_train"]).astype(str)).values
        y_test = pd.get_dummies(pd.DataFrame(iris_data["y_test"]).astype(str)).values
    else:
        y_train, y_test = iris_data["y_train"], iris_data["y_test"]

    ax = plot_metric_growth_per_labeled_instances(
        iris_data["x_train"], y_train, iris_data["x_test"], y_test,
        mock_classifiers, n_samples=n_samples, quantiles=quantiles, random_state=random_state
    )

    # Assert that the number of lines in the plot matches the number of mock classifiers
    assert len(ax.lines) == len(mock_classifiers)

    # Assert that the x-axis label is correct
    assert ax.get_xlabel() == "Number of training samples"

    # Assert that the y-axis label is correct
    assert ax.get_ylabel() == "Metric score"

    return plt.gcf()


def test_plot_metric_growth_per_labeled_instances_no_n_samples_no_quantiles(iris_data):
    mock_classifiers = {
        "MockClassifier": MockClassifier(),
        "AnotherMockClassifier": AnotherMockClassifier()
    }
    with pytest.raises(ValueError):
        plot_metric_growth_per_labeled_instances(
            iris_data["x_train"], iris_data["y_train"],
            iris_data["x_test"], iris_data["y_test"],
            mock_classifiers, n_samples=None, quantiles=None
        )


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR)
def test_plot_metric_growth_per_labeled_instances_exists_ax(iris_data):
    mock_classifiers = {
        "MockClassifier": MockClassifier(),
        "AnotherMockClassifier": AnotherMockClassifier()
    }
    fig, ax = plt.subplots()
    ax.set_title("My ax")
    plot_metric_growth_per_labeled_instances(
        iris_data["x_train"], iris_data["y_train"],
        iris_data["x_test"], iris_data["y_test"],
        mock_classifiers, ax=ax, random_state=42
    )

    assert ax.get_title() == "My ax"

    return fig


def test_plot_metric_growth_per_labeled_instances_verbose(iris_data, capsys):
    mock_classifiers = {
        "MockClassifier": MockClassifier(),
        "AnotherMockClassifier": AnotherMockClassifier()
    }
    plot_metric_growth_per_labeled_instances(
        iris_data["x_train"], iris_data["y_train"],
        iris_data["x_test"], iris_data["y_test"],
        mock_classifiers, verbose=1
    )
    captured = capsys.readouterr().out
    expected = ("Fitting classifier MockClassifier for 20 times\nFitting classifier AnotherMockClassifier"
                " for 20 times\n")
    assert captured == expected


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR)
@pytest.mark.parametrize("display_breakdown, bins, threshold", [
    (False, None, 0.5),
    (True, None, 0.5),
    (False, [0, 0.3, 0.5, 0.8, 1], 0.5),
    (False, None, 0.3)
], ids=["default", "with_breakdown", "custom_bins", "custom_threshold"])
def test_visualize_accuracy_grouped_by_probability(mocker, display_breakdown, bins, threshold):
    mock_df = pd.DataFrame({
        'loan_condition_cat': [0, 1, 0, 1, 0, 1, 1, 0, 1, 0],
        'probabilities': [0.1, 0.8, 0.3, 0.9, 0.4, 0.7, 0.65, 0.2, 0.95, 0.15]
    })
    mocker.patch("pandas.read_csv", return_value=mock_df)

    # The Path object is now just a placeholder as read_csv is mocked
    class_with_probabilities = pd.read_csv("dummy_path.csv")
    ax = visualize_accuracy_grouped_by_probability(
        class_with_probabilities["loan_condition_cat"], 1,
        class_with_probabilities["probabilities"],
        display_breakdown=display_breakdown, bins=bins, threshold=threshold
    )

    # Assert that the x-axis label is correct
    assert ax.get_xlabel() == "Probability Range"

    # Assert that the y-axis label is correct
    assert ax.get_ylabel() == "Count"

    # Assert that the title is correct
    assert ax.get_title() == "Accuracy Distribution for 1 Class"

    figure = plt.gcf()
    figure.set_size_inches(10, 8)
    return figure


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR)
def test_visualize_accuracy_grouped_by_probability_exists_ax(mocker):
    mock_df = pd.DataFrame({
        'loan_condition_cat': [0, 1, 0, 1, 0, 1, 1, 0, 1, 0],
        'probabilities': [0.1, 0.8, 0.3, 0.9, 0.4, 0.7, 0.65, 0.2, 0.95, 0.15]
    })
    mocker.patch("pandas.read_csv", return_value=mock_df)

    fig, ax = plt.subplots()
    ax.set_title("My ax")

    # The Path object is now just a placeholder as read_csv is mocked
    class_with_probabilities = pd.read_csv("dummy_path.csv")
    visualize_accuracy_grouped_by_probability(
        class_with_probabilities["loan_condition_cat"], 1,
        class_with_probabilities["probabilities"], ax=ax
    )

    assert ax.get_title() == "My ax"

    figure = plt.gcf()
    figure.set_size_inches(10, 8)
    return figure


# @pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR, tolerance=14)
@pytest.mark.parametrize("add_random_classifier_line", [True, False], ids=["default", "without_random_classifier"])
def test_plot_roc_curve_with_thresholds_annotations(mocker, request, add_random_classifier_line, plotly_models_dict):
    y_true = np.array(plotly_models_dict["y_true"])
    classifiers_names_and_scores_dict = {name: np.array(data["y_scores"]) for name, data in plotly_models_dict.items()
                                         if name != "y_true"}

    def _mock_roc_curve(y_true, y_score, **kwargs):
        for classifier, scores in classifiers_names_and_scores_dict.items():
            if np.array_equal(scores, y_score):
                data = plotly_models_dict[classifier]["roc_curve"]
                return np.array(data["fpr_array"]), np.array(data["tpr_array"]), np.array(data["thresholds"])

    mocker.patch("ds_utils.metrics.roc_curve", side_effect=_mock_roc_curve)

    def _mock_roc_auc_score(y_true, y_score, **kwargs):
        for classifier, scores in classifiers_names_and_scores_dict.items():
            if np.array_equal(scores, y_score):
                return np.float64(plotly_models_dict[classifier]["roc_auc_score"])

    mocker.patch("ds_utils.metrics.roc_auc_score", side_effect=_mock_roc_auc_score)

    fig = plot_roc_curve_with_thresholds_annotations(
        y_true,
        classifiers_names_and_scores_dict,
        add_random_classifier_line=add_random_classifier_line
    )

    assert fig.layout.showlegend
    assert fig.layout.xaxis.title.text == 'False Positive Rate'
    assert fig.layout.yaxis.title.text == 'True Positive Rate'
    assert not fig.layout.title.text
    assert len(fig.data) == len(classifiers_names_and_scores_dict) + (1 if add_random_classifier_line else 0)
    # Check if the random classifier line is present when it should be
    random_classifier_traces = [trace for trace in fig.data if trace.name == "Random Classifier"]
    assert len(random_classifier_traces) == (1 if add_random_classifier_line else 0)
    # Check if all classifiers are present in the plot
    for classifier_name in classifiers_names_and_scores_dict.keys():
        assert any(classifier_name in trace.name for trace in fig.data)
        # Check if AUC scores are present in the legend
        for trace in fig.data:
            if trace.name != "Random Classifier":
                assert "AUC =" in trace.name

    # return save_plotly_figure_and_return_matplot(fig, RESULT_DIR / f"{request.node.name}.png")


# @pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR, tolerance=16)
def test_plot_roc_curve_with_thresholds_annotations_exist_figure(mocker, request, plotly_models_dict):
    fig = go.Figure()
    fig.update_layout(title="Receiver Operating Characteristic (ROC) Curve")

    y_true = np.array(plotly_models_dict["y_true"])
    classifiers_names_and_scores_dict = {name: np.array(data["y_scores"]) for name, data in plotly_models_dict.items()
                                         if name != "y_true"}

    def _mock_roc_curve(y_true, y_score, **kwargs):
        for classifier, scores in classifiers_names_and_scores_dict.items():
            if np.array_equal(scores, y_score):
                data = plotly_models_dict[classifier]["roc_curve"]
                return np.array(data["fpr_array"]), np.array(data["tpr_array"]), np.array(data["thresholds"])

    mocker.patch("ds_utils.metrics.roc_curve", side_effect=_mock_roc_curve)

    def _mock_roc_auc_score(y_true, y_score, **kwargs):
        for classifier, scores in classifiers_names_and_scores_dict.items():
            if np.array_equal(scores, y_score):
                return np.float64(plotly_models_dict[classifier]["roc_auc_score"])

    mocker.patch("ds_utils.metrics.roc_auc_score", side_effect=_mock_roc_auc_score)

    fig = plot_roc_curve_with_thresholds_annotations(
        y_true,
        classifiers_names_and_scores_dict,
        add_random_classifier_line=True,
        fig=fig
    )

    assert fig.layout.title.text == "Receiver Operating Characteristic (ROC) Curve"

    # return save_plotly_figure_and_return_matplot(fig, RESULT_DIR / f"{request.node.name}.png")


@pytest.mark.parametrize("plotly_graph_method", [plot_roc_curve_with_thresholds_annotations,
                                                 plot_precision_recall_curve_with_thresholds_annotations],
                         ids=["plot_roc_curve_with_thresholds_annotations",
                              "plot_precision_recall_curve_with_thresholds_annotations"])
def test_plotly_graph_method_shape_mismatch(plotly_graph_method, plotly_models_dict):
    y_true = np.array(plotly_models_dict["y_true"][:1])
    classifiers_names_and_scores_dict = {name: np.array(data["y_scores"]) for name, data in plotly_models_dict.items()
                                         if name != "y_true"}
    with pytest.raises(ValueError,
                       match=r"Shape mismatch: y_true \(1,\) and y_scores \(2239,\) for classifier Decision Tree"):
        plotly_graph_method(
            y_true,
            classifiers_names_and_scores_dict
        )


@pytest.mark.parametrize("error, message",
                         [(ValueError, "Error calculating ROC curve for classifier Decision Tree:"),
                          (ValueError, "Error calculating AUC score for classifier Decision Tree:")],
                         ids=["roc_calc_fail", "auc_calc_fail"])
def test_plot_roc_curve_with_thresholds_annotations_fail_calc(mocker, request, error, message, plotly_models_dict):
    y_true = np.array(plotly_models_dict["y_true"])
    classifiers_names_and_scores_dict = {name: np.array(data["y_scores"]) for name, data in plotly_models_dict.items()
                                         if name != "y_true"}
    if request.node.callspec.id == "roc_calc_fail":
        mocker.patch("ds_utils.metrics.roc_curve", side_effect=ValueError)
    elif request.node.callspec.id == "auc_calc_fail":
        def _mock_roc_curve(y_true, y_score, **kwargs):
            for classifier, scores in classifiers_names_and_scores_dict.items():
                if np.array_equal(scores, y_score):
                    data = plotly_models_dict[classifier]["roc_curve"]
                    return np.array(data["fpr_array"]), np.array(data["tpr_array"]), np.array(data["thresholds"])

        mocker.patch("ds_utils.metrics.roc_curve", side_effect=_mock_roc_curve)

        mocker.patch("ds_utils.metrics.roc_auc_score", side_effect=ValueError)
    with pytest.raises(error, match=message):
        plot_roc_curve_with_thresholds_annotations(
            y_true,
            classifiers_names_and_scores_dict
        )


# @pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR, tolerance=18)
@pytest.mark.parametrize("add_random_classifier_line", [False, True], ids=["default", "with_random_classifier_line"])
def test_plot_precision_recall_curve_with_thresholds_annotations(mocker, request, add_random_classifier_line,
                                                                 plotly_models_dict):
    y_true = np.array(plotly_models_dict["y_true"])
    classifiers_names_and_scores_dict = {name: np.array(data["y_scores"]) for name, data in plotly_models_dict.items()
                                         if name != "y_true"}

    def _mock_precision_recall_curve(y_true, y_score, **kwargs):
        for classifier, scores in classifiers_names_and_scores_dict.items():
            if np.array_equal(scores, y_score):
                data = plotly_models_dict[classifier]["precision_recall_curve"]
                return np.array(data["precision_array"]), np.array(data["recall_array"]), np.array(data["thresholds"])

    mocker.patch("ds_utils.metrics.precision_recall_curve", side_effect=_mock_precision_recall_curve)

    fig = plot_precision_recall_curve_with_thresholds_annotations(
        y_true,
        classifiers_names_and_scores_dict,
        add_random_classifier_line=add_random_classifier_line
    )

    assert fig.layout.showlegend
    assert fig.layout.xaxis.title.text == 'Recall'
    assert fig.layout.yaxis.title.text == 'Precision'
    assert not fig.layout.title.text
    assert len(fig.data) == len(classifiers_names_and_scores_dict) + (1 if add_random_classifier_line else 0)
    # Check if the random classifier line is present when it should be
    random_classifier_traces = [trace for trace in fig.data if trace.name == "Random Classifier"]
    assert len(random_classifier_traces) == (1 if add_random_classifier_line else 0)
    # Check if all classifiers are present in the plot
    for classifier_name in classifiers_names_and_scores_dict.keys():
        assert any(classifier_name in trace.name for trace in fig.data)
        # Check if AUC scores are present in the legend
        for trace in fig.data:
            if trace.name != "Random Classifier":
                assert "AUC =" not in trace.name

    # return save_plotly_figure_and_return_matplot(fig, RESULT_DIR / f"{request.node.name}.png")


# @pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR, tolerance=15)
def test_plot_precision_recall_curve_with_thresholds_annotations_exists_figure(mocker, request, plotly_models_dict):
    fig = go.Figure()
    fig.update_layout(title="Precision-Recall Curve")

    y_true = np.array(plotly_models_dict["y_true"])
    classifiers_names_and_scores_dict = {name: np.array(data["y_scores"]) for name, data in plotly_models_dict.items()
                                         if name != "y_true"}

    def _mock_precision_recall_curve(y_true, y_score, **kwargs):
        for classifier, scores in classifiers_names_and_scores_dict.items():
            if np.array_equal(scores, y_score):
                data = plotly_models_dict[classifier]["precision_recall_curve"]
                return np.array(data["precision_array"]), np.array(data["recall_array"]), np.array(data["thresholds"])

    mocker.patch("ds_utils.metrics.precision_recall_curve", side_effect=_mock_precision_recall_curve)

    fig = plot_precision_recall_curve_with_thresholds_annotations(
        y_true,
        classifiers_names_and_scores_dict,
        fig=fig
    )

    assert fig.layout.title.text == "Precision-Recall Curve"

    # return save_plotly_figure_and_return_matplot(fig, RESULT_DIR / f"{request.node.name}.png")


def test_plot_precision_recall_curve_with_thresholds_annotations_fail_calc(mocker, plotly_models_dict):
    y_true = np.array(plotly_models_dict["y_true"])
    classifiers_names_and_scores_dict = {name: np.array(data["y_scores"]) for name, data in plotly_models_dict.items()
                                         if name != "y_true"}

    mocker.patch("ds_utils.metrics.precision_recall_curve", side_effect=ValueError)
    with pytest.raises(ValueError, match="Error calculating Precision-Recall curve for classifier Decision Tree:"):
        plot_precision_recall_curve_with_thresholds_annotations(
            y_true,
            classifiers_names_and_scores_dict
        )
