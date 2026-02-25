"""Tests for curves metrics visualization."""

import json
from pathlib import Path
from typing import Any, Dict

from matplotlib import pyplot as plt
import numpy as np
from plotly import graph_objects as go
import pytest

from ds_utils.metrics.curves import (
    plot_precision_recall_curve_with_thresholds_annotations,
    plot_roc_curve_with_thresholds_annotations,
)

BASELINE_DIR = Path(__file__).parents[1] / "baseline_images" / Path(__file__).parent.name / Path(__file__).stem
RESULT_DIR = Path(__file__).parents[1] / "result_images" / Path(__file__).parent.name / Path(__file__).stem
RESOURCES_DIR = Path(__file__).parents[1] / "resources"

RESULT_DIR.mkdir(exist_ok=True, parents=True)


@pytest.fixture
def plotly_models_dict() -> Dict[str, Any]:
    """Load plotly models data from JSON file."""
    with (RESOURCES_DIR / "plotly_models.json").open("r") as file:
        return json.load(file)


def save_plotly_figure_and_return_matplot(fig: go.Figure, path_to_save: Path) -> plt.Figure:
    """Save plotly figure and convert to a matplotlib figure for comparison."""
    fig.write_image(str(path_to_save))
    img = plt.imread(path_to_save)
    figure, ax = plt.subplots()
    ax.imshow(img)
    ax.axis("off")
    return figure


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR, tolerance=18)
@pytest.mark.parametrize("add_random_classifier_line", [True, False], ids=["default", "without_random_classifier"])
def test_plot_roc_curve_with_thresholds_annotations(mocker, request, add_random_classifier_line, plotly_models_dict):
    """Test ROC curve plotting when underlying calculations fail."""
    y_true = np.array(plotly_models_dict["y_true"])
    classifiers_names_and_scores_dict = {
        name: np.array(data["y_scores"]) for name, data in plotly_models_dict.items() if name != "y_true"
    }

    def _mock_roc_curve(y_true, y_score, **kwargs):
        for classifier, scores in classifiers_names_and_scores_dict.items():
            if np.array_equal(scores, y_score):
                data = plotly_models_dict[classifier]["roc_curve"]
                return np.array(data["fpr_array"]), np.array(data["tpr_array"]), np.array(data["thresholds"])

    mocker.patch("ds_utils.metrics.curves.roc_curve", side_effect=_mock_roc_curve)

    def _mock_roc_auc_score(y_true, y_score, **kwargs):
        for classifier, scores in classifiers_names_and_scores_dict.items():
            if np.array_equal(scores, y_score):
                return np.float64(plotly_models_dict[classifier]["roc_auc_score"])

    mocker.patch("ds_utils.metrics.curves.roc_auc_score", side_effect=_mock_roc_auc_score)

    fig = plot_roc_curve_with_thresholds_annotations(
        y_true, classifiers_names_and_scores_dict, add_random_classifier_line=add_random_classifier_line
    )

    return save_plotly_figure_and_return_matplot(fig, RESULT_DIR / f"{request.node.name}.png")


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR, tolerance=16)
def test_plot_roc_curve_with_thresholds_annotations_exist_figure(mocker, request, plotly_models_dict):
    """Test plotting ROC curve on an existing Figure object."""
    fig = go.Figure()
    fig.update_layout(title="Receiver Operating Characteristic (ROC) Curve")

    y_true = np.array(plotly_models_dict["y_true"])
    classifiers_names_and_scores_dict = {
        name: np.array(data["y_scores"]) for name, data in plotly_models_dict.items() if name != "y_true"
    }

    def _mock_roc_curve(y_true, y_score, **kwargs):
        for classifier, scores in classifiers_names_and_scores_dict.items():
            if np.array_equal(scores, y_score):
                data = plotly_models_dict[classifier]["roc_curve"]
                return np.array(data["fpr_array"]), np.array(data["tpr_array"]), np.array(data["thresholds"])

    mocker.patch("ds_utils.metrics.curves.roc_curve", side_effect=_mock_roc_curve)

    def _mock_roc_auc_score(y_true, y_score, **kwargs):
        for classifier, scores in classifiers_names_and_scores_dict.items():
            if np.array_equal(scores, y_score):
                return np.float64(plotly_models_dict[classifier]["roc_auc_score"])

    mocker.patch("ds_utils.metrics.curves.roc_auc_score", side_effect=_mock_roc_auc_score)

    fig = plot_roc_curve_with_thresholds_annotations(
        y_true, classifiers_names_and_scores_dict, add_random_classifier_line=True, fig=fig
    )

    return save_plotly_figure_and_return_matplot(fig, RESULT_DIR / f"{request.node.name}.png")


@pytest.mark.parametrize(
    "plotly_graph_method",
    [plot_roc_curve_with_thresholds_annotations, plot_precision_recall_curve_with_thresholds_annotations],
    ids=["plot_roc_curve_with_thresholds_annotations", "plot_precision_recall_curve_with_thresholds_annotations"],
)
def test_plotly_graph_method_shape_mismatch(plotly_graph_method, plotly_models_dict):
    """Test that plotly graph methods raise ValueError for shape mismatches."""
    y_true = np.array(plotly_models_dict["y_true"][:1])
    classifiers_names_and_scores_dict = {
        name: np.array(data["y_scores"]) for name, data in plotly_models_dict.items() if name != "y_true"
    }
    with pytest.raises(
        ValueError, match=r"Shape mismatch: y_true \(1,\) and y_scores \(2239,\) for classifier Decision Tree"
    ):
        plotly_graph_method(y_true, classifiers_names_and_scores_dict)


@pytest.mark.parametrize(
    ("error", "message"),
    [
        (ValueError, "Error calculating ROC curve for classifier Decision Tree:"),
        (ValueError, "Error calculating AUC score for classifier Decision Tree:"),
    ],
    ids=["roc_calc_fail", "auc_calc_fail"],
)
def test_plot_roc_curve_with_thresholds_annotations_fail_calc(mocker, request, error, message, plotly_models_dict):
    """Test ROC curve plotting when underlying calculations fail."""
    y_true = np.array(plotly_models_dict["y_true"])
    classifiers_names_and_scores_dict = {
        name: np.array(data["y_scores"]) for name, data in plotly_models_dict.items() if name != "y_true"
    }
    if request.node.callspec.id == "roc_calc_fail":
        mocker.patch("ds_utils.metrics.curves.roc_curve", side_effect=ValueError)
    elif request.node.callspec.id == "auc_calc_fail":

        def _mock_roc_curve(y_true, y_score, **kwargs):
            for classifier, scores in classifiers_names_and_scores_dict.items():
                if np.array_equal(scores, y_score):
                    data = plotly_models_dict[classifier]["roc_curve"]
                    return np.array(data["fpr_array"]), np.array(data["tpr_array"]), np.array(data["thresholds"])

        mocker.patch("ds_utils.metrics.curves.roc_curve", side_effect=_mock_roc_curve)

        mocker.patch("ds_utils.metrics.curves.roc_auc_score", side_effect=ValueError)
    with pytest.raises(error, match=message):
        plot_roc_curve_with_thresholds_annotations(y_true, classifiers_names_and_scores_dict)


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR, tolerance=18)
@pytest.mark.parametrize("add_random_classifier_line", [False, True], ids=["default", "with_random_classifier_line"])
def test_plot_precision_recall_curve_with_thresholds_annotations(
    mocker, request, add_random_classifier_line, plotly_models_dict
):
    """Test plotting a Precision-Recall curve with threshold annotations."""
    y_true = np.array(plotly_models_dict["y_true"])
    classifiers_names_and_scores_dict = {
        name: np.array(data["y_scores"]) for name, data in plotly_models_dict.items() if name != "y_true"
    }

    def _mock_precision_recall_curve(y_true, y_score, **kwargs):
        for classifier, scores in classifiers_names_and_scores_dict.items():
            if np.array_equal(scores, y_score):
                data = plotly_models_dict[classifier]["precision_recall_curve"]
                return np.array(data["precision_array"]), np.array(data["recall_array"]), np.array(data["thresholds"])

    mocker.patch("ds_utils.metrics.curves.precision_recall_curve", side_effect=_mock_precision_recall_curve)

    fig = plot_precision_recall_curve_with_thresholds_annotations(
        y_true, classifiers_names_and_scores_dict, add_random_classifier_line=add_random_classifier_line
    )

    return save_plotly_figure_and_return_matplot(fig, RESULT_DIR / f"{request.node.name}.png")


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR, tolerance=15)
def test_plot_precision_recall_curve_with_thresholds_annotations_exists_figure(mocker, request, plotly_models_dict):
    """Test plotting a Precision-Recall curve on an existing Figure object."""
    fig = go.Figure()
    fig.update_layout(title="Precision-Recall Curve")

    y_true = np.array(plotly_models_dict["y_true"])
    classifiers_names_and_scores_dict = {
        name: np.array(data["y_scores"]) for name, data in plotly_models_dict.items() if name != "y_true"
    }

    def _mock_precision_recall_curve(y_true, y_score, **kwargs):
        for classifier, scores in classifiers_names_and_scores_dict.items():
            if np.array_equal(scores, y_score):
                data = plotly_models_dict[classifier]["precision_recall_curve"]
                return np.array(data["precision_array"]), np.array(data["recall_array"]), np.array(data["thresholds"])

    mocker.patch("ds_utils.metrics.curves.precision_recall_curve", side_effect=_mock_precision_recall_curve)

    fig = plot_precision_recall_curve_with_thresholds_annotations(y_true, classifiers_names_and_scores_dict, fig=fig)

    return save_plotly_figure_and_return_matplot(fig, RESULT_DIR / f"{request.node.name}.png")


def test_plot_precision_recall_curve_with_thresholds_annotations_fail_calc(mocker, plotly_models_dict):
    """Test Precision-Recall curve plotting when underlying calculations fail."""
    y_true = np.array(plotly_models_dict["y_true"])
    classifiers_names_and_scores_dict = {
        name: np.array(data["y_scores"]) for name, data in plotly_models_dict.items() if name != "y_true"
    }

    mocker.patch("ds_utils.metrics.curves.precision_recall_curve", side_effect=ValueError)
    with pytest.raises(ValueError, match="Error calculating Precision-Recall curve for classifier Decision Tree:"):
        plot_precision_recall_curve_with_thresholds_annotations(y_true, classifiers_names_and_scores_dict)
