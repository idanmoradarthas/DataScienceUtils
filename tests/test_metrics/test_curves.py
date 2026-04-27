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


@pytest.fixture
def plotly_models_pr_curve_dict() -> Dict[str, Any]:
    """Load plotly models data for precision-recall curves from JSON file."""
    with (RESOURCES_DIR / "plotly_models_pr_curve.json").open("r") as file:
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
@pytest.mark.parametrize(
    ("add_random_classifier_line", "random_classifier_line_kw"),
    [
        (True, None),
        (False, None),
        (True, {"line": dict(dash="dot", color="red", width=3), "name": "Custom Chance Level Name"}),
    ],
    ids=["default", "without_random_classifier", "random_classifier_styling"],
)
def test_plot_roc_curve_with_thresholds_annotations(
    mocker, request, add_random_classifier_line, random_classifier_line_kw, plotly_models_dict
):
    """Test ROC curve plotting with various random classifier line configurations."""
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
        y_true,
        classifiers_names_and_scores_dict,
        add_random_classifier_line=add_random_classifier_line,
        random_classifier_line_kw=random_classifier_line_kw,
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
    ("plotly_graph_method", "fixture_name", "expected_length"),
    [
        (plot_roc_curve_with_thresholds_annotations, "plotly_models_dict", 2239),
        (plot_precision_recall_curve_with_thresholds_annotations, "plotly_models_pr_curve_dict", 1500),
    ],
    ids=["plot_roc_curve_with_thresholds_annotations", "plot_precision_recall_curve_with_thresholds_annotations"],
)
def test_plotly_graph_method_shape_mismatch(plotly_graph_method, fixture_name, expected_length, request):
    """Test that plotly graph methods raise ValueError for shape mismatches."""
    models_dict = request.getfixturevalue(fixture_name)
    y_true = np.array(models_dict["y_true"][:1])
    classifiers_names_and_scores_dict = {
        name: np.array(data["y_scores"]) for name, data in models_dict.items() if name != "y_true"
    }
    with pytest.raises(
        ValueError,
        match=rf"Shape mismatch: y_true \(1,\) and y_scores \({expected_length},\) for classifier Decision Tree",
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
@pytest.mark.parametrize(
    ("plot_chance_level", "chance_level_kw", "use_weights"),
    [
        (False, None, False),
        (True, None, False),
        (True, {"line": dict(dash="dot", color="red", width=3), "name": "Custom Chance Level Name"}, False),
        (True, None, True),
    ],
    ids=[
        "default",
        "with_chance_level",
        "chance_level_styling",
        "chance_level_sample_weights",
    ],
)
def test_plot_precision_recall_curve_with_thresholds_annotations(
    mocker, request, plot_chance_level, chance_level_kw, use_weights, plotly_models_pr_curve_dict
):
    """Test plotting a Precision-Recall curve with threshold annotations."""
    y_true = np.array(plotly_models_pr_curve_dict["y_true"])
    classifiers_names_and_scores_dict = {
        name: np.array(data["y_scores"]) for name, data in plotly_models_pr_curve_dict.items() if name != "y_true"
    }

    weights = np.random.RandomState(42).random(len(y_true)) if use_weights else None

    def _mock_precision_recall_curve(y_true, y_score, **kw):
        for classifier, scores in classifiers_names_and_scores_dict.items():
            if np.array_equal(scores, y_score):
                data = plotly_models_pr_curve_dict[classifier]["precision_recall_curve"]
                return np.array(data["precision_array"]), np.array(data["recall_array"]), np.array(data["thresholds"])

    mocker.patch("ds_utils.metrics.curves.precision_recall_curve", side_effect=_mock_precision_recall_curve)

    def _mock_average_precision_score(y_true, y_score, **kw):
        for classifier, scores in classifiers_names_and_scores_dict.items():
            if np.array_equal(scores, y_score):
                return np.float64(plotly_models_pr_curve_dict[classifier]["average_precision_score"])

    mocker.patch("ds_utils.metrics.curves.average_precision_score", side_effect=_mock_average_precision_score)

    fig_out = plot_precision_recall_curve_with_thresholds_annotations(
        y_true,
        classifiers_names_and_scores_dict,
        plot_chance_level=plot_chance_level,
        chance_level_kw=chance_level_kw,
        sample_weight=weights,
    )

    return save_plotly_figure_and_return_matplot(fig_out, RESULT_DIR / f"{request.node.name}.png")


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR, tolerance=17)
def test_plot_precision_recall_curve_with_thresholds_annotations_exists_figure(
    mocker, request, plotly_models_pr_curve_dict
):
    """Test plotting a Precision-Recall curve on an existing Figure object."""
    fig = go.Figure()
    fig.update_layout(title="Precision-Recall Curve")

    y_true = np.array(plotly_models_pr_curve_dict["y_true"])
    classifiers_names_and_scores_dict = {
        name: np.array(data["y_scores"]) for name, data in plotly_models_pr_curve_dict.items() if name != "y_true"
    }

    def _mock_precision_recall_curve(y_true, y_score, **kwargs):
        for classifier, scores in classifiers_names_and_scores_dict.items():
            if np.array_equal(scores, y_score):
                data = plotly_models_pr_curve_dict[classifier]["precision_recall_curve"]
                return np.array(data["precision_array"]), np.array(data["recall_array"]), np.array(data["thresholds"])

    mocker.patch("ds_utils.metrics.curves.precision_recall_curve", side_effect=_mock_precision_recall_curve)

    def _mock_average_precision_score(y_true, y_score, **kw):
        for classifier, scores in classifiers_names_and_scores_dict.items():
            if np.array_equal(scores, y_score):
                return np.float64(plotly_models_pr_curve_dict[classifier]["average_precision_score"])

    mocker.patch("ds_utils.metrics.curves.average_precision_score", side_effect=_mock_average_precision_score)

    fig = plot_precision_recall_curve_with_thresholds_annotations(y_true, classifiers_names_and_scores_dict, fig=fig)

    return save_plotly_figure_and_return_matplot(fig, RESULT_DIR / f"{request.node.name}.png")


def test_plot_precision_recall_curve_with_thresholds_annotations_fail_calc(mocker, plotly_models_pr_curve_dict):
    """Test Precision-Recall curve plotting when underlying calculations fail."""
    y_true = np.array(plotly_models_pr_curve_dict["y_true"])
    classifiers_names_and_scores_dict = {
        name: np.array(data["y_scores"]) for name, data in plotly_models_pr_curve_dict.items() if name != "y_true"
    }

    mocker.patch("ds_utils.metrics.curves.precision_recall_curve", side_effect=ValueError)
    with pytest.raises(ValueError, match="Error calculating Precision-Recall curve for classifier Decision Tree:"):
        plot_precision_recall_curve_with_thresholds_annotations(y_true, classifiers_names_and_scores_dict)


def test_plot_precision_recall_curve_with_thresholds_annotations_fail_ap_calc(mocker, plotly_models_pr_curve_dict):
    """Test PR curve plotting when average_precision_score fails."""
    y_true = np.array(plotly_models_pr_curve_dict["y_true"])
    classifiers_names_and_scores_dict = {
        name: np.array(data["y_scores"]) for name, data in plotly_models_pr_curve_dict.items() if name != "y_true"
    }

    def _mock_precision_recall_curve(y_true, y_score, **kw):
        for classifier, scores in classifiers_names_and_scores_dict.items():
            if np.array_equal(scores, y_score):
                data = plotly_models_pr_curve_dict[classifier]["precision_recall_curve"]
                return np.array(data["precision_array"]), np.array(data["recall_array"]), np.array(data["thresholds"])

    mocker.patch("ds_utils.metrics.curves.precision_recall_curve", side_effect=_mock_precision_recall_curve)
    mocker.patch("ds_utils.metrics.curves.average_precision_score", side_effect=ValueError)

    with pytest.raises(ValueError, match="Error calculating Average Precision for classifier Decision Tree:"):
        plot_precision_recall_curve_with_thresholds_annotations(y_true, classifiers_names_and_scores_dict)


def test_plot_precision_recall_curve_chance_level_non_binary(mocker):
    """Test that chance level plotting raises ValueError if y_true is not binary."""
    mock_prc = mocker.patch("ds_utils.metrics.curves.precision_recall_curve")
    y_true = np.array([0, 1, 2])
    classifiers_names_and_scores_dict = {"Model": np.array([0.1, 0.4, 0.9])}
    with pytest.raises(ValueError, match="y_true must be binary for plotting chance level"):
        plot_precision_recall_curve_with_thresholds_annotations(
            y_true, classifiers_names_and_scores_dict, plot_chance_level=True
        )
    mock_prc.assert_not_called()


def test_plot_precision_recall_curve_chance_level_prevalence(mocker, plotly_models_pr_curve_dict):
    """Test that chance level line is plotted at correct prevalence."""
    y_true = np.array(plotly_models_pr_curve_dict["y_true"])
    classifiers_names_and_scores_dict = {
        name: np.array(data["y_scores"]) for name, data in plotly_models_pr_curve_dict.items() if name != "y_true"
    }

    prevalence = np.sum(y_true == 1) / len(y_true)

    def _mock_precision_recall_curve(y_true, y_score, **kwargs):
        for classifier, scores in classifiers_names_and_scores_dict.items():
            if np.array_equal(scores, y_score):
                data = plotly_models_pr_curve_dict[classifier]["precision_recall_curve"]
                return np.array(data["precision_array"]), np.array(data["recall_array"]), np.array(data["thresholds"])

    mocker.patch("ds_utils.metrics.curves.precision_recall_curve", side_effect=_mock_precision_recall_curve)
    mocker.patch("ds_utils.metrics.curves.average_precision_score", return_value=0.5)

    fig = plot_precision_recall_curve_with_thresholds_annotations(
        y_true, classifiers_names_and_scores_dict, plot_chance_level=True
    )

    chance_level_trace = next(trace for trace in fig.data if "Chance level" in trace.name)
    assert np.allclose(chance_level_trace.x, [0, 1])
    assert np.allclose(chance_level_trace.y, [prevalence, prevalence])


def test_plot_precision_recall_curve_chance_level_explicit_positive_label(mocker):
    """Test that prevalence is correctly calculated with an explicit positive label."""
    y_true = np.array([0, 0, 1])
    classifiers_names_and_scores_dict = {"Model": np.array([0.1, 0.2, 0.9])}

    # Test with positive_label=0 (prevalence should be 2/3)
    prevalence_0 = 2 / 3

    mock_prc = mocker.patch(
        "ds_utils.metrics.curves.precision_recall_curve",
        return_value=(np.array([0, 1]), np.array([1, 0]), np.array([0.5])),
    )
    mocker.patch("ds_utils.metrics.curves.average_precision_score", return_value=0.5)

    fig = plot_precision_recall_curve_with_thresholds_annotations(
        y_true, classifiers_names_and_scores_dict, plot_chance_level=True, positive_label=0
    )

    mock_prc.assert_called_once()
    _, call_kwargs = mock_prc.call_args
    assert call_kwargs.get("pos_label") == 0

    chance_level_trace = next(trace for trace in fig.data if "Chance level" in trace.name)
    assert np.allclose(chance_level_trace.y, [prevalence_0, prevalence_0])

    # Note: AP here refers to the chance-level Average Precision, which equals prevalence.
    # This naming convention is consistent with scikit-learn's PrecisionRecallDisplay.
    expected_name = f"Chance level (AP = {prevalence_0:0.2f})"
    assert chance_level_trace.name == expected_name
