"""Regression test verifying function and class signatures in skill API references."""

import ast
import importlib
import inspect
from pathlib import Path
import re
import pytest

# Map of function/class names to import module in ds_utils
SYMBOL_MODULES = {
    "plot_confusion_matrix": "ds_utils.metrics.confusion_matrix",
    "plot_roc_curve_with_thresholds_annotations": "ds_utils.metrics.curves",
    "plot_precision_recall_curve_with_thresholds_annotations": "ds_utils.metrics.curves",
    "visualize_accuracy_grouped_by_probability": "ds_utils.metrics.probability_analysis",
    "plot_metric_growth_per_labeled_instances": "ds_utils.metrics.learning_curves",
    "plot_error_analysis_chart": "ds_utils.metrics.probability_analysis",
    "generate_error_analysis_report": "ds_utils.metrics.error_analysis",
    "directional_accuracy_score": "ds_utils.metrics.time_series",
    "directional_bias_score": "ds_utils.metrics.time_series",
    "regression_auc_score": "ds_utils.metrics.regression",
    "plot_rec_curve_with_annotations": "ds_utils.metrics.regression",
    "visualize_feature": "ds_utils.preprocess.visualization",
    "visualize_correlations": "ds_utils.preprocess.visualization",
    "plot_correlation_dendrogram": "ds_utils.preprocess.visualization",
    "plot_features_interaction": "ds_utils.preprocess.visualization",
    "plot_pca_explained_variance": "ds_utils.preprocess.visualization",
    "get_correlated_features": "ds_utils.preprocess.statistics",
    "extract_statistics_dataframe_per_label": "ds_utils.preprocess.statistics",
    "compute_mutual_information": "ds_utils.preprocess.statistics",
    "append_tags_to_frame": "ds_utils.strings",
    "extract_significant_terms_from_subset": "ds_utils.strings",
    "MultiLabelBinarizerTransformer": "ds_utils.transformers.multi_label_binarizer",
    "SentenceEmbeddingTransformer": "ds_utils.transformers.sentence_embedding",
    "plot_loss_vs_cluster_number": "ds_utils.unsupervised",
    "plot_cluster_cardinality": "ds_utils.unsupervised",
    "plot_cluster_magnitude": "ds_utils.unsupervised",
    "plot_magnitude_vs_cardinality": "ds_utils.unsupervised",
    "plot_features_importance": "ds_utils.xai",
    "draw_dot_data": "ds_utils.xai",
    "draw_tree": "ds_utils.xai",
    "generate_decision_paths": "ds_utils.xai",
}


def find_signature_blocks():
    """Extract (skill_name, symbol_name, signature_ast) from all api_reference.md files."""
    repo_root = Path(__file__).resolve().parent.parent
    ref_files = sorted(repo_root.glob("skills/*/references/api_reference.md"))
    cases = []

    for ref_path in ref_files:
        skill_name = ref_path.parts[-3]
        content = ref_path.read_text(encoding="utf-8")

        # Find python blocks following Signature or Signatures
        # We can find blocks with ```python ... ```
        pattern = r"### Signatures?\n```python\n(.*?)\n```"
        matches = re.findall(pattern, content, re.DOTALL)
        for block in matches:
            # A block might contain one or multiple function signatures
            # We split by top-level function names (starts at line beginning with identifier()
            sub_blocks = re.split(r"\n(?=[a-zA-Z_]\w*\()", block.strip())
            for sub in sub_blocks:
                sub = sub.strip()
                if not sub:
                    continue
                # Prepend 'def ' and append ': pass' if it has return annotation
                if "->" in sub:
                    def_code = "def " + sub + ": pass"
                else:
                    def_code = "def " + sub + ": pass"
                parsed = ast.parse(def_code)
                fn_def = parsed.body[0]
                cases.append((skill_name, fn_def.name, fn_def.args))

    return cases


SIG_CASES = find_signature_blocks()


@pytest.mark.parametrize(("skill_name", "symbol_name", "doc_args"), SIG_CASES)
def test_skill_api_reference_signatures(skill_name, symbol_name, doc_args):
    """Verify that every documented signature matches inspect.signature in ds_utils."""
    assert symbol_name in SYMBOL_MODULES, f"Unknown symbol {symbol_name} in {skill_name}"
    mod = importlib.import_module(SYMBOL_MODULES[symbol_name])
    obj = getattr(mod, symbol_name)

    real_sig = inspect.signature(obj)

    # 1. Positional / args
    doc_pos_args = [arg.arg for arg in doc_args.args]
    doc_kwonly_args = [arg.arg for arg in doc_args.kwonlyargs]
    doc_vararg = doc_args.vararg.arg if doc_args.vararg else None
    doc_kwarg = doc_args.kwarg.arg if doc_args.kwarg else None

    real_pos_args = []
    real_kwonly_args = []
    real_vararg = None
    real_kwarg = None

    for param in real_sig.parameters.values():
        if param.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            real_pos_args.append(param.name)
        elif param.kind == inspect.Parameter.KEYWORD_ONLY:
            real_kwonly_args.append(param.name)
        elif param.kind == inspect.Parameter.VAR_POSITIONAL:
            real_vararg = param.name
        elif param.kind == inspect.Parameter.VAR_KEYWORD:
            real_kwarg = param.name

    # Assert names and order
    assert doc_pos_args == real_pos_args, (
        f"[{skill_name}:{symbol_name}] Positional args mismatch. Doc: {doc_pos_args}, Real: {real_pos_args}"
    )
    assert doc_kwonly_args == real_kwonly_args, (
        f"[{skill_name}:{symbol_name}] Keyword-only args mismatch. Doc: {doc_kwonly_args}, Real: {real_kwonly_args}"
    )
    assert doc_vararg == real_vararg, (
        f"[{skill_name}:{symbol_name}] Varargs mismatch. Doc: {doc_vararg}, Real: {real_vararg}"
    )
    assert doc_kwarg == real_kwarg, (
        f"[{skill_name}:{symbol_name}] Kwargs mismatch. Doc: {doc_kwarg}, Real: {real_kwarg}"
    )

    # 2. Presence of defaults
    num_pos_defaults = len(doc_args.defaults)
    doc_pos_has_default = {name: (i >= len(doc_pos_args) - num_pos_defaults) for i, name in enumerate(doc_pos_args)}

    doc_kwonly_has_default = {
        name: (default is not None) for name, default in zip(doc_kwonly_args, doc_args.kw_defaults, strict=False)
    }

    for name in real_pos_args:
        real_has_default = real_sig.parameters[name].default is not inspect.Parameter.empty
        assert doc_pos_has_default[name] == real_has_default, (
            f"[{skill_name}:{symbol_name}] Default presence mismatch for arg '{name}'. "
            f"Doc: {doc_pos_has_default[name]}, Real: {real_has_default}"
        )

    for name in real_kwonly_args:
        real_has_default = real_sig.parameters[name].default is not inspect.Parameter.empty
        assert doc_kwonly_has_default[name] == real_has_default, (
            f"[{skill_name}:{symbol_name}] Default presence mismatch for kwonly arg '{name}'. "
            f"Doc: {doc_kwonly_has_default[name]}, Real: {real_has_default}"
        )
