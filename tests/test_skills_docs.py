"""Regression test verifying function and class signatures in skill API references."""

import ast
import importlib
import inspect
from pathlib import Path
import pkgutil
import re

from collections import Counter

import ds_utils
import pytest

# exact duplicate sections inflate agent context and are reported HIGH by SkillEvaluator Tier 2 dedup-scan
MIN_DUPLICATE_SECTION_CHARS = 60

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
    "plot_clusters": "ds_utils.unsupervised",
    "plot_clusters_plotly": "ds_utils.unsupervised",
    "plot_features_importance": "ds_utils.xai",
    "draw_dot_data": "ds_utils.xai",
}

# Deprecated functions stay in the library for backwards compatibility but are not documented in skills or Sphinx
DEPRECATED_API = {"draw_tree", "generate_decision_paths"}


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

    # 3. Literal default value validation
    offset = len(doc_pos_args) - num_pos_defaults
    for i, default_node in enumerate(doc_args.defaults):
        arg_name = doc_pos_args[offset + i]
        real_def = real_sig.parameters[arg_name].default
        try:
            doc_val = ast.literal_eval(default_node)
            if isinstance(real_def, (int, float, str, bool, list, tuple, dict, type(None))):
                if isinstance(real_def, list) and isinstance(doc_val, list):
                    assert len(real_def) == len(doc_val), (
                        f"[{skill_name}:{symbol_name}] Default list length mismatch for '{arg_name}'"
                    )
                else:
                    assert doc_val == real_def, (
                        f"[{skill_name}:{symbol_name}] Default value mismatch for '{arg_name}'. "
                        f"Doc: {doc_val!r}, Real: {real_def!r}"
                    )
        except (ValueError, SyntaxError):
            pass

    for arg, default_node in zip(doc_args.kwonlyargs, doc_args.kw_defaults, strict=False):
        if default_node is not None:
            arg_name = arg.arg
            real_def = real_sig.parameters[arg_name].default
            try:
                doc_val = ast.literal_eval(default_node)
                if isinstance(real_def, (int, float, str, bool, list, tuple, dict, type(None))):
                    assert doc_val == real_def, (
                        f"[{skill_name}:{symbol_name}] Default value mismatch for kwonly '{arg_name}'. "
                        f"Doc: {doc_val!r}, Real: {real_def!r}"
                    )
            except (ValueError, SyntaxError):
                pass


def test_symbol_modules_coverage():
    """Verify bidirectional 1:1 correspondence between SYMBOL_MODULES and documented signatures."""
    documented_symbols = {case[1] for case in SIG_CASES}
    registered_symbols = set(SYMBOL_MODULES.keys())
    assert documented_symbols == registered_symbols, (
        f"Coverage mismatch: diff = {documented_symbols ^ registered_symbols}"
    )


# Internal math helper not exposed as part of the public analytical API
INTENTIONALLY_UNDOCUMENTED = {"safe_percentile"}


def test_all_public_api_is_documented():
    """Verify all public ds_utils functions and classes are documented in skill API references."""
    public_symbols = set()
    for module_info in pkgutil.walk_packages(ds_utils.__path__, ds_utils.__name__ + "."):
        mod = importlib.import_module(module_info.name)
        for name, obj in inspect.getmembers(mod):
            if not name.startswith("_") and (inspect.isfunction(obj) or inspect.isclass(obj)):
                if getattr(obj, "__module__", None) == module_info.name:
                    public_symbols.add(name)

    expected_symbols = public_symbols - INTENTIONALLY_UNDOCUMENTED - DEPRECATED_API
    documented_symbols = {case[1] for case in SIG_CASES}
    missing = sorted(expected_symbols - documented_symbols)
    assert not missing, f"Public API symbols missing from skill documentation: {missing}"


def test_deprecated_api_is_registered():
    """Verify that all functions emitting deprecation warnings in ds_utils match DEPRECATED_API."""
    repo_root = Path(__file__).resolve().parent.parent
    ds_utils_dir = repo_root / "ds_utils"
    categories = {"DeprecationWarning", "PendingDeprecationWarning", "FutureWarning"}
    detected_deprecated = set()

    for py_file in ds_utils_dir.rglob("*.py"):
        content = py_file.read_text(encoding="utf-8")
        parsed = ast.parse(content, filename=str(py_file))
        for node in ast.walk(parsed):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for child in ast.walk(node):
                    if isinstance(child, ast.Call):
                        func = child.func
                        is_warn = (isinstance(func, ast.Attribute) and func.attr == "warn") or (
                            isinstance(func, ast.Name) and func.id == "warn"
                        )
                        if is_warn:
                            matched = False
                            if len(child.args) >= 2:
                                arg2 = child.args[1]
                                if isinstance(arg2, ast.Name) and arg2.id in categories:
                                    matched = True
                            for kw in child.keywords:
                                if (
                                    kw.arg == "category"
                                    and isinstance(kw.value, ast.Name)
                                    and kw.value.id in categories
                                ):
                                    matched = True
                            if matched:
                                detected_deprecated.add(node.name)

    assert detected_deprecated == DEPRECATED_API, (
        f"Mismatch between detected deprecated API and DEPRECATED_API registry: {detected_deprecated ^ DEPRECATED_API}"
    )


def test_deprecated_api_not_in_skills():
    """Verify none of the DEPRECATED_API names appears as a whole word in skill documentation."""
    repo_root = Path(__file__).resolve().parent.parent
    skill_files = list(repo_root.glob("skills/*/SKILL.md")) + list(repo_root.glob("skills/*/references/*.md"))
    patterns = {name: re.compile(rf"\b{re.escape(name)}\b") for name in DEPRECATED_API}

    violations = []
    for doc_file in skill_files:
        content = doc_file.read_text(encoding="utf-8")
        for name, pattern in patterns.items():
            if pattern.search(content):
                violations.append(f"{doc_file.relative_to(repo_root)} mentions deprecated symbol '{name}'")

    assert not violations, "Deprecated API found in skill documentation:\n" + "\n".join(violations)


def test_deprecated_api_not_in_sphinx_docs():
    """Verify none of DEPRECATED_API names or '.. deprecated::' appears in Sphinx documentation."""
    repo_root = Path(__file__).resolve().parent.parent
    docs_dir = repo_root / "docs" / "source"
    patterns = {name: re.compile(rf"\b{re.escape(name)}\b") for name in DEPRECATED_API}

    violations = []
    for rst_file in docs_dir.rglob("*.rst"):
        content = rst_file.read_text(encoding="utf-8")
        for name, pattern in patterns.items():
            if pattern.search(content):
                violations.append(f"{rst_file.relative_to(repo_root)} mentions deprecated symbol '{name}'")
        if ".. deprecated::" in content:
            violations.append(f"{rst_file.relative_to(repo_root)} contains '.. deprecated::' directive")

    assert not violations, "Deprecated API or directive found in Sphinx documentation:\n" + "\n".join(violations)


# Regex patterns copied from skillspector 2.12.0 references.py:
# _QUOTED_OR_CODE_PATH and _PLAIN_RELATIVE_PATH
QUOTED_OR_CODE_PATH = re.compile(
    r"(?:`|'|\")((?:\./)?(?:[A-Za-z0-9_.-]+/)*[A-Za-z0-9_.-]+\.[A-Za-z0-9]{1,12})(?:`|'|\")"
)
PLAIN_RELATIVE_PATH = re.compile(
    r"(?<![\w:/.-])((?:\./(?:[A-Za-z0-9_.-]+/)*[A-Za-z0-9_.-]+|"
    r"(?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]+\.[A-Za-z0-9]{1,12}))(?![\w/.-])"
)

SKILL_NAMES = [
    "ds-utils-metrics",
    "ds-utils-preprocess",
    "ds-utils-strings",
    "ds-utils-transformers",
    "ds-utils-unsupervised",
    "ds-utils-xai",
]


@pytest.mark.parametrize("skill_name", SKILL_NAMES)
def test_skill_md_has_no_unresolved_path_like_tokens(skill_name):
    """Ensure SKILL.md prose and frontmatter contain no path-like tokens that fail resolution.

    Skillspector treats any token matching QUOTED_OR_CODE_PATH or PLAIN_RELATIVE_PATH in
    SKILL.md (outside fenced code blocks) as a bundled file reference. If it does not resolve
    to an existing file relative to the skill directory, Tier 1 static evaluation reports
    INCOMPLETE.
    """
    repo_root = Path(__file__).resolve().parent.parent
    skill_dir = repo_root / "skills" / skill_name
    skill_md = skill_dir / "SKILL.md"
    assert skill_md.is_file(), f"Missing SKILL.md at {skill_md}"

    lines = skill_md.read_text(encoding="utf-8").splitlines()
    in_code_block = False
    fence_char = None
    fence_len = 0
    failures = []

    for line_idx, line in enumerate(lines, 1):
        # Check fenced code block start/end (3 or more backticks or tildes)
        fence_match = re.match(r"^[ ]{0,3}(`{3,}|~{3,})", line)
        if fence_match:
            marker = fence_match.group(1)
            char = marker[0]
            length = len(marker)
            if not in_code_block:
                in_code_block = True
                fence_char = char
                fence_len = length
                continue
            elif char == fence_char and length >= fence_len:
                in_code_block = False
                fence_char = None
                fence_len = 0
                continue

        if in_code_block:
            continue

        # Check quoted/code and plain relative matches
        candidates = []
        for m in QUOTED_OR_CODE_PATH.finditer(line):
            candidates.append(m.group(1))
        for m in PLAIN_RELATIVE_PATH.finditer(line):
            candidates.append(m.group(1))

        for token in candidates:
            # Strip leading ./ if present
            clean_token = token[2:] if token.startswith("./") else token
            target_path = (skill_dir / clean_token).resolve()
            # If target does not exist relative to skill_dir, flag failure
            try:
                target_path.relative_to(skill_dir.resolve())
            except ValueError:
                # Outside skill dir
                failures.append(
                    f"{skill_md.name}:{line_idx}:{token} -> escapes skill directory. "
                    "Rephrase to avoid backticked/quoted dotted tokens or use plain text."
                )
                continue

            if not target_path.exists():
                failures.append(
                    f"{skill_md.name}:{line_idx}:{token} -> unresolved file reference. "
                    "Rephrase token in prose/frontmatter (e.g. unquote version, remove backticks, "
                    "or include spaces/parentheses) so skillspector does not treat it as a bundled file."
                )

    assert not failures, f"Found {len(failures)} unresolved path-like token(s) in {skill_name}/SKILL.md:\n" + "\n".join(
        failures
    )


def test_skill_api_references_have_no_identical_subsections():
    """Verify that no API reference markdown file contains identical non-trivial sections."""
    repo_root = Path(__file__).resolve().parent.parent
    ref_files = sorted(repo_root.glob("skills/*/references/api_reference.md"))

    duplicate_findings = []

    for ref_path in ref_files:
        lines = ref_path.read_text(encoding="utf-8").splitlines()
        in_code_block = False
        current_heading = None
        current_body_lines = []
        sections = []

        for line in lines:
            if line.startswith("```"):
                in_code_block = not in_code_block
                current_body_lines.append(line)
                continue

            if not in_code_block and (re.match(r"^#{1,6}\s", line) or line.strip() == "---"):
                if current_heading is not None:
                    body = "\n".join(current_body_lines).strip()
                    if len(body) >= MIN_DUPLICATE_SECTION_CHARS:
                        sections.append((current_heading, body))
                current_heading = line.strip()
                current_body_lines = []
                continue

            current_body_lines.append(line)

        if current_heading is not None:
            body = "\n".join(current_body_lines).strip()
            if len(body) >= MIN_DUPLICATE_SECTION_CHARS:
                sections.append((current_heading, body))

        counts = Counter(sections)
        for (heading, body), count in counts.items():
            if count > 1:
                rel_path = ref_path.relative_to(repo_root).as_posix()
                duplicate_findings.append(f"{rel_path}: heading '{heading}' appears {count} times with identical body")

    assert not duplicate_findings, "Found duplicate subsections in skill API references:\n" + "\n".join(
        duplicate_findings
    )
