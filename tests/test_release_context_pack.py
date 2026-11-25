"""Tests for the release-notes context pack extractor (.github/release/context_pack.py).

A small git repository is built with the situations that made the 1.10.0 notes hard to write from commit messages
alone: a module turned into a package (old imports break), a renamed parameter, a parameter inserted before an existing
one, a bug fix hidden in a function body, a commit message that names the wrong module, dependency bumps, tests-only
and version-bump commits, support-matrix changes and new workflows.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess
import sys

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / ".github" / "release"))
import context_pack as cp  # noqa: E402

PYPROJECT_V1 = """[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "lib"
requires-python = ">=3.10"
dependencies = ["pandas>=2.1.4", "numpy"]
classifiers = [
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
]
"""

PYPROJECT_V2 = """[build-system]
requires = ["uv_build>=0.13.8,<0.14"]
build-backend = "uv_build"

[project]
name = "lib"
requires-python = ">=3.11"
dependencies = ["pandas>=3.0.0", "numpy", "pyparsing<3.3.0"]
classifiers = [
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
]

[project.optional-dependencies]
nlp = ["sentence-transformers>=3.0.0"]
"""

WORKFLOW_V1 = """name: Build
on: [push]
jobs:
  test:
    name: Tests
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest]
        python-version: ["3.10", "3.11"]
    runs-on: ${{ matrix.os }}
    steps:
      - run: echo hi
"""

WORKFLOW_V2 = WORKFLOW_V1.replace('["3.10", "3.11"]', '["3.11", "3.12"]')

SKILLS_WORKFLOW = """name: Skills check
on: [pull_request]
jobs:
  static:
    name: Static validation (blocking)
    runs-on: ubuntu-latest
    steps:
      - run: echo hi
  dedup:
    name: Dedup (advisory)
    runs-on: ubuntu-latest
    steps:
      - run: echo hi
"""

METRICS_V1 = '''"""Metrics."""


def plot_a(x, y, add_line=False):
    """Plot A."""
    return x


def plot_b(x):
    """Plot B."""
    return x
'''

CURVES = '''"""Curves."""


def plot_a(x, y, plot_chance_level=False, chance_level_kw=None):
    """Plot A.

    :param plot_chance_level: bool. Whether to plot the chance level.
    """
    return x
'''

BASIC = '''"""Basic."""


def plot_b(x):
    """Plot B."""
    return x
'''

STRINGS_V1 = '''"""Strings."""


def tags(x, tokenizer=None):
    """Tag it."""
    return x


def old_function(x):
    """Old."""
    return x
'''

STRINGS_V2 = '''"""Strings."""
import warnings


def tags(x, sparse=False, tokenizer=None):
    """Tag it.

    :param sparse: bool. Return sparse columns.
    """
    if sparse:
        return list(x)
    return x


def old_function(x):
    """Old."""
    warnings.warn("old_function is deprecated", DeprecationWarning, stacklevel=2)
    return x
'''

EXTRAS = '''"""Extras."""


def plot_new(x, *, color="red"):
    """Plot something new."""
    return x


class Transformer:
    """Transform things."""

    def __init__(self, *, size=3):
        self.size = size

    def fit(self, x):
        """Fit."""
        return self
'''


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=T",
            "-c",
            "user.email=t@example.com",
            "-c",
            "commit.gpgsign=false",
            "-c",
            "tag.gpgsign=false",
            "-c",
            "core.autocrlf=false",
            *args,
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    return result.stdout


def _commit(repo: Path, message: str, files: dict[str, str | None]) -> None:
    for relative, content in files.items():
        target = repo / relative
        if content is None:
            target.unlink()
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content.encode("utf-8"))
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", message)


@pytest.fixture(scope="module")
def release_repo(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Create a repository with tag 1.0.0 and the release commits described in the module docstring."""
    repo = tmp_path_factory.mktemp("repo")
    _git(repo, "init", "-q", "-b", "main")
    _commit(
        repo,
        "Initial release",
        {
            "pyproject.toml": PYPROJECT_V1,
            "CHANGELOG.md": "# Changelog\n\n## [1.0.0] - 2026-01-01\n\n### Added\n\n- **First**: Everything.\n",
            "lib/__init__.py": '__version__ = "1.0.0"\n',
            "lib/metrics.py": METRICS_V1,
            "lib/strings.py": STRINGS_V1,
            "tests/test_version.py": 'VERSION = "1.0.0"\n',
            "uv.lock": "lock v1\n",
            ".github/workflows/test.yml": WORKFLOW_V1,
        },
    )
    _git(repo, "tag", "1.0.0")
    _git(repo, "tag", "pr129_r7")
    _commit(
        repo,
        "Refactor metrics into a package (#11)\n\n"
        "* Split the metrics module\n\n"
        "- Added plot_new to lib.strings for reuse.\n"
        "- Moved plot_a to curves.\n"
        "Co-Authored-By: Someone <s@example.com>\n"
        "Claude-Session: https://example.com/s\n",
        {
            "lib/metrics.py": None,
            "lib/metrics/__init__.py": '"""Metrics package."""\n',
            "lib/metrics/curves.py": CURVES,
            "lib/metrics/basic.py": BASIC,
            "lib/metrics/extras.py": EXTRAS,
        },
    )
    _commit(repo, "Add sparse option to tags (#12)\n\n- Added `sparse`.\n", {"lib/strings.py": STRINGS_V2})
    _commit(repo, "Bump pandas from 2.1.4 to 2.2.0 (#13)", {"uv.lock": "lock v2\n"})
    _commit(repo, "Refresh the lockfile (#14)", {"uv.lock": "lock v3\n"})
    _commit(repo, "test: more coverage (#15)", {"tests/test_more.py": "def test_x():\n    pass\n"})
    _commit(
        repo,
        "Add skills and their validation (#16)",
        {
            "skills/one/SKILL.md": "# One\n",
            "install.sh": "#!/bin/sh\n",
            ".github/workflows/skills.yml": SKILLS_WORKFLOW,
            ".github/workflows/test.yml": WORKFLOW_V2,
        },
    )
    _commit(repo, "Drop Python 3.10 (#17)", {"pyproject.toml": PYPROJECT_V2})
    _commit(
        repo,
        "Update version to 1.1.0rc2 (#18)",
        {"lib/__init__.py": '__version__ = "1.1.0rc2"\n', "tests/test_version.py": 'VERSION = "1.1.0rc2"\n'},
    )
    return repo


@pytest.fixture(scope="module")
def analysis(release_repo: Path) -> cp.Analysis:
    """Analyze the whole range of the fixture repository."""
    return cp.analyze(release_repo, "1.0.0", "HEAD", "lib")


def test_latest_release_tag_uses_numeric_order_and_ignores_other_tags(release_repo: Path):
    """Only X.Y.Z and vX.Y.Z tags count, compared numerically, with an optional exclusion."""
    repo = release_repo
    _git(repo, "tag", "1.0.2")
    _git(repo, "tag", "v1.0.10")
    try:
        assert cp.latest_release_tag(repo) == "v1.0.10"
        assert cp.latest_release_tag(repo, exclude="1.0.10") == "1.0.2"
    finally:
        _git(repo, "tag", "-d", "1.0.2", "v1.0.10")


def test_latest_release_tag_returns_none_without_release_tags(tmp_path: Path):
    """A repository without release tags has no base."""
    _git(tmp_path, "init", "-q")
    assert cp.latest_release_tag(tmp_path) is None


def test_moved_symbols_are_matched_by_name_and_grouped(analysis: cp.Analysis):
    """A function that changed module is a move, not a removal plus an addition."""
    moves = {(move.old, move.new) for move in analysis.api_diff.moved}
    assert ("lib.metrics.plot_a", "lib.metrics.curves.plot_a") in moves
    assert ("lib.metrics.plot_b", "lib.metrics.basic.plot_b") in moves
    assert analysis.api_diff.removed == []


def test_old_import_paths_are_reported_as_broken(analysis: cp.Analysis):
    """The metrics package no longer re-exports its functions, so the old import fails at head."""
    outcomes = {item.statement: item.outcome for item in analysis.api_diff.legacy_imports}
    assert outcomes["from lib.metrics import plot_a"] == "ImportError"
    assert outcomes["from lib.metrics import plot_b"] == "ImportError"


def test_signature_changes_show_rename_and_positional_shift(analysis: cp.Analysis):
    """A rename keeps its position, and a parameter inserted before an existing one is flagged."""
    changes = {change.new: change.diff for change in analysis.api_diff.signature_changes}
    assert "add_line -> plot_chance_level" in changes["lib.metrics.curves.plot_a"].possible_renames
    assert any("chance_level_kw" in text for text in changes["lib.metrics.curves.plot_a"].added)
    assert "sparse inserted before tokenizer" in changes["lib.strings.tags"].positional_shift
    assert "lib.metrics.basic.plot_b" not in changes


def test_added_symbols_include_functions_and_classes_with_parameters(analysis: cp.Analysis):
    """New public functions and classes are listed with their parameters."""
    added = analysis.api_diff.added
    assert added == ["lib.metrics.extras.Transformer", "lib.metrics.extras.plot_new"]
    transformer = analysis.api_new.symbols["lib.metrics.extras.Transformer"]
    assert transformer.kind == "class"
    assert [param.name for param in transformer.params] == ["size"]
    assert transformer.methods == ["fit"]
    assert analysis.api_new.symbols["lib.metrics.extras.plot_new"].signature() == "(x, *, color = 'red')"


def test_implementation_changes_show_code_diff_without_docstring_noise(analysis: cp.Analysis):
    """A fix hidden in a function body is listed with the changed code and the changed parameter docs."""
    changes = {item.new: item for item in analysis.api_diff.implementation_changes}
    tags = changes["lib.strings.tags"]
    assert tags.code_changed
    assert any("if sparse" in line for line in tags.diff_lines)
    assert any(line.startswith("doc +:param sparse") for line in tags.diff_lines)
    assert not any('"""' in line for line in tags.diff_lines)


def test_deprecation_is_detected_from_warnings_in_the_body(analysis: cp.Analysis):
    """A function that warns with DeprecationWarning at head is marked deprecated."""
    assert "lib.strings.old_function" in analysis.api_diff.deprecated_now


def test_packaging_diff_reports_support_matrix_dependencies_and_backend(analysis: cp.Analysis):
    """Python support, minimum versions, extras and the build backend come from pyproject.toml."""
    text = "\n".join(analysis.packaging)
    assert "requires-python: >=3.10 -> >=3.11" in text
    assert "python classifiers: 3.10, 3.11 -> 3.11, 3.12" in text
    assert "build backend: hatchling.build -> uv_build" in text
    assert "dependency changed: pandas >=2.1.4 -> >=3.0.0" in text
    assert "dependency added: pyparsing <3.3.0" in text
    assert "optional extra added: nlp = sentence-transformers>=3.0.0" in text


def test_workflow_diff_reports_new_workflows_jobs_and_matrix(analysis: cp.Analysis):
    """Workflow names, job names (with their blocking/advisory wording) and matrix changes are reported."""
    text = "\n".join(analysis.workflows)
    assert "workflow added: Skills check" in text
    assert "static: Static validation (blocking)" in text
    assert "dedup: Dedup (advisory)" in text
    assert "matrix in Build" in text
    assert '"3.10", "3.11"' in text
    assert '"3.11", "3.12"' in text


def test_new_top_level_paths_are_listed(analysis: cp.Analysis):
    """Top-level paths that did not exist at the base are listed."""
    assert "install.sh" in analysis.new_paths
    assert any(path.startswith("skills/") for path in analysis.new_paths)


def test_pull_requests_are_classified_by_the_files_they_touch(analysis: cp.Analysis):
    """Noise (dependency bump, lockfile only, tests only, version bump) is told apart from user-facing work."""
    by_number = {change.numbers[0]: change for change in analysis.changes}
    assert [change.numbers[0] for change in analysis.changes] == [11, 12, 13, 14, 15, 16, 17, 18]
    assert by_number[11].relevance == "user"
    assert by_number[12].relevance == "user"
    assert (by_number[13].relevance, by_number[13].reason) == ("noise", "dependency bump")
    assert (by_number[14].relevance, by_number[14].reason) == ("noise", "lockfile only")
    assert (by_number[15].relevance, by_number[15].reason) == ("noise", "tests only")
    assert (by_number[16].relevance, by_number[16].reason) == ("user", "skills")
    assert by_number[17].relevance == "maintainer"
    assert (by_number[18].relevance, by_number[18].reason) == ("noise", "version bump")


def test_commit_message_naming_the_wrong_module_is_flagged(analysis: cp.Analysis):
    """PR #11 says plot_new was added to lib.strings, but it lives in lib.metrics.extras at head."""
    change = next(change for change in analysis.changes if change.numbers == [11])
    assert change.stale_claims == ["plot_new: the message says 'lib.strings', at head it lives in lib.metrics.extras"]
    other = next(change for change in analysis.changes if change.numbers == [12])
    assert other.stale_claims == []


def test_stale_claim_check_ignores_case_and_consistent_mentions(analysis: cp.Analysis):
    """A module word in another case, or a symbol in its true module, is not a contradiction."""
    api = analysis.api_new
    assert cp.find_stale_claims("Added plot_new to the EXTRAS module", api, "lib") == []
    assert cp.find_stale_claims("plot_new now lives in lib.metrics.extras", api, "lib") == []
    assert cp.find_stale_claims("plot_new moved to the strings module", api, "lib") != []


def test_clean_body_drops_noise_and_caps_length():
    """Trailers, version lines, housekeeping blocks, rules and repeated lines are removed before capping."""
    body = (
        "* Update dependencies and versioning\n- Upgrade pytest to 9.0.1.\n"
        "* Add the thing\n- It does X.\n- It does X.\n---------\n"
        "- Updated package version from 1.10.0rc5 to 1.10.0rc6.\n"
        "Co-authored-by: A <a@example.com>\n- Generated with a tool.\n"
    )
    cleaned = cp.clean_body(body, 1000)
    assert cleaned == "* Add the thing\n- It does X."
    assert cp.clean_body("x" * 50, 10).endswith("...")
    assert len(cp.clean_body("x" * 50, 10)) <= 10


@pytest.mark.parametrize(
    ("old", "new", "expected"),
    [
        ("def f(a, b=1): ...", "def f(a, b=2): ...", {"default_changed": ["b: 1 -> 2"]}),
        ("def f(a, b=1): ...", "def f(a, b): ...", {"became_required": ["b"]}),
        ("def f(a, b): ...", "def f(a): ...", {"removed": ["b"]}),
        ("def f(a, *, b=1): ...", "def f(a, *, b=1, c=None): ...", {"added": ["c = None"]}),
        ("def f(a, c=None): ...", "def f(a, b=0, c=None): ...", {"positional_shift": ["b inserted before c"]}),
        ("def f(a, b=1): ...", "def f(a, *, b=1): ...", {"kind_changed": ["b: positional-or-keyword -> keyword-only"]}),
        ("def f(a, b): ...", "def f(a, /, b): ...", {"kind_changed": ["a: positional-or-keyword -> positional-only"]}),
        ("def f(*, a): ...", "def f(a): ...", {"kind_changed": ["a: keyword-only -> positional-or-keyword"]}),
    ],
)
def test_diff_params_cases(old: str, new: str, expected: dict[str, list[str]]):
    """Parameter differences are reported precisely."""
    import ast

    before = cp.extract_params(ast.parse(old).body[0].args)
    after = cp.extract_params(ast.parse(new).body[0].args)
    diff = cp.diff_params(before, after)
    for key, value in expected.items():
        assert getattr(diff, key) == value
    assert not diff.empty()
    assert cp.diff_params(before, before).empty()


def test_a_parameter_that_became_keyword_only_is_reported_in_the_pack(tmp_path: Path):
    """Callers that passed the parameter by position break; the pack says so next to the other signature changes."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _commit(repo, "Initial", {"pkg/__init__.py": "", "pkg/m.py": "def f(a, b=1):\n    return a\n"})
    _git(repo, "tag", "1.0.0")
    _commit(repo, "Make b keyword-only (#2)", {"pkg/m.py": "def f(a, *, b=1):\n    return a\n"})
    analysis = cp.analyze(repo, "1.0.0", "HEAD", "pkg")
    pack, _ = cp.build_pack(analysis, 4000)
    assert "pkg.m.f: parameter kind changed: b: positional-or-keyword -> keyword-only" in pack


def test_a_docstring_change_after_the_first_paragraph_is_an_implementation_change(tmp_path: Path):
    """Only the parameter list of a docstring changed (the summary line did not): it must not be missed."""
    repo = tmp_path / "repo"
    repo.mkdir()
    doc = '"""Summary line.\n\n    :param x: {}\n    """'
    _git(repo, "init", "-q", "-b", "main")
    _commit(repo, "Initial", {"pkg/__init__.py": "", "pkg/m.py": f"def f(x):\n    {doc.format('old')}\n    return x\n"})
    _git(repo, "tag", "1.0.0")
    _commit(repo, "Reword the docs (#2)", {"pkg/m.py": f"def f(x):\n    {doc.format('new')}\n    return x\n"})
    changes = cp.analyze(repo, "1.0.0", "HEAD", "pkg").api_diff.implementation_changes
    assert [(item.new, item.code_changed, item.doc_changed) for item in changes] == [("pkg.m.f", False, True)]


def test_new_top_level_folders_are_taken_from_the_release_not_from_the_checkout(tmp_path: Path):
    """The range can be compared while another commit is checked out; a folder is recognized by its files' paths."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _commit(repo, "Initial", {"pkg/__init__.py": ""})
    _git(repo, "tag", "1.0.0")
    _git(repo, "checkout", "-q", "-b", "release")
    _commit(repo, "Add things (#2)", {"newdir/one.py": "", "notes.txt": "", "many/a.py": "", "many/b.py": ""})
    _git(repo, "checkout", "-q", "1.0.0")  # the folder does not exist in this checkout
    assert not (repo / "newdir").exists()
    assert cp.new_top_level_paths(repo, "1.0.0", "release") == ["many/ (2 files)", "newdir/ (1 files)", "notes.txt"]


def test_resolve_import_follows_star_imports_and_flags_dynamic_modules():
    """Re-exports through a star import work, a module-level __getattr__ makes the answer unknown."""
    api = cp.Api()
    api.modules = {
        "pkg": cp.ModuleInfo(names=set(), star_imports=["pkg.impl"], is_package=True),
        "pkg.impl": cp.ModuleInfo(names={"thing"}),
        "dyn": cp.ModuleInfo(names=set(), has_getattr=True),
    }
    assert cp.resolve_import(api, "pkg", "thing") == "ok"
    assert cp.resolve_import(api, "pkg", "other") == "ImportError"
    assert cp.resolve_import(api, "dyn", "anything") == "unknown"
    assert cp.resolve_import(api, "missing", "x") == "ModuleNotFoundError"


def test_pack_contains_every_section_and_the_key_facts(analysis: cp.Analysis):
    """The rendered pack is the prompt-ready text with all sections in the documented order."""
    text, info = cp.build_pack(analysis, budget_tokens=16000)
    headings = [line for line in text.splitlines() if re.match(r"## \d+\. ", line)]
    assert [heading.split(". ", 1)[1].split(" (")[0] for heading in headings] == [
        "RANGE",
        "PUBLIC API CHANGES",
        "SUPPORT MATRIX AND DEPENDENCIES",
        "CI AND TOOLING",
        "NEW TOP-LEVEL PATHS",
        "PULL REQUESTS, oldest first",
        "PREVIOUS CHANGELOG SECTION",
    ]
    assert "from lib.metrics import <2 names>: ImportError at head" in text
    assert "possible rename: add_line -> plot_chance_level" in text
    assert "(!) plot_new: the message says 'lib.strings'" in text
    assert "Co-Authored-By" not in text
    assert "Claude-Session" not in text
    assert "## [1.0.0] - 2026-01-01" in text
    assert info["fits"] is True
    assert info["reduced"] is False


def test_pack_shrinks_to_the_budget_but_keeps_the_facts(analysis: cp.Analysis):
    """A small budget removes bodies and diffs first; the API, packaging and CI facts stay."""
    full, _ = cp.build_pack(analysis, budget_tokens=16000)
    small, info = cp.build_pack(analysis, budget_tokens=cp.estimate_tokens(full) - 120)
    assert info["reduced"] is True
    assert cp.estimate_tokens(small) < cp.estimate_tokens(full)
    assert "ImportError at head" in small
    assert "requires-python: >=3.10 -> >=3.11" in small
    tiny, tiny_info = cp.build_pack(analysis, budget_tokens=10)
    assert tiny_info["fits"] is False
    assert "ImportError at head" in tiny


def test_facts_are_json_serializable_and_list_public_symbols(analysis: cp.Analysis):
    """facts.json carries what later steps need to verify names written by a model."""
    facts = json.loads(json.dumps(cp.facts_for_verification(analysis)))
    assert facts["public_symbols"]["lib.metrics.curves.plot_a"]["params"] == [
        "x",
        "y",
        "plot_chance_level",
        "chance_level_kw",
    ]
    assert facts["moved"][0] == {"old": "lib.metrics.plot_a", "new": "lib.metrics.curves.plot_a"}
    assert [pr["numbers"] for pr in facts["pull_requests"]][0] == [11]


def test_cli_writes_pack_facts_and_info(release_repo: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    """The command line entry point writes the three output files and reports the token estimate."""
    out_dir = tmp_path / "out"
    code = cp.main(["--repo", str(release_repo), "--base", "1.0.0", "--package", "lib", "--out-dir", str(out_dir)])
    assert code == 0
    assert (out_dir / "context_pack.md").read_text(encoding="utf-8").startswith("<release_context")
    assert json.loads((out_dir / "facts.json").read_text(encoding="utf-8"))["base"] == "1.0.0"
    assert json.loads((out_dir / "pack_info.json").read_text(encoding="utf-8"))["fits"] is True
    assert "context pack: ~" in capsys.readouterr().out


def test_cli_in_a_folder_that_is_not_a_repository_exits_with_an_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    """Looking for the newest tag runs git; when git fails the tool prints the reason and exits with code 2."""
    folder = tmp_path / "plain"
    folder.mkdir()
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))  # never find a repository above the temp folder
    assert cp.main(["--repo", str(folder)]) == 2
    assert "error: git tag --list failed" in capsys.readouterr().err


def test_cli_without_a_release_tag_exits_with_an_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    """Without --base and without any release tag the tool stops with exit code 2 and a clear message."""
    _git(tmp_path, "init", "-q")
    assert cp.main(["--repo", str(tmp_path)]) == 2
    assert "no release tag" in capsys.readouterr().err
