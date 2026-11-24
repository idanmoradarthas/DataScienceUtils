"""Tests for the notes checks and the changelog generator of the release tooling.

A fake Ollama server (see ``fake_ollama.py``) stands in for the model. These tests verify the plumbing, the checks and
the printed result; they cannot say anything about how good a real model's notes are.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

_TESTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_TESTS_DIR.parent / ".github" / "release"))
sys.path.insert(0, str(_TESTS_DIR))
import fake_ollama as fo  # noqa: E402
import generate_changelog as gc  # noqa: E402
import llm_client as lc  # noqa: E402
import notes_checks as nc  # noqa: E402
import prompts  # noqa: E402
import release_repo as rr  # noqa: E402

GOOD_NOTES = {
    "release_title": "Zebra Curves and Faster Statistics",
    "breaking_changes": [],
    "added": [
        {"title": "Zebra Curve", "text": "Added `plot_zebra_curve`, which plots the zebra curve of the values."},
    ],
    "changed": [],
    "fixed": [
        {"title": "Mean Absolute Value", "text": "`mean_abs` now returns the mean absolute value instead of 0."},
    ],
}
SCRIPT = [("changed_files", {}), ("find_symbol", {"name": "plot_zebra_curve"})]


@pytest.fixture
def release_repo(tmp_path: Path) -> Path:
    """Create the synthetic release repository (tag 1.0.0, then ``plot_zebra_curve`` is added)."""
    rr.make_release_repo(tmp_path / "repo")
    return tmp_path / "repo"


# ---------------------------------------------------------------- prompts and rendering


def test_system_prompt_adds_the_tool_guidance_only_for_tool_modes():
    """The tool guidance is part of the prompt only when the model has tools."""
    assert "TOOLS." not in prompts.system_prompt(tools=False)
    assert "find_symbol" in prompts.system_prompt(tools=True)
    assert prompts.system_prompt(tools=True).startswith(prompts.NOTES_SYSTEM)


def test_system_prompt_carries_the_maintainers_style_rules():
    """The standing rules from the 1.10.0 baseline review are in the prompt, as rules and not as release facts."""
    text = prompts.NOTES_SYSTEM
    for fragment in (
        "Data scientists",
        "Not here: renamed parameters",
        "reported here, as a fix",
        "mandatory (blocking)",
        "Leave out: documentation, build system, dependency bumps",
        'only when the API section lists it under "Added public symbols"',
        "untrusted data",
    ):
        assert fragment in text
    assert "1.10.0" not in text
    assert "ds_utils.metrics" not in text


def test_notes_schema_validates_good_notes_and_rejects_bad_ones():
    """The schema requires every section and a title and text for every item."""
    assert lc.validate_schema(prompts.NOTES_SCHEMA, GOOD_NOTES) == []
    broken = {**GOOD_NOTES, "added": [{"title": "x"}]}
    assert lc.validate_schema(prompts.NOTES_SCHEMA, broken)
    assert lc.validate_schema(prompts.NOTES_SCHEMA, {"release_title": "A title with words"})


def test_render_orders_sections_and_leaves_empty_ones_out():
    """Markdown is rendered by code: fixed section order, bullet format, collapsed whitespace, no empty sections."""
    notes = {
        "release_title": "T",
        "breaking_changes": [{"title": "Package Layout", "text": "Imports moved.\n  Use submodules."}],
        "added": [],
        "changed": [{"title": "Plots", "text": "Plots changed."}],
        "fixed": [{"title": "Fix", "text": "Fixed it."}],
    }
    assert nc.render_changelog(notes, "1.2.3", "2026-11-21") == (
        "## [1.2.3] - 2026-11-21\n\n"
        "### Breaking Changes\n\n- **Package Layout**: Imports moved. Use submodules.\n\n"
        "### Changed\n\n- **Plots**: Plots changed.\n\n"
        "### Fixed\n\n- **Fix**: Fixed it.\n"
    )


# ---------------------------------------------------------------- notes checks

FACTS = {
    "public_symbols": {"pkg.curves.plot_zebra_curve": {"params": ["values", "smooth"]}},
    "added": ["pkg.curves.plot_zebra_curve"],
    "legacy_imports": [],
    "packaging": [],
}
PACK = "<release_context>plot_zebra_curve in pkg.curves; extra data-science-utils[umap]; UserWarning</release_context>"


def _failed(checks: list[nc.Check]) -> list[str]:
    return [check.text for check in checks if not check.ok]


def test_a_clean_note_passes_every_check():
    """Good notes about known names produce no warning."""
    assert _failed(nc.verify_notes(GOOD_NOTES, FACTS, PACK + " mean_abs")) == []


def test_names_that_are_not_in_the_context_are_flagged():
    """A backticked name that appears nowhere in the pack, the facts or the tool results is reported."""
    notes = {
        **GOOD_NOTES,
        "added": [{"title": "Invented", "text": "Added `plot_zebra_curve` and `plot_unicorn`, `pkg.curves.nothing`."}],
    }
    problems = _failed(nc.verify_notes(notes, FACTS, PACK + " mean_abs"))
    assert any(
        "plot_unicorn" in text and "pkg.curves.nothing" in text and "plot_zebra_curve" not in text for text in problems
    )


def test_names_seen_only_in_a_tool_result_count_as_known():
    """What the model learned from a tool is allowed in the notes."""
    notes = {**GOOD_NOTES, "fixed": [{"title": "Fix", "text": "`mean_abs` and `sneaky_helper` were corrected."}]}
    assert (
        _failed(nc.verify_notes(notes, FACTS, PACK, observed=["def sneaky_helper(): ...", "def mean_abs(): ..."])) == []
    )
    assert _failed(nc.verify_notes(notes, FACTS, PACK))


def test_non_identifier_tokens_are_matched_verbatim():
    """Extras and quoted values are looked up as text, not as identifiers."""
    notes = {
        **GOOD_NOTES,
        "added": [
            {
                "title": "Extras",
                "text": "Install `data-science-utils[umap]` for plot_zebra_curve; `x[nope]` is not real.",
            }
        ],
    }
    problems = _failed(nc.verify_notes(notes, FACTS, PACK + " mean_abs"))
    assert problems
    assert "x[nope]" in problems[0]
    assert "umap" not in problems[0]


def test_new_public_symbols_that_are_not_mentioned_are_reported():
    """Coverage: every symbol the API diff lists as added must appear in the notes."""
    notes = {**GOOD_NOTES, "added": []}
    assert any("not mentioned: plot_zebra_curve" in text for text in _failed(nc.verify_notes(notes, FACTS, PACK)))


def test_failing_old_imports_require_a_breaking_change():
    """Old import paths that fail at head must show up under breaking changes."""
    facts = {**FACTS, "legacy_imports": [{"statement": "from pkg import a", "outcome": "ImportError"}]}
    assert any(
        "breaking_changes is empty" in text for text in _failed(nc.verify_notes(GOOD_NOTES, facts, PACK + " mean_abs"))
    )
    notes = {**GOOD_NOTES, "breaking_changes": [{"title": "Layout", "text": "Import from the submodules instead."}]}
    assert not any("breaking_changes" in text for text in _failed(nc.verify_notes(notes, facts, PACK + " mean_abs")))


def test_a_python_support_change_must_be_a_breaking_change():
    """A changed ``requires-python`` is reported as breaking only when the text names Python."""
    facts = {**FACTS, "packaging": ["requires-python: >=3.10 -> >=3.11"]}
    assert any("Python support" in text for text in _failed(nc.verify_notes(GOOD_NOTES, facts, PACK + " mean_abs")))
    notes = {
        **GOOD_NOTES,
        "breaking_changes": [{"title": "Supported Versions", "text": "Python 3.10 is no longer supported."}],
    }
    assert not any("Python support" in text for text in _failed(nc.verify_notes(notes, facts, PACK + " mean_abs")))


def test_style_rules_are_checked():
    """Bullets of left-out kinds, pull request numbers, links and a title of the wrong size are warnings."""
    notes = {
        **GOOD_NOTES,
        "release_title": "Tiny",
        "changed": [
            {"title": "Documentation", "text": "Docs were updated, see #12 and https://example.com for details."},
            {"title": "Build System", "text": "Switched to a new build backend for the package."},
        ],
    }
    problems = " | ".join(_failed(nc.verify_notes(notes, FACTS, PACK + " mean_abs")))
    assert "Documentation, Build System" in problems
    assert "pull request numbers and links" in problems
    assert "release title has 1 words" in problems


# ---------------------------------------------------------------- generator


def _generate(repo: Path, out: Path, *extra: str, host: str | None = None) -> int:
    args = ["--repo", str(repo), "--base", "1.0.0", "--head", "HEAD", "--package", "pkg", "--version", "1.1.0"]
    args += ["--date", "2026-11-21", "--out-dir", str(out), *extra]
    if host:
        args += ["--model", "fake:1b", "--host", host]
    return gc.main(args)


def test_version_comes_from_pyproject_and_loses_the_rc_suffix(tmp_path: Path):
    """The release version is read from pyproject.toml at the release ref, or from ``__version__``."""
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main", str(repo)], check=True)
    (repo / "pyproject.toml").write_text('[project]\nname = "x"\nversion = "1.10.0rc3"\n', encoding="utf-8")
    (repo / "pkg").mkdir()
    (repo / "pkg" / "__init__.py").write_text('__version__ = "2.5.1.dev1"\n', encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=T", "-c", "user.email=t@e.com", "commit", "-q", "-m", "x"], check=True
    )
    assert gc.default_version(repo, "HEAD", "pkg") == "1.10.0"
    (repo / "pyproject.toml").unlink()
    subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=T", "-c", "user.email=t@e.com", "commit", "-q", "-m", "y"], check=True
    )
    assert gc.default_version(repo, "HEAD", "pkg") == "2.5.1"


def test_pack_budget_shrinks_with_the_window_and_the_mode():
    """The pack gets what fits next to the answer and the instructions; tool modes keep room for exploration."""
    system = "x" * 7000
    big = {mode: gc.pack_budget(mode, 131072, 3000, system) for mode in gc.MODES}
    assert big == {"pack": 16000, "hybrid": 10000, "agent": 6000}
    small = {mode: gc.pack_budget(mode, 16384, 3000, system) for mode in gc.MODES}
    assert small["pack"] > small["hybrid"] > small["agent"]
    assert small["pack"] < 16000
    assert gc.pack_budget("agent", 2048, 3000, system) == 1000


def test_dry_run_sizes_the_prompt_without_a_model(
    release_repo: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    """``--dry-run`` needs no server and no model; it writes the pack and facts."""
    assert _generate(release_repo, tmp_path / "out", "--dry-run") == 0
    assert "dry run: prompt ~" in capsys.readouterr().out
    assert (tmp_path / "out" / "context_pack.md").read_text(encoding="utf-8").startswith("<release_context")
    assert "plot_zebra_curve" in (tmp_path / "out" / "facts.json").read_text(encoding="utf-8")


def test_pack_mode_prints_the_changelog_and_writes_the_files(
    release_repo: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
):
    """One request with the pack attached; the log gets the release, the checks and the run numbers."""
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    with fo.FakeOllama(fo.ScriptedModel(notes=GOOD_NOTES)) as server:
        code = _generate(release_repo, tmp_path / "out", "--strict", host=server.host)
    out = capsys.readouterr().out
    assert code == 0
    assert out.index("===== RELEASE") < out.index("===== CHECKS") < out.index("===== RUN") < out.index("===== END")
    assert "Tag: v1.1.0" in out
    assert "Release title: Zebra Curves and Faster Statistics" in out
    assert "## [1.1.0] - 2026-11-21" in out
    assert "### Added\n\n- **Zebra Curve**: Added `plot_zebra_curve`" in out
    assert "### Fixed" in out
    assert "### Breaking Changes" not in out
    assert "WARN" not in out
    assert "model fake:1b, mode pack (asked for pack), 1 model call(s), 0 tool call(s)" in out
    assert "context window 32768 requested, loaded 32768" in out
    assert len(server.requests) == 1
    request = server.requests[0]
    assert "tools" not in request
    assert request["format"] == prompts.NOTES_SCHEMA
    assert request["options"]["num_ctx"] == 32768
    assert request["messages"][0]["content"] == prompts.system_prompt(tools=False)
    assert "Write the CHANGELOG section for version 1.1.0." in request["messages"][1]["content"]
    assert "<release_context" in request["messages"][1]["content"]
    files = {path.name for path in (tmp_path / "out").iterdir()}
    assert {"changelog.md", "notes.json", "run_report.json", "context_pack.md", "facts.json"} <= files
    report = json.loads((tmp_path / "out" / "run_report.json").read_text(encoding="utf-8"))
    assert (report["mode"], report["requested_mode"], report["agent"]) == ("pack", "pack", None)
    assert report["server"]["capabilities"] == ["completion", "tools"]
    assert report["context_window_full"] is False
    assert (tmp_path / "out" / "changelog.md").read_text(encoding="utf-8").startswith("## [1.1.0] - 2026-11-21")
    assert summary.read_text(encoding="utf-8").startswith("### Release notes draft for v1.1.0")


def test_threads_and_server_settings_are_sent_and_recorded_without_leaking_other_variables(
    release_repo: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
):
    """The thread count reaches the model; only the known Ollama settings are recorded, never other variables."""
    monkeypatch.setenv("OLLAMA_FLASH_ATTENTION", "1")
    monkeypatch.setenv("OLLAMA_KV_CACHE_TYPE", "q8_0")
    monkeypatch.setenv("OLLAMA_API_KEY", "super-secret-value")
    with fo.FakeOllama(fo.ScriptedModel(notes=GOOD_NOTES)) as server:
        _generate(release_repo, tmp_path / "out", "--num-thread", "4", host=server.host)
    out = capsys.readouterr().out
    assert server.requests[0]["options"]["num_thread"] == 4
    report_text = (tmp_path / "out" / "run_report.json").read_text(encoding="utf-8")
    report = json.loads(report_text)
    assert report["options"]["num_thread"] == 4
    assert report["server_settings"] == {"OLLAMA_FLASH_ATTENTION": "1", "OLLAMA_KV_CACHE_TYPE": "q8_0"}
    assert "threads 4" in out
    assert "OLLAMA_KV_CACHE_TYPE" in out
    assert "super-secret-value" not in report_text + out


def test_threads_are_left_to_ollama_by_default(release_repo: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    """Without --num-thread nothing is sent and the log says so."""
    with fo.FakeOllama(fo.ScriptedModel(notes=GOOD_NOTES)) as server:
        _generate(release_repo, tmp_path / "out", host=server.host)
    assert "num_thread" not in server.requests[0]["options"]
    assert "threads left to Ollama" in capsys.readouterr().out


def test_agent_mode_gives_the_model_tools_and_reports_the_steps(
    release_repo: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    """In agent mode the model explores first; the report lists the tool calls."""
    with fo.FakeOllama(fo.ScriptedModel(notes=GOOD_NOTES, tool_script=SCRIPT)) as server:
        code = _generate(release_repo, tmp_path / "out", "--mode", "agent", host=server.host)
    assert code == 0
    out = capsys.readouterr()
    assert "mode agent (asked for agent), 4 model call(s), 2 tool call(s)" in out.out
    assert "[agent] round 2 step 2: find_symbol(" in out.err
    assert server.requests[0]["messages"][0]["content"] == prompts.system_prompt(tools=True)
    assert "tools" in server.requests[0]
    assert "tools" not in server.requests[-1]
    report = json.loads((tmp_path / "out" / "run_report.json").read_text(encoding="utf-8"))
    assert [step["name"] for step in report["agent"]["steps"]] == ["changed_files", "find_symbol"]
    assert report["agent"]["stop_reason"] == "explored"


def test_a_model_without_the_tools_capability_falls_back_to_the_pack(
    release_repo: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    """The server says the model has no tools: mode agent becomes mode pack, with a warning."""
    with fo.FakeOllama(
        fo.ScriptedModel(notes=GOOD_NOTES, supports_tools=False), capabilities=("completion",)
    ) as server:
        code = _generate(release_repo, tmp_path / "out", "--mode", "hybrid", host=server.host)
    out = capsys.readouterr()
    assert code == 0
    assert "does not declare the 'tools' capability" in out.err
    assert "mode pack (asked for hybrid)" in out.out
    assert all("tools" not in request for request in server.requests)


def test_a_server_that_rejects_tools_after_declaring_them_also_falls_back(
    release_repo: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    """The capability list can be wrong; a 400 about tools triggers the same fallback."""
    with fo.FakeOllama(fo.ScriptedModel(notes=GOOD_NOTES, supports_tools=False)) as server:
        code = _generate(release_repo, tmp_path / "out", "--mode", "agent", host=server.host)
    out = capsys.readouterr()
    assert code == 0
    assert "rejected tools" in out.err
    assert "mode pack (asked for agent)" in out.out
    assert "format" in server.requests[-1]
    assert "tools" not in server.requests[-1]


def test_invented_names_warn_and_strict_turns_the_warning_into_an_exit_code(
    release_repo: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    """A name that is not in the context is a WARN in the log; ``--strict`` makes it exit code 3."""
    invented = {
        **GOOD_NOTES,
        "added": [{"title": "Unicorn", "text": "Added `plot_zebra_curve` and `plot_unicorn` for horses."}],
    }
    with fo.FakeOllama(fo.ScriptedModel(notes=invented)) as server:
        assert _generate(release_repo, tmp_path / "a", host=server.host) == 0
        assert "WARN names not found in the context: plot_unicorn" in capsys.readouterr().out
        assert _generate(release_repo, tmp_path / "b", "--strict", host=server.host) == 3


def test_the_window_full_warning_appears_when_the_prompt_fills_the_context(
    release_repo: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    """A prompt that reaches the window is flagged, because Ollama may have dropped the start of it."""
    with fo.FakeOllama(fo.ScriptedModel(notes=GOOD_NOTES, prompt_tokens_override=32768)) as server:
        _generate(release_repo, tmp_path / "out", host=server.host)
    assert "WINDOW FULL" in capsys.readouterr().out


def test_setup_errors_return_exit_code_2(release_repo: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    """No model, no tag, an unknown model and an unreachable server are setup errors, not crashes."""
    assert _generate(release_repo, tmp_path / "a") == 2
    assert "--model is required" in capsys.readouterr().err
    empty = tmp_path / "empty"
    empty.mkdir()
    subprocess.run(["git", "init", "-q", str(empty)], check=True)
    assert gc.main(["--repo", str(empty), "--model", "fake:1b", "--out-dir", str(tmp_path / "b")]) == 2
    assert "no release tag" in capsys.readouterr().err
    with fo.FakeOllama(fo.ScriptedModel(), models=("other:7b",)) as server:
        assert _generate(release_repo, tmp_path / "c", host=server.host) == 2
        assert "not available" in capsys.readouterr().err
    assert _generate(release_repo, tmp_path / "d", host=server.host) == 2
    assert "Cannot reach Ollama" in capsys.readouterr().err


def test_a_model_that_never_returns_valid_json_gives_exit_code_1(
    release_repo: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    """Two invalid answers end the run with a clear message and exit code 1."""
    with fo.FakeOllama(lambda payload, server: fo.reply("I cannot do that.")) as server:
        assert _generate(release_repo, tmp_path / "out", host=server.host) == 1
    assert "no valid JSON" in capsys.readouterr().err
