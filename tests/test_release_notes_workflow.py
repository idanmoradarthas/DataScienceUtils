"""Guards for ``.github/workflows/release-notes.yml``: the first workflow of the release chain.

The workflow cannot run in the tests (it needs a runner and a model), so these tests read its text. They protect what
matters most: it is listed first, it can only be started by hand, it has no write access and no secret, user input never
reaches a shell script as text, and every file and option it uses exists.
"""

from __future__ import annotations

from pathlib import Path
import re
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "release-notes.yml"
sys.path.insert(0, str(REPO_ROOT / ".github" / "release"))
import generate_changelog as gc  # noqa: E402

TEXT = WORKFLOW.read_text(encoding="utf-8")


def _run_blocks() -> list[str]:
    """Return the text of every ``run:`` step (single line or block scalar)."""
    blocks: list[str] = []
    lines = TEXT.splitlines()
    for index, line in enumerate(lines):
        match = re.match(r"^(\s*)(?:- )?run:\s*(.*)$", line)
        if not match:
            continue
        indent, rest = len(match.group(1)), match.group(2)
        if rest and rest not in {"|", "|-", ">"}:
            blocks.append(rest)
            continue
        body = []
        for follower in lines[index + 1 :]:
            if follower.strip() and len(follower) - len(follower.lstrip()) <= indent:
                break
            body.append(follower)
        blocks.append("\n".join(body))
    return blocks


def test_the_name_puts_it_first_in_the_actions_list_and_says_to_run_it_first():
    """The Actions sidebar lists workflows by name, so a leading ``1.`` puts it before the later release steps."""
    name = re.search(r'^name:\s*"?(.+?)"?\s*$', TEXT, re.MULTILINE)
    assert name
    assert name.group(1).startswith("1. Release")
    assert "run first" in name.group(1).lower()


def test_it_starts_only_by_hand_and_has_read_only_access_and_no_secrets():
    """No push, pull request or schedule trigger, ``contents: read`` only, and no secret or token anywhere."""
    triggers = re.search(r"^on:\n((?:[ \t]+.*\n|\n)+)", TEXT, re.MULTILINE)
    assert triggers
    assert re.findall(r"^  ([a-z_]+):", triggers.group(1), re.MULTILINE) == ["workflow_dispatch"]
    permissions = re.search(r"^permissions:\n((?:  .*\n)+)", TEXT, re.MULTILINE)
    assert permissions
    assert permissions.group(1).strip() == "contents: read"
    assert "secrets." not in TEXT
    assert "GITHUB_TOKEN" not in TEXT


def test_inputs_reach_the_shell_only_through_environment_variables():
    """No ``${{ ... }}`` expression is written inside a ``run:`` script, so an input cannot inject a command."""
    blocks = _run_blocks()
    assert len(blocks) >= 6
    for block in blocks:
        assert "${{" not in block, f"an expression is interpolated into a script:\n{block}"
    for name in ("model", "base", "threads", "max_answer_tokens"):
        assert re.search(rf"^\s+\w+: \$\{{\{{ inputs\.{name} \}}\}}$", TEXT, re.MULTILINE), name


def test_the_model_name_is_validated_before_anything_runs():
    """The first step rejects model names that are not ``name:tag`` shaped, before checkout and before any install."""
    steps = re.findall(r"^      - (?:name: (.+)|uses: (.+))$", TEXT, re.MULTILINE)
    assert steps[0][0] == "Validate the inputs"
    assert "[A-Za-z0-9][A-Za-z0-9._:/@-]" in TEXT


def test_every_action_is_pinned_to_a_version_tag():
    """``uses:`` entries carry a version tag, in line with the other workflows."""
    uses = re.findall(r"^\s+(?:- )?uses: (\S+)$", TEXT, re.MULTILINE)
    assert {"actions/checkout", "actions/setup-python", "actions/upload-artifact"} <= {u.split("@")[0] for u in uses}
    for action in uses:
        assert re.fullmatch(r"[\w./-]+@v\d+(\.\d+)*", action), action


def test_it_uses_the_shared_ollama_installer_and_the_documented_server_settings():
    """The server comes from the pinned installer, one request at a time, one model, flash attention on."""
    assert "bash .github/tier2/install_ollama.sh" in TEXT
    assert "bash .github/tier2/pull_llm.sh" in TEXT
    for setting in (
        'OLLAMA_NUM_PARALLEL: "1"',
        'OLLAMA_MAX_LOADED_MODELS: "1"',
        'OLLAMA_FLASH_ATTENTION: "1"',
        "OLLAMA_KV_CACHE_TYPE: ${{ inputs.kv_cache }}",
        "OLLAMA_CONTEXT_LENGTH: ${{ inputs.num_ctx }}",
    ):
        assert setting in TEXT, setting


def test_every_file_it_calls_exists():
    """Scripts named in the workflow are in the repository."""
    paths = set(re.findall(r"\.github/[\w./-]+\.(?:sh|py)", TEXT))
    assert paths >= {".github/release/generate_changelog.py", ".github/tier2/install_ollama.sh"}
    for path in paths:
        assert (REPO_ROOT / path).is_file(), path


def test_every_generator_option_it_passes_is_accepted_by_the_generator():
    """Renaming an option of ``generate_changelog.py`` must not silently break the workflow."""
    step = re.search(r"- name: Draft the release notes\n(.*?)(?=\n      - name:)", TEXT, re.DOTALL)
    assert step
    used = set(re.findall(r"(?<![\w-])(--[a-z][a-z-]*)", step.group(1)))
    accepted = {option for action in gc.build_parser()._actions for option in action.option_strings}
    assert {"--model", "--mode", "--num-ctx", "--num-predict", "--num-thread", "--think", "--base"} <= used
    assert used <= accepted, used - accepted


def test_the_inputs_offered_for_choices_are_accepted_by_the_generator():
    """The ``mode`` and ``think`` choices are values the generator understands."""
    parser = gc.build_parser()
    actions = {action.dest: action for action in parser._actions}
    for name, dest in (("mode", "mode"), ("think", "think")):
        block = re.search(rf"^      {name}:\n(.*?)(?=^      \w+:\n|^permissions:)", TEXT, re.DOTALL | re.MULTILINE)
        assert block
        options = re.findall(r'^          - "?([\w]+)"?$', block.group(1), re.MULTILINE)
        assert options
        assert set(options) <= set(actions[dest].choices), (name, options)
