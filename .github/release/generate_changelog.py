"""Draft the CHANGELOG section and the release title of a release with a model served by Ollama.

Three modes decide how the model gets to know the repository:

* ``pack``   the context pack (``context_pack.py``) is attached to the prompt, one request, no tools;
* ``agent``  a lean context pack plus read-only repository tools the model can call (``repo_tools.py``);
* ``hybrid`` the full pack plus the tools, to verify and dig deeper.

The result is printed to the log between marker lines, together with the checks of ``notes_checks.py`` and the numbers
of the run (tokens, speeds, context window). Files are written to ``--out-dir`` and, on GitHub Actions, the same text
is appended to the job summary. Nothing is committed, tagged or published, and no secret is read or printed.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sys
import time
import tomllib
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import context_pack as cp  # noqa: E402
from agent import AgentRun, estimate_tokens, run_agent, tools_unsupported  # noqa: E402
from llm_client import LLMError, OllamaClient  # noqa: E402
from notes_checks import one_line, render_changelog, verify_notes  # noqa: E402
from prompts import NOTES_SCHEMA, system_prompt, user_message  # noqa: E402
from repo_tools import RepoTools  # noqa: E402

MODES = ("pack", "agent", "hybrid")
PACK_TARGETS = {"pack": cp.DEFAULT_BUDGET_TOKENS, "hybrid": 10000, "agent": 6000}
CONTEXT_SHARE = {"pack": 1.0, "hybrid": 0.55, "agent": 0.35}
FULL_WINDOW = 0.98
# Server settings worth recording in the run report. Only these names are read: the environment may hold secrets.
OLLAMA_SERVER_SETTINGS = (
    "OLLAMA_NUM_PARALLEL",
    "OLLAMA_MAX_LOADED_MODELS",
    "OLLAMA_KEEP_ALIVE",
    "OLLAMA_CONTEXT_LENGTH",
    "OLLAMA_FLASH_ATTENTION",
    "OLLAMA_KV_CACHE_TYPE",
)


def log(message: str) -> None:
    """Write a progress line to standard error (standard output carries the result)."""
    print(message, file=sys.stderr, flush=True)


def default_version(repo: Path, head: str, package: str) -> str | None:
    """Return the release version from ``pyproject.toml`` at ``head`` (or ``__version__``), without rc suffixes."""
    text = cp.git_show(repo, head, "pyproject.toml")
    if text:
        try:
            version = tomllib.loads(text).get("project", {}).get("version")
        except tomllib.TOMLDecodeError:
            version = None
        if version:
            return _release_part(str(version))
    init = cp.git_show(repo, head, f"{package}/__init__.py") or ""
    match = re.search(r"__version__\s*=\s*[\"']([^\"']+)[\"']", init)
    return _release_part(match.group(1)) if match else None


def _release_part(version: str) -> str | None:
    """Return ``X.Y.Z`` from a version such as ``1.10.0rc3``."""
    match = re.match(r"^v?(\d+\.\d+\.\d+)", version)
    return match.group(1) if match else None


def pack_budget(mode: str, num_ctx: int, num_predict: int, system: str) -> int:
    """Return the token budget of the context pack: what fits the window after the answer and the instructions."""
    free = int((num_ctx - num_predict - estimate_tokens(system) - 400) * 0.9)
    return max(1000, min(PACK_TARGETS[mode], int(free * CONTEXT_SHARE[mode])))


def build_parser() -> argparse.ArgumentParser:
    """Create the command line parser."""
    parser = argparse.ArgumentParser(description="Draft release notes with an Ollama model and print them to the log.")
    parser.add_argument("--repo", type=Path, default=Path("."), help="repository root (full history and tags needed)")
    parser.add_argument("--base", help="previous release ref (default: newest X.Y.Z tag)")
    parser.add_argument("--head", default="HEAD", help="ref of the release")
    parser.add_argument("--package", default="ds_utils")
    parser.add_argument("--version", help="release version (default: version in pyproject.toml at --head)")
    parser.add_argument("--date", help="release date YYYY-MM-DD (default: today, UTC)")
    parser.add_argument("--model", help="Ollama model name, for example granite4.1:3b")
    parser.add_argument("--mode", choices=MODES, default="pack", help="how the model learns about the repository")
    parser.add_argument("--host", help="Ollama host (default: OLLAMA_HOST or 127.0.0.1:11434)")
    parser.add_argument("--num-ctx", type=int, default=32768, help="context window requested from Ollama")
    parser.add_argument("--num-predict", type=int, default=3000, help="maximum answer tokens")
    parser.add_argument("--num-thread", type=int, default=0, help="CPU threads for the model (0 = Ollama decides)")
    parser.add_argument("--max-rounds", type=int, default=12, help="maximum model calls while exploring (tool modes)")
    parser.add_argument("--think", choices=("auto", "on", "off"), default="auto", help="thinking flag for the model")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--timeout", type=float, default=1800.0, help="seconds to wait for one model answer")
    parser.add_argument("--out-dir", type=Path, default=Path("release-notes-out"))
    parser.add_argument("--dry-run", action="store_true", help="build and size the prompt, do not call a model")
    parser.add_argument("--strict", action="store_true", help="exit with code 3 when a check warns")
    return parser


def _agent_report(run: AgentRun | None) -> dict[str, Any] | None:
    """Return the tool-loop part of the run report."""
    if run is None:
        return None
    return {
        "stop_reason": run.stop_reason,
        "rounds": run.rounds,
        "peak_prompt_tokens": run.peak_prompt_tokens,
        "dropped_tool_calls": run.dropped_calls,
        "exploration_note": run.exploration_note[:2000],
        "steps": [asdict(step) for step in run.steps],
    }


def _server_facts(client: OllamaClient) -> dict[str, Any]:
    """Ask the server what it can tell about the model before the first request; failures become a note."""
    facts: dict[str, Any] = {}
    for key, getter in (
        ("server_version", client.version),
        ("capabilities", client.capabilities),
        ("trained_context", client.trained_context_length),
    ):
        try:
            facts[key] = getter()
        except LLMError as error:
            facts[key] = f"unavailable ({str(error)[:80]})"
    return facts


def _write_summary(text: str) -> None:
    """Append ``text`` to the GitHub Actions job summary when running there."""
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if path:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(text + "\n")


def main(argv: list[str] | None = None) -> int:
    """Run the generator and return a process exit code (0 ok, 1 model failure, 2 setup error, 3 strict warnings)."""
    args = build_parser().parse_args(argv)
    if not args.model and not args.dry_run:
        log("error: --model is required (or use --dry-run)")
        return 2
    repo = args.repo.resolve()
    try:
        base = args.base or cp.latest_release_tag(repo)
    except RuntimeError as error:  # git failed, for example because --repo is not a repository
        log(f"error: {error}")
        return 2
    if not base:
        log("error: no release tag found, pass --base")
        return 2
    version = args.version or default_version(repo, args.head, args.package)
    if not version:
        log("error: cannot read the version from pyproject.toml, pass --version")
        return 2
    date = args.date or datetime.now(timezone.utc).date().isoformat()
    started = time.monotonic()
    try:
        analysis = cp.analyze(repo, base, args.head, args.package)
        tools = RepoTools(repo, base, args.head, args.package)
    except RuntimeError as error:
        log(f"error: {error}")
        return 2
    facts = cp.facts_for_verification(analysis)
    requested = effective = args.mode
    use_tools = effective != "pack"

    client = OllamaClient(
        args.model or "dry-run",
        host=args.host,
        temperature=args.temperature,
        seed=args.seed,
        num_ctx=args.num_ctx,
        num_predict=args.num_predict,
        num_thread=args.num_thread,
        think={"auto": None, "on": True, "off": False}[args.think],
        timeout=args.timeout,
    )
    server: dict[str, Any] = {}
    if not args.dry_run:
        try:
            client.ensure_ready()
        except LLMError as error:
            log(f"error: {error}")
            return 2
        server = _server_facts(client)
        capabilities = server.get("capabilities")
        if use_tools and isinstance(capabilities, list) and capabilities and "tools" not in capabilities:
            log(f"warning: {args.model} does not declare the 'tools' capability ({capabilities}); using mode pack")
            effective, use_tools = "pack", False

    def prepare(mode: str, with_tools: bool) -> tuple[str, str, dict[str, Any]]:
        system = system_prompt(tools=with_tools)
        pack_text, info = cp.build_pack(analysis, pack_budget(mode, args.num_ctx, args.num_predict, system))
        return system, user_message(pack_text, version), {"pack_text": pack_text, **info}

    system, user, pack_info = prepare(effective, use_tools)
    prompt_tokens = estimate_tokens(system) + estimate_tokens(user)
    log(
        f"[generate] {base}..{args.head} -> {version}; mode {effective}; prompt ~{prompt_tokens} tokens of "
        f"{args.num_ctx} (pack ~{pack_info['tokens']}, budget {pack_info['budget']}, fits={pack_info['fits']})"
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pack_file = args.out_dir / "context_pack.md"
    pack_file.write_text(pack_info["pack_text"], encoding="utf-8")
    (args.out_dir / "facts.json").write_text(json.dumps(facts, indent=2), encoding="utf-8")
    if args.dry_run:
        print(f"dry run: prompt ~{prompt_tokens} tokens (context window {args.num_ctx}); files in {args.out_dir}")
        return 0

    run: AgentRun | None = None
    last_result = None
    try:
        if use_tools:
            try:
                run = run_agent(client, tools, system, user, NOTES_SCHEMA, max_rounds=args.max_rounds, log=log)
                notes, last_result = run.final, run.final_result
            except LLMError as error:
                if not tools_unsupported(error):
                    raise
                log(f"warning: {args.model} rejected tools ({error.body[:120]}); using mode pack")
                effective, use_tools, run = "pack", False, None
                system, user, pack_info = prepare(effective, use_tools)
                pack_file.write_text(pack_info["pack_text"], encoding="utf-8")  # the file shows what the model read
        if not use_tools:
            notes, last_result = client.chat_json_messages(
                [{"role": "system", "content": system}, {"role": "user", "content": user}], NOTES_SCHEMA
            )
    except LLMError as error:
        log(f"error: {error}")
        return 1

    try:
        server["loaded_context"] = client.loaded_context_length()  # only known once the model is loaded
    except LLMError as error:
        server["loaded_context"] = f"unavailable ({str(error)[:80]})"
    checks = verify_notes(notes, facts, pack_info["pack_text"], tools.observed)
    changelog = render_changelog(notes, version, date)
    release_title = one_line(notes["release_title"])
    peak = max(run.peak_prompt_tokens if run else 0, last_result.prompt_tokens if last_result else 0)
    usage = client.usage
    report = {
        "model": args.model,
        "requested_mode": requested,
        "mode": effective,
        "version": version,
        "date": date,
        "range": {"base": base, "head": args.head, "base_sha": analysis.base_sha, "head_sha": analysis.head_sha},
        "options": client.options,
        "server": server,
        "server_settings": {name: os.environ[name] for name in OLLAMA_SERVER_SETTINGS if name in os.environ},
        "cpu_count": os.cpu_count(),
        "pack": {key: value for key, value in pack_info.items() if key != "pack_text"},
        "usage": asdict(usage),
        "peak_prompt_tokens": peak,
        "context_window_full": peak >= FULL_WINDOW * args.num_ctx,
        "agent": _agent_report(run),
        "checks": [{"ok": check.ok, "text": check.text} for check in checks],
        "seconds": round(time.monotonic() - started, 1),
        "release_title": release_title,
    }
    (args.out_dir / "changelog.md").write_text(changelog, encoding="utf-8")
    (args.out_dir / "notes.json").write_text(json.dumps(notes, indent=2, ensure_ascii=False), encoding="utf-8")
    (args.out_dir / "run_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    check_lines = [str(check) for check in checks]
    speed = (
        f"read {usage.prompt_tokens} tokens at {usage.prompt_tokens / usage.prompt_seconds:.1f}/s, "
        f"wrote {usage.output_tokens} tokens at {usage.output_tokens / usage.eval_seconds:.1f}/s"
        if usage.prompt_seconds and usage.eval_seconds
        else f"read {usage.prompt_tokens} tokens, wrote {usage.output_tokens} tokens"
    )
    run_lines = [
        f"model {args.model}, mode {effective} (asked for {requested}), {usage.calls} model call(s), "
        f"{usage.tool_calls} tool call(s)",
        f"context window {args.num_ctx} requested, loaded {server.get('loaded_context')}, trained "
        f"{server.get('trained_context')}; largest prompt {peak} tokens"
        + (" (WINDOW FULL: Ollama may have dropped the start of the prompt)" if report["context_window_full"] else ""),
        f"{speed}; total {report['seconds']} s",
        f"threads {args.num_thread or 'left to Ollama'} ({report['cpu_count']} vCPUs here); server settings "
        f"{report['server_settings'] or 'not set in this environment'}",
    ]
    out = [
        "===== RELEASE (copy from here) =====",
        f"Tag: v{version}",
        f"Release title: {release_title}",
        "",
        changelog.rstrip(),
        "===== CHECKS =====",
        *check_lines,
        "===== RUN =====",
        *run_lines,
        "===== END =====",
    ]
    print("\n".join(out))
    _write_summary(
        "\n".join(
            [
                f"### Release notes draft for v{version} ({args.model}, mode {effective})",
                f"**Release title:** {release_title}",
                "",
                "```markdown",
                changelog.rstrip(),
                "```",
                "",
                *(f"- {line}" for line in check_lines),
                "",
                *(f"- {line}" for line in run_lines),
            ]
        )
    )
    return 3 if args.strict and any(not check.ok for check in checks) else 0


if __name__ == "__main__":
    raise SystemExit(main())
