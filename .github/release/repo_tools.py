"""Read-only repository tools a model can call while it writes release notes.

Ollama cannot explore a repository on its own; a client has to run the tools the model asks for. This module is that
client side. It answers the questions a person would ask with git while writing the notes: which files changed, what
did the diff of this file look like, what does the file say now, where is this name used, what did this pull request
say, and is this symbol really new in the release.

Safety rules (the model is untrusted and so is every byte it reads from the repository):

* The model never passes a git ref or an option. It chooses ``"base"`` (the previous release) or ``"head"`` (the new
  release), which are resolved once to commit hashes when the tools are created.
* Paths are checked (relative, no ``..``, no pathspec magic) and passed to git as literal pathspecs after ``--``.
* Search text is passed with ``-e`` and ``-F`` (a plain string, not a regular expression).
* Every answer is size-limited so a tool can never flood the context window; larger results are paged by line.
* Nothing here writes: only ``git diff``, ``git show``, ``git grep`` and ``git log`` are used.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parent))
import context_pack as cp  # noqa: E402

MAX_FILE_BYTES = 2_000_000
PAGE_LINES = 120
MAX_PAGE_LINES = 300
MAX_LIST_ROWS = 120
MAX_SEARCH_RESULTS = 30
MAX_LOG_ROWS = 40
MAX_SYMBOL_MATCHES = 8
SHORT_HASH = 8  # length of the commit hashes shown to the model
MIN_HASH_PREFIX = 4  # shortest prefix of a commit hash that is accepted
VERSIONS = ("base", "head")
COMMIT_ID = re.compile(r"^(?:#(?P<explicit>\d{1,6})|(?P<plain>\d{1,6})|(?P<hex>[0-9a-fA-F]{4,40}))$")
DRIVE_LETTER = re.compile(r"^[A-Za-z]:")
ELLIPSIS = "..."


def _function(name: str, description: str, properties: dict[str, dict[str, Any]], required: list[str]) -> dict:
    """Build one tool definition in the format of the Ollama ``tools`` request field."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


VERSION_PROPERTY = {
    "type": "string",
    "enum": list(VERSIONS),
    "description": "'head' is the new release (default), 'base' is the previous release.",
}
PATH_PREFIX_PROPERTY = {"type": "string", "description": "Only this folder or file, for example 'ds_utils/metrics'."}

TOOL_DEFINITIONS = [
    _function(
        "changed_files",
        "List the files that differ between the previous release and the new release, with status (A added, "
        "M modified, D deleted, R renamed) and the number of added and removed lines.",
        {"path_prefix": PATH_PREFIX_PROPERTY},
        [],
    ),
    _function(
        "diff_file",
        "Show the unified diff of one file between the previous release and the new release.",
        {
            "path": {"type": "string", "description": "Repository-relative file path."},
            "start_line": {"type": "integer", "description": "First diff line to show (default 1), for paging."},
        },
        ["path"],
    ),
    _function(
        "read_file",
        "Read a text file at the previous release ('base') or the new release ('head'), with line numbers.",
        {
            "path": {"type": "string", "description": "Repository-relative file path."},
            "version": VERSION_PROPERTY,
            "start_line": {"type": "integer", "description": "First line to show (default 1), for paging."},
        },
        ["path"],
    ),
    _function(
        "search",
        "Find a plain text (case-insensitive, not a regular expression) in the files of a release.",
        {
            "text": {"type": "string", "description": "Text to look for, for example a function name."},
            "version": VERSION_PROPERTY,
            "path_prefix": PATH_PREFIX_PROPERTY,
        },
        ["text"],
    ),
    _function(
        "commit_log",
        "List the merged pull requests (one line each, newest first) of this release, optionally only those that "
        "touched a folder or file.",
        {"path_prefix": PATH_PREFIX_PROPERTY},
        [],
    ),
    _function(
        "show_pull_request",
        "Show the commit message and the changed files of one merged pull request of this release.",
        {"number": {"type": "string", "description": "Pull request number such as 162, or a commit hash."}},
        ["number"],
    ),
    _function(
        "find_symbol",
        "Check a public function or class of the package by name: where it lives now, its parameters, and whether it "
        "is new in this release, moved, or already existed. Use it before you call anything 'new' or 'removed'.",
        {"name": {"type": "string", "description": "Function or class name, for example 'plot_clusters'."}},
        ["name"],
    ),
]


class ToolError(ValueError):
    """The model passed arguments the tool cannot use; the message tells it how to correct the call."""


def safe_path(value: Any) -> str:
    """Return ``value`` as a clean relative path, or raise :class:`ToolError`."""
    if not isinstance(value, str) or not value.strip():
        raise ToolError("path must be a non-empty string")
    path = value.strip().replace("\\", "/")
    while path.startswith("./"):
        path = path[2:]
    segments = path.split("/")
    invalid = (
        path.startswith(("/", "-", ":"))
        or DRIVE_LETTER.match(path)
        or any(segment == ".." for segment in segments)
        or any(char in path for char in "\0\n\r")
    )
    if invalid:
        raise ToolError(f"'{value}' is not a valid repository-relative path")
    return path.rstrip("/") if path != "/" else ""


def _as_int(value: Any, name: str, default: int, *, minimum: int = 1, maximum: int | None = None) -> int:
    """Read an integer argument; small models often send numbers as strings."""
    if value in (None, ""):
        return default
    try:
        number = int(str(value).strip())
    except ValueError as error:
        raise ToolError(f"{name} must be an integer, got '{value}'") from error
    number = max(number, minimum)
    return min(number, maximum) if maximum else number


def _version(value: Any) -> str:
    """Read the ``version`` argument (``base`` or ``head``)."""
    text = str(value or "head").strip().lower()
    if text not in VERSIONS:
        raise ToolError("version must be 'base' or 'head'")
    return text


def _page(lines: list[str], start: int, count: int, *, numbered: bool, limit: int) -> str:
    """Return up to ``count`` lines from ``start`` (1-based), at most ``limit`` characters, with a footer.

    The footer names the lines that are really shown and the line to continue from, so the model can page through a
    file without a gap even when the character limit ends a page early.
    """
    total = len(lines)
    if total == 0:
        return "(empty)"
    if start > total:
        return f"(no more lines; the text has {total} lines)"
    last = min(start - 1 + count, total)
    width = len(str(last))
    # The longest footer this page can get (all lines shown), so the body can never take the room the footer needs.
    footer_size = len(f"[lines {start}-{last} of {total}; call again with start_line={last + 1} for more]") + 1
    budget = max(limit - footer_size, 1)
    body: list[str] = []
    size = 0
    for number in range(start, last + 1):
        text = f"{number:>{width}}  {lines[number - 1]}" if numbered else lines[number - 1]
        if body and size + len(text) + 1 > budget:
            break
        if len(text) > budget:  # one very long line (minified or generated text) still has to fit
            text = text[: budget - len(ELLIPSIS)] + ELLIPSIS if budget > len(ELLIPSIS) else text[:budget]
        body.append(text)
        size += len(text) + 1
    end = start - 1 + len(body)
    footer = f"[lines {start}-{end} of {total}]"
    if end < total:
        footer = f"[lines {start}-{end} of {total}; call again with start_line={end + 1} for more]"
    return "\n".join([*body, footer])


def _cap(text: str, limit: int) -> str:
    """Return ``text`` unchanged when it fits, else cut it (at a line end when one is near) and say so.

    The result, including the note, is never longer than ``limit`` characters.
    """
    if len(text) <= limit:
        return text
    note = f"\n[output cut at {limit} characters; ask for a smaller part]"
    room = max(limit - len(note), 0)
    head = text[:room]
    line_end = head.rfind("\n")
    if line_end >= room // 2:
        head = head[:line_end]
    return (head + note)[:limit]


def _import_note(symbol: cp.Symbol, outcome: str) -> str:
    """Say whether the old import of a symbol still works at head, from :func:`context_pack.resolve_import`."""
    statement = f"from {symbol.module} import {symbol.name}"
    if outcome == "ok":
        return f" ('{statement}' still works at head, so users are not affected)"
    if outcome == "unknown":
        return f" (cannot tell whether '{statement}' still works: the module resolves names dynamically)"
    return ""


def _parse_name_status(raw: str) -> dict[str, tuple[str, str]]:
    """Parse ``git diff --name-status -z -M`` into ``{new path: (status letter, old path)}``."""
    tokens = raw.split("\0")
    result: dict[str, tuple[str, str]] = {}
    index = 0
    while index < len(tokens) and tokens[index]:
        status = tokens[index]
        if status[0] in "RC":
            result[tokens[index + 2]] = (status[0], tokens[index + 1])
            index += 3
        else:
            result[tokens[index + 1]] = (status[0], tokens[index + 1])
            index += 2
    return result


def _parse_numstat(raw: str) -> dict[str, tuple[str, str]]:
    """Parse ``git diff --numstat -z -M`` into ``{new path: (added, deleted)}``."""
    tokens = raw.split("\0")
    result: dict[str, tuple[str, str]] = {}
    index = 0
    while index < len(tokens) and tokens[index]:
        added, deleted, path = tokens[index].split("\t", 2)
        if path:
            result[path] = (added, deleted)
            index += 1
        else:
            result[tokens[index + 2]] = (added, deleted)
            index += 3
    return result


class RepoTools:
    """The tool set for one release range: ``base`` is the previous release, ``head`` the new one."""

    def __init__(self, repo: Path, base: str, head: str, package: str = "ds_utils", *, max_output_chars: int = 6000):
        """Resolve both refs to commit hashes; the model can only choose between them."""
        self.repo = Path(repo)
        self.package = package
        self.max_output_chars = max_output_chars
        self.shas = {
            "base": cp.git(self.repo, "rev-parse", "--verify", f"{base}^{{commit}}").strip(),
            "head": cp.git(self.repo, "rev-parse", "--verify", f"{head}^{{commit}}").strip(),
        }
        self.calls = 0
        self.output_chars = 0
        self.observed: list[str] = []
        self._apis: dict[str, cp.Api] = {}
        self._diff: cp.ApiDiff | None = None
        self._handlers: dict[str, Callable[[dict[str, Any]], str]] = {
            "changed_files": self._changed_files,
            "diff_file": self._diff_file,
            "read_file": self._read_file,
            "search": self._search,
            "commit_log": self._commit_log,
            "show_pull_request": self._show_pull_request,
            "find_symbol": self._find_symbol,
        }

    def schemas(self) -> list[dict[str, Any]]:
        """Return the tool definitions for the ``tools`` field of a chat request."""
        return TOOL_DEFINITIONS

    def call(self, name: str, arguments: dict[str, Any]) -> str:
        """Run one tool and return its text, never longer than ``max_output_chars``.

        Never raises: a problem is returned as ``error: ...`` text, and that text is capped like any other answer
        because it can repeat what the model sent (a very long path, for example).
        """
        self.calls += 1
        handler = self._handlers.get(name)
        try:
            if handler is None:
                raise ToolError(f"unknown tool '{name}'. Available tools: {', '.join(self._handlers)}.")
            text = handler(arguments or {})
        except ToolError as error:
            text = f"error: {error}"
        except RuntimeError as error:
            text = f"error: git could not answer ({str(error)[:160]})"
        text = _cap(text, self.max_output_chars)
        self.output_chars += len(text)
        self.observed.append(text)
        return text

    def _git(self, *args: str) -> str:
        """Run git with literal pathspecs, so a path can never act as a glob or pathspec magic."""
        return cp.git(self.repo, "--literal-pathspecs", *args)

    def _pathspec(self, prefix: Any) -> list[str]:
        """Return the ``-- <path>`` arguments for an optional path prefix."""
        if prefix in (None, ""):
            return []
        path = safe_path(prefix)
        return ["--", path] if path else []

    def _changed_files(self, arguments: dict[str, Any]) -> str:
        spec = self._pathspec(arguments.get("path_prefix"))
        base, head = self.shas["base"], self.shas["head"]
        status = _parse_name_status(self._git("diff", "--name-status", "-z", "-M", base, head, *spec))
        numbers = _parse_numstat(self._git("diff", "--numstat", "-z", "-M", base, head, *spec))
        rows = []
        for path in sorted(status):
            letter, old = status[path]
            added, deleted = numbers.get(path, ("?", "?"))
            suffix = f"  (from {old})" if letter in "RC" else ""
            rows.append(f"{letter}  +{added} -{deleted}  {path}{suffix}")
        if not rows:
            return "(no files differ)"
        shown = rows[:MAX_LIST_ROWS]
        if len(rows) > len(shown):
            shown.append(f"[{len(rows) - len(shown)} more files; pass path_prefix to narrow the list]")
        return f"{len(rows)} files changed\n" + "\n".join(shown)

    def _diff_file(self, arguments: dict[str, Any]) -> str:
        path = safe_path(arguments.get("path"))
        start = _as_int(arguments.get("start_line"), "start_line", 1)
        text = self._git("diff", "-U2", "-M", self.shas["base"], self.shas["head"], "--", path)
        if not text.strip():
            return f"(no differences for '{path}'; if the file moved, diff the new path or use changed_files)"
        lines = [line for line in text.splitlines() if not line.startswith(("index ", "--- ", "+++ "))]
        return _page(lines, start, PAGE_LINES, numbered=False, limit=self.max_output_chars)

    def _read_file(self, arguments: dict[str, Any]) -> str:
        path = safe_path(arguments.get("path"))
        version = _version(arguments.get("version"))
        start = _as_int(arguments.get("start_line"), "start_line", 1)
        size = self._git("cat-file", "-s", f"{self.shas[version]}:{path}") if self._exists(version, path) else ""
        if not size:
            return f"(file '{path}' does not exist at {version})"
        if int(size) > MAX_FILE_BYTES:
            return f"(file '{path}' is {int(size)} bytes, too large to read)"
        text = cp.git_show(self.repo, self.shas[version], path) or ""
        if "\0" in text[:8000]:
            return f"(file '{path}' is binary)"
        return _page(text.splitlines(), start, PAGE_LINES, numbered=True, limit=self.max_output_chars)

    def _exists(self, version: str, path: str) -> bool:
        """Return True when ``path`` is a file at the given version."""
        listing = self._git("ls-tree", "-z", "--name-only", self.shas[version], "--", path)  # -z: names are not quoted
        return path in listing.split("\0")

    def _search(self, arguments: dict[str, Any]) -> str:
        text = str(arguments.get("text") or "").strip()
        if not text or "\n" in text or "\0" in text:
            raise ToolError("text must be a single line of plain text")
        version = _version(arguments.get("version"))
        sha = self.shas[version]
        raw = cp.git(
            self.repo,
            "--literal-pathspecs",
            "grep",
            "-n",
            "-I",
            "-i",
            "-F",
            "-e",
            text,
            sha,
            *self._pathspec(arguments.get("path_prefix")),
            check=False,
        )
        rows = []
        for line in raw.splitlines():
            location = line[len(sha) + 1 :] if line.startswith(f"{sha}:") else line
            path, _, rest = location.partition(":")
            number, _, content = rest.partition(":")
            rows.append(f"{path}:{number}: {content.strip()[:160]}")
        if not rows:
            return f"(no match for '{text}' at {version})"
        shown = rows[:MAX_SEARCH_RESULTS]
        if len(rows) > len(shown):
            shown.append(f"[{len(rows) - len(shown)} more matches; pass path_prefix to narrow the search]")
        return f"{len(rows)} matches at {version}\n" + "\n".join(shown)

    def _range_commits(self, spec: list[str]) -> list[tuple[str, str]]:
        """Return ``(full hash, subject)`` of the first-parent commits of the range, newest first."""
        raw = self._git(
            "log", "--first-parent", "--format=%H%x1f%s", f"{self.shas['base']}..{self.shas['head']}", *spec
        )
        return [tuple(line.split("\x1f", 1)) for line in raw.splitlines() if "\x1f" in line]  # type: ignore[misc]

    def _commit_log(self, arguments: dict[str, Any]) -> str:
        rows = self._range_commits(self._pathspec(arguments.get("path_prefix")))
        if not rows:
            return "(no pull requests touched that path)"
        shown = [f"{sha[:SHORT_HASH]}  {subject}" for sha, subject in rows[:MAX_LOG_ROWS]]
        if len(rows) > len(shown):
            shown.append(f"[{len(rows) - len(shown)} more; pass path_prefix to narrow the list]")
        return f"{len(rows)} pull requests\n" + "\n".join(shown)

    def _show_pull_request(self, arguments: dict[str, Any]) -> str:
        value = str(arguments.get("number") or "").strip()
        match = COMMIT_ID.match(value)
        if not match:
            raise ToolError("number must be a pull request number such as 162 or a commit hash")
        plain = match.group("plain") or ""
        number = match.group("explicit") or plain
        # Digits alone can be a pull request number or the start of a hash; '#162' is always a pull request.
        prefix = (match.group("hex") or (plain if len(plain) >= MIN_HASH_PREFIX else "")).lower()
        commits = self._range_commits([])
        by_number = [sha for sha, subject in commits if number and number in cp.PR_SUFFIX.findall(subject)][:1]
        by_prefix = [sha for sha, _ in commits if prefix and sha.startswith(prefix)]
        candidates = list(dict.fromkeys([*by_number, *by_prefix]))
        if not candidates:
            return f"(no pull request '{value}' in this release; use commit_log to list them)"
        if len(candidates) > 1:
            subjects = dict(commits)
            rows = [f"{sha[:SHORT_HASH]}  {subjects[sha]}" for sha in candidates[:MAX_SYMBOL_MATCHES]]
            return (
                f"'{value}' fits more than one pull request of this release:\n"
                + "\n".join(rows)
                + "\nwrite '#' and the number for a pull request number, or more characters of the hash"
            )
        found = candidates[0]
        subject, _, body = self._git("show", "-s", "--format=%s%x1f%b", found).partition("\x1f")
        status = _parse_name_status(self._git("diff", "--name-status", "-z", "-M", f"{found}^1", found))
        files = [f"{letter}  {path}" for path, (letter, _) in sorted(status.items())]
        shown = files[:60] + ([f"[{len(files) - 60} more files]"] if len(files) > 60 else [])
        cleaned = cp.clean_body(body, 1500) or "(no description)"
        return f"{subject}\n\n{cleaned}\n\nfiles ({len(files)}):\n" + "\n".join(shown)

    def _api(self, version: str) -> cp.Api:
        """Return the public API of a release, computed once."""
        if version not in self._apis:
            self._apis[version] = cp.collect_api(self.repo, self.shas[version], self.package)
        return self._apis[version]

    def _api_diff(self) -> cp.ApiDiff:
        """Return the net public API difference of the range, computed once."""
        if self._diff is None:
            self._diff = cp.diff_api(self._api("base"), self._api("head"))
        return self._diff

    def _find_symbol(self, arguments: dict[str, Any]) -> str:
        name = str(arguments.get("name") or "").strip().strip("`").rsplit(".", 1)[-1].rstrip("()")
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            raise ToolError("name must be a function or class name such as 'plot_clusters'")
        head, base, diff = self._api("head"), self._api("base"), self._api_diff()
        moved = {move.new: move.old for move in diff.moved}
        added = set(diff.added)
        matches = [s for s in head.symbols.values() if s.name == name]
        if not matches:
            # An exact name that only the previous release has is a removal; a partial match must not hide that.
            matches = [s for s in head.symbols.values() if name.lower() in s.name.lower()]
            earlier = sorted((s for s in base.symbols.values() if s.name == name), key=lambda item: item.qualname)
            if earlier:
                # The definition is gone from the public modules, but the old import can still work (re-export).
                outcomes = [cp.resolve_import(head, s.module, s.name) for s in earlier]
                broken = all(outcome not in {"ok", "unknown"} for outcome in outcomes)
                listing = ", ".join(s.qualname + _import_note(s, outcome) for s, outcome in zip(earlier, outcomes))
                headline = "is not public any more" if broken else "is no longer defined in a public module"
                answer = f"'{name}' {headline}; the previous release had {listing}"
                if matches:
                    similar = sorted(s.qualname for s in matches)[:MAX_SYMBOL_MATCHES]
                    answer += f"\nsimilar names at head: {', '.join(similar)}"
                return answer
        rows = []
        for symbol in sorted(matches, key=lambda item: item.qualname)[:MAX_SYMBOL_MATCHES]:
            if symbol.qualname in moved:
                status = f"moved from {moved[symbol.qualname]}"
            elif symbol.qualname in added:
                status = "NEW in this release"
            elif symbol.qualname in base.symbols:
                status = "already existed in the previous release, same module"
            else:
                status = "status unknown"
            flags = f"{symbol.kind}; {status}" + ("; deprecated" if symbol.deprecated else "")
            doc = f"\n    {symbol.doc[:200]}" if symbol.doc else ""
            rows.append(f"{symbol.qualname}{symbol.signature()}  [{flags}]{doc}")
        return "\n".join(rows) if rows else f"(no public function or class named '{name}' in either release)"
