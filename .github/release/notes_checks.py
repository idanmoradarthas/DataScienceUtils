"""Render the model's notes as Markdown and check them against the facts computed from the code.

Rendering is deterministic: the model never writes headings, dates or bullet markup. The checks are advisory and
return text lines (``OK`` or ``WARN``) for the log; the caller decides whether a warning fails a run.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from prompts import SECTIONS, SKIPPED_BULLET_TITLES

BACKTICKED = re.compile(r"`([^`\n]+)`")
IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*(?:\(\))?$")
WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
PR_REFERENCE = re.compile(r"(?<!\w)#\d{1,6}\b")
LINK = re.compile(r"https?://|\]\(")


def one_line(text: str) -> str:
    """Return ``text`` on one line: a line break written by the model must not start a heading or a bullet."""
    return " ".join(text.split())


def render_changelog(notes: dict[str, Any], version: str, date: str) -> str:
    """Return the CHANGELOG section for ``version``; empty sections are left out."""
    lines = [f"## [{version}] - {date}"]
    for key, heading in SECTIONS:
        items = notes.get(key) or []
        if not items:
            continue
        lines += ["", f"### {heading}", ""]
        lines += [f"- **{one_line(item['title'])}**: {one_line(item['text'])}" for item in items]
    return "\n".join(lines) + "\n"


def _items(notes: dict[str, Any]) -> list[tuple[str, dict[str, str]]]:
    """Return ``(section key, item)`` for every item of the notes."""
    return [(key, item) for key, _ in SECTIONS for item in notes.get(key) or []]


def notes_text(notes: dict[str, Any]) -> str:
    """Return all the words the model wrote, for searching."""
    return "\n".join(
        [notes.get("release_title", ""), *(f"{item['title']}\n{item['text']}" for _, item in _items(notes))]
    )


def known_words(pack_text: str, facts: dict[str, Any], observed: list[str]) -> set[str]:
    """Return every identifier-like word that appears in the context, the facts or a tool result."""
    words = set(WORD.findall(pack_text))
    for qualname, symbol in facts.get("public_symbols", {}).items():
        words.update(qualname.split("."))
        words.update(symbol.get("params", []))
    for text in observed:
        words.update(WORD.findall(text))
    return words


@dataclass
class Check:
    """One check result for the log."""

    ok: bool
    text: str

    def __str__(self) -> str:
        """Format as ``OK`` or ``WARN`` followed by the message."""
        return f"{'OK  ' if self.ok else 'WARN'} {self.text}"


def verify_notes(
    notes: dict[str, Any], facts: dict[str, Any], pack_text: str, observed: list[str] | None = None
) -> list[Check]:
    """Check names, coverage, breaking changes and style rules of the notes; never raises."""
    observed = observed or []
    text = notes_text(notes)
    words = known_words(pack_text, facts, observed)
    full_text = "\n".join([pack_text, *observed])
    checks: list[Check] = []

    unknown = []
    for token in dict.fromkeys(BACKTICKED.findall(text)):
        name = token.removesuffix("()")
        if IDENTIFIER.match(token):
            found = all(part in words for part in name.split("."))
        else:
            found = token in full_text
        if not found:
            unknown.append(token)
    checks.append(
        Check(
            not unknown,
            "every name in backticks exists in the context"
            if not unknown
            else f"names not found in the context: {', '.join(unknown)}",
        )
    )

    added = [qualname.rsplit(".", 1)[-1] for qualname in facts.get("added", [])]
    missing = [name for name in added if not re.search(rf"(?<!\w){re.escape(name)}(?!\w)", text)]
    checks.append(
        Check(
            not missing,
            f"all {len(added)} new public symbols are mentioned"
            if not missing
            else f"new public symbols not mentioned: {', '.join(missing)}",
        )
    )

    broken = [
        item
        for item in facts.get("legacy_imports", [])
        if item.get("outcome") in {"ImportError", "ModuleNotFoundError"}
    ]
    has_breaking = bool(notes.get("breaking_changes"))
    checks.append(
        Check(
            has_breaking or not broken,
            "breaking changes are reported"
            if has_breaking or not broken
            else f"{len(broken)} old import paths fail at head but breaking_changes is empty",
        )
    )

    python_changed = any(line.startswith("requires-python") for line in facts.get("packaging", []))
    breaking_text = " ".join(item["text"] for item in notes.get("breaking_changes") or [])
    checks.append(
        Check(
            not python_changed or "python" in breaking_text.lower(),
            "Python support change is reported under breaking changes"
            if python_changed
            else "no Python support change in the range",
        )
    )

    skipped = [item["title"] for _, item in _items(notes) if item["title"].strip().lower() in SKIPPED_BULLET_TITLES]
    checks.append(
        Check(
            not skipped,
            "no bullets of left-out kinds" if not skipped else f"bullets the project leaves out: {', '.join(skipped)}",
        )
    )

    noise = [
        name for name, pattern in (("pull request numbers", PR_REFERENCE), ("links", LINK)) if pattern.search(text)
    ]
    checks.append(
        Check(
            not noise, "no pull request numbers or links" if not noise else f"found {' and '.join(noise)} in the text"
        )
    )

    words_in_title = len(notes.get("release_title", "").split())
    checks.append(Check(4 <= words_in_title <= 12, f"release title has {words_in_title} words"))
    return checks
