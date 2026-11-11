"""Generate pull request report and GitHub annotations for Tier 2 intra-skill dedup.

This script parses SkillEvaluator dedup JSON reports for skills scoped in the PR,
classifies findings (candidate duplicates, refused LLM judgments, incomplete scans),
determines if candidates intersect modified line ranges in the PR, emits GitHub
workflow annotations (up to --max-annotations), and appends a Markdown summary table
and details to --summary-file.

The script always exits 0, even on malformed inputs or missing files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

_MEMBER_RE = re.compile(r"in (\S+) \(lines (\d+)-(\d+)\)")


def _escape_data(text: str) -> str:
    """Escape text for GitHub Actions workflow command messages."""
    return text.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def _escape_property(text: str) -> str:
    """Escape text for GitHub Actions workflow command properties."""
    return text.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A").replace(":", "%3A").replace(",", "%2C")


def _emit_warning(
    message: str,
    title: str = "Tier 2 dedup (advisory)",
    file_path: str | None = None,
    line: int | None = None,
    end_line: int | None = None,
) -> None:
    """Emit a workflow command warning annotation to stdout."""
    props: list[str] = []
    if file_path is not None:
        props.append(f"file={_escape_property(file_path)}")
    if line is not None:
        props.append(f"line={line}")
    if end_line is not None:
        props.append(f"endLine={end_line}")
    props.append(f"title={_escape_property(title)}")
    prop_str = ",".join(props)
    print(f"::warning {prop_str}::{_escape_data(message)}")


def _first_sentence(text: str) -> str:
    """Extract the first sentence from text, ending with a period."""
    text = text.strip()
    if not text:
        return ""
    # Match up to the first period followed by whitespace or end of string
    match = re.search(r"^.*?\.(?:\s|$)", text, re.DOTALL)
    if match:
        return match.group(0).strip()
    return text


def _range_intersects(r1_start: int, r1_end: int, r2_start: int, r2_end: int) -> bool:
    """Return True if two inclusive ranges [r1_start, r1_end] and [r2_start, r2_end] intersect."""
    return max(r1_start, r2_start) <= min(r1_end, r2_end)


def parse_cluster_members(message: str, skill_name: str, skills_dir: str) -> list[tuple[str, int, int]]:
    """Parse cluster members from a dedup finding message.

    Returns a list of (repo_relative_path, start_line, end_line).
    """
    members: list[tuple[str, int, int]] = []
    for match in _MEMBER_RE.finditer(message):
        file_in_skill, s_str, e_str = match.groups()
        repo_path = f"{skills_dir}/{skill_name}/{file_in_skill}"
        members.append((repo_path, int(s_str), int(e_str)))
    return members


class Candidate:
    """Represents a duplicate candidate finding from an intra-skill dedup report."""

    def __init__(
        self,
        skill: str,
        finding: dict,
        members: list[tuple[str, int, int]],
        touches_pr: bool,
    ) -> None:
        """Initialize candidate with finding data and computed PR touch flag."""
        self.skill = skill
        self.finding = finding
        self.members = members
        self.touches_pr = touches_pr
        meta = finding.get("metadata", {})
        self.confidence = meta.get("confidence", "unknown")
        self.reasoning = meta.get("reasoning") or meta.get("rationale") or ""
        self.suggestion = finding.get("suggestion") or ""


def process_dedup_report(
    scope_file: Path,
    reports_dir: Path,
    model: str,
    digest: str,
    summary_file: Path | None,
    max_annotations: int = 10,
) -> None:
    """Process dedup scan reports and produce annotations and Markdown summary."""
    summary_lines: list[str] = ["## Tier 2 dedup (advisory)", ""]

    pin_label = digest[:12] if digest else "unpinned"
    summary_lines.append(f"Judge: `{model}` ({pin_label})")
    summary_lines.append("")

    try:
        if not scope_file.is_file():
            msg = f"Scope file '{scope_file}' not found."
            _emit_warning(msg)
            summary_lines.append(f"Tier 2 advisory scan did not run: {msg}")
            _write_summary(summary_lines, summary_file)
            return

        scope_data = json.loads(scope_file.read_text(encoding="utf-8"))
    except Exception as exc:
        msg = f"Failed to read scope file '{scope_file}': {exc}"
        _emit_warning(msg)
        summary_lines.append(f"Tier 2 advisory scan did not run: {msg}")
        _write_summary(summary_lines, summary_file)
        return

    skills_dir = scope_data.get("skills_dir", "skills")
    changed_lines = scope_data.get("changed_lines", {})
    dedup_skills = scope_data.get("dedup_skills", [])

    if not dedup_skills:
        summary_lines.append("No skills eligible for Tier 2 dedup in this PR.")
        _write_summary(summary_lines, summary_file)
        return

    if not reports_dir.is_dir():
        msg = f"Tier 2 reports directory '{reports_dir}' not found."
        _emit_warning(msg)
        summary_lines.append(f"Tier 2 advisory scan did not run: {msg}")
        _write_summary(summary_lines, summary_file)
        return

    skill_statuses: list[dict] = []
    all_candidates: list[Candidate] = []
    incomplete_msgs: list[str] = []

    for skill in dedup_skills:
        report_path = reports_dir / f"dedup-{skill}" / "skillevaluator-dedup.json"
        if not report_path.is_file():
            msg = f"report missing for {skill}"
            _emit_warning(f"Tier 2 dedup scan report missing for {skill}: file '{report_path}' not found.")
            incomplete_msgs.append(msg)
            skill_statuses.append(
                {
                    "skill": skill,
                    "status": "missing",
                    "on_changed": 0,
                    "elsewhere": 0,
                }
            )
            continue

        try:
            report_data = json.loads(report_path.read_text(encoding="utf-8"))
        except Exception as exc:
            msg = f"report corrupt for {skill}: {exc}"
            _emit_warning(f"Tier 2 dedup scan report corrupt for {skill}: {exc}")
            incomplete_msgs.append(msg)
            skill_statuses.append(
                {
                    "skill": skill,
                    "status": "corrupt",
                    "on_changed": 0,
                    "elsewhere": 0,
                }
            )
            continue

        overall_status = report_data.get("overall_status", "").lower()
        if overall_status == "incomplete":
            inc_detail = ""
            for r in report_data.get("results", []):
                work_limit = r.get("metadata", {}).get("work_limit")
                if work_limit:
                    chk = work_limit.get("check_name", "limit")
                    actual = work_limit.get("actual")
                    limit = work_limit.get("limit")
                    inc_detail = f"tripped work limit {chk} (actual {actual}, limit {limit})"
                    break
                errors = r.get("errors", [])
                if errors:
                    inc_detail = errors[0]
                    break
            if not inc_detail:
                inc_detail = "scan reported incomplete"
            inc_msg = f"Tier 2 dedup scan incomplete for {skill}: {inc_detail}"
            _emit_warning(inc_msg)
            incomplete_msgs.append(f"{skill}: {inc_detail}")

        # Process findings
        skill_candidates_changed = 0
        skill_candidates_elsewhere = 0

        for r in report_data.get("results", []):
            for f in r.get("findings", []):
                check_name = f.get("check_name", "")
                if check_name == "llm_refused":
                    # LLM declined to judge, not a candidate duplicate
                    continue
                meta = f.get("metadata", {})
                if meta.get("confidence") is not None:
                    # Duplicate candidate
                    raw_msg = f.get("message", "")
                    members = parse_cluster_members(raw_msg, skill, skills_dir)
                    touches_pr = False
                    for m_file, m_start, m_end in members:
                        if m_file in changed_lines:
                            for c_start, c_end in changed_lines[m_file]:
                                if _range_intersects(m_start, m_end, c_start, c_end):
                                    touches_pr = True
                                    break
                        if touches_pr:
                            break

                    cand = Candidate(skill=skill, finding=f, members=members, touches_pr=touches_pr)
                    all_candidates.append(cand)
                    if touches_pr:
                        skill_candidates_changed += 1
                    else:
                        skill_candidates_elsewhere += 1

        skill_statuses.append(
            {
                "skill": skill,
                "status": overall_status or "passed",
                "on_changed": skill_candidates_changed,
                "elsewhere": skill_candidates_elsewhere,
            }
        )

    # Output table
    summary_lines.append("| Skill | Status | Candidates on changed lines | Candidates elsewhere |")
    summary_lines.append("|---|---|---|---|")
    for s_info in skill_statuses:
        summary_lines.append(
            f"| `{s_info['skill']}` | {s_info['status']} | {s_info['on_changed']} | {s_info['elsewhere']} |"
        )
    summary_lines.append("")

    if incomplete_msgs:
        for im in incomplete_msgs:
            summary_lines.append(f"- Tier 2 dedup scan incomplete for {im}")
        summary_lines.append("")

    # Annotations for touching candidates
    emitted_annotations = 0
    total_touching_members = 0

    touching_candidates = [c for c in all_candidates if c.touches_pr]
    for cand in touching_candidates:
        sugg_first = _first_sentence(cand.suggestion)
        for idx, (m_file, m_start, m_end) in enumerate(cand.members):
            total_touching_members += 1
            if emitted_annotations < max_annotations:
                other_locs = [
                    f"{other_file}:{other_start}-{other_end}"
                    for o_idx, (other_file, other_start, other_end) in enumerate(cand.members)
                    if o_idx != idx
                ]
                other_str = ", ".join(other_locs) if other_locs else "other locations"
                ann_msg = (
                    f"Possible duplicate of {other_str} (judge {model}, confidence {cand.confidence}). "
                    f"{sugg_first} Advisory: sections of different functions can look alike."
                )
                _emit_warning(
                    message=ann_msg,
                    title="Tier 2 dedup (advisory)",
                    file_path=m_file,
                    line=m_start,
                    end_line=m_end,
                )
                emitted_annotations += 1

    if total_touching_members > max_annotations:
        omitted = total_touching_members - max_annotations
        summary_lines.append(f"> **Note**: {omitted} additional warning annotation(s) were omitted.")
        summary_lines.append("")

    # Candidate details
    if all_candidates:
        summary_lines.append("### Duplicate Candidates")
        summary_lines.append("")
        for cand in all_candidates:
            label = "touches this PR" if cand.touches_pr else "pre-existing"
            member_strs = [f"`{m_file}:{m_start}-{m_end}`" for m_file, m_start, m_end in cand.members]
            summary_lines.append(f"#### `{cand.skill}`: {', '.join(member_strs)} ({label})")
            summary_lines.append(f"- **Confidence**: {cand.confidence}")
            if cand.reasoning:
                summary_lines.append(f"- **Reasoning**: {cand.reasoning[:300]}")
            if cand.suggestion:
                summary_lines.append(f"- **Suggestion**: {cand.suggestion[:200]}")
            summary_lines.append("")
    else:
        summary_lines.append("No duplicate candidates for the skills changed in this PR.")
        summary_lines.append("")

    summary_lines.append(
        "Advisory only: the judge sees headings and text but not function names, so sibling "
        "functions' sections (### Parameters, ### Example) can be reported as duplicates."
    )

    _write_summary(summary_lines, summary_file)


def _write_summary(lines: list[str], summary_file: Path | None) -> None:
    """Write or print Markdown summary lines."""
    content = "\n".join(lines) + "\n"
    if summary_file:
        try:
            with summary_file.open("a", encoding="utf-8") as f:
                f.write(content)
        except Exception as exc:
            print(f"Warning: Could not append to summary file '{summary_file}': {exc}", file=sys.stderr)
            print(content)
    else:
        print(content)


def main() -> None:
    """CLI entry point for pr_dedup_report."""
    parser = argparse.ArgumentParser(description="Generate Tier 2 intra-skill dedup report and annotations.")
    parser.add_argument("--scope", required=True, help="Path to scope.json")
    parser.add_argument("--reports", required=True, help="Path to tier2_reports directory")
    parser.add_argument("--model", required=True, help="LLM model tag")
    parser.add_argument("--digest", default="", help="LLM pinned digest sha256")
    parser.add_argument("--summary-file", help="Path to append Markdown summary")
    parser.add_argument("--max-annotations", type=int, default=10, help="Max annotations to emit (default 10)")

    args = parser.parse_args()

    process_dedup_report(
        scope_file=Path(args.scope),
        reports_dir=Path(args.reports),
        model=args.model,
        digest=args.digest,
        summary_file=Path(args.summary_file) if args.summary_file else None,
        max_annotations=args.max_annotations,
    )


if __name__ == "__main__":
    main()
