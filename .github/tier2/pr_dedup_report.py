"""Generate pull request report, annotations and comment content for Tier 2 intra-skill dedup.

Two sub-commands:

``select-second-pass``
    Reads the first judge's dedup reports and lists the skills that need the second judge: skills with at least one
    duplicate candidate that touches the PR (or any candidate when the Tier 2 pipeline itself changed).

``report``
    Parses SkillEvaluator dedup JSON reports for the skills scoped in the PR, classifies findings (candidate duplicates,
    refused LLM judgments, incomplete scans), decides whether each candidate touches the PR (intersects a changed line
    range, or the skill had a file deleted), and, when a second judge ran, whether both judges agree. It emits GitHub
    workflow annotations (up to --max-annotations), appends a Markdown summary to --summary-file and writes a
    machine-readable result (--result-file) with the content for the sticky PR comment and the neutral check row.

A candidate is raised (annotation, PR comment, neutral check row) only when it touches the PR and, in two-judge mode,
both judges flagged the same cluster. The result status is ``clean``, ``findings``, ``cannot_run`` (a scan or judge did
not run or finished incomplete) or ``skipped`` (no skill to scan). The script itself always exits 0.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

_MEMBER_RE = re.compile(r"in (\S+) \(lines (\d+)-(\d+)\)")

COMMENT_MARKER = "<!-- tier2-dedup-advisory -->"
_MAX_COMMENT_ROWS = 15
_ADVISORY_NOTE = (
    "Advisory only: the judge sees headings and text but not function names, so sibling "
    "functions' sections (### Parameters, ### Example) can be reported as duplicates."
)


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


def _members_overlap(a: list[tuple[str, int, int]], b: list[tuple[str, int, int]]) -> bool:
    """Return True if any member of a overlaps a member of b in the same file (same embedding cluster)."""
    return any(fa == fb and _range_intersects(sa, ea, sb, eb) for fa, sa, ea in a for fb, sb, eb in b)


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
        # "single": one judge only; "confirmed": both judges flagged the cluster; "first_only": the second judge
        # scanned the skill and did not flag it; "unverified": the second judge could not scan the skill;
        # "not_cross_checked": the second judge was not needed for this skill.
        self.verdict = "single"
        self.second: Candidate | None = None

    @property
    def raised(self) -> bool:
        """Return True if the candidate touches the PR and every required judge flagged it."""
        return self.touches_pr and self.verdict in {"single", "confirmed"}


class SkillScan:
    """Parsed dedup report of one skill for one judge."""

    def __init__(self, skill: str) -> None:
        """Initialize an empty scan; ``status`` stays ``clean`` until a report says otherwise."""
        self.skill = skill
        self.status = "clean"  # clean | findings | incomplete | missing | corrupt
        self.problem = ""  # why the scan is unusable (missing, corrupt, incomplete)
        self.candidates: list[Candidate] = []
        self.refused: list[dict] = []
        self.other: list[dict] = []

    @property
    def usable(self) -> bool:
        """Return True if the scan produced a complete report."""
        return self.status in {"clean", "findings"}


def _incomplete_detail(report_data: dict) -> str:
    """Describe why a SkillEvaluator dedup report is incomplete."""
    for r in report_data.get("results", []):
        work_limit = r.get("work_limit")
        if work_limit and isinstance(work_limit, dict):
            chk = work_limit.get("check_name", "limit")
            other_items = [f"{k}={v}" for k, v in work_limit.items() if k != "check_name"]
            return f"tripped work limit {chk} ({', '.join(other_items)})"
        legacy = r.get("legacy")
        if legacy and isinstance(legacy, dict):
            errors = legacy.get("errors", [])
            if errors and isinstance(errors, list):
                return str(errors[0])[:300]
    return "scan reported incomplete"


def load_skill_scan(
    skill: str,
    reports_dir: Path,
    scope: dict,
    judge_label: str = "",
) -> SkillScan:
    """Load and classify the dedup report of one skill, emitting warning annotations for unusable reports."""
    skills_dir = scope.get("skills_dir", "skills")
    changed_lines = scope.get("changed_lines", {})
    deletion_skills = set(scope.get("deletion_skills", []))
    who = f" ({judge_label})" if judge_label else ""

    scan = SkillScan(skill)
    report_path = reports_dir / f"dedup-{skill}" / "skillevaluator-dedup.json"
    if not report_path.is_file():
        scan.status = "missing"
        scan.problem = f"Tier 2 dedup report missing for {skill}{who}"
        _emit_warning(f"Tier 2 dedup scan report missing for {skill}{who}: file '{report_path}' not found.")
        return scan

    try:
        report_data = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception as exc:
        scan.status = "corrupt"
        scan.problem = f"Tier 2 dedup report corrupt for {skill}{who}: {exc}"
        _emit_warning(f"Tier 2 dedup scan report corrupt for {skill}{who}: {exc}")
        return scan

    raw_overall_status = str(report_data.get("overall_status", "")).lower()
    if raw_overall_status == "incomplete":
        detail = _incomplete_detail(report_data)
        scan.status = "incomplete"
        scan.problem = f"Tier 2 dedup scan incomplete for {skill}{who}: {detail}"
        _emit_warning(f"Tier 2 dedup scan incomplete for {skill}{who}: {detail}")
        return scan

    scan.status = "findings" if raw_overall_status == "failed" else "clean"
    if raw_overall_status not in {"", "passed", "failed"}:
        scan.status = raw_overall_status

    for r in report_data.get("results", []):
        for f in r.get("findings", []):
            if f.get("check_name", "") == "llm_refused":
                scan.refused.append(f)
                continue
            if f.get("metadata", {}).get("confidence") is None:
                scan.other.append(f)
                continue
            members = parse_cluster_members(f.get("message", ""), skill, skills_dir)
            touches_pr = skill in deletion_skills or any(
                _range_intersects(m_start, m_end, c_start, c_end)
                for m_file, m_start, m_end in members
                for c_start, c_end in changed_lines.get(m_file, [])
            )
            scan.candidates.append(Candidate(skill=skill, finding=f, members=members, touches_pr=touches_pr))
    return scan


def _load_scope(scope_file: Path) -> tuple[dict | None, str]:
    """Read scope.json; return (scope, error message)."""
    try:
        if not scope_file.is_file():
            return None, f"Scope file '{scope_file}' not found."
        return json.loads(scope_file.read_text(encoding="utf-8")), ""
    except Exception as exc:
        return None, f"Failed to read scope file '{scope_file}': {exc}"


def skills_needing_second_pass(scans: dict[str, SkillScan], scope: dict) -> list[str]:
    """Return the skills the first judge flagged in a way that matters for this PR.

    A skill needs the second judge when it has at least one candidate touching the PR. When the Tier 2 pipeline itself
    changed, any candidate counts, so the second-judge step is exercised as well.
    """
    infra = bool(scope.get("infra_changed"))
    needed = []
    for skill, scan in scans.items():
        if any(c.touches_pr or infra for c in scan.candidates):
            needed.append(skill)
    return sorted(needed)


def select_second_pass(scope_file: Path, reports_dir: Path, github_output: Path | None = None) -> list[str]:
    """List skills for the second judge and optionally append GitHub step outputs."""
    scope, _error = _load_scope(scope_file)
    skills: list[str] = []
    if scope is not None:
        scans = {
            skill: load_skill_scan(skill, reports_dir, scope, judge_label="first judge")
            for skill in scope.get("dedup_skills", [])
        }
        skills = skills_needing_second_pass(scans, scope)
    if github_output is not None:
        with github_output.open("a", encoding="utf-8") as fh:
            fh.write(f"run_second_pass={'true' if skills else 'false'}\n")
            fh.write(f"second_pass_skills={' '.join(skills)}\n")
    print(f"Second judge needed for: {' '.join(skills) if skills else '(no skills)'}")
    return skills


def _location_link(member: tuple[str, int, int], repo: str, sha: str, skill_prefix: str = "") -> str:
    """Return a Markdown reference (linked when repo and sha are known) to a cluster member.

    The visible label drops ``skill_prefix`` (``skills/<skill>/``) because the skill is shown in its own column.
    """
    m_file, m_start, m_end = member
    shown = m_file[len(skill_prefix) :] if skill_prefix and m_file.startswith(skill_prefix) else m_file
    label = f"`{shown}:{m_start}-{m_end}`"
    if repo and sha:
        return f"[{label}](https://github.com/{repo}/blob/{sha}/{m_file}#L{m_start}-L{m_end})"
    return label


def _confidence_text(cand: Candidate) -> str:
    """Return confidence of both judges (``0.91 / 0.88``) or of the single judge."""
    if cand.second is not None:
        return f"{cand.confidence} / {cand.second.confidence}"
    return str(cand.confidence)


def _build_comment(result: dict, raised: list[Candidate], first_only: list[Candidate], ctx: dict) -> str:
    """Build the sticky PR comment body for the given status."""
    status = result["status"]
    short_sha = ctx["sha"][:7]
    commit = f" (commit `{short_sha}`)" if short_sha else ""
    judges = ctx["judges_text"]
    lines = [COMMENT_MARKER, "## Tier 2 dedup (advisory)", ""]

    if status == "skipped":
        lines.append(f"No skill files changed in the latest push{commit}; nothing to scan.")
    elif status == "clean":
        lines.append(f"No possible duplicate on lines changed in this PR{commit}. {judges}")
    else:
        if status == "cannot_run":
            lines.append(f"**The dedup scan could not complete{commit}, so this PR has no complete dedup result.**")
            lines.append("")
            lines.extend(f"- {reason}" for reason in result["reasons"])
            if raised:
                lines.append("")
                lines.append("Findings already confirmed:")
        if raised:
            if status == "findings":
                noun = "possible duplicate" + ("" if len(raised) == 1 else "s")
                lines.append(f"**{len(raised)} {noun} on lines changed in this PR**{commit}. {judges}")
            lines.append("")
            lines.append("| Skill | Locations | Confidence | Suggestion |")
            lines.append("|---|---|---|---|")
            for cand in raised[:_MAX_COMMENT_ROWS]:
                prefix = f"{ctx['skills_dir']}/{cand.skill}/"
                locs = " ↔ ".join(_location_link(m, ctx["repo"], ctx["sha"], prefix) for m in cand.members)
                sugg = _first_sentence(cand.suggestion).replace("|", "\\|").replace("\n", " ")
                lines.append(f"| `{cand.skill}` | {locs} | {_confidence_text(cand)} | {sugg} |")
            if len(raised) > _MAX_COMMENT_ROWS:
                lines.append("")
                lines.append(f"…and {len(raised) - _MAX_COMMENT_ROWS} more; see the full report.")
        if first_only:
            lines.append("")
            lines.append(
                "<details><summary>Flagged by the first judge, not confirmed by the second "
                f"({len(first_only)})</summary>"
            )
            lines.append("")
            for cand in first_only[:_MAX_COMMENT_ROWS]:
                prefix = f"{ctx['skills_dir']}/{cand.skill}/"
                locs = " ↔ ".join(_location_link(m, ctx["repo"], ctx["sha"], prefix) for m in cand.members)
                lines.append(f"- `{cand.skill}`: {locs} (confidence {cand.confidence})")
            lines.append("")
            lines.append("</details>")

    lines.append("")
    lines.append(_ADVISORY_NOTE)
    if ctx["run_url"]:
        lines.append(f"This comment is updated on every push. [Full report]({ctx['run_url']})")
    return "\n".join(lines) + "\n"


def process_dedup_report(
    scope_file: Path,
    reports_dir: Path,
    model: str,
    digest: str,
    summary_file: Path | None,
    max_annotations: int = 10,
    *,
    second_reports_dir: Path | None = None,
    second_model: str = "",
    second_digest: str = "",
    blockers: tuple[str, ...] | list[str] = (),
    result_file: Path | None = None,
    repo: str = "",
    sha: str = "",
    run_url: str = "",
) -> dict:
    """Process dedup scan reports; emit annotations, a Markdown summary and the machine-readable result.

    Single-judge mode (``second_model`` empty): every candidate touching the PR is raised.
    Two-judge mode: a candidate touching the PR is raised only if the second judge flagged the same cluster.
    Returns the result dictionary (also written to ``result_file``).
    """
    two_judge = bool(second_model)
    pin_label = digest[:12] if digest else "unpinned"
    summary_lines: list[str] = ["## Tier 2 dedup (advisory)", ""]
    summary_lines.append(f"Judge: `{model}` ({pin_label})")
    if two_judge:
        second_pin = second_digest[:12] if second_digest else "unpinned"
        summary_lines.append("")
        summary_lines.append(
            f"Second judge: `{second_model}` ({second_pin}), run only on skills where the first judge flagged "
            "a candidate that matters for this PR; a candidate is raised only when both judges flag it."
        )
    summary_lines.append("")

    reasons: list[str] = [str(b) for b in blockers]
    judges_text = f"Judges: `{model}`, `{second_model}`." if two_judge else f"Judge: `{model}`."
    ctx = {"repo": repo, "sha": sha, "run_url": run_url, "judges_text": judges_text, "skills_dir": "skills"}

    def finish(status: str, raised: list[Candidate], first_only: list[Candidate], counts: dict) -> dict:
        """Assemble, write and return the result dictionary."""
        result = {
            "status": status,
            "reasons": reasons,
            "raised": len(raised),
            "first_judge_only": len(first_only),
            **counts,
        }
        result["comment_markdown"] = _build_comment(result, raised, first_only, ctx)
        if status == "findings":
            noun = "possible duplicate" + ("" if len(raised) == 1 else "s")
            result["check_title"] = f"{len(raised)} {noun} on changed lines"
        elif status == "cannot_run":
            result["check_title"] = "Scan could not complete"
        else:
            result["check_title"] = "No possible duplicate on changed lines"
        result["check_conclusion"] = {"findings": "neutral", "clean": "success", "cannot_run": "failure"}.get(
            status, "skipped"
        )
        result["comment_needed"] = status in {"findings", "cannot_run"}
        if result_file is not None:
            result_file.parent.mkdir(parents=True, exist_ok=True)
            result_file.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        return result

    scope, scope_error = _load_scope(scope_file)
    if scope is None:
        _emit_warning(scope_error)
        reasons.append(scope_error)
        summary_lines.append(f"Tier 2 advisory scan did not run: {scope_error}")
        summary_lines.extend(f"- {b}" for b in blockers)
        _write_summary(summary_lines, summary_file)
        return finish("cannot_run", [], [], {})

    dedup_skills = scope.get("dedup_skills", [])
    ctx["skills_dir"] = scope.get("skills_dir", "skills")

    if not dedup_skills:
        summary_lines.append("No skills eligible for Tier 2 dedup in this PR.")
        _write_summary(summary_lines, summary_file)
        return finish("skipped", [], [], {})

    if not reports_dir.is_dir():
        msg = f"Tier 2 reports directory '{reports_dir}' not found."
        _emit_warning(msg)
        reasons.append(msg)
        summary_lines.append(f"Tier 2 advisory scan did not run: {msg}")
        summary_lines.extend(f"- {b}" for b in blockers)
        _write_summary(summary_lines, summary_file)
        return finish("cannot_run", [], [], {})

    label = "first judge" if two_judge else ""
    scans = {skill: load_skill_scan(skill, reports_dir, scope, judge_label=label) for skill in dedup_skills}
    reasons.extend(s.problem for s in scans.values() if not s.usable)

    # Second judge: cross-check the candidates of the skills that needed it.
    second_scans: dict[str, SkillScan] = {}
    needed = skills_needing_second_pass(scans, scope) if two_judge else []
    if two_judge:
        second_dir = second_reports_dir if second_reports_dir is not None else Path("__no_second_reports__")
        for skill in needed:
            second_scans[skill] = load_skill_scan(skill, second_dir, scope, judge_label="second judge")
        reasons.extend(s.problem for s in second_scans.values() if not s.usable)

    all_candidates: list[Candidate] = []
    skill_rows: list[dict] = []
    for skill in dedup_skills:
        scan = scans[skill]
        second_scan = second_scans.get(skill)
        for cand in scan.candidates:
            if two_judge:
                if second_scan is None:
                    cand.verdict = "not_cross_checked"
                elif not second_scan.usable:
                    cand.verdict = "unverified"
                else:
                    match = next((c for c in second_scan.candidates if _members_overlap(cand.members, c.members)), None)
                    cand.second = match
                    cand.verdict = "confirmed" if match is not None else "first_only"
            all_candidates.append(cand)
        on_changed = sum(1 for c in scan.candidates if c.touches_pr)
        skill_rows.append(
            {
                "skill": skill,
                "status": scan.status,
                "on_changed": on_changed,
                "elsewhere": len(scan.candidates) - on_changed,
                "confirmed": sum(1 for c in scan.candidates if c.raised),
                "second": (
                    "not needed"
                    if second_scan is None
                    else ("incomplete" if not second_scan.usable else second_scan.status)
                ),
            }
        )

    # Table
    if two_judge:
        summary_lines.append(
            "| Skill | Status | Candidates on changed lines | Candidates elsewhere | Second judge | Confirmed by both |"
        )
        summary_lines.append("|---|---|---|---|---|---|")
        for row in skill_rows:
            summary_lines.append(
                f"| `{row['skill']}` | {row['status']} | {row['on_changed']} | {row['elsewhere']} "
                f"| {row['second']} | {row['confirmed']} |"
            )
    else:
        summary_lines.append("| Skill | Status | Candidates on changed lines | Candidates elsewhere |")
        summary_lines.append("|---|---|---|---|")
        for row in skill_rows:
            summary_lines.append(f"| `{row['skill']}` | {row['status']} | {row['on_changed']} | {row['elsewhere']} |")
    summary_lines.append("")

    unusable = [s.problem for s in list(scans.values()) + list(second_scans.values()) if not s.usable]
    summary_lines.extend(f"- {problem}" for problem in unusable)
    summary_lines.extend(f"- {blocker}" for blocker in blockers)
    if unusable or blockers:
        summary_lines.append("")

    # Annotations: only candidates that are raised (touch the PR and are confirmed in two-judge mode)
    raised = [c for c in all_candidates if c.raised]
    first_only = [c for c in all_candidates if c.touches_pr and c.verdict in {"first_only", "unverified"}]
    emitted_annotations = 0
    total_touching_members = 0
    for cand in raised:
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
                if cand.second is not None:
                    judge_text = f"judges {model} and {second_model}, confidence {_confidence_text(cand)}"
                else:
                    judge_text = f"judge {model}, confidence {cand.confidence}"
                ann_msg = (
                    f"Possible duplicate of {other_str} ({judge_text}). "
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

    verdict_labels = {
        "single": "",
        "confirmed": ", confirmed by both judges",
        "first_only": ", first judge only",
        "unverified": ", second judge unavailable",
        "not_cross_checked": ", not cross-checked",
    }
    if all_candidates:
        summary_lines.append("### Duplicate Candidates")
        summary_lines.append("")
        for cand in all_candidates:
            label_pr = "touches this PR" if cand.touches_pr else "pre-existing"
            member_strs = [f"`{m_file}:{m_start}-{m_end}`" for m_file, m_start, m_end in cand.members]
            summary_lines.append(
                f"#### `{cand.skill}`: {', '.join(member_strs)} ({label_pr}{verdict_labels[cand.verdict]})"
            )
            summary_lines.append(f"- **Confidence**: {_confidence_text(cand)}")
            if cand.reasoning:
                summary_lines.append(f"- **Reasoning**: {cand.reasoning[:300]}")
            if cand.second is not None and cand.second.reasoning:
                summary_lines.append(f"- **Second judge reasoning**: {cand.second.reasoning[:300]}")
            if cand.suggestion:
                summary_lines.append(f"- **Suggestion**: {cand.suggestion[:200]}")
            summary_lines.append("")
    else:
        summary_lines.append("No duplicate candidates for the skills changed in this PR.")
        summary_lines.append("")

    refused_findings = [(s.skill, f, model) for s in scans.values() for f in s.refused]
    refused_findings += [(s.skill, f, second_model) for s in second_scans.values() for f in s.refused]
    if refused_findings:
        summary_lines.append("### LLM declined to judge")
        summary_lines.append("")
        for s_name, f, judge in refused_findings:
            fp = f.get("file_path", "")
            ln = f.get("line_number")
            loc = f"{fp}:{ln}" if ln is not None else fp
            raw_msg = f.get("message", "")
            msg_first = raw_msg.splitlines()[0] if raw_msg else ""
            msg_snippet = msg_first[:200]
            judge_model = f.get("metadata", {}).get("llm_model") or judge
            summary_lines.append(f"- `{s_name}`: `{loc}` - {msg_snippet} - {judge_model}")
        summary_lines.append("")
        summary_lines.append("These overlaps were not classified; review them manually or rerun with another judge.")
        summary_lines.append("")

    other_findings = [(s.skill, f) for s in scans.values() for f in s.other]
    if other_findings:
        summary_lines.append("### Other findings")
        summary_lines.append("")
        for s_name, f in other_findings:
            chk = f.get("check_name", "finding")
            sev = f.get("severity", "unknown")
            raw_msg = f.get("message", "")
            msg_first = raw_msg.splitlines()[0] if raw_msg else ""
            msg_snippet = msg_first[:200]
            summary_lines.append(f"- `{s_name}`: {chk} ({sev}) - {msg_snippet}")
        summary_lines.append("")

    summary_lines.append(_ADVISORY_NOTE)
    _write_summary(summary_lines, summary_file)

    if reasons:
        status = "cannot_run"
    elif raised:
        status = "findings"
    else:
        status = "clean"
    counts = {
        "candidates_total": len(all_candidates),
        "candidates_touching": sum(1 for c in all_candidates if c.touches_pr),
        "second_pass_skills": needed,
    }
    return finish(status, raised, first_only, counts)


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


def _append_outputs(github_output: Path | None, values: dict[str, str]) -> None:
    """Append key=value lines to the GitHub step output file."""
    if github_output is None:
        return
    with github_output.open("a", encoding="utf-8") as fh:
        for key, value in values.items():
            fh.write(f"{key}={value}\n")


def main() -> None:
    """CLI entry point for pr_dedup_report."""
    parser = argparse.ArgumentParser(description="Tier 2 intra-skill dedup report and second-judge selection.")
    sub = parser.add_subparsers(dest="command", required=True)

    sel = sub.add_parser("select-second-pass", help="List the skills that need the second judge.")
    sel.add_argument("--scope", required=True, help="Path to scope.json")
    sel.add_argument("--reports", required=True, help="Path to the first judge's tier2_reports directory")
    sel.add_argument("--github-output", help="Path to append GitHub step outputs")

    rep = sub.add_parser("report", help="Write annotations, the job summary and the result file.")
    rep.add_argument("--scope", required=True, help="Path to scope.json")
    rep.add_argument("--reports", required=True, help="Path to tier2_reports directory")
    rep.add_argument("--model", required=True, help="First judge model tag")
    rep.add_argument("--digest", default="", help="First judge pinned digest sha256")
    rep.add_argument("--second-reports", help="Path to the second judge's reports directory")
    rep.add_argument("--second-model", default="", help="Second judge model tag (enables two-judge mode)")
    rep.add_argument("--second-digest", default="", help="Second judge pinned digest sha256")
    rep.add_argument("--blocker", action="append", default=[], help="Reason the scan could not run (repeatable)")
    rep.add_argument("--summary-file", help="Path to append Markdown summary")
    rep.add_argument("--result-file", help="Path to write the machine-readable result JSON")
    rep.add_argument("--github-output", help="Path to append GitHub step outputs")
    rep.add_argument("--repo", default="", help="owner/name, for links in the PR comment")
    rep.add_argument("--sha", default="", help="PR head commit SHA, for links in the PR comment")
    rep.add_argument("--run-url", default="", help="URL of the workflow run, linked from the PR comment")
    rep.add_argument("--max-annotations", type=int, default=10, help="Max annotations to emit (default 10)")

    args = parser.parse_args()

    if args.command == "select-second-pass":
        select_second_pass(
            Path(args.scope),
            Path(args.reports),
            Path(args.github_output) if args.github_output else None,
        )
        return

    result = process_dedup_report(
        scope_file=Path(args.scope),
        reports_dir=Path(args.reports),
        model=args.model,
        digest=args.digest,
        summary_file=Path(args.summary_file) if args.summary_file else None,
        max_annotations=args.max_annotations,
        second_reports_dir=Path(args.second_reports) if args.second_reports else None,
        second_model=args.second_model,
        second_digest=args.second_digest,
        blockers=args.blocker,
        result_file=Path(args.result_file) if args.result_file else None,
        repo=args.repo,
        sha=args.sha,
        run_url=args.run_url,
    )
    _append_outputs(Path(args.github_output) if args.github_output else None, {"dedup_status": result["status"]})


if __name__ == "__main__":
    main()
