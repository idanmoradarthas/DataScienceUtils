"""Determine changed files, affected skills, and Tier 2 execution scope for a PR.

This script parses git diff output between a base commit and a head revision to identify:
- Changed files and line ranges (new-side)
- Changed skills under the skills directory
- Skills eligible for intra-skill deduplication (dedup_skills)
- Whether Tier 2 similarity (cross-skill) and dedup checks should run
Outputs results as JSON and optionally appends key=value lines to GITHUB_OUTPUT.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess

_HUNK_HEADER_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")
_VALID_SKILL_NAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")

_INFRA_EXACT = {
    ".github/workflows/skills-tier2-pr.yml",
    "eval-skills-tier2.sh",
    "requirements-skills.txt",
}


def _is_infra_path(posix_path: str) -> bool:
    """Return True if posix_path is part of Tier 2 CI/runner infrastructure."""
    if posix_path in _INFRA_EXACT:
        return True
    return posix_path.startswith(".github/tier2/")


def parse_git_diff_output(diff_text: str) -> tuple[dict[str, list[list[int]]], set[str], set[str]]:
    """Parse git diff --no-renames -U0 output into changed lines, touched files, and deleted files.

    Returns:
        (changed_lines, touched_files, deleted_files)
        changed_lines: mapping from POSIX path to list of [start, end] inclusive line ranges.
        touched_files: set of all POSIX paths appearing in the diff.
        deleted_files: set of POSIX paths deleted in the diff (status 'deleted file mode').

    """
    changed_lines: dict[str, list[list[int]]] = {}
    touched_files: set[str] = set()
    deleted_files: set[str] = set()

    current_file: str | None = None
    is_deleted = False

    for line in diff_text.splitlines():
        if line.startswith("diff --git "):
            current_file = None
            is_deleted = False
        elif line.startswith("deleted file mode "):
            is_deleted = True
        elif line.startswith("+++ "):
            raw_path = line[4:].strip()
            if raw_path == "/dev/null":
                current_file = None
            else:
                if raw_path.startswith("b/"):
                    current_file = raw_path[2:]
                else:
                    current_file = raw_path
                touched_files.add(current_file)
                if is_deleted:
                    deleted_files.add(current_file)
                if current_file not in changed_lines:
                    changed_lines[current_file] = []
        elif line.startswith("--- "):
            raw_path = line[4:].strip()
            if is_deleted and raw_path != "/dev/null":
                path_a = raw_path[2:] if raw_path.startswith("a/") else raw_path
                touched_files.add(path_a)
                deleted_files.add(path_a)
        elif line.startswith("@@ ") and current_file is not None and not is_deleted:
            match = _HUNK_HEADER_RE.match(line)
            if match:
                start_str, count_str = match.groups()
                start = int(start_str)
                count = int(count_str) if count_str is not None else 1
                if count > 0:
                    changed_lines[current_file].append([start, start + count - 1])

    return changed_lines, touched_files, deleted_files


def compute_scope(
    base: str,
    head: str = "HEAD",
    skills_dir: str = "skills",
    repo_root: Path | None = None,
) -> dict:
    """Compute Tier 2 scope dictionary comparing base and head revisions."""
    root = repo_root or Path.cwd()
    cmd = ["git", "diff", "--no-renames", "-U0", base, head, "--"]
    res = subprocess.run(cmd, cwd=root, capture_output=True, text=True, check=True)

    changed_lines, touched_files, deleted_files = parse_git_diff_output(res.stdout)

    skills_posix = skills_dir.replace("\\", "/").strip("/")

    changed_skills: set[str] = set()
    skills_with_add_mod: set[str] = set()

    skills_changed = False
    infra_changed = False

    for path in touched_files:
        if path.startswith(skills_posix + "/"):
            skills_changed = True
            rel_under_skills = path[len(skills_posix) + 1 :]
            parts = rel_under_skills.split("/")
            if parts and parts[0]:
                skill_name = parts[0]
                changed_skills.add(skill_name)
                if path not in deleted_files:
                    skills_with_add_mod.add(skill_name)
        elif _is_infra_path(path):
            infra_changed = True

    # dedup_skills: changed skills whose directory exists on disk, has at least one added/modified file,
    # and whose name matches ^[A-Za-z0-9._-]+$. Invalid names are put in skipped_skills.
    dedup_skills: list[str] = []
    skipped_skills: list[str] = []
    for skill_name in sorted(skills_with_add_mod):
        skill_path = root / skills_dir / skill_name
        if skill_path.is_dir():
            if _VALID_SKILL_NAME_RE.match(skill_name):
                dedup_skills.append(skill_name)
            else:
                skipped_skills.append(skill_name)

    run_similarity = skills_changed or infra_changed
    run_dedup = len(dedup_skills) > 0

    return {
        "skills_dir": skills_dir,
        "changed_files": sorted(touched_files),
        "changed_skills": sorted(changed_skills),
        "dedup_skills": dedup_skills,
        "skipped_skills": sorted(skipped_skills),
        "run_similarity": run_similarity,
        "run_dedup": run_dedup,
        "changed_lines": {k: v for k, v in sorted(changed_lines.items()) if v},
    }


def main() -> None:
    """Entry point for pr_scope CLI."""
    parser = argparse.ArgumentParser(description="Compute Tier 2 PR change scope.")
    parser.add_argument("--base", required=True, help="Base commit SHA or branch")
    parser.add_argument("--head", default="HEAD", help="Head commit or revision (default: HEAD)")
    parser.add_argument("--skills-dir", default="skills", help="Directory containing skills (default: skills)")
    parser.add_argument("--output", help="Path to write JSON output file")
    parser.add_argument("--github-output", help="Path to append GitHub Action outputs")

    args = parser.parse_args()

    scope = compute_scope(base=args.base, head=args.head, skills_dir=args.skills_dir)

    scope_json = json.dumps(scope, indent=2)
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(scope_json + "\n", encoding="utf-8")
    else:
        print(scope_json)

    if args.github_output:
        gh_path = Path(args.github_output)
        lines = [
            f"run_similarity={'true' if scope['run_similarity'] else 'false'}\n",
            f"run_dedup={'true' if scope['run_dedup'] else 'false'}\n",
            f"dedup_skills={' '.join(scope['dedup_skills'])}\n",
        ]
        with gh_path.open("a", encoding="utf-8") as f:
            f.writelines(lines)


if __name__ == "__main__":
    main()
