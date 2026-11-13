"""Unit tests for Tier 2 pull request helper scripts.

Tests cover:
- pr_scope.py in temporary git repositories (modifications, additions, deletions, renames, infra, non-skill files,
  invalid skill names)
- pr_dedup_report.py with JSON fixtures (touching vs non-touching candidates, >10 annotations truncation, escaping,
  incomplete scans with real work_limit and legacy.errors shapes, real llm_refused, other findings, corrupt files)
- pr_similarity_report.py (enforced pass/fail, margin near-threshold extraction, missing margin, empty near-threshold)
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

# Anchor repo root
_REPO_ROOT = Path(__file__).resolve().parent.parent

# Import script modules
sys.path.insert(0, str(_REPO_ROOT / ".github" / "tier2"))
import pr_dedup_publish  # noqa: E402
import pr_dedup_report  # noqa: E402
import pr_scope  # noqa: E402
import pr_similarity_report  # noqa: E402


def _run_git(args: list[str], cwd: Path) -> str:
    """Run git command in directory cwd with core.autocrlf=false."""
    base_cmd = ["git", "-c", "core.autocrlf=false"]
    res = subprocess.run(base_cmd + args, cwd=cwd, capture_output=True, text=True, check=True)
    return res.stdout


@pytest.fixture
def temp_git_repo(tmp_path: Path) -> Path:
    """Create a minimal git repo with initial commit and LF line endings."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _run_git(["init", "-b", "master"], cwd=repo)
    _run_git(["config", "user.name", "Test User"], cwd=repo)
    _run_git(["config", "user.email", "test@example.com"], cwd=repo)

    # Initial structure
    skills_dir = repo / "skills" / "skill-a"
    skills_dir.mkdir(parents=True)
    (skills_dir / "SKILL.md").write_text("# Skill A\nLine 2\nLine 3\nLine 4\n", encoding="utf-8", newline="\n")
    (skills_dir / "api.md").write_text("API doc\n", encoding="utf-8", newline="\n")

    skill_b_dir = repo / "skills" / "skill-b"
    skill_b_dir.mkdir(parents=True)
    (skill_b_dir / "SKILL.md").write_text("# Skill B\n", encoding="utf-8", newline="\n")

    infra_dir = repo / ".github" / "tier2"
    infra_dir.mkdir(parents=True)
    (infra_dir / "pins.env").write_text("TEST=1\n", encoding="utf-8", newline="\n")

    (repo / "README.md").write_text("Readme\n", encoding="utf-8", newline="\n")

    _run_git(["add", "."], cwd=repo)
    _run_git(["commit", "-m", "Initial commit"], cwd=repo)
    return repo


class TestPrScope:
    """Tests for pr_scope.py."""

    def test_modify_skill_file(self, temp_git_repo: Path) -> None:
        """Modify existing skill file: runs similarity and dedup for skill-a, parses hunk ranges."""
        skill_file = temp_git_repo / "skills" / "skill-a" / "SKILL.md"
        skill_file.write_text("# Skill A\nModified Line 2\nLine 3\nLine 4\nAdded 5\n", encoding="utf-8", newline="\n")
        _run_git(["commit", "-am", "Modify skill-a"], cwd=temp_git_repo)

        scope = pr_scope.compute_scope(base="HEAD~1", head="HEAD", repo_root=temp_git_repo)
        assert scope["run_similarity"] is True
        assert scope["run_dedup"] is True
        assert scope["changed_skills"] == ["skill-a"]
        assert scope["dedup_skills"] == ["skill-a"]
        assert scope["skipped_skills"] == []
        assert "skills/skill-a/SKILL.md" in scope["changed_lines"]
        ranges = scope["changed_lines"]["skills/skill-a/SKILL.md"]
        assert len(ranges) >= 1
        # Range should be inclusive [start, end]
        assert all(r[0] <= r[1] for r in ranges)

    def test_add_new_skill(self, temp_git_repo: Path) -> None:
        """Add a brand new skill: flags True and dedup_skills contains new skill."""
        new_skill = temp_git_repo / "skills" / "skill-c"
        new_skill.mkdir(parents=True)
        (new_skill / "SKILL.md").write_text("# Skill C\nContent\n", encoding="utf-8", newline="\n")
        _run_git(["add", "."], cwd=temp_git_repo)
        _run_git(["commit", "-m", "Add skill-c"], cwd=temp_git_repo)

        scope = pr_scope.compute_scope(base="HEAD~1", head="HEAD", repo_root=temp_git_repo)
        assert scope["run_similarity"] is True
        assert scope["run_dedup"] is True
        assert "skill-c" in scope["changed_skills"]
        assert "skill-c" in scope["dedup_skills"]
        assert scope["skipped_skills"] == []

    def test_delete_skill(self, temp_git_repo: Path) -> None:
        """Delete an entire skill: run_similarity is True, but run_dedup is False (dir deleted)."""
        _run_git(["rm", "-r", "skills/skill-b"], cwd=temp_git_repo)
        _run_git(["commit", "-m", "Remove skill-b"], cwd=temp_git_repo)

        scope = pr_scope.compute_scope(base="HEAD~1", head="HEAD", repo_root=temp_git_repo)
        assert scope["run_similarity"] is True
        assert scope["run_dedup"] is False
        assert scope["changed_skills"] == ["skill-b"]
        assert scope["dedup_skills"] == []
        assert scope["skipped_skills"] == []

    def test_infra_only_change(self, temp_git_repo: Path) -> None:
        """Change only infra file: similarity runs and dedup scans every skill so the pipeline is exercised."""
        pins = temp_git_repo / ".github" / "tier2" / "pins.env"
        pins.write_text("TEST=2\n", encoding="utf-8", newline="\n")
        _run_git(["commit", "-am", "Update pins.env"], cwd=temp_git_repo)

        scope = pr_scope.compute_scope(base="HEAD~1", head="HEAD", repo_root=temp_git_repo)
        assert scope["run_similarity"] is True
        assert scope["run_dedup"] is True
        assert scope["infra_changed"] is True
        assert scope["changed_skills"] == []
        assert scope["dedup_skills"] == ["skill-a", "skill-b"]
        assert scope["deletion_skills"] == []
        assert scope["skipped_skills"] == []

    def test_infra_change_skips_hidden_directories_and_files(self, temp_git_repo: Path) -> None:
        """Infra change scans every real skill directory but not dot-directories or loose files."""
        (temp_git_repo / "skills" / ".hidden").mkdir()
        (temp_git_repo / "skills" / ".hidden" / "x.md").write_text("x\n", encoding="utf-8", newline="\n")
        (temp_git_repo / "skills" / "README.md").write_text("readme\n", encoding="utf-8", newline="\n")
        _run_git(["add", "."], cwd=temp_git_repo)
        _run_git(["commit", "-m", "Add noise"], cwd=temp_git_repo)
        (temp_git_repo / "eval-skills-tier2.sh").write_text("#!/bin/sh\n", encoding="utf-8", newline="\n")
        _run_git(["add", "."], cwd=temp_git_repo)
        _run_git(["commit", "-m", "Touch runner"], cwd=temp_git_repo)

        scope = pr_scope.compute_scope(base="HEAD~1", head="HEAD", repo_root=temp_git_repo)
        assert scope["infra_changed"] is True
        assert scope["dedup_skills"] == ["skill-a", "skill-b"]

    def test_delete_file_in_remaining_skill_runs_dedup(self, temp_git_repo: Path) -> None:
        """Deleting one file of a skill that remains still scans that skill and records the deletion."""
        _run_git(["rm", "skills/skill-a/api.md"], cwd=temp_git_repo)
        _run_git(["commit", "-m", "Delete api.md"], cwd=temp_git_repo)

        scope = pr_scope.compute_scope(base="HEAD~1", head="HEAD", repo_root=temp_git_repo)
        assert scope["run_similarity"] is True
        assert scope["run_dedup"] is True
        assert scope["changed_skills"] == ["skill-a"]
        assert scope["dedup_skills"] == ["skill-a"]
        assert scope["deletion_skills"] == ["skill-a"]
        assert scope["infra_changed"] is False

    def test_modify_without_deletion_has_no_deletion_skills(self, temp_git_repo: Path) -> None:
        """A plain edit does not mark the skill as having a deletion."""
        (temp_git_repo / "skills" / "skill-a" / "SKILL.md").write_text(
            "# Skill A\nNew\n", encoding="utf-8", newline="\n"
        )
        _run_git(["commit", "-am", "Edit"], cwd=temp_git_repo)

        scope = pr_scope.compute_scope(base="HEAD~1", head="HEAD", repo_root=temp_git_repo)
        assert scope["deletion_skills"] == []

    def test_non_skill_change(self, temp_git_repo: Path) -> None:
        """Change unrelated file: neither similarity nor dedup run."""
        readme = temp_git_repo / "README.md"
        readme.write_text("Updated Readme\n", encoding="utf-8", newline="\n")
        _run_git(["commit", "-am", "Update readme"], cwd=temp_git_repo)

        scope = pr_scope.compute_scope(base="HEAD~1", head="HEAD", repo_root=temp_git_repo)
        assert scope["run_similarity"] is False
        assert scope["run_dedup"] is False
        assert scope["changed_skills"] == []
        assert scope["dedup_skills"] == []
        assert scope["skipped_skills"] == []

    def test_pure_deletion_hunk(self) -> None:
        """Hunk with count 0 (e.g. @@ -5,2 +5,0 @@) records no changed line ranges."""
        diff_text = """diff --git a/skills/skill-a/SKILL.md b/skills/skill-a/SKILL.md
index 111..222 100644
--- a/skills/skill-a/SKILL.md
+++ b/skills/skill-a/SKILL.md
@@ -5,2 +5,0 @@
-line to remove 1
-line to remove 2
"""
        changed_lines, touched_files, _deleted_files = pr_scope.parse_git_diff_output(diff_text)
        assert "skills/skill-a/SKILL.md" in touched_files
        assert changed_lines.get("skills/skill-a/SKILL.md") == []

    def test_invalid_skill_name_skipped(self, temp_git_repo: Path) -> None:
        """Skill directory named 'bad name' is excluded from dedup_skills and listed in skipped_skills."""
        bad_dir = temp_git_repo / "skills" / "bad name"
        bad_dir.mkdir(parents=True)
        (bad_dir / "SKILL.md").write_text("# Bad Name\n", encoding="utf-8", newline="\n")
        _run_git(["add", "."], cwd=temp_git_repo)
        _run_git(["commit", "-m", "Add bad name skill"], cwd=temp_git_repo)

        scope = pr_scope.compute_scope(base="HEAD~1", head="HEAD", repo_root=temp_git_repo)
        assert scope["run_similarity"] is True
        assert "bad name" in scope["changed_skills"]
        assert "bad name" not in scope["dedup_skills"]
        assert scope["skipped_skills"] == ["bad name"]


class TestPrDedupReport:
    """Tests for pr_dedup_report.py."""

    def test_touching_and_non_touching_candidates(self, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
        """Distinguish touching candidates from pre-existing candidates based on changed_lines."""
        scope_data = {
            "skills_dir": "skills",
            "dedup_skills": ["skill-a"],
            "changed_lines": {
                "skills/skill-a/references/api.md": [[90, 106]],
            },
        }
        scope_file = tmp_path / "scope.json"
        scope_file.write_text(json.dumps(scope_data), encoding="utf-8")

        reports_dir = tmp_path / "reports"
        skill_dir = reports_dir / "dedup-skill-a"
        skill_dir.mkdir(parents=True)

        report_data = {
            "overall_status": "failed",
            "results": [
                {
                    "findings": [
                        {
                            "category": "DUPLICATE",
                            "severity": "high",
                            "check_name": "duplicate",
                            "message": (
                                "Duplicate content found within references/api.md:\n"
                                '  "### Parameters" in references/api.md (lines 90-106)\n'
                                '  vs "### Parameters" in references/api.md (lines 134-146)'
                            ),
                            "suggestion": "Consolidate parameters. Additional info.",
                            "metadata": {
                                "confidence": 0.95,
                                "reasoning": "Identical parameters section",
                            },
                        },
                        {
                            "category": "DUPLICATE",
                            "severity": "medium",
                            "check_name": "duplicate",
                            "message": (
                                "Duplicate content found within references/api.md:\n"
                                '  "### Notes" in references/api.md (lines 200-210)\n'
                                '  vs "### Notes" in references/api.md (lines 220-230)'
                            ),
                            "suggestion": "Consolidate notes.",
                            "metadata": {
                                "confidence": 0.85,
                                "reasoning": "Identical notes section",
                            },
                        },
                    ]
                }
            ],
        }
        (skill_dir / "skillevaluator-dedup.json").write_text(json.dumps(report_data), encoding="utf-8")

        summary_file = tmp_path / "summary.md"
        pr_dedup_report.process_dedup_report(
            scope_file=scope_file,
            reports_dir=reports_dir,
            model="granite4.1:3b",
            digest="abcdef1234567890",
            summary_file=summary_file,
            max_annotations=10,
        )

        captured = capsys.readouterr()
        # Exactly 2 annotations emitted for candidate 1 (which has 2 members touching changed lines)
        assert captured.out.count("::warning") == 2
        assert "file=skills/skill-a/references/api.md,line=90,endLine=106" in captured.out
        assert "file=skills/skill-a/references/api.md,line=134,endLine=146" in captured.out

        summary_text = summary_file.read_text(encoding="utf-8")
        assert "| `skill-a` | findings | 1 | 1 |" in summary_text
        assert "(touches this PR)" in summary_text
        assert "(pre-existing)" in summary_text
        assert "Judge: `granite4.1:3b` (abcdef123456)" in summary_text

    def test_max_annotations_truncation(self, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
        """When touching members exceed max_annotations, truncate to limit and note in summary."""
        scope_data = {
            "skills_dir": "skills",
            "dedup_skills": ["skill-a"],
            "changed_lines": {
                "skills/skill-a/ref.md": [[1, 1000]],
            },
        }
        scope_file = tmp_path / "scope.json"
        scope_file.write_text(json.dumps(scope_data), encoding="utf-8")

        reports_dir = tmp_path / "reports"
        skill_dir = reports_dir / "dedup-skill-a"
        skill_dir.mkdir(parents=True)

        # Create finding with 12 members
        members_str = "\n  vs ".join([f'"Section {i}" in ref.md (lines {i * 10}-{i * 10 + 5})' for i in range(12)])
        report_data = {
            "overall_status": "failed",
            "results": [
                {
                    "findings": [
                        {
                            "category": "DUPLICATE",
                            "severity": "high",
                            "check_name": "duplicate",
                            "message": f"Duplicate content found within ref.md:\n  {members_str}",
                            "suggestion": "Merge sections.",
                            "metadata": {"confidence": 0.90},
                        }
                    ]
                }
            ],
        }
        (skill_dir / "skillevaluator-dedup.json").write_text(json.dumps(report_data), encoding="utf-8")

        summary_file = tmp_path / "summary.md"
        pr_dedup_report.process_dedup_report(
            scope_file=scope_file,
            reports_dir=reports_dir,
            model="granite4.1:3b",
            digest="",
            summary_file=summary_file,
            max_annotations=10,
        )

        captured = capsys.readouterr()
        assert captured.out.count("::warning") == 10

        summary_text = summary_file.read_text(encoding="utf-8")
        assert "2 additional warning annotation(s) were omitted" in summary_text
        assert "unpinned" in summary_text

    def test_escaping_workflow_commands(self, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
        """Verify %, newline, carriage return, colon, and comma are escaped properly."""
        scope_data = {
            "skills_dir": "skills",
            "dedup_skills": ["skill-a"],
            "changed_lines": {"skills/skill-a/ref:1,2.md": [[10, 20]]},
        }
        scope_file = tmp_path / "scope.json"
        scope_file.write_text(json.dumps(scope_data), encoding="utf-8")

        reports_dir = tmp_path / "reports"
        skill_dir = reports_dir / "dedup-skill-a"
        skill_dir.mkdir(parents=True)

        report_data = {
            "overall_status": "failed",
            "results": [
                {
                    "findings": [
                        {
                            "category": "DUPLICATE",
                            "severity": "high",
                            "check_name": "duplicate",
                            "message": (
                                "Duplicate content found within ref:1,2.md:\n"
                                '  "H1" in ref:1,2.md (lines 10-20)\n'
                                '  vs "H2" in ref:1,2.md (lines 30-40)'
                            ),
                            "suggestion": "Suggestion with 100% discount\nand newline.",
                            "metadata": {"confidence": 0.99},
                        }
                    ]
                }
            ],
        }
        (skill_dir / "skillevaluator-dedup.json").write_text(json.dumps(report_data), encoding="utf-8")

        pr_dedup_report.process_dedup_report(
            scope_file=scope_file,
            reports_dir=reports_dir,
            model="test-model",
            digest="pin123",
            summary_file=None,
        )

        captured = capsys.readouterr()
        # In properties, : -> %3A and , -> %2C
        assert "file=skills/skill-a/ref%3A1%2C2.md" in captured.out
        # In message, % -> %25 and \n -> %0A
        assert "100%25 discount%0Aand newline." in captured.out

    def test_incomplete_scan_work_limit_real_shape(self, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
        """Handle real SkillEvaluator work_limit shape when overall_status is incomplete."""
        scope_data = {
            "skills_dir": "skills",
            "dedup_skills": ["skill-a"],
            "changed_lines": {"skills/skill-a/SKILL.md": [[1, 10]]},
        }
        scope_file = tmp_path / "scope.json"
        scope_file.write_text(json.dumps(scope_data), encoding="utf-8")

        reports_dir = tmp_path / "reports"
        skill_dir = reports_dir / "dedup-skill-a"
        skill_dir.mkdir(parents=True)

        report_data = {
            "overall_status": "incomplete",
            "results": [
                {
                    "status": "incomplete",
                    "findings": [],
                    "work_limit": {
                        "check_name": "chunk_count_limit",
                        "file_path": "SKILL.md",
                        "actual": 120,
                        "limit": 100,
                    },
                }
            ],
        }
        (skill_dir / "skillevaluator-dedup.json").write_text(json.dumps(report_data), encoding="utf-8")

        summary_file = tmp_path / "summary.md"
        pr_dedup_report.process_dedup_report(
            scope_file=scope_file,
            reports_dir=reports_dir,
            model="granite4.1:3b",
            digest="abc",
            summary_file=summary_file,
        )

        captured = capsys.readouterr()
        expected_msg = (
            "::warning title=Tier 2 dedup (advisory)::Tier 2 dedup scan incomplete for skill-a: "
            "tripped work limit chunk_count_limit (file_path=SKILL.md, actual=120, limit=100)"
        )
        assert expected_msg in captured.out
        summary_text = summary_file.read_text(encoding="utf-8")
        expected_summary = (
            "- Tier 2 dedup scan incomplete for skill-a: "
            "tripped work limit chunk_count_limit (file_path=SKILL.md, actual=120, limit=100)"
        )
        assert expected_summary in summary_text
        assert "| `skill-a` | incomplete | 0 | 0 |" in summary_text

    def test_incomplete_scan_legacy_errors_real_shape(self, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
        """Handle real SkillEvaluator legacy.errors shape when overall_status is incomplete."""
        scope_data = {
            "skills_dir": "skills",
            "dedup_skills": ["skill-a"],
            "changed_lines": {"skills/skill-a/SKILL.md": [[1, 10]]},
        }
        scope_file = tmp_path / "scope.json"
        scope_file.write_text(json.dumps(scope_data), encoding="utf-8")

        reports_dir = tmp_path / "reports"
        skill_dir = reports_dir / "dedup-skill-a"
        skill_dir.mkdir(parents=True)

        err_msg = (
            "LLM analysis did not complete for 4 of 4 content clusters (provider: openai-compatible; "
            "model: granite4.1:3b). 4 cluster(s): LLM connection failed. Check provider connectivity "
            "and endpoint configuration."
        )
        report_data = {
            "overall_status": "incomplete",
            "results": [
                {
                    "status": "incomplete",
                    "findings": [],
                    "legacy": {
                        "errors": [err_msg],
                    },
                    "llm_analysis": {
                        "clusters_total": 4,
                        "clusters_completed": 0,
                        "clusters_refused": 0,
                        "clusters_failed": 4,
                    },
                }
            ],
        }
        (skill_dir / "skillevaluator-dedup.json").write_text(json.dumps(report_data), encoding="utf-8")

        summary_file = tmp_path / "summary.md"
        pr_dedup_report.process_dedup_report(
            scope_file=scope_file,
            reports_dir=reports_dir,
            model="granite4.1:3b",
            digest="abc",
            summary_file=summary_file,
        )

        captured = capsys.readouterr()
        assert f"::warning title=Tier 2 dedup (advisory)::Tier 2 dedup scan incomplete for skill-a: {err_msg}" in (
            captured.out
        )
        summary_text = summary_file.read_text(encoding="utf-8")
        assert f"- Tier 2 dedup scan incomplete for skill-a: {err_msg}" in summary_text
        assert "| `skill-a` | incomplete | 0 | 0 |" in summary_text

    def test_llm_refused_and_other_findings_summary(self, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
        """Verify real llm_refused and other findings appear in summary, not in annotations or candidates."""
        scope_data = {
            "skills_dir": "skills",
            "dedup_skills": ["skill-a"],
            "changed_lines": {"skills/skill-a/references/api_reference.md": [[90, 106]]},
        }
        scope_file = tmp_path / "scope.json"
        scope_file.write_text(json.dumps(scope_data), encoding="utf-8")

        reports_dir = tmp_path / "reports"
        skill_dir = reports_dir / "dedup-skill-a"
        skill_dir.mkdir(parents=True)

        report_data = {
            "overall_status": "passed",
            "results": [
                {
                    "findings": [
                        {
                            "category": "CONTENT_DEDUP",
                            "severity": "medium",
                            "check_name": "llm_refused",
                            "message": (
                                "The LLM declined to classify overlapping content within references/api_reference.md:\n"
                                '  "### Parameters" in references/api_reference.md (lines 90-106)\n'
                                '  vs "### Parameters" in references/api_reference.md (lines 134-146)'
                            ),
                            "file_path": "references/api_reference.md",
                            "line_number": 90,
                            "suggestion": "Review this overlap manually.",
                            "metadata": {
                                "source_files": ["references/api_reference.md"],
                                "llm_provider": "openai-compatible",
                                "llm_model": "granite4.1:3b",
                            },
                        },
                        {
                            "category": "CONTENT_DEDUP",
                            "severity": "low",
                            "check_name": "trivial_intra_file",
                            "message": "Trivial comment repeated\nline 2",
                            "file_path": "SKILL.md",
                        },
                    ]
                }
            ],
        }
        (skill_dir / "skillevaluator-dedup.json").write_text(json.dumps(report_data), encoding="utf-8")

        summary_file = tmp_path / "summary.md"
        pr_dedup_report.process_dedup_report(
            scope_file=scope_file,
            reports_dir=reports_dir,
            model="granite4.1:3b",
            digest="abcdef",
            summary_file=summary_file,
        )

        captured = capsys.readouterr()
        # No annotations emitted
        assert "::warning" not in captured.out

        summary_text = summary_file.read_text(encoding="utf-8")
        assert "| `skill-a` | clean | 0 | 0 |" in summary_text
        assert "No duplicate candidates" in summary_text
        assert "### LLM declined to judge" in summary_text
        expected_refused_line = (
            "- `skill-a`: `references/api_reference.md:90` - "
            "The LLM declined to classify overlapping content within references/api_reference.md: - granite4.1:3b"
        )
        assert expected_refused_line in summary_text
        assert "These overlaps were not classified; review them manually or rerun with another judge." in summary_text
        assert "### Other findings" in summary_text
        assert "- `skill-a`: trivial_intra_file (low) - Trivial comment repeated" in summary_text

    def test_missing_report_and_corrupt_json(self, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
        """Handle missing report file and malformed JSON gracefully without raising exceptions."""
        scope_data = {
            "skills_dir": "skills",
            "dedup_skills": ["skill-missing", "skill-corrupt"],
            "changed_lines": {},
        }
        scope_file = tmp_path / "scope.json"
        scope_file.write_text(json.dumps(scope_data), encoding="utf-8")

        reports_dir = tmp_path / "reports"
        corrupt_dir = reports_dir / "dedup-skill-corrupt"
        corrupt_dir.mkdir(parents=True)
        (corrupt_dir / "skillevaluator-dedup.json").write_text("{corrupt", encoding="utf-8")

        summary_file = tmp_path / "summary.md"
        pr_dedup_report.process_dedup_report(
            scope_file=scope_file,
            reports_dir=reports_dir,
            model="granite4.1:3b",
            digest="abc",
            summary_file=summary_file,
        )

        captured = capsys.readouterr()
        assert "report missing for skill-missing" in captured.out
        assert "report corrupt for skill-corrupt" in captured.out


def _dup_finding(
    first: tuple[int, int], second: tuple[int, int], confidence: float = 0.9, file: str = "ref.md"
) -> dict:
    """Return a SkillEvaluator duplicate finding for two sections of one file."""
    return {
        "category": "DUPLICATE",
        "severity": "high",
        "check_name": "duplicate",
        "message": (
            f"Duplicate content found within {file}:\n"
            f'  "### Example" in {file} (lines {first[0]}-{first[1]})\n'
            f'  vs "### Example" in {file} (lines {second[0]}-{second[1]})'
        ),
        "suggestion": "Merge the sections. More detail.",
        "metadata": {"confidence": confidence, "reasoning": "Same text"},
    }


def _write_dedup_report(reports_dir: Path, skill: str, findings: list[dict], status: str | None = None) -> None:
    """Write a SkillEvaluator dedup report for a skill under reports_dir."""
    skill_dir = reports_dir / f"dedup-{skill}"
    skill_dir.mkdir(parents=True, exist_ok=True)
    overall = status or ("failed" if findings else "passed")
    data = {"overall_status": overall, "results": [{"findings": findings}]}
    (skill_dir / "skillevaluator-dedup.json").write_text(json.dumps(data), encoding="utf-8")


def _write_scope(tmp_path: Path, **overrides: object) -> Path:
    """Write a scope.json where lines 10-20 of skills/skill-a/ref.md changed."""
    scope = {
        "skills_dir": "skills",
        "dedup_skills": ["skill-a"],
        "deletion_skills": [],
        "infra_changed": False,
        "changed_lines": {"skills/skill-a/ref.md": [[10, 20]]},
    }
    scope.update(overrides)
    scope_file = tmp_path / "scope.json"
    scope_file.write_text(json.dumps(scope), encoding="utf-8")
    return scope_file


def _run_report(tmp_path: Path, scope_file: Path, *, second: bool = True, **kwargs: object) -> dict:
    """Run process_dedup_report in two-judge mode (or single-judge mode) and return the result."""
    extra: dict = {}
    if second:
        extra = {"second_reports_dir": tmp_path / "second", "second_model": "phi4-mini:3.8b", "second_digest": "b9db5d"}
    extra.update(kwargs)
    return pr_dedup_report.process_dedup_report(
        scope_file=scope_file,
        reports_dir=tmp_path / "first",
        model="granite4.1:3b",
        digest="59b95aeac3cf",
        summary_file=tmp_path / "summary.md",
        result_file=tmp_path / "result.json",
        repo="owner/repo",
        sha="abcdef1234567890",
        run_url="https://example.test/run/1",
        **extra,
    )


class TestPrDedupTwoJudges:
    """Tests for the second judge, deletion handling and the machine-readable result."""

    def test_confirmed_by_both_is_raised(self, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
        """A touching candidate flagged by both judges is annotated and sets status findings."""
        _write_dedup_report(tmp_path / "first", "skill-a", [_dup_finding((12, 18), (50, 56), 0.91)])
        _write_dedup_report(tmp_path / "second", "skill-a", [_dup_finding((12, 18), (50, 56), 0.88)])

        result = _run_report(tmp_path, _write_scope(tmp_path))

        assert result["status"] == "findings"
        assert result["raised"] == 1
        assert result["check_conclusion"] == "neutral"
        assert result["comment_needed"] is True
        out = capsys.readouterr().out
        assert out.count("::warning") == 2
        assert "judges granite4.1:3b and phi4-mini:3.8b, confidence 0.91 / 0.88" in out
        assert (
            "[`ref.md:12-18`](https://github.com/owner/repo/blob/abcdef1234567890/skills/skill-a/ref.md#L12-L18)"
            in (result["comment_markdown"])
        )
        assert result["comment_markdown"].startswith(pr_dedup_report.COMMENT_MARKER)
        summary = (tmp_path / "summary.md").read_text(encoding="utf-8")
        assert "confirmed by both judges" in summary
        assert "| `skill-a` | findings | 1 | 0 | findings | 1 |" in summary
        assert json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))["status"] == "findings"

    def test_first_judge_only_is_not_raised(self, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
        """A candidate the second judge did not flag stays out of annotations, comment table and check."""
        _write_dedup_report(tmp_path / "first", "skill-a", [_dup_finding((12, 18), (50, 56))])
        _write_dedup_report(tmp_path / "second", "skill-a", [])

        result = _run_report(tmp_path, _write_scope(tmp_path))

        assert result["status"] == "clean"
        assert result["raised"] == 0
        assert result["first_judge_only"] == 1
        assert result["check_conclusion"] == "success"
        assert result["comment_needed"] is False
        assert "::warning" not in capsys.readouterr().out
        assert "first judge only" in (tmp_path / "summary.md").read_text(encoding="utf-8")

    def test_second_judge_matches_clusters_by_overlap(self, tmp_path: Path) -> None:
        """Clusters match when a member overlaps, even if the second judge's ranges differ slightly."""
        _write_dedup_report(tmp_path / "first", "skill-a", [_dup_finding((12, 18), (50, 56))])
        _write_dedup_report(tmp_path / "second", "skill-a", [_dup_finding((14, 19), (51, 57))])

        assert _run_report(tmp_path, _write_scope(tmp_path))["status"] == "findings"

    def test_second_judge_other_cluster_does_not_confirm(self, tmp_path: Path) -> None:
        """A second-judge finding in an unrelated cluster does not confirm the first judge's candidate."""
        _write_dedup_report(tmp_path / "first", "skill-a", [_dup_finding((12, 18), (50, 56))])
        _write_dedup_report(tmp_path / "second", "skill-a", [_dup_finding((100, 110), (120, 130))])

        result = _run_report(tmp_path, _write_scope(tmp_path))
        assert result["status"] == "clean"
        assert result["first_judge_only"] == 1

    def test_preexisting_candidate_skips_second_judge(self, tmp_path: Path) -> None:
        """A candidate away from changed lines needs no second judge and is never raised."""
        _write_dedup_report(tmp_path / "first", "skill-a", [_dup_finding((200, 210), (220, 230))])

        result = _run_report(tmp_path, _write_scope(tmp_path))

        assert result["status"] == "clean"
        assert result["second_pass_skills"] == []
        assert "not cross-checked" in (tmp_path / "summary.md").read_text(encoding="utf-8")

    def test_missing_second_report_cannot_run(self, tmp_path: Path) -> None:
        """If the first judge flagged something that matters and the second judge produced no report, fail loudly."""
        _write_dedup_report(tmp_path / "first", "skill-a", [_dup_finding((12, 18), (50, 56))])

        result = _run_report(tmp_path, _write_scope(tmp_path))

        assert result["status"] == "cannot_run"
        assert result["check_conclusion"] == "failure"
        assert result["comment_needed"] is True
        assert any("second judge" in r for r in result["reasons"])
        assert "second judge unavailable" in (tmp_path / "summary.md").read_text(encoding="utf-8")

    def test_incomplete_second_report_cannot_run(self, tmp_path: Path) -> None:
        """An incomplete second-judge scan is not a confirmation and not a pass."""
        _write_dedup_report(tmp_path / "first", "skill-a", [_dup_finding((12, 18), (50, 56))])
        _write_dedup_report(tmp_path / "second", "skill-a", [], status="incomplete")

        assert _run_report(tmp_path, _write_scope(tmp_path))["status"] == "cannot_run"

    def test_incomplete_first_scan_cannot_run(self, tmp_path: Path) -> None:
        """An incomplete first-judge scan means the scan could not run."""
        _write_dedup_report(tmp_path / "first", "skill-a", [], status="incomplete")

        result = _run_report(tmp_path, _write_scope(tmp_path))
        assert result["status"] == "cannot_run"
        assert any("incomplete for skill-a" in r for r in result["reasons"])

    def test_blockers_make_status_cannot_run(self, tmp_path: Path) -> None:
        """Setup failures passed by the workflow are reported even when reports are absent."""
        scope_file = _write_scope(tmp_path)
        (tmp_path / "first").mkdir()

        result = _run_report(tmp_path, scope_file, blockers=["Ollama setup failed"])

        assert result["status"] == "cannot_run"
        assert "Ollama setup failed" in result["reasons"]
        assert "Ollama setup failed" in result["comment_markdown"]
        assert "could not complete" in result["comment_markdown"]

    def test_missing_reports_dir_cannot_run(self, tmp_path: Path) -> None:
        """No reports directory at all is a cannot_run result, not a crash; blockers are listed in the summary."""
        result = _run_report(tmp_path, _write_scope(tmp_path), blockers=["Ollama or model setup failed."])
        assert result["status"] == "cannot_run"
        assert "- Ollama or model setup failed." in (tmp_path / "summary.md").read_text(encoding="utf-8")

    def test_missing_scope_cannot_run(self, tmp_path: Path) -> None:
        """A missing scope file is a cannot_run result."""
        result = _run_report(tmp_path, tmp_path / "nope.json")
        assert result["status"] == "cannot_run"

    def test_no_dedup_skills_is_skipped(self, tmp_path: Path) -> None:
        """Nothing to scan: status skipped, no comment needed, no check row."""
        result = _run_report(tmp_path, _write_scope(tmp_path, dedup_skills=[]))
        assert result["status"] == "skipped"
        assert result["comment_needed"] is False
        assert result["check_conclusion"] == "skipped"
        assert "nothing to scan" in result["comment_markdown"]

    def test_deletion_skill_counts_all_candidates_as_touching(self, tmp_path: Path) -> None:
        """When a file of the skill was deleted every candidate in that skill touches the PR."""
        _write_dedup_report(tmp_path / "first", "skill-a", [_dup_finding((200, 210), (220, 230))])
        _write_dedup_report(tmp_path / "second", "skill-a", [_dup_finding((200, 210), (220, 230))])

        result = _run_report(tmp_path, _write_scope(tmp_path, deletion_skills=["skill-a"], changed_lines={}))

        assert result["status"] == "findings"

    def test_infra_change_second_judge_runs_but_preexisting_is_not_raised(self, tmp_path: Path) -> None:
        """Infra-only PR: second judge still runs on flagged skills (pipeline exercised) but nothing is raised."""
        _write_dedup_report(tmp_path / "first", "skill-a", [_dup_finding((200, 210), (220, 230))])
        _write_dedup_report(tmp_path / "second", "skill-a", [_dup_finding((200, 210), (220, 230))])

        result = _run_report(tmp_path, _write_scope(tmp_path, infra_changed=True, changed_lines={}))

        assert result["second_pass_skills"] == ["skill-a"]
        assert result["status"] == "clean"

    def test_second_judge_refusal_is_listed_not_confirming(self, tmp_path: Path) -> None:
        """A second-judge llm_refused finding is shown in the summary and does not confirm."""
        _write_dedup_report(tmp_path / "first", "skill-a", [_dup_finding((12, 18), (50, 56))])
        refused = {
            "check_name": "llm_refused",
            "file_path": "ref.md",
            "line_number": 12,
            "message": "Model declined",
            "metadata": {"llm_model": "phi4-mini:3.8b"},
        }
        _write_dedup_report(tmp_path / "second", "skill-a", [refused], status="passed")

        result = _run_report(tmp_path, _write_scope(tmp_path))

        assert result["status"] == "clean"
        assert "### LLM declined to judge" in (tmp_path / "summary.md").read_text(encoding="utf-8")

    def test_comment_row_limit(self, tmp_path: Path) -> None:
        """The comment lists at most 15 rows and says how many more exist."""
        findings = [_dup_finding((12, 13), (1000 + i * 10, 1005 + i * 10)) for i in range(20)]
        # make each cluster distinct on the changed side as well
        for i, f in enumerate(findings):
            f["message"] = f["message"].replace("lines 12-13", f"lines {10 + (i % 10)}-{10 + (i % 10)}")
            f["message"] = f["message"].replace(f"lines {1000 + i * 10}-", f"lines {2000 + i * 10}-")
        _write_dedup_report(tmp_path / "first", "skill-a", findings)
        _write_dedup_report(tmp_path / "second", "skill-a", findings)

        result = _run_report(tmp_path, _write_scope(tmp_path))

        assert result["status"] == "findings"
        assert "…and " in result["comment_markdown"]

    def test_single_judge_mode_raises_without_second(self, tmp_path: Path) -> None:
        """Without a second model (local run) a touching candidate is raised by the single judge."""
        _write_dedup_report(tmp_path / "first", "skill-a", [_dup_finding((12, 18), (50, 56))])

        result = _run_report(tmp_path, _write_scope(tmp_path), second=False)

        assert result["status"] == "findings"

    def test_select_second_pass(self, tmp_path: Path) -> None:
        """Only skills with a candidate touching the PR are selected; outputs are written for the workflow."""
        _write_dedup_report(tmp_path / "first", "skill-a", [_dup_finding((12, 18), (50, 56))])
        _write_dedup_report(tmp_path / "first", "skill-b", [_dup_finding((200, 210), (220, 230))])
        _write_dedup_report(tmp_path / "first", "skill-c", [])
        scope_file = _write_scope(tmp_path, dedup_skills=["skill-a", "skill-b", "skill-c"])
        scope_data = json.loads(scope_file.read_text(encoding="utf-8"))
        scope_data["changed_lines"]["skills/skill-b/ref.md"] = [[1, 5]]
        scope_file.write_text(json.dumps(scope_data), encoding="utf-8")
        gh_output = tmp_path / "gh_output"

        skills = pr_dedup_report.select_second_pass(scope_file, tmp_path / "first", gh_output)

        assert skills == ["skill-a"]
        assert gh_output.read_text(encoding="utf-8") == "run_second_pass=true\nsecond_pass_skills=skill-a\n"

    def test_select_second_pass_none(self, tmp_path: Path) -> None:
        """No flagged skills: run_second_pass is false."""
        _write_dedup_report(tmp_path / "first", "skill-a", [])
        gh_output = tmp_path / "gh_output"

        assert pr_dedup_report.select_second_pass(_write_scope(tmp_path), tmp_path / "first", gh_output) == []
        assert gh_output.read_text(encoding="utf-8") == "run_second_pass=false\nsecond_pass_skills=\n"

    def test_cli_report_writes_outputs(self, tmp_path: Path) -> None:
        """The report sub-command writes the result file and the dedup_status step output."""
        _write_dedup_report(tmp_path / "first", "skill-a", [])
        gh_output = tmp_path / "gh_output"
        script = _REPO_ROOT / ".github" / "tier2" / "pr_dedup_report.py"

        res = subprocess.run(
            [
                sys.executable,
                str(script),
                "report",
                "--scope",
                str(_write_scope(tmp_path)),
                "--reports",
                str(tmp_path / "first"),
                "--model",
                "granite4.1:3b",
                "--second-model",
                "phi4-mini:3.8b",
                "--second-reports",
                str(tmp_path / "second"),
                "--result-file",
                str(tmp_path / "result.json"),
                "--github-output",
                str(gh_output),
                "--summary-file",
                str(tmp_path / "summary.md"),
                "--blocker",
                "x failed",
            ],
            capture_output=True,
            text=True,
            check=False,
        )

        assert res.returncode == 0
        assert gh_output.read_text(encoding="utf-8") == "dedup_status=cannot_run\n"
        assert json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))["reasons"] == ["x failed"]


class FakeApi:
    """Records GitHub API calls and serves canned responses."""

    def __init__(self, comments: list[dict] | None = None, check_runs: list[dict] | None = None) -> None:
        """Create a fake with optional existing comments and check runs."""
        self.comments = comments or []
        self.check_runs = check_runs or []
        self.calls: list[tuple[str, str, dict | None]] = []
        self.fail_with: pr_dedup_publish.ApiError | None = None

    def request(self, method: str, path: str, payload: dict | None = None) -> object:
        """Return canned data or raise the configured error."""
        self.calls.append((method, path, payload))
        if self.fail_with is not None and method != "GET":
            raise self.fail_with
        if method == "GET" and "/comments" in path:
            return self.comments if "page=1" in path else []
        if method == "GET" and "/check-runs" in path:
            return {"check_runs": self.check_runs}
        return {}


def _result(status: str, **overrides: object) -> dict:
    """Return a result dictionary shaped like pr_dedup_report output."""
    base = {
        "status": status,
        "comment_markdown": f"{pr_dedup_report.COMMENT_MARKER}\n## Tier 2 dedup (advisory)\n\nbody {status}\n",
        "comment_needed": status in {"findings", "cannot_run"},
        "check_conclusion": {"findings": "neutral", "clean": "success", "cannot_run": "failure"}.get(status, "skipped"),
        "check_title": f"title {status}",
    }
    base.update(overrides)
    return base


class TestPrDedupPublish:
    """Tests for pr_dedup_publish.py."""

    def test_findings_create_comment_and_neutral_check(self) -> None:
        """Findings with no earlier comment create one comment and one neutral check row."""
        api = FakeApi()

        outcomes = pr_dedup_publish.publish(api, "o/r", 7, "sha1", _result("findings"), "https://run")

        assert outcomes == ["comment created", "check row created"]
        post_comment = next(c for c in api.calls if c[0] == "POST" and "/issues/7/comments" in c[1])
        assert pr_dedup_report.COMMENT_MARKER in post_comment[2]["body"]
        post_check = next(c for c in api.calls if c[0] == "POST" and c[1] == "/repos/o/r/check-runs")
        assert post_check[2]["conclusion"] == "neutral"
        assert post_check[2]["name"] == "Tier 2 dedup findings (advisory)"
        assert post_check[2]["head_sha"] == "sha1"
        assert post_check[2]["details_url"] == "https://run"
        assert pr_dedup_report.COMMENT_MARKER not in post_check[2]["output"]["summary"]

    def test_existing_sticky_comment_is_updated(self) -> None:
        """The bot's marker comment is patched in place instead of creating a second one."""
        existing = {
            "id": 99,
            "body": f"{pr_dedup_report.COMMENT_MARKER}\nold",
            "user": {"login": "github-actions[bot]"},
        }
        api = FakeApi(comments=[existing])

        assert pr_dedup_publish.publish_comment(api, "o/r", 7, _result("findings")) == "comment updated"
        assert ("PATCH", "/repos/o/r/issues/comments/99") == api.calls[-1][:2]
        assert not any(c[0] == "POST" for c in api.calls)

    def test_marker_from_another_author_is_ignored(self) -> None:
        """A human pasting the marker is not edited; a fresh comment is created."""
        spoof = {"id": 5, "body": f"{pr_dedup_report.COMMENT_MARKER} hi", "user": {"login": "someone"}}
        api = FakeApi(comments=[spoof])

        assert pr_dedup_publish.publish_comment(api, "o/r", 7, _result("findings")) == "comment created"

    def test_clean_without_comment_creates_none(self) -> None:
        """A clean result with no earlier comment stays silent but still records a success check row."""
        api = FakeApi()

        outcomes = pr_dedup_publish.publish(api, "o/r", 7, "sha1", _result("clean"))

        assert outcomes == ["no comment needed", "check row created"]
        assert not any(c[0] == "POST" and "/comments" in c[1] for c in api.calls)

    def test_clean_updates_existing_comment_to_resolved(self) -> None:
        """A clean push after a findings push rewrites the sticky comment."""
        existing = {
            "id": 1,
            "body": f"{pr_dedup_report.COMMENT_MARKER}\nfindings",
            "user": {"login": "github-actions[bot]"},
        }
        api = FakeApi(comments=[existing])

        assert pr_dedup_publish.publish_comment(api, "o/r", 7, _result("clean")) == "comment updated"

    def test_unchanged_comment_is_not_patched(self) -> None:
        """Identical body: no write."""
        result = _result("findings")
        existing = {"id": 1, "body": result["comment_markdown"], "user": {"login": "github-actions[bot]"}}
        api = FakeApi(comments=[existing])

        assert pr_dedup_publish.publish_comment(api, "o/r", 7, result) == "comment unchanged"
        assert [c[0] for c in api.calls] == ["GET"]

    def test_skipped_updates_existing_comment_but_has_no_check_row(self) -> None:
        """No skill files left in scope: stale comment is refreshed, no check row is written."""
        existing = {
            "id": 1,
            "body": f"{pr_dedup_report.COMMENT_MARKER}\nstale",
            "user": {"login": "github-actions[bot]"},
        }
        api = FakeApi(comments=[existing])

        outcomes = pr_dedup_publish.publish(api, "o/r", 7, "sha1", _result("skipped"))

        assert outcomes == ["comment updated", "no check row needed"]

    def test_existing_check_row_is_patched(self) -> None:
        """A re-run on the same commit updates the existing row instead of adding another."""
        api = FakeApi(check_runs=[{"id": 42, "name": "Tier 2 dedup findings (advisory)"}])

        assert pr_dedup_publish.publish_check_run(api, "o/r", "sha1", _result("clean")) == "check row updated"
        assert api.calls[-1][:2] == ("PATCH", "/repos/o/r/check-runs/42")

    def test_cannot_run_conclusion_is_failure(self) -> None:
        """A scan that could not run is a failed check row and a comment."""
        api = FakeApi()

        pr_dedup_publish.publish(api, "o/r", 7, "sha1", _result("cannot_run"))

        post_check = next(c for c in api.calls if c[1] == "/repos/o/r/check-runs" and c[0] == "POST")
        assert post_check[2]["conclusion"] == "failure"

    def test_read_only_token_is_a_notice_not_an_error(self, capsys: pytest.CaptureFixture) -> None:
        """HTTP 403 (fork or Dependabot PR) downgrades to a notice."""
        api = FakeApi()
        api.fail_with = pr_dedup_publish.ApiError(403, "Resource not accessible by integration")

        outcomes = pr_dedup_publish.publish(api, "o/r", 7, "sha1", _result("findings"))

        assert all("read-only" in o for o in outcomes)
        assert capsys.readouterr().out.count("::notice") == 2

    def test_other_api_errors_propagate(self) -> None:
        """Any other API failure is raised so the job goes red."""
        api = FakeApi()
        api.fail_with = pr_dedup_publish.ApiError(500, "boom")

        with pytest.raises(pr_dedup_publish.ApiError):
            pr_dedup_publish.publish(api, "o/r", 7, "sha1", _result("findings"))

    def test_main_requires_token(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ) -> None:
        """Without GITHUB_TOKEN main returns 1 with an error annotation."""
        result_file = tmp_path / "result.json"
        result_file.write_text(json.dumps(_result("clean")), encoding="utf-8")
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        monkeypatch.setattr(
            sys, "argv", ["x", "--result", str(result_file), "--repo", "o/r", "--pr", "1", "--sha", "s"]
        )

        assert pr_dedup_publish.main() == 1
        assert "::error" in capsys.readouterr().out

    def test_retries_transient_errors(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """5xx responses are retried with backoff and then raised."""
        import urllib.error

        calls: list[int] = []

        def fake_urlopen(*_args: object, **_kwargs: object) -> object:
            calls.append(1)
            raise urllib.error.HTTPError("u", 502, "bad gateway", {}, None)  # type: ignore[arg-type]

        monkeypatch.setattr(pr_dedup_publish.urllib.request, "urlopen", fake_urlopen)
        sleeps: list[float] = []
        api = pr_dedup_publish.GitHubApi("t", sleep=sleeps.append)

        with pytest.raises(pr_dedup_publish.ApiError):
            api.request("GET", "/x")
        assert len(calls) == 3
        assert sleeps == [1, 2]


class TestPrSimilarityReport:
    """Tests for pr_similarity_report.py."""

    def test_enforced_pass_with_margin(self, tmp_path: Path) -> None:
        """Enforced check passed and margin probe extracts pairs sorted by similarity."""
        enforced = tmp_path / "enforced.json"
        enforced.write_text(json.dumps({"overall_status": "passed", "results": []}), encoding="utf-8")

        margin = tmp_path / "margin.json"
        margin_data = {
            "overall_status": "passed",
            "results": [
                {
                    "findings": [
                        {
                            "category": "SIMILARITY",
                            "metadata": {
                                "score": 0.68,
                                "entry_a": "ds-utils-metrics",
                                "entry_b": "ds-utils-xai",
                            },
                        },
                        {
                            "category": "SIMILARITY",
                            "metadata": {
                                "score": 0.72,
                                "entry_a": "ds-utils-preprocess",
                                "entry_b": "ds-utils-transformers",
                            },
                        },
                        {
                            "category": "SIMILARITY",
                            "metadata": {
                                "score": 0.78,  # Above 0.75 -> omitted from near-threshold list
                                "entry_a": "pairA",
                                "entry_b": "pairB",
                            },
                        },
                    ]
                }
            ],
        }
        margin.write_text(json.dumps(margin_data), encoding="utf-8")

        summary_file = tmp_path / "sim_summary.md"
        pr_similarity_report.process_similarity_report(
            enforced_file=enforced,
            margin_file=margin,
            threshold=0.75,
            summary_file=summary_file,
            margin_threshold=0.65,
        )

        content = summary_file.read_text(encoding="utf-8")
        assert "- **Enforced Check Status**: `PASSED` (threshold: `0.75`)" in content
        # Check sorted descending: 0.720 before 0.680
        idx_72 = content.find("`ds-utils-preprocess <-> ds-utils-transformers` : `0.720`")
        idx_68 = content.find("`ds-utils-metrics <-> ds-utils-xai` : `0.680`")
        assert idx_72 != -1
        assert idx_68 != -1
        assert idx_72 < idx_68
        assert "pairA" not in content

    def test_margin_empty_near_threshold(self, tmp_path: Path) -> None:
        """When margin probe has no pairs below enforced threshold, output formatted sentence."""
        enforced = tmp_path / "enforced.json"
        enforced.write_text(json.dumps({"overall_status": "passed", "results": []}), encoding="utf-8")
        margin = tmp_path / "margin.json"
        margin.write_text(json.dumps({"overall_status": "passed", "results": []}), encoding="utf-8")

        summary_file = tmp_path / "sim_summary.md"
        pr_similarity_report.process_similarity_report(
            enforced_file=enforced,
            margin_file=margin,
            threshold=0.75,
            summary_file=summary_file,
            margin_threshold=0.65,
        )

        content = summary_file.read_text(encoding="utf-8")
        assert "No pair scored between 0.65 and 0.75." in content

    def test_missing_margin(self, tmp_path: Path) -> None:
        """Missing margin report outputs 'margin unavailable'."""
        enforced = tmp_path / "enforced.json"
        enforced.write_text(json.dumps({"overall_status": "failed", "results": []}), encoding="utf-8")
        margin = tmp_path / "missing.json"

        summary_file = tmp_path / "sim_summary.md"
        pr_similarity_report.process_similarity_report(
            enforced_file=enforced,
            margin_file=margin,
            threshold=0.75,
            summary_file=summary_file,
        )

        content = summary_file.read_text(encoding="utf-8")
        assert "- **Enforced Check Status**: `FAILED` (threshold: `0.75`)" in content
        assert "margin unavailable" in content
