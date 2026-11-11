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
        """Change only infra file: run_similarity is True, but run_dedup is False."""
        pins = temp_git_repo / ".github" / "tier2" / "pins.env"
        pins.write_text("TEST=2\n", encoding="utf-8", newline="\n")
        _run_git(["commit", "-am", "Update pins.env"], cwd=temp_git_repo)

        scope = pr_scope.compute_scope(base="HEAD~1", head="HEAD", repo_root=temp_git_repo)
        assert scope["run_similarity"] is True
        assert scope["run_dedup"] is False
        assert scope["changed_skills"] == []
        assert scope["dedup_skills"] == []
        assert scope["skipped_skills"] == []

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
