# Agent Skills Evaluation Guide

This guide details the evaluation setup, tooling versions, and validation commands used to verify the Agent Skills in `skills/` using **NVIDIA SkillEvaluator**.

## Tooling & Pinned Dependencies

To ensure 100% reproducible evaluation runs, tool dependencies are pinned in `requirements-skills.txt`:

| Tool | Pinned Version / Commit | Purpose | Upstream Reference |
|---|---|---|---|
| **SkillEvaluator** | commit [`a61e1f3ff609118c0c07e5f569aaf58fa5a76be0`](https://github.com/NVIDIA/SkillEvaluator/commit/a61e1f3ff609118c0c07e5f569aaf58fa5a76be0) (v0.6.0) | Static Tier 1 validation & quality scoring | [NVIDIA/SkillEvaluator](https://github.com/NVIDIA/SkillEvaluator) |
| **SkillSpector** | tag [`v2.12.0`](https://github.com/NVIDIA/SkillSpector/releases/tag/v2.12.0) | Skill-specific security & prompt vulnerability scanner | [NVIDIA/SkillSpector](https://github.com/NVIDIA/SkillSpector) |

*Note*: PyPI currently hosts only an empty `0.0.0a0` placeholder for `skillevaluator`. Therefore, installations must use the pinned Git commit archive via `uv tool install` or `pip`.

### Installing the Python Evaluator Tools

Using [`uv`](https://docs.astral.sh/uv/) (recommended):

```bash
uv tool install --python 3.13 "skillevaluator @ git+https://github.com/NVIDIA/SkillEvaluator.git@a61e1f3ff609118c0c07e5f569aaf58fa5a76be0"
uv tool install git+https://github.com/NVIDIA/SkillSpector.git@v2.12.0
```

Verify installation:
```bash
skillevaluator --version    # Expected: skillevaluator, version 0.6.0
skillspector --version      # Expected: 2.12.0
```

---

## External Security Scanners (Platform-Specific)

Python dependencies alone do not provide all external static security binaries required for full static Tier 1 validation:

### 1. Gitleaks (Secrets Detection)
- **Linux (Ubuntu/Debian)**: Install via binary release from [Gitleaks GitHub Releases](https://github.com/gitleaks/gitleaks/releases) or via Go: `go install github.com/gitleaks/gitleaks/v8@latest`.
- **macOS**: `brew install gitleaks`
- **Windows**: `winget install Gitleaks.Gitleaks` or `choco install gitleaks`.

### 2. Semgrep (Code Risk Analysis)
- **Linux / macOS**: `uv tool install semgrep` or `brew install semgrep`.
- **Windows**: Semgrep has limited native Windows support; the official recommendation from [Semgrep Docs](https://semgrep.dev/docs/getting-started/) is running Semgrep inside WSL2 (Windows Subsystem for Linux) or via its official Docker container (`returntocorp/semgrep`).

---

## Evaluation Architecture: Understanding Tiers, Scanners, and Statuses

SkillEvaluator defines three distinct evaluation tiers. It is critical to distinguish between keyless static checks, scanner availability, and optional LLM-backed analysis:

### 1. Keyless Static Checks (`eval-skills.sh` & `eval-skills.bat`)
The evaluation runner scripts execute:
- `skillevaluator quality-check <skill-dir>`: Scores quality across Correctness (35%), Discoverability (25%), Reliability (25%), and Efficiency (15%) against a pass threshold of 70.
- `skillevaluator tier1 --no-llm <skill-dir>`: Evaluates schema (frontmatter strings, name-directory equality, line limits, approved subdirectories), semantic versioning, PII patterns, license compliance, code hygiene, and script linting.

**No API key is required** for any of these static checks.

### 2. External Scanner Availability & `INCOMPLETE` Status
Under Tier 1 static validation:
- When an external scanner is missing from `PATH` (such as `gitleaks`), Tier 1 reports `[INCOMPLETE] gitleaks did not complete` and exits non-zero (`exit 1`).
- When `skillspector` runs in standalone keyless mode without an LLM verification pass, it reports `analysis_completeness: partial`. SkillEvaluator treats partial analysis as `INCOMPLETE` (`exit 1`).
- **Important Disclosure**: A run reporting `INCOMPLETE` is **not** a pass. In environments lacking external binaries (e.g. Gitleaks) or LLM credentials, all six skills pass schema, hygiene, quality, version, license, and PII checks with 0 HIGH errors, but the aggregate Tier 1 exit status truthfully reports `INCOMPLETE` (exit 1).

### 3. Optional LLM-Backed Checks (Not run by `--no-llm`)
- Invoking `skillevaluator rubric-eval` or adding `--llm` / `--llm-verify` requires a model provider.
- Configure credentials locally via environment variables:
  ```bash
  export SKILL_EVAL_LLM_PROVIDER=nv_build
  export NVIDIA_API_KEY="nvapi-your-key-here"   # Use placeholder, never commit real keys!
  ```
- Adding an API key will **not** turn a static `--no-llm` command into an LLM run. `--no-llm` explicitly disables LLM evaluation.

---

## Running the Evaluation Scripts

### macOS / Linux
```bash
./eval-skills.sh
```

### Windows
```cmd
eval-skills.bat
```

Both scripts:
1. Verify `skillevaluator` is present in `PATH` (exiting with an informative error if absent).
2. Evaluate all six skills in sequence without stopping prematurely on the first error.
3. Capture genuine exit codes for `quality-check` and `tier1 --no-llm`.
4. Output a comprehensive outcome table.
5. Exit with a non-zero exit code if any check fails or reports incomplete scanner evidence.

---

## Tested Environment Matrix

- **Evaluator Version Tested**: `skillevaluator` 0.6.0 (git commit `a61e1f3ff609118c0c07e5f569aaf58fa5a76be0`).
- **Platform Tested**: Ubuntu 24.04 LTS (x86_64, Linux kernel 6.6) with Python 3.13.12.
- **PowerShell (`install.ps1`, `eval-skills.bat`)**: Code-reviewed against PowerShell 5.1/7+ specifications and batch syntax standards.
