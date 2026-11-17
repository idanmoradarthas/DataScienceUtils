"""Consistency checks between pyproject.toml, the conda recipe, the deploy scripts and the CI matrix.

pyproject.toml is the single source of truth for the version, the supported Python versions and the runtime
dependencies. These tests fail when the conda recipe or the deploy scripts drift away from it.
"""

import re
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

import ds_utils

REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
META_YAML = (REPO_ROOT / "data-science-utils" / "meta.yaml").read_text(encoding="utf-8")
DEPLOY_SH = (REPO_ROOT / "deploy-conda.sh").read_text(encoding="utf-8")
DEPLOY_BAT = (REPO_ROOT / "deploy-conda.bat").read_text(encoding="utf-8")
TEST_WORKFLOW = (REPO_ROOT / ".github" / "workflows" / "test.yml").read_text(encoding="utf-8")


def _normalize(spec: str) -> str:
    """Remove all whitespace so that 'python >=3.11' and 'python>=3.11' compare equal."""
    return re.sub(r"\s+", "", spec)


def _meta_requirements(section: str) -> list[str]:
    """Return the normalized entries of requirements/<section> in meta.yaml."""
    entries: list[str] = []
    in_requirements = False
    in_section = False
    for line in META_YAML.splitlines():
        if re.match(r"^requirements:\s*$", line):
            in_requirements = True
            continue
        if in_requirements and re.match(r"^\S", line):
            break
        if not in_requirements:
            continue
        header = re.match(r"^  (\w+):\s*$", line)
        if header:
            in_section = header.group(1) == section
            continue
        item = re.match(r"^    - (.+?)\s*$", line)
        if in_section and item:
            entries.append(_normalize(item.group(1)))
    return entries


def _meta_version() -> str:
    match = re.search(r'^  version:\s*"([^"]+)"', META_YAML, re.MULTILINE)
    assert match, "version not found in data-science-utils/meta.yaml"
    return match.group(1)


def _classifier_python_versions() -> list[str]:
    versions = []
    for classifier in PYPROJECT["project"]["classifiers"]:
        match = re.fullmatch(r"Programming Language :: Python :: (3\.\d+)", classifier)
        if match:
            versions.append(match.group(1))
    return sorted(versions, key=lambda v: int(v.split(".")[1]))


def test_version_is_identical_in_pyproject_conda_recipe_and_package():
    """The version lives in pyproject.toml; the conda recipe and the imported package must report the same one."""
    project_version = PYPROJECT["project"]["version"]
    assert _meta_version() == project_version, "data-science-utils/meta.yaml version differs from pyproject.toml"
    assert ds_utils.__version__ == project_version, (
        "ds_utils.__version__ differs from pyproject.toml; re-install the package (uv sync) after bumping the version"
    )


def test_pyproject_does_not_use_a_dynamic_version():
    """uv_build needs a static version, so a leftover dynamic version entry would break the build."""
    assert "version" not in PYPROJECT["project"].get("dynamic", [])
    assert PYPROJECT["build-system"]["build-backend"] == "uv_build"


def test_requires_python_matches_the_oldest_classifier():
    """requires-python must start at the oldest Python version that is advertised in the classifiers."""
    versions = _classifier_python_versions()
    assert versions, "no 'Programming Language :: Python :: 3.x' classifiers found"
    assert PYPROJECT["project"]["requires-python"] == f">={versions[0]}"


def test_conda_recipe_python_constraint_matches_requires_python():
    """The conda recipe must not build or allow Python versions that pyproject.toml excludes."""
    expected = _normalize("python" + PYPROJECT["project"]["requires-python"])
    for section in ("host", "run"):
        assert expected in _meta_requirements(section), f"meta.yaml requirements/{section} must contain '{expected}'"


@pytest.mark.parametrize("section", ["host", "run"])
def test_conda_recipe_dependencies_match_pyproject(section):
    """Every runtime dependency of pyproject.toml appears, with the same specifier, in the conda recipe."""
    expected = {_normalize(dep) for dep in PYPROJECT["project"]["dependencies"]}
    actual = {entry for entry in _meta_requirements(section) if not entry.startswith(("python", "pip"))}
    assert actual == expected, (
        f"meta.yaml requirements/{section} differs from pyproject.toml dependencies; "
        f"missing: {sorted(expected - actual)}, unexpected: {sorted(actual - expected)}"
    )


def test_deploy_scripts_build_for_the_classifier_python_versions():
    """deploy-conda.sh and deploy-conda.bat must build exactly the Python versions listed in the classifiers."""
    expected = _classifier_python_versions()

    sh_match = re.search(r'^PYTHON_VERSIONS="([^"]+)"', DEPLOY_SH, re.MULTILINE)
    assert sh_match, "PYTHON_VERSIONS not defined in deploy-conda.sh"
    assert sh_match.group(1).split() == expected

    bat_versions = re.search(r'^set "PYTHON_VERSIONS=([^"]+)"', DEPLOY_BAT, re.MULTILINE)
    bat_tags = re.search(r'^set "PYTHON_TAGS=([^"]+)"', DEPLOY_BAT, re.MULTILINE)
    assert bat_versions, "PYTHON_VERSIONS not defined in deploy-conda.bat"
    assert bat_tags, "PYTHON_TAGS not defined in deploy-conda.bat"
    assert bat_versions.group(1).split() == expected
    assert bat_tags.group(1).split() == [v.replace(".", "") for v in expected]


def test_ci_test_matrix_covers_the_classifier_python_versions():
    """The first python-version matrix of the test workflow must be the list of classifier Python versions."""
    match = re.search(r"python-version:\s*\[([^\]]+)\]", TEST_WORKFLOW)
    assert match, "python-version matrix not found in .github/workflows/test.yml"
    matrix = [value.strip().strip("\"'") for value in match.group(1).split(",")]
    assert matrix == _classifier_python_versions()


@pytest.mark.skipif(
    sys.platform == "win32" or shutil.which("bash") is None,
    reason="deploy-conda.sh is executed by bash inside the Linux conda container",
)
def test_deploy_conda_sh_reads_the_version_from_pyproject():
    """The version extraction command of deploy-conda.sh returns the version of pyproject.toml."""
    match = re.search(r"^version=\$\((.+)\)$", DEPLOY_SH, re.MULTILINE)
    assert match, "version extraction line not found in deploy-conda.sh"
    result = subprocess.run(
        ["bash", "-c", match.group(1)],
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == PYPROJECT["project"]["version"]
