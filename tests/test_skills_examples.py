"""Test executing python blocks under ## Examples across all skill documents."""

from pathlib import Path
import re
import warnings

import matplotlib
import matplotlib.pyplot as plt
import pytest

matplotlib.use("Agg")

SKILL_NAMES = [
    "ds-utils-metrics",
    "ds-utils-preprocess",
    "ds-utils-strings",
    "ds-utils-transformers",
    "ds-utils-unsupervised",
    "ds-utils-xai",
]


def extract_examples_code(skill_name: str) -> str:
    """Extract code from all python code fences under ## Examples in SKILL.md."""
    repo_root = Path(__file__).resolve().parent.parent
    skill_md = repo_root / "skills" / skill_name / "SKILL.md"
    assert skill_md.is_file(), f"Missing SKILL.md for {skill_name}"

    content = skill_md.read_text(encoding="utf-8")
    # Find ## Examples section up to the next top-level or second-level heading
    match = re.search(r"^## Examples\s*\n(.*?)(?=^## |\Z)", content, re.DOTALL | re.MULTILINE)
    assert match, f"No '## Examples' section found in {skill_name}/SKILL.md"

    examples_text = match.group(1)
    # Extract all ```python ... ``` blocks
    code_blocks = re.findall(r"```python\s*\n(.*?)```", examples_text, re.DOTALL)
    assert code_blocks, f"No python code blocks found under ## Examples in {skill_name}/SKILL.md"

    return "\n\n".join(code_blocks)


@pytest.mark.parametrize("skill_name", SKILL_NAMES)
def test_skill_examples_execution(skill_name):
    """Execute the python blocks under ## Examples in an isolated namespace."""
    code = extract_examples_code(skill_name)

    # Global execution namespace
    namespace = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        exec(code, namespace)

    plt.close("all")
