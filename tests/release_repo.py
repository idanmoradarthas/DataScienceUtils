"""A tiny synthetic git repository for the release tooling tests: tag ``1.0.0``, then ``plot_zebra_curve`` is added."""

from __future__ import annotations

from pathlib import Path
import subprocess

MEAN_BEFORE = "\n".join(["def mean_abs(values):", '    """Mean absolute value."""', "    return 0", ""])
MEAN_AFTER = "\n".join(
    [
        "def mean_abs(values):",
        '    """Mean absolute value."""',
        "    return sum(abs(v) for v in values) / len(values)",
        "",
    ]
)
ZEBRA = "\n".join(
    ["def plot_zebra_curve(values, *, smooth=True):", '    """Plot the zebra curve."""', "    return values", ""]
)
REPO_FILES = {
    "base": {"pkg/__init__.py": "", "pkg/stats.py": MEAN_BEFORE},
    "head": {"pkg/curves.py": ZEBRA, "pkg/stats.py": MEAN_AFTER},
}


def _git(repo: Path, *args: str) -> None:
    """Run git in ``repo`` with a fixed identity (used only for the synthetic repository)."""
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=release-tests",
            "-c",
            "user.email=release-tests@example.com",
            "-c",
            "commit.gpgsign=false",
            "-c",
            "tag.gpgsign=false",
            *args,
        ],
        check=True,
        capture_output=True,
    )


def make_release_repo(path: Path) -> None:
    """Create the repository: tag 1.0.0, then a commit that adds ``plot_zebra_curve``."""
    path.mkdir(parents=True, exist_ok=True)
    _git(path, "init", "-q", "-b", "main")
    for stage, message in (("base", "Initial release"), ("head", "Add the zebra curve (#7)")):
        for relative, content in REPO_FILES[stage].items():
            target = path / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        _git(path, "add", "-A")
        _git(path, "commit", "-q", "-m", message)
        if stage == "base":
            _git(path, "tag", "1.0.0")
