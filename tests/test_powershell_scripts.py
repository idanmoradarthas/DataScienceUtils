"""Regression test verifying all PowerShell and batch scripts are pure ASCII.

Windows PowerShell 5.1 and cmd.exe read BOM-less scripts using the ANSI code page
(e.g., Windows-1252), corrupting non-ASCII UTF-8 bytes and causing parse errors.
"""

from pathlib import Path
import pytest


def get_shell_scripts():
    """Collect all .ps1 and .bat scripts in the repository root."""
    repo_root = Path(__file__).resolve().parent.parent
    scripts = sorted(repo_root.glob("*.ps1")) + sorted(repo_root.glob("*.bat"))
    return [pytest.param(p, id=p.name) for p in scripts]


@pytest.mark.parametrize("script_path", get_shell_scripts())
def test_scripts_are_pure_ascii(script_path):
    """Assert that script files contain only ASCII bytes."""
    raw_bytes = script_path.read_bytes()
    if not raw_bytes.isascii():
        lines = raw_bytes.splitlines()
        offending = []
        for line_num, line in enumerate(lines, 1):
            if not line.isascii():
                bad_chars = [chr(b) if 32 <= b <= 126 else f"\\x{b:02x}" for b in line if b > 127]
                offending.append(f"  Line {line_num}: {bad_chars}")
        offending_str = "\n".join(offending)
        pytest.fail(
            f"{script_path.name} contains non-ASCII characters:\n{offending_str}\n"
            "Windows PowerShell 5.1 and cmd.exe read BOM-less scripts as ANSI, "
            "causing fatal syntax/parse errors when non-ASCII UTF-8 sequences are interpreted."
        )
