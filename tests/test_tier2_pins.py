"""Tests for Tier 2 deduplication judge pins and llm_pin_for.sh helper."""

from pathlib import Path
import re
import shutil
import subprocess
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
PINS_FILE = REPO_ROOT / ".github" / "tier2" / "pins.env"
PIN_HELPER = REPO_ROOT / ".github" / "tier2" / "llm_pin_for.sh"


def parse_env_file(path: Path) -> dict[str, str]:
    """Parse simple KEY="VALUE" environment file."""
    pins = {}
    pattern = re.compile(r'^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*"([^"]*)"\s*$')
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            m = pattern.match(line)
            if m:
                pins[m.group(1)] = m.group(2)
    return pins


def test_pins_env_parses_and_llm_pins_are_valid():
    """Verify pins.env parses and LLM models and digests are non-empty, distinct, and valid sha256."""
    assert PINS_FILE.is_file(), f"pins.env missing at {PINS_FILE}"
    pins = parse_env_file(PINS_FILE)

    assert "OLLAMA_LLM_MODEL" in pins
    assert "OLLAMA_LLM_MANIFEST_DIGEST" in pins
    assert "OLLAMA_LLM_ALT_MODEL" in pins
    assert "OLLAMA_LLM_ALT_MANIFEST_DIGEST" in pins

    model = pins["OLLAMA_LLM_MODEL"]
    alt_model = pins["OLLAMA_LLM_ALT_MODEL"]
    digest = pins["OLLAMA_LLM_MANIFEST_DIGEST"]
    alt_digest = pins["OLLAMA_LLM_ALT_MANIFEST_DIGEST"]

    assert model, "OLLAMA_LLM_MODEL is empty"
    assert alt_model, "OLLAMA_LLM_ALT_MODEL is empty"
    assert model != alt_model, "OLLAMA_LLM_MODEL and OLLAMA_LLM_ALT_MODEL must be different"

    sha256_pattern = re.compile(r"^[0-9a-f]{64}$")
    assert sha256_pattern.match(digest), f"OLLAMA_LLM_MANIFEST_DIGEST '{digest}' is not a valid 64-char hex sha256"
    assert sha256_pattern.match(alt_digest), (
        f"OLLAMA_LLM_ALT_MANIFEST_DIGEST '{alt_digest}' is not a valid 64-char hex sha256"
    )


@pytest.mark.skipif(
    sys.platform == "win32" or shutil.which("bash") is None,
    reason="llm_pin_for.sh only runs on Linux CI runners; on Windows 'bash' is the WSL launcher stub",
)
class TestLlmPinForHelper:
    """Tests for .github/tier2/llm_pin_for.sh."""

    def run_helper(self, model: str, pins_file: Path | str | None = None) -> subprocess.CompletedProcess:
        """Run llm_pin_for.sh for a model, optionally against a custom pins file."""
        args = ["bash", str(PIN_HELPER), model]
        if pins_file is not None:
            args.append(str(pins_file))
        return subprocess.run(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )

    def test_default_model_returns_default_digest(self):
        """The default model resolves to the default manifest digest."""
        pins = parse_env_file(PINS_FILE)
        proc = self.run_helper(pins["OLLAMA_LLM_MODEL"])
        assert proc.returncode == 0
        assert proc.stdout.strip() == pins["OLLAMA_LLM_MANIFEST_DIGEST"]
        assert proc.stderr == ""

    def test_alt_model_returns_alt_digest(self):
        """The alternate model resolves to the alternate manifest digest."""
        pins = parse_env_file(PINS_FILE)
        proc = self.run_helper(pins["OLLAMA_LLM_ALT_MODEL"])
        assert proc.returncode == 0
        assert proc.stdout.strip() == pins["OLLAMA_LLM_ALT_MANIFEST_DIGEST"]
        assert proc.stderr == ""

    def test_unpinned_model_returns_empty_and_exits_zero(self):
        """A model without a pin prints nothing and exits 0."""
        proc = self.run_helper("gemma4:e4b")
        assert proc.returncode == 0
        assert proc.stdout == ""

    def test_empty_digest_fails_closed(self, tmp_path):
        """A matching model with an empty digest exits 1 and prints nothing."""
        temp_pins = tmp_path / "pins.env"
        temp_pins.write_text(
            'OLLAMA_LLM_MODEL="granite4.1:3b"\n'
            'OLLAMA_LLM_MANIFEST_DIGEST=""\n'
            'OLLAMA_LLM_ALT_MODEL="phi4-mini:3.8b"\n'
            'OLLAMA_LLM_ALT_MANIFEST_DIGEST="b9db5daa27583d278796ac1fceaa934d09d09b3688a82e5889d089535b8c7a38"\n',
            encoding="utf-8",
        )
        proc = self.run_helper("granite4.1:3b", temp_pins)
        assert proc.returncode == 1
        assert proc.stdout == ""
        assert "unset or empty" in proc.stderr

        # Also test alt model with empty digest
        temp_pins_alt = tmp_path / "pins_alt.env"
        temp_pins_alt.write_text(
            'OLLAMA_LLM_MODEL="granite4.1:3b"\n'
            'OLLAMA_LLM_MANIFEST_DIGEST="59b95aeac3cf33b93391c475aa78a53230cdfb06ea1063485e4d28d6c48ae2a8"\n'
            'OLLAMA_LLM_ALT_MODEL="phi4-mini:3.8b"\n'
            'OLLAMA_LLM_ALT_MANIFEST_DIGEST=""\n',
            encoding="utf-8",
        )
        proc_alt = self.run_helper("phi4-mini:3.8b", temp_pins_alt)
        assert proc_alt.returncode == 1
        assert proc_alt.stdout == ""
        assert "unset or empty" in proc_alt.stderr

    def test_missing_pins_file_exits_one(self, tmp_path):
        """A missing pins file exits 1 with a clear message."""
        non_existent = tmp_path / "does_not_exist.env"
        proc = self.run_helper("granite4.1:3b", non_existent)
        assert proc.returncode == 1
        assert proc.stdout == ""
        assert "pins file not found" in proc.stderr
