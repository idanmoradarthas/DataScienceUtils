"""Tests for the single Ollama pin: one version in ``pins.env``, one installer, and every workflow goes through it."""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
GITHUB_DIR = REPO_ROOT / ".github"
TIER2_DIR = GITHUB_DIR / "tier2"
PINS_FILE = TIER2_DIR / "pins.env"
INSTALLER = TIER2_DIR / "install_ollama.sh"
SETUP = TIER2_DIR / "setup_ollama.sh"
WORKFLOWS = sorted((GITHUB_DIR / "workflows").glob("*.yml"))
PIN_LINE = re.compile(r'^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*"([^"]*)"\s*$')

linux_only = pytest.mark.skipif(
    not sys.platform.startswith("linux") or shutil.which("bash") is None,
    reason="the Ollama installer is written for the Linux runners",
)


def read_pins(path: Path = PINS_FILE) -> dict[str, str]:
    """Return the KEY="VALUE" pairs of a pins file."""
    pins = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = PIN_LINE.match(line)
        if match:
            pins[match.group(1)] = match.group(2)
    return pins


def test_the_ollama_pin_is_a_release_tag_and_a_sha256():
    """The version looks like ``vX.Y.Z`` and the tarball digest is 64 hexadecimal characters."""
    pins = read_pins()
    assert re.fullmatch(r"v\d+\.\d+\.\d+", pins["OLLAMA_VERSION"])
    assert re.fullmatch(r"[0-9a-f]{64}", pins["OLLAMA_LINUX_AMD64_SHA256"])


def test_the_pinned_version_and_digest_appear_nowhere_else_in_github_files():
    """Moving to another Ollama release means editing ``pins.env`` only."""
    pins = read_pins()
    others = [
        path
        for path in GITHUB_DIR.rglob("*")
        if path.is_file() and path != PINS_FILE and "__pycache__" not in path.parts and path.suffix != ".pyc"
    ]
    for path in others:
        text = path.read_text(encoding="utf-8", errors="ignore")
        assert pins["OLLAMA_VERSION"] not in text, f"{path} repeats the pinned Ollama version"
        assert pins["OLLAMA_LINUX_AMD64_SHA256"] not in text, f"{path} repeats the pinned Ollama digest"


def test_only_the_installer_downloads_or_starts_ollama():
    """Workflows and scripts install Ollama through ``install_ollama.sh``; nobody else fetches a release or serves."""
    for path in [*WORKFLOWS, *sorted(TIER2_DIR.glob("*.sh")), *sorted(TIER2_DIR.glob("*.py"))]:
        if path in {INSTALLER, SETUP}:
            continue
        text = path.read_text(encoding="utf-8")
        assert "ollama/ollama/releases" not in text, f"{path.name} downloads Ollama itself"
        assert "ollama.com/install" not in text, f"{path.name} runs the Ollama install script"
        assert "ollama serve" not in text, f"{path.name} starts the Ollama server itself"
    assert "ollama/ollama/releases/download" in INSTALLER.read_text(encoding="utf-8")
    setup_text = SETUP.read_text(encoding="utf-8")  # its --discover mode only prints candidate pins
    assert "tar --zstd" not in setup_text
    assert "ollama serve" not in setup_text


def test_every_workflow_that_uses_ollama_goes_through_the_installer():
    """A workflow that mentions Ollama calls ``install_ollama.sh`` or ``setup_ollama.sh``, which source the pin."""
    users = [path for path in WORKFLOWS if re.search(r"\bollama\b", path.read_text(encoding="utf-8"), re.IGNORECASE)]
    assert {path.name for path in users} >= {"release-notes.yml", "skills-tier2-pr.yml"}
    for path in users:
        text = path.read_text(encoding="utf-8")
        assert "install_ollama.sh" in text or "setup_ollama.sh" in text, f"{path.name} does not use the installer"
    assert 'source "$SCRIPT_DIR/install_ollama.sh"' in SETUP.read_text(encoding="utf-8")


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class _VersionServer:
    """A server that answers ``/api/version`` like a running Ollama."""

    def __init__(self) -> None:
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args: object) -> None:
                return

            def do_GET(self) -> None:  # noqa: N802 - http.server API
                body = b'{"version": "9.9.9"}'
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self) -> str:
        self.thread.start()
        return f"127.0.0.1:{self.server.server_address[1]}"

    def __exit__(self, *_exc: object) -> None:
        self.server.shutdown()
        self.server.server_close()


@linux_only
class TestInstaller:
    """Run ``install_ollama.sh`` with a fake ``ollama`` and a fake ``curl``; nothing is downloaded."""

    @staticmethod
    def _bin(tmp_path: Path, name: str, body: str) -> Path:
        directory = tmp_path / "bin"
        directory.mkdir(exist_ok=True)
        script = directory / name
        script.write_text("#!/usr/bin/env bash\n" + body, encoding="utf-8")
        script.chmod(0o755)
        return directory

    @staticmethod
    def _pins(tmp_path: Path, text: str = 'OLLAMA_VERSION="v9.9.9"\nOLLAMA_LINUX_AMD64_SHA256="{}"\n') -> Path:
        path = tmp_path / "pins.env"
        path.write_text(text.format("a" * 64), encoding="utf-8")
        return path

    @staticmethod
    def _run(tmp_path: Path, host: str, pins: Path) -> subprocess.CompletedProcess[str]:
        (tmp_path / "home").mkdir(exist_ok=True)
        env = {
            "PATH": f"{tmp_path / 'bin'}:/usr/bin:/bin",
            "HOME": str(tmp_path / "home"),
            "PINS_FILE": str(pins),
            "OLLAMA_HOST": host,
            "LANG": "C.UTF-8",
        }
        return subprocess.run(
            ["bash", str(INSTALLER)], env=env, capture_output=True, text=True, check=False, timeout=60
        )

    def test_an_installed_ollama_is_reused_and_the_server_is_left_alone(self, tmp_path: Path):
        """With the pinned version installed and a server answering, nothing is installed or started."""
        self._bin(tmp_path, "ollama", 'echo "ollama version is 9.9.9"\n')
        with _VersionServer() as host:
            result = self._run(tmp_path, host, self._pins(tmp_path))
        assert result.returncode == 0, result.stderr
        assert "Installing Ollama" not in result.stdout
        assert "Starting Ollama server" not in result.stdout
        assert "Ollama version: ollama version is 9.9.9" in result.stdout
        assert "Warning" not in result.stderr

    def test_a_different_installed_version_is_reported(self, tmp_path: Path):
        """An Ollama that is not the pinned version is used, with a warning."""
        self._bin(tmp_path, "ollama", 'echo "ollama version is 1.0.0"\n')
        with _VersionServer() as host:
            result = self._run(tmp_path, host, self._pins(tmp_path))
        assert result.returncode == 0, result.stderr
        assert "not the pinned v9.9.9" in result.stderr

    def test_the_server_is_started_when_none_is_running(self, tmp_path: Path):
        """The installer starts ``ollama serve`` in the background and waits until it answers."""
        port = _free_port()
        server_code = (
            "import http.server,json,os\n"
            "class H(http.server.BaseHTTPRequestHandler):\n"
            "    def log_message(self,*a): pass\n"
            "    def do_GET(self):\n"
            "        b=json.dumps({'version':'9.9.9'}).encode()\n"
            "        self.send_response(200); self.send_header('Content-Length',str(len(b))); self.end_headers()\n"
            "        self.wfile.write(b)\n"
            f"http.server.ThreadingHTTPServer(('127.0.0.1',{port}),H).serve_forever()\n"
        )
        script = tmp_path / "fake_server.py"
        script.write_text(server_code, encoding="utf-8")
        self._bin(
            tmp_path,
            "ollama",
            f'if [ "$1" = "serve" ]; then exec "{sys.executable}" "{script}"; fi\necho "ollama version is 9.9.9"\n',
        )
        result = self._run(tmp_path, f"127.0.0.1:{port}", self._pins(tmp_path))
        match = re.search(r"\(PID (\d+)\)", result.stdout)
        try:
            assert result.returncode == 0, result.stderr
            assert "Ollama server is responding" in result.stdout
            assert match, result.stdout
        finally:
            if match:
                try:
                    os.kill(int(match.group(1)), signal.SIGKILL)
                except ProcessLookupError:
                    pass

    def test_a_tarball_with_the_wrong_digest_is_refused(self, tmp_path: Path):
        """When Ollama is missing, the downloaded archive must match the pinned sha256 or the install stops."""
        if shutil.which("ollama", path="/usr/bin:/bin"):
            pytest.skip("a system Ollama is installed, so the installer would not download anything")
        self._bin(
            tmp_path,
            "curl",
            'while [ $# -gt 0 ]; do if [ "$1" = "-o" ]; then out="$2"; fi; shift; done\n'
            'echo "not the real tarball" > "$out"\n',
        )
        result = self._run(tmp_path, f"127.0.0.1:{_free_port()}", self._pins(tmp_path))
        assert result.returncode == 1
        assert "Ollama archive sha256 verification failed" in result.stderr
        assert not (tmp_path / "home" / ".local" / "bin").exists()

    def test_a_missing_pin_stops_the_installer(self, tmp_path: Path):
        """An empty pins file fails with the name of the missing pin."""
        self._bin(tmp_path, "ollama", 'echo "ollama version is 9.9.9"\n')
        result = self._run(tmp_path, f"127.0.0.1:{_free_port()}", self._pins(tmp_path, 'OLLAMA_VERSION="v9.9.9"\n'))
        assert result.returncode == 1
        assert "Pin 'OLLAMA_LINUX_AMD64_SHA256' is not set" in result.stderr
