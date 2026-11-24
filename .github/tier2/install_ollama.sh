#!/usr/bin/env bash
#
# Install the pinned Ollama release and start its server in the background.
#
# The Ollama version and the sha256 of its Linux tarball are pinned in pins.env (OLLAMA_VERSION and
# OLLAMA_LINUX_AMD64_SHA256) and nowhere else. Every workflow that needs Ollama goes through this script, directly or
# through setup_ollama.sh, so they all run the same release. To change the version, edit pins.env only.
#
# Usage: .github/tier2/install_ollama.sh        (as a workflow step; the server keeps running for later steps)
#        source .github/tier2/install_ollama.sh (setup_ollama.sh does this, so PATH and OLLAMA_* reach its shell)
#
# Server settings come from the environment; these three have defaults: OLLAMA_HOST, OLLAMA_NUM_PARALLEL (1),
# OLLAMA_MAX_LOADED_MODELS (2) and OLLAMA_KEEP_ALIVE (30m). Any other OLLAMA_* variable that is set (for example
# OLLAMA_CONTEXT_LENGTH, OLLAMA_FLASH_ATTENTION, OLLAMA_KV_CACHE_TYPE) is inherited by the server.
#

set -euo pipefail

_INSTALL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PINS_FILE="${PINS_FILE:-$_INSTALL_DIR/pins.env}"

if [ ! -f "$PINS_FILE" ]; then
  echo "Error: pins.env file not found at $PINS_FILE" >&2
  exit 1
fi

# shellcheck source=/dev/null
source "$PINS_FILE"

for var in OLLAMA_VERSION OLLAMA_LINUX_AMD64_SHA256; do
  if [ -z "${!var:-}" ]; then
    echo "Error: Pin '$var' is not set in $PINS_FILE." >&2
    exit 1
  fi
done

MODELS_CACHE_DIR="${MODELS_CACHE_DIR:-$HOME/.cache/tier2_models}"
mkdir -p "$MODELS_CACHE_DIR"

# 1. Install Ollama when it is not on PATH. An Ollama that is already installed is used as it is (a developer's own
# machine), and a version that differs from the pin is reported.
if ! command -v ollama >/dev/null 2>&1; then
  echo "Installing Ollama $OLLAMA_VERSION ..."
  OLLAMA_TAR_URL="https://github.com/ollama/ollama/releases/download/$OLLAMA_VERSION/ollama-linux-amd64.tar.zst"
  OLLAMA_ARCHIVE="$MODELS_CACHE_DIR/ollama-linux-amd64.tar.zst"
  curl -L -f -s "$OLLAMA_TAR_URL" -o "$OLLAMA_ARCHIVE"
  ACTUAL_TAR_SHA=$(sha256sum "$OLLAMA_ARCHIVE" | awk '{print $1}')
  if [ "$ACTUAL_TAR_SHA" != "$OLLAMA_LINUX_AMD64_SHA256" ]; then
    echo "Error: Ollama archive sha256 verification failed!" >&2
    echo "  Expected: $OLLAMA_LINUX_AMD64_SHA256" >&2
    echo "  Actual:   $ACTUAL_TAR_SHA" >&2
    exit 1
  fi
  mkdir -p "$HOME/.local/bin" "$HOME/.local/lib/ollama"
  tar --zstd -xf "$OLLAMA_ARCHIVE" -C "$HOME/.local"
  export PATH="$HOME/.local/bin:$PATH"
elif ! ollama --version 2>&1 | grep -qF "${OLLAMA_VERSION#v}"; then
  echo "Warning: the Ollama already installed is not the pinned $OLLAMA_VERSION; using it as it is." >&2
fi

# 2. Start the server in the background if it is not already running.
export OLLAMA_HOST="${OLLAMA_HOST:-127.0.0.1:11434}"
export OLLAMA_NUM_PARALLEL="${OLLAMA_NUM_PARALLEL:-1}"
export OLLAMA_MAX_LOADED_MODELS="${OLLAMA_MAX_LOADED_MODELS:-2}"
export OLLAMA_KEEP_ALIVE="${OLLAMA_KEEP_ALIVE:-30m}"

if ! curl -s "http://$OLLAMA_HOST/api/version" >/dev/null 2>&1; then
  echo "Starting Ollama server on $OLLAMA_HOST in background..."
  OLLAMA_LOG="${OLLAMA_LOG:-$MODELS_CACHE_DIR/ollama_server.log}"
  ollama serve > "$OLLAMA_LOG" 2>&1 &
  SERVER_PID=$!
  echo "Ollama server started (PID $SERVER_PID). Waiting for ready..."

  RETRIES=30
  until curl -s "http://$OLLAMA_HOST/api/version" >/dev/null 2>&1 || [ $RETRIES -eq 0 ]; do
    sleep 1
    RETRIES=$((RETRIES - 1))
  done

  if [ $RETRIES -eq 0 ]; then
    echo "Error: Ollama server failed to start within 30 seconds." >&2
    [ -f "$OLLAMA_LOG" ] && cat "$OLLAMA_LOG" >&2
    exit 1
  fi
  echo "✓ Ollama server is responding."
fi

echo "Ollama version: $(ollama --version 2>&1 || true)"
