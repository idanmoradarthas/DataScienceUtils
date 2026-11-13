#!/usr/bin/env bash
#
# Pull a dedup LLM judge into a running Ollama server and verify it against its pinned manifest digest.
#
# Usage: pull_llm.sh <model:tag> [pins_file]
#
# Used by setup_ollama.sh for the first judge and by the PR workflow for the second judge. Models without a pin run
# unpinned and their local manifest sha256 is printed for provenance.
#

set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <model:tag> [pins_file]" >&2
  exit 1
fi

LLM_MODEL_CHOICE="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PINS_FILE="${2:-$SCRIPT_DIR/pins.env}"

if [ ! -f "$PINS_FILE" ]; then
  echo "Error: pins.env file not found at $PINS_FILE" >&2
  exit 1
fi

# shellcheck source=/dev/null
source "$PINS_FILE"

export OLLAMA_HOST="${OLLAMA_HOST:-127.0.0.1:11434}"

# setup_ollama.sh installs Ollama into ~/.local/bin and only exports PATH inside its own shell, so a later step needs it.
if ! command -v ollama >/dev/null 2>&1; then
  export PATH="$HOME/.local/bin:$PATH"
fi
if ! command -v ollama >/dev/null 2>&1; then
  echo "Error: 'ollama' executable not found; run setup_ollama.sh first." >&2
  exit 1
fi

MODELS_CACHE_DIR="${MODELS_CACHE_DIR:-$HOME/.cache/tier2_models}"
mkdir -p "$MODELS_CACHE_DIR"

echo "Pulling LLM model '$LLM_MODEL_CHOICE'..."
PULL_LOG="$MODELS_CACHE_DIR/ollama_pull.log"
if ! ollama pull "$LLM_MODEL_CHOICE" > "$PULL_LOG" 2>&1; then
  echo "Error: 'ollama pull' failed. Full output:" >&2
  sed -E 's/\x1b\[[0-9;?]*[a-zA-Z]//g' "$PULL_LOG" | tr '\r' '\n' >&2
  exit 1
fi
sed -E 's/\x1b\[[0-9;?]*[a-zA-Z]//g' "$PULL_LOG" | tr '\r' '\n' | tail -n 15

LLM_PIN=$("$SCRIPT_DIR/llm_pin_for.sh" "$LLM_MODEL_CHOICE" "$PINS_FILE")
if [ -n "$LLM_PIN" ]; then
  echo "Verifying LLM manifest digest against pin ($LLM_PIN)..."
  "$SCRIPT_DIR/verify_llm_manifest.sh" "$LLM_MODEL_CHOICE" "$LLM_PIN"

  MODEL_INFO=$(curl -s "http://$OLLAMA_HOST/api/show" -d "{\"model\": \"$LLM_MODEL_CHOICE\"}")
  echo "Model details: $(echo "$MODEL_INFO" | jq -c '.details // empty' 2>/dev/null || true)"
  echo "✓ Pulled LLM: $LLM_MODEL_CHOICE (verified against pinned digest $LLM_PIN)"
else
  echo "LLM '$LLM_MODEL_CHOICE' has no pinned digest (pinned models: ${OLLAMA_LLM_MODEL:-}, ${OLLAMA_LLM_ALT_MODEL:-}); recording its manifest digest for provenance:"
  "$SCRIPT_DIR/verify_llm_manifest.sh" "$LLM_MODEL_CHOICE" || echo "Warning: could not read the local manifest for $LLM_MODEL_CHOICE" >&2
  MODEL_INFO=$(curl -s "http://$OLLAMA_HOST/api/show" -d "{\"model\": \"$LLM_MODEL_CHOICE\"}")
  echo "Model details: $(echo "$MODEL_INFO" | jq -c '.details // empty' 2>/dev/null || true)"
fi
