#!/usr/bin/env bash
#
# Look up the pinned manifest digest for a given Ollama LLM model tag.
#
# Usage: llm_pin_for.sh <model:tag> [pins_file]
#

set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <model:tag> [pins_file]" >&2
  exit 1
fi

MODEL_SPEC="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PINS_FILE="${2:-$SCRIPT_DIR/pins.env}"

if [ ! -f "$PINS_FILE" ]; then
  echo "Error: pins file not found at '$PINS_FILE'" >&2
  exit 1
fi

# Load pins
# shellcheck source=/dev/null
source "$PINS_FILE"

if [ "$MODEL_SPEC" = "${OLLAMA_LLM_MODEL:-}" ]; then
  DIGEST="${OLLAMA_LLM_MANIFEST_DIGEST:-}"
  if [ -z "$DIGEST" ]; then
    echo "Error: OLLAMA_LLM_MANIFEST_DIGEST is unset or empty in '$PINS_FILE'" >&2
    exit 1
  fi
  printf "%s\n" "$DIGEST"
  exit 0
elif [ "$MODEL_SPEC" = "${OLLAMA_LLM_ALT_MODEL:-}" ]; then
  DIGEST="${OLLAMA_LLM_ALT_MANIFEST_DIGEST:-}"
  if [ -z "$DIGEST" ]; then
    echo "Error: OLLAMA_LLM_ALT_MANIFEST_DIGEST is unset or empty in '$PINS_FILE'" >&2
    exit 1
  fi
  printf "%s\n" "$DIGEST"
  exit 0
else
  exit 0
fi
