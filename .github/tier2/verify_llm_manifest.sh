#!/usr/bin/env bash
#
# Verify local Ollama LLM manifest digest against expected sha256 pin.
#
# Usage: verify_llm_manifest.sh <model:tag> [expected_sha256]
#

set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <model:tag> [expected_sha256]" >&2
  exit 1
fi

MODEL_SPEC="$1"
HAVE_PIN=0
EXPECTED_SHA256=""
if [ "$#" -ge 2 ]; then
  HAVE_PIN=1
  EXPECTED_SHA256="$2"
  if [ -z "$EXPECTED_SHA256" ]; then
    echo "Error: expected sha256 pin for $MODEL_SPEC is empty; refusing to skip verification." >&2
    exit 1
  fi
fi

LLM_NAME="${MODEL_SPEC%%:*}"
LLM_TAG="${MODEL_SPEC##*:}"
[ "$LLM_TAG" = "$LLM_NAME" ] && LLM_TAG="latest"

OLLAMA_MODELS_DIR="${OLLAMA_MODELS:-$HOME/.ollama/models}"
MANIFEST_PATH="$OLLAMA_MODELS_DIR/manifests/registry.ollama.ai/library/$LLM_NAME/$LLM_TAG"

if [ ! -f "$MANIFEST_PATH" ]; then
  echo "Error: Local Ollama manifest not found at $MANIFEST_PATH" >&2
  exit 1
fi

ACTUAL_SHA256=$(sha256sum "$MANIFEST_PATH" | awk '{print $1}')
echo "Local manifest sha256: $ACTUAL_SHA256"

if command -v jq >/dev/null 2>&1; then
  jq -r '.layers[]? | "layer \(.digest) \(.size) \(.mediaType)"' "$MANIFEST_PATH" 2>/dev/null || true
  CONFIG_DIGEST=$(jq -r '.config.digest // empty' "$MANIFEST_PATH" 2>/dev/null || true)
  if [ -n "$CONFIG_DIGEST" ]; then
    echo "config $CONFIG_DIGEST"
  fi
fi

if [ "$HAVE_PIN" -eq 0 ]; then
  echo "No pin given: manifest NOT verified."
  exit 0
fi

if [ "$ACTUAL_SHA256" != "$EXPECTED_SHA256" ]; then
  echo "Error: LLM manifest digest mismatch for $MODEL_SPEC!" >&2
  echo "  Expected: $EXPECTED_SHA256" >&2
  echo "  Actual:   $ACTUAL_SHA256" >&2
  if command -v jq >/dev/null 2>&1; then
    jq -c . "$MANIFEST_PATH" 2>/dev/null || cat "$MANIFEST_PATH"
  else
    cat "$MANIFEST_PATH"
  fi
  exit 1
fi

exit 0
