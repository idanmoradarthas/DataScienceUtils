#!/usr/bin/env bash
#
# Setup script for Ollama, Nemotron-3 embedding GGUF, and dedup LLM.
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PINS_FILE="$SCRIPT_DIR/pins.env"

# Discovery mode
if [ "${1:-}" = "--discover" ]; then
  echo "=== Discovering current pins from upstream sources ==="
  echo "--- Ollama latest release ---"
  OLLAMA_LATEST=$(curl -s https://api.github.com/repos/ollama/ollama/releases/latest)
  OLLAMA_VER=$(echo "$OLLAMA_LATEST" | jq -r '.tag_name')
  SHA256SUMS=$(curl -sL "https://github.com/ollama/ollama/releases/download/$OLLAMA_VER/sha256sum.txt")
  OLLAMA_SHA=$(echo "$SHA256SUMS" | grep -F "ollama-linux-amd64.tar.zst" | awk '{print $1}')
  echo "OLLAMA_VERSION=\"$OLLAMA_VER\""
  echo "OLLAMA_LINUX_AMD64_SHA256=\"$OLLAMA_SHA\""

  echo ""
  echo "--- GGUF blobs & commit ---"
  GGUF_INFO=$(curl -s "https://huggingface.co/api/models/NeoRoth/nemotron-3-embed-1b-gguf?blobs=true")
  GGUF_SHA=$(echo "$GGUF_INFO" | jq -r '.sha')
  echo "GGUF_REPO=\"NeoRoth/nemotron-3-embed-1b-gguf\""
  echo "GGUF_REVISION=\"$GGUF_SHA\""
  for s in q8_0 f16 q4_k_m; do
    fname="nemotron-3-embed-1b-$s.gguf"
    fsha=$(echo "$GGUF_INFO" | jq -r --arg f "$fname" '.siblings[] | select(.rfilename == $f) | .lfs.sha256')
    upper_s=$(echo "$s" | tr '[:lower:]' '[:upper:]')
    echo "GGUF_SHA256_${upper_s}=\"$fsha\""
  done

  echo ""
  echo "--- Upstream BF16 repo ---"
  BF16_INFO=$(curl -s "https://huggingface.co/api/models/nvidia/Nemotron-3-Embed-1B-BF16")
  BF16_SHA=$(echo "$BF16_INFO" | jq -r '.sha')
  echo "UPSTREAM_BF16_REPO=\"nvidia/Nemotron-3-Embed-1B-BF16\""
  echo "UPSTREAM_BF16_REVISION=\"$BF16_SHA\""

  echo ""
  echo "--- LLM tag digest ---"
  LLM_TAG="granite4.1:3b"
  echo "OLLAMA_LLM_MODEL=\"$LLM_TAG\""
  echo "OLLAMA_LLM_ALT_MODEL=\"phi4-mini:3.8b\""
  echo "# OLLAMA_LLM_MANIFEST_DIGEST: take it from the CI log line \"Local manifest sha256:\" after a run (see pins.env comment)."
  exit 0
fi

if [ ! -f "$PINS_FILE" ]; then
  echo "Error: pins.env file not found at $PINS_FILE" >&2
  exit 1
fi

# Load pins
# shellcheck source=/dev/null
source "$PINS_FILE"

# Validate that pins are non-empty
for var in OLLAMA_VERSION OLLAMA_LINUX_AMD64_SHA256 GGUF_REPO GGUF_REVISION \
           GGUF_SHA256_Q8_0 GGUF_SHA256_F16 GGUF_SHA256_Q4_K_M \
           UPSTREAM_BF16_REPO UPSTREAM_BF16_REVISION \
           OLLAMA_LLM_MODEL OLLAMA_LLM_MANIFEST_DIGEST \
           OLLAMA_LLM_ALT_MODEL OLLAMA_LLM_ALT_MANIFEST_DIGEST; do
  if [ -z "${!var:-}" ]; then
    echo "Error: Pin '$var' is not set in $PINS_FILE. Run $0 --discover or set pins manually." >&2
    exit 1
  fi
done

EMBEDDING_FILE_CHOICE="${1:-q8_0}"
case "$EMBEDDING_FILE_CHOICE" in
  q8_0)
    GGUF_FILENAME="nemotron-3-embed-1b-q8_0.gguf"
    EXPECTED_GGUF_SHA256="$GGUF_SHA256_Q8_0"
    ;;
  f16)
    GGUF_FILENAME="nemotron-3-embed-1b-f16.gguf"
    EXPECTED_GGUF_SHA256="$GGUF_SHA256_F16"
    ;;
  q4_k_m)
    GGUF_FILENAME="nemotron-3-embed-1b-q4_k_m.gguf"
    EXPECTED_GGUF_SHA256="$GGUF_SHA256_Q4_K_M"
    ;;
  *)
    echo "Error: Unknown embedding_file choice '$EMBEDDING_FILE_CHOICE'. Must be q8_0, f16, or q4_k_m." >&2
    exit 1
    ;;
esac

LLM_MODEL_CHOICE="${2:-$OLLAMA_LLM_MODEL}"
TIER2_SKIP_LLM="${TIER2_SKIP_LLM:-0}"

MODELS_CACHE_DIR="${MODELS_CACHE_DIR:-$HOME/.cache/tier2_models}"
mkdir -p "$MODELS_CACHE_DIR"

echo "=========================================================="
echo "Setting up Ollama and Models for SkillEvaluator Tier 2"
echo "  Embedding GGUF: $GGUF_FILENAME"
if [ "$TIER2_SKIP_LLM" = "1" ]; then
  echo "  LLM Model:      (skipped)"
else
  echo "  LLM Model:      $LLM_MODEL_CHOICE"
fi
echo "=========================================================="

# 1. Download & verify GGUF
GGUF_PATH="$MODELS_CACHE_DIR/$GGUF_FILENAME"
if [ -f "$GGUF_PATH" ]; then
  echo "Checking existing cached GGUF: $GGUF_PATH"
  ACTUAL_SHA=$(sha256sum "$GGUF_PATH" | awk '{print $1}')
  if [ "$ACTUAL_SHA" != "$EXPECTED_GGUF_SHA256" ]; then
    echo "Cached GGUF sha256 mismatch. Re-downloading..."
    rm -f "$GGUF_PATH"
  fi
fi

if [ ! -f "$GGUF_PATH" ]; then
  GGUF_URL="https://huggingface.co/$GGUF_REPO/resolve/$GGUF_REVISION/$GGUF_FILENAME"
  echo "Downloading GGUF from $GGUF_URL ..."
  curl -L -f --progress-bar "$GGUF_URL" -o "$GGUF_PATH.tmp"
  ACTUAL_SHA=$(sha256sum "$GGUF_PATH.tmp" | awk '{print $1}')
  if [ "$ACTUAL_SHA" != "$EXPECTED_GGUF_SHA256" ]; then
    echo "Error: GGUF sha256 verification failed!" >&2
    echo "  Expected: $EXPECTED_GGUF_SHA256" >&2
    echo "  Actual:   $ACTUAL_SHA" >&2
    rm -f "$GGUF_PATH.tmp"
    exit 1
  fi
  mv "$GGUF_PATH.tmp" "$GGUF_PATH"
  echo "✓ GGUF verified and saved to $GGUF_PATH"
fi

# 2. Install Ollama if not present or version mismatch
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
fi

# 3. Start Ollama server in background if not already running
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

# Check TIER2_EMBED_ADD_BOS mode
TIER2_EMBED_ADD_BOS="${TIER2_EMBED_ADD_BOS:-disabled}"
if [ "$TIER2_EMBED_ADD_BOS" != "default" ] && [ "$TIER2_EMBED_ADD_BOS" != "disabled" ]; then
  echo "Error: TIER2_EMBED_ADD_BOS must be 'default' or 'disabled' (got '$TIER2_EMBED_ADD_BOS')" >&2
  exit 1
fi
# The models cache key is shared by both BOS modes, so a sha file from an earlier run may be restored from cache.
rm -f "$MODELS_CACHE_DIR/patched_gguf.sha256"
MODELFILE_GGUF_PATH="$GGUF_PATH"
PATCHED_GGUF_PATH=""

if [ "$TIER2_EMBED_ADD_BOS" = "disabled" ]; then
  echo "Opt-in: patching GGUF to disable implicit BOS token..."
  python3 -m pip install --quiet "gguf==0.19.0"
  PATCHED_GGUF_PATH="$MODELS_CACHE_DIR/$(basename "$GGUF_PATH" .gguf).no-bos.gguf"
  python3 "$SCRIPT_DIR/patch_gguf_add_bos.py" "$GGUF_PATH" "$PATCHED_GGUF_PATH"
  sha256sum "$PATCHED_GGUF_PATH" | awk '{print $1}' > "$MODELS_CACHE_DIR/patched_gguf.sha256"
  MODELFILE_GGUF_PATH="$PATCHED_GGUF_PATH"
fi

# 4. Create embedding model in Ollama
MODELFILE="$MODELS_CACHE_DIR/Modelfile.nemotron"
cat << EOF > "$MODELFILE"
FROM $MODELFILE_GGUF_PATH
# Embedder context is intentionally 4096 (observed embedding prompts peak below 1k tokens, no truncation); OLLAMA_CONTEXT_LENGTH only affects the LLM.
PARAMETER num_ctx 4096
EOF

echo "Creating Ollama model 'nemotron-3-embed-1b' from $MODELFILE_GGUF_PATH ..."
CREATE_LOG="$MODELS_CACHE_DIR/ollama_create.log"
if ! ollama create nemotron-3-embed-1b -f "$MODELFILE" > "$CREATE_LOG" 2>&1; then
  echo "Error: 'ollama create' failed. Full output:" >&2
  sed -E 's/\x1b\[[0-9;?]*[a-zA-Z]//g' "$CREATE_LOG" | tr '\r' '\n' >&2
  [ -n "$PATCHED_GGUF_PATH" ] && [ -f "$PATCHED_GGUF_PATH" ] && rm -f "$PATCHED_GGUF_PATH"
  exit 1
fi
sed -E 's/\x1b\[[0-9;?]*[a-zA-Z]//g' "$CREATE_LOG" | tr '\r' '\n' | tail -n 15
echo "✓ Ollama model 'nemotron-3-embed-1b' created."

# Delete patched GGUF if created so it does not end up in actions/cache
if [ -n "$PATCHED_GGUF_PATH" ] && [ -f "$PATCHED_GGUF_PATH" ]; then
  echo "Cleaning up temporary patched GGUF: $PATCHED_GGUF_PATH"
  rm -f "$PATCHED_GGUF_PATH"
fi

# 5. Pull and verify LLM model
if [ "$TIER2_SKIP_LLM" = "1" ]; then
  echo "Skipping LLM pull (TIER2_SKIP_LLM=1)"
else
  "$SCRIPT_DIR/pull_llm.sh" "$LLM_MODEL_CHOICE" "$PINS_FILE"
fi

echo ""
echo "=== Setup complete! Available Ollama models: ==="
ollama list
