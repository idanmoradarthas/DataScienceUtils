#!/bin/bash
# Comprehensive installer test suite for install.sh
set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEST_TMP=$(mktemp -d)

cleanup() {
  if [ -n "${SERVER_PID:-}" ]; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  rm -rf "$TEST_TMP"
}
trap cleanup EXIT

echo "=== Running Comprehensive Installer Test Suite ==="

# Start mock HTTP server
PORT=18893
python3 -m http.server "$PORT" --directory "$REPO_ROOT" >/dev/null 2>&1 &
SERVER_PID=$!
sleep 1

# Scenario 1: Complete success
echo ""
echo "--- Scenario 1: Complete Success ---"
DIR1="$TEST_TMP/scenario1"
mkdir -p "$DIR1"
(
  cd "$DIR1"
  RAW_URL="http://127.0.0.1:$PORT" bash "$REPO_ROOT/install.sh" --skills-only --silent --tools claude
)
RC1=$?
if [ $RC1 -eq 0 ] && \
   [ -f "$DIR1/.claude/skills/ds-utils-metrics/SKILL.md" ] && \
   [ -f "$DIR1/.claude/skills/ds-utils-metrics/references/api_reference.md" ] && \
   [ -f "$DIR1/.agents/skills/ds-utils-xai/references/api_reference.md" ]; then
  echo "✓ Scenario 1 passed: All skills and references installed, exited 0."
else
  echo "✗ Scenario 1 FAILED! rc=$RC1" >&2
  exit 1
fi

# Scenario 2: Already-present skips (preserve existing user content)
echo ""
echo "--- Scenario 2: Already-Present Skips ---"
echo "custom_user_note" > "$DIR1/.claude/skills/ds-utils-metrics/custom.txt"
(
  cd "$DIR1"
  RAW_URL="http://127.0.0.1:$PORT" bash "$REPO_ROOT/install.sh" --skills-only --silent --tools claude
)
RC2=$?
if [ $RC2 -eq 0 ] && [ -f "$DIR1/.claude/skills/ds-utils-metrics/custom.txt" ]; then
  echo "✓ Scenario 2 passed: Pre-existing valid skills skipped without overwriting extra files, exited 0."
else
  echo "✗ Scenario 2 FAILED! rc=$RC2" >&2
  exit 1
fi

# Scenario 3: One early failure followed by later successes
echo ""
echo "--- Scenario 3: Early Failure Followed by Later Successes ---"
MOCK_ROOT="$TEST_TMP/mock_repo"
mkdir -p "$MOCK_ROOT/skills"
cp -r "$REPO_ROOT/skills/"* "$MOCK_ROOT/skills/"
# Sabotage metrics skill by removing SKILL.md
rm -f "$MOCK_ROOT/skills/ds-utils-metrics/SKILL.md"

PORT_MOCK=18894
python3 -m http.server "$PORT_MOCK" --directory "$MOCK_ROOT" >/dev/null 2>&1 &
MOCK_PID=$!
sleep 1

DIR3="$TEST_TMP/scenario3"
mkdir -p "$DIR3"
(
  cd "$DIR3"
  RAW_URL="http://127.0.0.1:$PORT_MOCK" bash "$REPO_ROOT/install.sh" --skills-only --silent --tools claude
)
RC3=$?
kill "$MOCK_PID" 2>/dev/null || true

# metrics must have failed and been cleaned up, while preprocess and other skills succeeded
if [ $RC3 -ne 0 ] && \
   [ ! -d "$DIR3/.claude/skills/ds-utils-metrics" ] && \
   [ -f "$DIR3/.claude/skills/ds-utils-preprocess/SKILL.md" ] && \
   [ -f "$DIR3/.claude/skills/ds-utils-preprocess/references/api_reference.md" ]; then
  echo "✓ Scenario 3 passed: Early failure continued to install later skills and exited non-zero ($RC3)."
else
  echo "✗ Scenario 3 FAILED! rc=$RC3" >&2
  exit 1
fi

# Scenario 4: All downloads failing
echo ""
echo "--- Scenario 4: All Downloads Failing ---"
DIR4="$TEST_TMP/scenario4"
mkdir -p "$DIR4"
(
  cd "$DIR4"
  RAW_URL="http://127.0.0.1:1" bash "$REPO_ROOT/install.sh" --skills-only --silent --tools claude
)
RC4=$?
if [ $RC4 -ne 0 ] && [ ! -d "$DIR4/.claude/skills/ds-utils-metrics" ]; then
  echo "✓ Scenario 4 passed: All failing exited non-zero ($RC4) and left no partial files."
else
  echo "✗ Scenario 4 FAILED! rc=$RC4" >&2
  exit 1
fi

echo ""
echo "=== All 4 installer scenarios passed with 100% truthful outcomes! ==="
