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
  if [ -n "${MOCK_PID:-}" ]; then
    kill "$MOCK_PID" 2>/dev/null || true
    wait "$MOCK_PID" 2>/dev/null || true
  fi
  rm -rf "$TEST_TMP"
}
trap cleanup EXIT

echo "=== Running Comprehensive Installer Test Suite ==="

# Get a random free port between 20000 and 40000
get_free_port() {
  python3 -c '
import socket
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind(("", 0))
    print(s.getsockname()[1])
'
}

# Wait for server on port
wait_for_port() {
  local port=$1
  for _ in $(seq 1 50); do
    if curl -s "http://127.0.0.1:$port" >/dev/null 2>&1; then
      return 0
    fi
    sleep 0.1
  done
  echo "Timeout waiting for port $port" >&2
  return 1
}

# Start mock HTTP server
PORT=$(get_free_port)
python3 -m http.server "$PORT" --directory "$REPO_ROOT" >/dev/null 2>&1 &
SERVER_PID=$!
wait_for_port "$PORT"

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

# Scenario 3: Upgrading legacy install that has only SKILL.md and no references/
echo ""
echo "--- Scenario 3: Refresh Legacy Install (missing references/) ---"
DIR_LEGACY="$TEST_TMP/scenario_legacy"
mkdir -p "$DIR_LEGACY/.claude/skills/ds-utils-metrics"
echo "old skill" > "$DIR_LEGACY/.claude/skills/ds-utils-metrics/SKILL.md"
(
  cd "$DIR_LEGACY"
  RAW_URL="http://127.0.0.1:$PORT" bash "$REPO_ROOT/install.sh" --skills-only --silent --tools claude
)
RC_LEGACY=$?
if [ $RC_LEGACY -eq 0 ] && [ -f "$DIR_LEGACY/.claude/skills/ds-utils-metrics/references/api_reference.md" ]; then
  echo "✓ Scenario 3 passed: Incomplete legacy installation refreshed with references without requiring --force."
else
  echo "✗ Scenario 3 FAILED! rc=$RC_LEGACY" >&2
  exit 1
fi

# Scenario 4: Overwrite with --force
echo ""
echo "--- Scenario 4: Overwrite with --force ---"
echo "modified content" > "$DIR1/.claude/skills/ds-utils-metrics/SKILL.md"
(
  cd "$DIR1"
  RAW_URL="http://127.0.0.1:$PORT" bash "$REPO_ROOT/install.sh" --skills-only --silent --tools claude --force
)
RC_FORCE=$?
if [ $RC_FORCE -eq 0 ] && ! grep -q "modified content" "$DIR1/.claude/skills/ds-utils-metrics/SKILL.md"; then
  echo "✓ Scenario 4 passed: --force successfully overwrites modified files."
else
  echo "✗ Scenario 4 FAILED! rc=$RC_FORCE" >&2
  exit 1
fi

# Scenario 5: One early failure followed by later successes
echo ""
echo "--- Scenario 5: Early Failure Followed by Later Successes ---"
MOCK_ROOT="$TEST_TMP/mock_repo"
mkdir -p "$MOCK_ROOT/skills"
cp -r "$REPO_ROOT/skills/"* "$MOCK_ROOT/skills/"
# Sabotage metrics skill by removing SKILL.md
rm -f "$MOCK_ROOT/skills/ds-utils-metrics/SKILL.md"

PORT_MOCK=$(get_free_port)
python3 -m http.server "$PORT_MOCK" --directory "$MOCK_ROOT" >/dev/null 2>&1 &
MOCK_PID=$!
wait_for_port "$PORT_MOCK"

DIR5="$TEST_TMP/scenario5"
mkdir -p "$DIR5"
(
  cd "$DIR5"
  RAW_URL="http://127.0.0.1:$PORT_MOCK" bash "$REPO_ROOT/install.sh" --skills-only --silent --tools claude
)
RC5=$?
kill "$MOCK_PID" 2>/dev/null || true
unset MOCK_PID

# metrics must have failed and been cleaned up, while preprocess and other skills succeeded
if [ $RC5 -ne 0 ] && \
   [ ! -d "$DIR5/.claude/skills/ds-utils-metrics" ] && \
   [ -f "$DIR5/.claude/skills/ds-utils-preprocess/SKILL.md" ] && \
   [ -f "$DIR5/.claude/skills/ds-utils-preprocess/references/api_reference.md" ]; then
  echo "✓ Scenario 5 passed: Early failure continued to install later skills and exited non-zero ($RC5)."
else
  echo "✗ Scenario 5 FAILED! rc=$RC5" >&2
  exit 1
fi

# Scenario 6: All downloads failing
echo ""
echo "--- Scenario 6: All Downloads Failing ---"
DIR6="$TEST_TMP/scenario6"
mkdir -p "$DIR6"
(
  cd "$DIR6"
  RAW_URL="http://127.0.0.1:1" bash "$REPO_ROOT/install.sh" --skills-only --silent --tools claude
)
RC6=$?
if [ $RC6 -ne 0 ] && [ ! -d "$DIR6/.claude/skills/ds-utils-metrics" ]; then
  echo "✓ Scenario 6 passed: All failing exited non-zero ($RC6) and left no partial files."
else
  echo "✗ Scenario 6 FAILED! rc=$RC6" >&2
  exit 1
fi

echo ""
echo "=== All 6 installer scenarios passed with 100% truthful outcomes! ==="
