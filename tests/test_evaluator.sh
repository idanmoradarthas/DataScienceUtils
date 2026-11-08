#!/usr/bin/env bash
# Test suite for eval-skills.sh: Scenarios 1-4 are hermetic using a stub skillevaluator;
# Scenario 5 is an integration check using real tools when available.
set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEST_TMP=$(mktemp -d)

cleanup() {
  rm -rf "$TEST_TMP"
}
trap cleanup EXIT

echo "=== Running Evaluator Test Suite (Hermetic & Integration) ==="

BIN_DIR="$TEST_TMP/bin"
mkdir -p "$BIN_DIR"

# Create stub skillevaluator
cat << 'EOF' > "$BIN_DIR/skillevaluator"
#!/usr/bin/env bash
cmd="$1"
shift

if [ "$cmd" = "--version" ]; then
  echo "skillevaluator, version 0.6.0 (stub)"
  exit 0
fi

if [ "$cmd" = "quality-check" ]; then
  target="$1"
  if [ -n "${STUB_FAIL_QUALITY_SKILL:-}" ] && [[ "$target" == *"$STUB_FAIL_QUALITY_SKILL"* ]]; then
    echo "[QUALITY-FAIL] Quality check failed for $target" >&2
    exit 1
  fi
  echo "Overall: 98.0/100 (Grade: A)"
  exit 0
fi

if [ "$cmd" = "tier1" ]; then
  # Flags: --no-llm <dir> [-o output_dir]
  target=""
  for arg in "$@"; do
    if [[ "$arg" != -* ]]; then
      target="$arg"
      break
    fi
  done
  if [ -n "${STUB_FAIL_SKILL:-}" ] && [[ "$target" == *"$STUB_FAIL_SKILL"* ]]; then
    echo "[TIER1-FAIL] Tier 1 checks failed for $target" >&2
    exit 1
  fi
  echo "Tier 1 validation passed for $target"
  exit 0
fi

echo "Unknown command: $cmd" >&2
exit 2
EOF
chmod +x "$BIN_DIR/skillevaluator"

ORIG_PATH="$PATH"

# Scenario 1: All pass -> eval-skills.sh exits 0 and all 6 rows show PASS
echo ""
echo "--- Scenario 1: All Pass ---"
OUT1=$(PATH="$BIN_DIR:$ORIG_PATH" "$REPO_ROOT/eval-skills.sh" 2>&1)
RC1=$?
if [ $RC1 -eq 0 ] && ! echo "$OUT1" | grep -q "FAIL"; then
  echo "✓ Scenario 1 passed: Exits 0 and all rows show PASS."
else
  echo "✗ Scenario 1 FAILED! rc=$RC1, output:\n$OUT1" >&2
  exit 1
fi

# Scenario 2: STUB_FAIL_SKILL=ds-utils-metrics -> non-zero exit, all six skills evaluated, only metrics shows FAIL/INCOMPL
echo ""
echo "--- Scenario 2: One Skill Fails Tier 1 ---"
OUT2=$(PATH="$BIN_DIR:$ORIG_PATH" STUB_FAIL_SKILL="ds-utils-metrics" "$REPO_ROOT/eval-skills.sh" 2>&1)
RC2=$?
if [ $RC2 -ne 0 ] && \
   echo "$OUT2" | grep -q "Evaluating: ds-utils-xai" && \
   echo "$OUT2" | grep "ds-utils-metrics" | grep -q "FAIL/INCOMPL" && \
   echo "$OUT2" | grep "ds-utils-preprocess" | grep -q "PASS (0)"; then
  echo "✓ Scenario 2 passed: Exited non-zero ($RC2), all skills evaluated, only metrics marked FAIL/INCOMPL."
else
  echo "✗ Scenario 2 FAILED! rc=$RC2, output:\n$OUT2" >&2
  exit 1
fi

# Scenario 3: Missing skillevaluator on PATH -> exit 1 with install hint
echo ""
echo "--- Scenario 3: Missing skillevaluator ---"
OUT3=$(PATH="/usr/bin:/bin" "$REPO_ROOT/eval-skills.sh" 2>&1)
RC3=$?
if [ $RC3 -eq 1 ] && echo "$OUT3" | grep -q "skillevaluator.*executable not found"; then
  echo "✓ Scenario 3 passed: Exits 1 with clear installation hint when executable is absent."
else
  echo "✗ Scenario 3 FAILED! rc=$RC3, output:\n$OUT3" >&2
  exit 1
fi

# Scenario 4: Pre-existing reports/ directory with sentinel file survives
echo ""
echo "--- Scenario 4: Pre-existing reports/ Directory Survives ---"
mkdir -p "$REPO_ROOT/reports"
SENTINEL="$REPO_ROOT/reports/sentinel_do_not_delete.txt"
echo "preserve_me" > "$SENTINEL"
OUT4=$(PATH="$BIN_DIR:$ORIG_PATH" "$REPO_ROOT/eval-skills.sh" 2>&1)
RC4=$?
if [ $RC4 -eq 0 ] && [ -f "$SENTINEL" ]; then
  echo "✓ Scenario 4 passed: Pre-existing reports/ directory and sentinel file survived."
  rm -rf "$REPO_ROOT/reports"
else
  echo "✗ Scenario 4 FAILED! Sentinel was lost or script failed. rc=$RC4" >&2
  rm -rf "$REPO_ROOT/reports"
  exit 1
fi

# Scenario 5: Real tools integration check
echo ""
echo "--- Scenario 5: Real Tools Integration Check ---"
MISSING_TOOL=""
if ! command -v skillevaluator >/dev/null 2>&1; then
  MISSING_TOOL="skillevaluator"
elif ! command -v skillspector >/dev/null 2>&1; then
  MISSING_TOOL="skillspector"
elif ! command -v gitleaks >/dev/null 2>&1; then
  MISSING_TOOL="gitleaks"
fi

if [ -n "$MISSING_TOOL" ]; then
  if [ "${REQUIRE_REAL_EVALUATOR:-0}" = "1" ]; then
    echo "✗ Scenario 5 FAILED: $MISSING_TOOL not found on PATH but REQUIRE_REAL_EVALUATOR=1 is set!" >&2
    exit 1
  else
    echo "Scenario 5 SKIPPED: $MISSING_TOOL not found on PATH."
  fi
else
  REAL_OUT_DIR="$TEST_TMP/real_reports"
  OUT5=$("$REPO_ROOT/eval-skills.sh" -o "$REAL_OUT_DIR" 2>&1)
  RC5=$?

  if [ $RC5 -eq 0 ]; then
    # Verify summary has exactly six ds-utils-* rows and all six have PASS (0) in both columns
    PASS_COUNT=$(echo "$OUT5" | grep -E '^ds-utils-[a-z]+ +\| +PASS \(0\) +\| +PASS \(0\)' | wc -l)
    if [ "$PASS_COUNT" -eq 6 ]; then
      echo "✓ Scenario 5 passed: Real evaluator succeeded with all 6 skills showing PASS (0) in Quality and Tier 1."
    else
      echo "✗ Scenario 5 FAILED: Expected 6 PASS (0) rows but found $PASS_COUNT. Output:" >&2
      echo "$OUT5" >&2
      exit 1
    fi
  else
    echo "✗ Scenario 5 FAILED! eval-skills.sh exited $RC5. Output:" >&2
    echo "$OUT5" >&2
    exit 1
  fi
fi

# Scenario 6: Tier 2 runner integration check with mock OpenAI-compatible server
echo ""
echo "--- Scenario 6: Tier 2 Runner Integration Check ---"
if ! command -v skillevaluator >/dev/null 2>&1; then
  if [ "${REQUIRE_REAL_EVALUATOR:-0}" = "1" ]; then
    echo "✗ Scenario 6 FAILED: skillevaluator not found on PATH but REQUIRE_REAL_EVALUATOR=1 is set!" >&2
    exit 1
  else
    echo "Scenario 6 SKIPPED: skillevaluator not found on PATH."
  fi
else
  # 6a. Start mock server
  SERVER_LOG="$TEST_TMP/mock_server.log"
  MOCK_FIFO="$TEST_TMP/mock_fifo"
  VERDICT_FILE="$TEST_TMP/mock_verdict.txt"
  echo "RELATED_BUT_DISTINCT" > "$VERDICT_FILE"
  rm -f "$MOCK_FIFO"
  mkfifo "$MOCK_FIFO"

  MOCK_VERDICT_FILE="$VERDICT_FILE" python3 "$REPO_ROOT/tests/mock_openai_server.py" 0 > "$MOCK_FIFO" 2> "$SERVER_LOG" &
  MOCK_PID=$!
  read -r MOCK_PORT < "$MOCK_FIFO"
  rm -f "$MOCK_FIFO"

  stop_mock() {
    if [ -n "${MOCK_PID:-}" ] && kill -0 "$MOCK_PID" 2>/dev/null; then
      kill "$MOCK_PID" 2>/dev/null || true
      wait "$MOCK_PID" 2>/dev/null || true
    fi
  }

  echo "Mock server started on port $MOCK_PORT (PID $MOCK_PID)"

  ENV_COMMON=(
    "SKILL_EVAL_EMBEDDING_PROVIDER=openai-compatible"
    "SKILL_EVAL_EMBEDDING_BASE_URL=http://127.0.0.1:$MOCK_PORT/v1"
    "SKILL_EVAL_EMBEDDING_API_KEY=ollama"
    "SKILL_EVAL_EMBEDDING_MODEL=nemotron-3-embed-1b"
    "SKILL_EVAL_LLM_PROVIDER=openai-compatible"
    "SKILL_EVAL_LLM_BASE_URL=http://127.0.0.1:$MOCK_PORT/v1"
    "SKILL_EVAL_LLM_API_KEY=ollama"
    "SKILL_EVAL_LLM_MODEL=gemma4:e2b"
  )

  # Test 6.1: RELATED_BUT_DISTINCT -> exits 0 with all PASS
  echo "RELATED_BUT_DISTINCT" > "$VERDICT_FILE"
  OUT6_1=$(env "${ENV_COMMON[@]}" "$REPO_ROOT/eval-skills-tier2.sh" 2>&1)
  RC6_1=$?
  if [ $RC6_1 -eq 0 ] && ! echo "$OUT6_1" | grep -q "FAIL" && ! echo "$OUT6_1" | grep -q "INCOMPLETE"; then
    echo "✓ Scenario 6.1 passed: RELATED_BUT_DISTINCT exits 0 and all rows show PASS."
  else
    stop_mock
    printf "✗ Scenario 6.1 FAILED! rc=%s, output:\n%s\n" "$RC6_1" "$OUT6_1" >&2
    if echo "$OUT6_1" | grep -q "package is required"; then
      echo 'hint: install SkillEvaluator with --with "openai>=2.21.0"' >&2
    fi
    exit 1
  fi

  # Test 6.2: DUPLICATE -> exits 1 with a FAIL row
  echo "DUPLICATE" > "$VERDICT_FILE"
  OUT6_2=$(env "${ENV_COMMON[@]}" "$REPO_ROOT/eval-skills-tier2.sh" 2>&1)
  RC6_2=$?
  if [ $RC6_2 -eq 1 ] && echo "$OUT6_2" | grep -q "FAIL"; then
    echo "✓ Scenario 6.2 passed: DUPLICATE exits 1 and reports FAIL row."
  else
    stop_mock
    printf "✗ Scenario 6.2 FAILED! rc=%s, output:\n%s\n" "$RC6_2" "$OUT6_2" >&2
    if echo "$OUT6_2" | grep -q "package is required"; then
      echo 'hint: install SkillEvaluator with --with "openai>=2.21.0"' >&2
    fi
    exit 1
  fi

  # Stop mock server before unreachable port test
  stop_mock

  # Test 6.3: Unreachable port -> exits 1 with an INCOMPLETE row
  ENV_UNREACHABLE=(
    "SKILL_EVAL_EMBEDDING_PROVIDER=openai-compatible"
    "SKILL_EVAL_EMBEDDING_BASE_URL=http://127.0.0.1:1/v1"
    "SKILL_EVAL_EMBEDDING_API_KEY=ollama"
    "SKILL_EVAL_EMBEDDING_MODEL=nemotron-3-embed-1b"
    "SKILL_EVAL_LLM_PROVIDER=openai-compatible"
    "SKILL_EVAL_LLM_BASE_URL=http://127.0.0.1:1/v1"
    "SKILL_EVAL_LLM_API_KEY=ollama"
    "SKILL_EVAL_LLM_MODEL=gemma4:e2b"
  )
  OUT6_3=$(env "${ENV_UNREACHABLE[@]}" "$REPO_ROOT/eval-skills-tier2.sh" 2>&1)
  RC6_3=$?
  if [ $RC6_3 -eq 1 ] && echo "$OUT6_3" | grep -q "INCOMPLETE"; then
    echo "✓ Scenario 6.3 passed: Unreachable port exits 1 and reports INCOMPLETE row."
  else
    printf "✗ Scenario 6.3 FAILED! rc=%s, output:\n%s\n" "$RC6_3" "$OUT6_3" >&2
    if echo "$OUT6_3" | grep -q "package is required"; then
      echo 'hint: install SkillEvaluator with --with "openai>=2.21.0"' >&2
    fi
    exit 1
  fi
fi

echo ""
echo "=== Evaluator test suite finished successfully! ==="
