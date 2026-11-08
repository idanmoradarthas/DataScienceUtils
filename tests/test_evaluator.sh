#!/usr/bin/env bash
# Test suite for eval-skills.sh: Scenarios 1-4 are hermetic using a stub skillevaluator;
# Scenario 5 is an integration check using real tools when available.
# Scenario 6 is a Tier 2 runner integration check with mock OpenAI-compatible server.
# Scenario 7 tests the passage-prefix proxy and mock recording.
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

# Scenario 2: STUB_FAIL_SKILL=ds-utils-metrics -> non-zero exit, all six skills evaluated, only metrics marked FAIL/INCOMPL
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

# wait_for_listening <logfile> <pid> <timeout_seconds>
# Prints the port on stdout and returns 0 as soon as "listening on <port>" appears in the log.
# Returns 1 (after dumping the log to stderr) if the process dies first or the wall-clock deadline passes.
wait_for_listening() {
  local log="$1" pid="$2" timeout="$3" deadline line
  deadline=$((SECONDS + timeout))
  while [ "$SECONDS" -lt "$deadline" ]; do
    line=$(grep -m1 "listening on " "$log" 2>/dev/null || true)
    if [ -n "$line" ]; then
      echo "${line##* }"
      return 0
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      break
    fi
    sleep 0.2
  done
  echo "proxy did not report a listening port within ${timeout}s (pid $pid); log follows:" >&2
  cat "$log" >&2 2>/dev/null || true
  return 1
}

# Scenario 7: Passage-Prefix Proxy and Mock Recording Check (Hermetic, no skillevaluator required)
echo ""
echo "--- Scenario 7: Passage-Prefix Proxy & Mock Recording Check ---"
PROXY_RECORD_FILE="$TEST_TMP/proxy_records.jsonl"
MOCK7_FIFO="$TEST_TMP/mock7_fifo"
MOCK7_LOG="$TEST_TMP/mock7.log"
PROXY_LOG="$TEST_TMP/proxy.log"

rm -f "$MOCK7_FIFO"
mkfifo "$MOCK7_FIFO"

MOCK_RECORD_FILE="$PROXY_RECORD_FILE" python3 "$REPO_ROOT/tests/mock_openai_server.py" 0 > "$MOCK7_FIFO" 2> "$MOCK7_LOG" &
MOCK7_PID=$!
read -r MOCK7_PORT < "$MOCK7_FIFO"
rm -f "$MOCK7_FIFO"

cleanup_proxy_test() {
  if [ -n "${PROXY_PID:-}" ] && kill -0 "$PROXY_PID" 2>/dev/null; then
    kill "$PROXY_PID" 2>/dev/null || true
    wait "$PROXY_PID" 2>/dev/null || true
  fi
  if [ -n "${MOCK7_PID:-}" ] && kill -0 "$MOCK7_PID" 2>/dev/null; then
    kill "$MOCK7_PID" 2>/dev/null || true
    wait "$MOCK7_PID" 2>/dev/null || true
  fi
}

TIER2_UPSTREAM="http://127.0.0.1:$MOCK7_PORT" TIER2_EMBED_PROXY_PORT=0 python3 "$REPO_ROOT/.github/tier2/embed_prefix_proxy.py" > "$PROXY_LOG" 2>&1 &
PROXY_PID=$!

PROXY_PORT=$(wait_for_listening "$PROXY_LOG" "$PROXY_PID" 120) || { cleanup_proxy_test; echo "✗ Scenario 7 FAILED: proxy did not start" >&2; exit 1; }

echo "Proxy started on port $PROXY_PORT, forwarding to mock on port $MOCK7_PORT"

# Test 7.1: ["a", "b"] with input_type=passage arrives as ["passage: a", "passage: b"] without input_type
REQ1_STATUS=$(python3 -c "
import urllib.request, json
req = urllib.request.Request(
    'http://127.0.0.1:$PROXY_PORT/v1/embeddings',
    data=json.dumps({'input': ['a', 'b'], 'input_type': 'passage', 'model': 'test-model'}).encode('utf-8'),
    headers={'Content-Type': 'application/json'},
    method='POST'
)
with urllib.request.urlopen(req) as resp:
    print(resp.status)
")

if [ "$REQ1_STATUS" != "200" ]; then
  cleanup_proxy_test
  echo "✗ Scenario 7.1 FAILED: Expected HTTP 200, got $REQ1_STATUS" >&2
  exit 1
fi

REC1_CHECK=$(python3 -c "
import json
with open('$PROXY_RECORD_FILE') as f:
    line = f.readline()
rec = json.loads(line)
inp = rec.get('input', [])
keys = rec.get('keys', [])
assert inp == ['passage: a', 'passage: b'], f'Unexpected input: {inp}'
assert 'input_type' not in keys, f'input_type found in keys: {keys}'
print('OK')
")

if [ "$REC1_CHECK" != "OK" ]; then
  cleanup_proxy_test
  echo "✗ Scenario 7.1 FAILED: Mock record mismatch: $REC1_CHECK" >&2
  exit 1
fi
echo "✓ Scenario 7.1 passed: ['a', 'b'] with input_type=passage rewritten to ['passage: a', 'passage: b'] without input_type."

# Test 7.2: Plain string input
REQ2_STATUS=$(python3 -c "
import urllib.request, json
req = urllib.request.Request(
    'http://127.0.0.1:$PROXY_PORT/v1/embeddings',
    data=json.dumps({'input': 'hello world', 'model': 'test-model'}).encode('utf-8'),
    headers={'Content-Type': 'application/json'},
    method='POST'
)
with urllib.request.urlopen(req) as resp:
    print(resp.status)
")

if [ "$REQ2_STATUS" != "200" ]; then
  cleanup_proxy_test
  echo "✗ Scenario 7.2 FAILED: Expected HTTP 200, got $REQ2_STATUS" >&2
  exit 1
fi

REC2_CHECK=$(python3 -c "
import json
with open('$PROXY_RECORD_FILE') as f:
    lines = f.readlines()
rec = json.loads(lines[1])
inp = rec.get('input', [])
assert inp == ['passage: hello world'], f'Unexpected input: {inp}'
print('OK')
")

if [ "$REC2_CHECK" != "OK" ]; then
  cleanup_proxy_test
  echo "✗ Scenario 7.2 FAILED: Plain string input mismatch: $REC2_CHECK" >&2
  exit 1
fi
echo "✓ Scenario 7.2 passed: Plain string input rewritten with passage prefix."

# Test 7.3: /v1/chat/completions passes through transparently
CHAT_RESP=$(python3 -c "
import urllib.request, json
req = urllib.request.Request(
    'http://127.0.0.1:$PROXY_PORT/v1/chat/completions',
    data=json.dumps({'model': 'mock-model', 'messages': [{'role': 'user', 'content': 'test'}]}).encode('utf-8'),
    headers={'Content-Type': 'application/json'},
    method='POST'
)
with urllib.request.urlopen(req) as resp:
    data = json.loads(resp.read().decode('utf-8'))
    print(data.get('choices', [{}])[0].get('message', {}).get('content', ''))
")

if ! echo "$CHAT_RESP" | grep -q "verdict"; then
  cleanup_proxy_test
  echo "✗ Scenario 7.3 FAILED: /v1/chat/completions did not return expected response: $CHAT_RESP" >&2
  exit 1
fi
echo "✓ Scenario 7.3 passed: /v1/chat/completions passed through transparently."

# Test 7.5: body [1, 2, 3] -> HTTP 400
REQ5_STATUS=$(python3 -c "
import urllib.request, urllib.error, json
req = urllib.request.Request(
    'http://127.0.0.1:$PROXY_PORT/v1/embeddings',
    data=json.dumps([1, 2, 3]).encode('utf-8'),
    headers={'Content-Type': 'application/json'},
    method='POST'
)
try:
    urllib.request.urlopen(req)
    print('FAILED_NOT_400')
except urllib.error.HTTPError as e:
    print(e.code)
except Exception as e:
    print(f'ERROR: {e}')
")

if [ "$REQ5_STATUS" = "400" ]; then
  echo "✓ Scenario 7.5 passed: Body [1, 2, 3] returns HTTP 400."
else
  cleanup_proxy_test
  echo "✗ Scenario 7.5 FAILED: Expected HTTP 400, got $REQ5_STATUS" >&2
  exit 1
fi

# Test 7.6: input [[1, 2, 3]] reaches the mock unchanged
REQ6_STATUS=$(python3 -c "
import urllib.request, json
req = urllib.request.Request(
    'http://127.0.0.1:$PROXY_PORT/v1/embeddings',
    data=json.dumps({'input': [[1, 2, 3]], 'model': 'test-model'}).encode('utf-8'),
    headers={'Content-Type': 'application/json'},
    method='POST'
)
with urllib.request.urlopen(req) as resp:
    print(resp.status)
")

if [ "$REQ6_STATUS" != "200" ]; then
  cleanup_proxy_test
  echo "✗ Scenario 7.6 FAILED: Expected HTTP 200, got $REQ6_STATUS" >&2
  exit 1
fi

REC6_CHECK=$(python3 -c "
import json
with open('$PROXY_RECORD_FILE') as f:
    lines = f.readlines()
rec = json.loads(lines[-1])
inp = rec.get('input', [])
assert inp == [[1, 2, 3]], f'Unexpected input: {inp}'
print('OK')
")

if [ "$REC6_CHECK" != "OK" ]; then
  cleanup_proxy_test
  echo "✗ Scenario 7.6 FAILED: Non-str input mismatch: $REC6_CHECK" >&2
  exit 1
fi
echo "✓ Scenario 7.6 passed: Input [[1, 2, 3]] reached mock unchanged."

# Test 7.7: with MOCK_GZIP=1 on mock and client sending Accept-Encoding: gzip through proxy,
# client gets plain, decodable JSON (proxy sent Accept-Encoding: identity), and recorded input is prefixed.
MOCK_GZIP_FIFO="$TEST_TMP/mock_gzip_fifo"
MOCK_GZIP_LOG="$TEST_TMP/mock_gzip.log"
MOCK_GZIP_RECORD="$TEST_TMP/mock_gzip_records.jsonl"
PROXY_GZIP_LOG="$TEST_TMP/proxy_gzip.log"

mkfifo "$MOCK_GZIP_FIFO"
MOCK_RECORD_FILE="$MOCK_GZIP_RECORD" MOCK_GZIP=1 python3 "$REPO_ROOT/tests/mock_openai_server.py" 0 > "$MOCK_GZIP_FIFO" 2> "$MOCK_GZIP_LOG" &
MOCK_GZIP_PID=$!
read -r MOCK_GZIP_PORT < "$MOCK_GZIP_FIFO"
rm -f "$MOCK_GZIP_FIFO"

TIER2_UPSTREAM="http://127.0.0.1:$MOCK_GZIP_PORT" TIER2_EMBED_PROXY_PORT=0 python3 "$REPO_ROOT/.github/tier2/embed_prefix_proxy.py" > "$PROXY_GZIP_LOG" 2>&1 &
PROXY_GZIP_PID=$!

PROXY_GZIP_PORT=$(wait_for_listening "$PROXY_GZIP_LOG" "$PROXY_GZIP_PID" 120) || { kill "$PROXY_GZIP_PID" "$MOCK_GZIP_PID" 2>/dev/null || true; wait "$PROXY_GZIP_PID" "$MOCK_GZIP_PID" 2>/dev/null || true; cleanup_proxy_test; echo "✗ Scenario 7.7 FAILED: gzip proxy did not start" >&2; exit 1; }

GZIP_CHECK=$(python3 -c "
import urllib.request, json
req = urllib.request.Request(
    'http://127.0.0.1:$PROXY_GZIP_PORT/v1/embeddings',
    data=json.dumps({'input': 'gzip test', 'model': 'test-model'}).encode('utf-8'),
    headers={'Content-Type': 'application/json', 'Accept-Encoding': 'gzip'},
    method='POST'
)
with urllib.request.urlopen(req) as resp:
    raw_body = resp.read()
    data = json.loads(raw_body.decode('utf-8'))
    assert data.get('object') == 'list'
    print('OK')
")

kill "$PROXY_GZIP_PID" "$MOCK_GZIP_PID" 2>/dev/null || true
wait "$PROXY_GZIP_PID" "$MOCK_GZIP_PID" 2>/dev/null || true

if [ "$GZIP_CHECK" != "OK" ]; then
  cleanup_proxy_test
  echo "✗ Scenario 7.7 FAILED: Client did not receive plain decodable JSON with MOCK_GZIP: $GZIP_CHECK" >&2
  exit 1
fi

REC_GZIP_CHECK=$(python3 -c "
import json
with open('$MOCK_GZIP_RECORD') as f:
    rec = json.loads(f.readline())
assert rec.get('input') == ['passage: gzip test']
print('OK')
")

if [ "$REC_GZIP_CHECK" != "OK" ]; then
  cleanup_proxy_test
  echo "✗ Scenario 7.7 FAILED: Recorded input for gzip test not prefixed: $REC_GZIP_CHECK" >&2
  exit 1
fi
echo "✓ Scenario 7.7 passed: MOCK_GZIP=1 upstream with client Accept-Encoding: gzip yielded plain decodable JSON and prefixed input."

# Test 7.8: one proxy log line exists per embeddings request and its inputs= matches
LOG_CHECK=$(python3 -c "
with open('$PROXY_LOG') as f:
    lines = [line.strip() for line in f if line.startswith('embeddings:')]
# Requests made to PROXY_PORT:
# 7.1: inputs=['a', 'b'] -> inputs=2
# 7.2: input='hello world' -> inputs=1
# 7.5: body=[1, 2, 3] -> 400 error before embeddings log (not logged)
# 7.6: input=[[1, 2, 3]] -> inputs=1
assert len(lines) == 3, f'Expected 3 embeddings log lines, found {len(lines)}: {lines}'
assert lines[0] == \"embeddings: inputs=2 prefix='passage: '\", f'Line 0 mismatch: {lines[0]}'
assert lines[1] == \"embeddings: inputs=1 prefix='passage: '\", f'Line 1 mismatch: {lines[1]}'
assert lines[2] == \"embeddings: inputs=1 prefix='passage: '\", f'Line 2 mismatch: {lines[2]}'
print('OK')
")

if [ "$LOG_CHECK" != "OK" ]; then
  cleanup_proxy_test
  echo "✗ Scenario 7.8 FAILED: Proxy embeddings log lines mismatch: $LOG_CHECK" >&2
  exit 1
fi
echo "✓ Scenario 7.8 passed: Exactly one proxy log line per embeddings request and inputs= matches."

# Test 7.4: Unreachable upstream gives HTTP 502
# Stop the mock upstream server
kill -9 "$MOCK7_PID" 2>/dev/null || true
wait "$MOCK7_PID" 2>/dev/null || true

HTTP_502_CHECK=$(python3 -c "
import urllib.request, urllib.error, json
req = urllib.request.Request(
    'http://127.0.0.1:$PROXY_PORT/v1/embeddings',
    data=json.dumps({'input': 'fail', 'model': 'test-model'}).encode('utf-8'),
    headers={'Content-Type': 'application/json'},
    method='POST'
)
try:
    urllib.request.urlopen(req)
    print('FAILED_NOT_502')
except urllib.error.HTTPError as e:
    print(e.code)
except Exception as e:
    print(f'ERROR: {e}')
")

if [ "$HTTP_502_CHECK" = "502" ]; then
  echo "✓ Scenario 7.4 passed: Unreachable upstream returned HTTP 502."
else
  cleanup_proxy_test
  echo "✗ Scenario 7.4 FAILED: Expected HTTP 502, got $HTTP_502_CHECK" >&2
  exit 1
fi

cleanup_proxy_test

echo ""
echo "=== Evaluator test suite finished successfully! ==="
