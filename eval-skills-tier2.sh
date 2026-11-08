#!/usr/bin/env bash
#
# Evaluates Agent Skills under skills/ (or a custom SKILLS_DIR) using NVIDIA SkillEvaluator Tier 2.
# Executes similarity-check across skills and dedup-scan per skill directory.
# Maps outcomes to PASS, FAIL, or INCOMPLETE, prints a summary table, lists findings,
# and exits 0 only if all checks pass, otherwise 1.
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

# Check for help flag
for arg in "$@"; do
  if [ "$arg" = "-h" ] || [ "$arg" = "--help" ]; then
    echo "Usage: ./eval-skills-tier2.sh"
    echo ""
    echo "Executes NVIDIA SkillEvaluator Tier 2 (similarity-check and dedup-scan)."
    echo ""
    echo "Required environment variables:"
    echo "  SKILL_EVAL_EMBEDDING_PROVIDER  e.g. openai-compatible"
    echo "  SKILL_EVAL_EMBEDDING_BASE_URL  e.g. http://127.0.0.1:11434/v1"
    echo "  SKILL_EVAL_EMBEDDING_API_KEY   e.g. ollama"
    echo "  SKILL_EVAL_EMBEDDING_MODEL     e.g. nemotron-3-embed-1b"
    echo "  SKILL_EVAL_LLM_PROVIDER        e.g. openai-compatible"
    echo "  SKILL_EVAL_LLM_BASE_URL        e.g. http://127.0.0.1:11434/v1"
    echo "  SKILL_EVAL_LLM_API_KEY         e.g. ollama"
    echo "  SKILL_EVAL_LLM_MODEL           e.g. gemma4:e2b"
    echo ""
    echo "Optional environment variables:"
    echo "  SKILLS_DIR                     Path to skills directory (default: skills)"
    exit 0
  fi
done

# 1. Verify prerequisite CLI
if ! command -v skillevaluator >/dev/null 2>&1; then
  echo "Error: 'skillevaluator' executable not found in PATH." >&2
  echo "Please install dependencies listed in requirements-skills.txt:" >&2
  echo '  uv tool install --python 3.13 --with "openai>=2.21.0" "skillevaluator @ git+https://github.com/NVIDIA/SkillEvaluator.git@a61e1f3ff609118c0c07e5f569aaf58fa5a76be0"' >&2
  exit 1
fi

# 2. Verify provider environment variables
MISSING_VARS=()
[ -z "${SKILL_EVAL_EMBEDDING_PROVIDER:-}" ] && MISSING_VARS+=("SKILL_EVAL_EMBEDDING_PROVIDER")
[ -z "${SKILL_EVAL_EMBEDDING_BASE_URL:-}" ] && MISSING_VARS+=("SKILL_EVAL_EMBEDDING_BASE_URL")
[ -z "${SKILL_EVAL_EMBEDDING_API_KEY:-}" ] && MISSING_VARS+=("SKILL_EVAL_EMBEDDING_API_KEY")
[ -z "${SKILL_EVAL_EMBEDDING_MODEL:-}" ] && MISSING_VARS+=("SKILL_EVAL_EMBEDDING_MODEL")
[ -z "${SKILL_EVAL_LLM_PROVIDER:-}" ] && MISSING_VARS+=("SKILL_EVAL_LLM_PROVIDER")
[ -z "${SKILL_EVAL_LLM_BASE_URL:-}" ] && MISSING_VARS+=("SKILL_EVAL_LLM_BASE_URL")
[ -z "${SKILL_EVAL_LLM_API_KEY:-}" ] && MISSING_VARS+=("SKILL_EVAL_LLM_API_KEY")
[ -z "${SKILL_EVAL_LLM_MODEL:-}" ] && MISSING_VARS+=("SKILL_EVAL_LLM_MODEL")

if [ ${#MISSING_VARS[@]} -ne 0 ]; then
  echo "Error: Missing required Tier 2 provider environment variable(s):" >&2
  for var in "${MISSING_VARS[@]}"; do
    echo "  - $var" >&2
  done
  echo "" >&2
  echo "Please configure both the embedding and LLM providers before running Tier 2 checks." >&2
  echo "For Ollama running locally, you can export:" >&2
  echo "  export SKILL_EVAL_EMBEDDING_PROVIDER=openai-compatible" >&2
  echo "  export SKILL_EVAL_EMBEDDING_BASE_URL=http://127.0.0.1:11434/v1" >&2
  echo "  export SKILL_EVAL_EMBEDDING_API_KEY=ollama" >&2
  echo "  export SKILL_EVAL_EMBEDDING_MODEL=nemotron-3-embed-1b" >&2
  echo "  export SKILL_EVAL_LLM_PROVIDER=openai-compatible" >&2
  echo "  export SKILL_EVAL_LLM_BASE_URL=http://127.0.0.1:11434/v1" >&2
  echo "  export SKILL_EVAL_LLM_API_KEY=ollama" >&2
  echo "  export SKILL_EVAL_LLM_MODEL=gemma4:e2b" >&2
  exit 1
fi

SKILLS_DIR_TARGET="${SKILLS_DIR:-skills}"
if [ ! -d "$SKILLS_DIR_TARGET" ]; then
  echo "Error: Skills directory '$SKILLS_DIR_TARGET' does not exist." >&2
  exit 1
fi

echo "=========================================================="
echo "    NVIDIA SkillEvaluator Tier 2 Runner - DataScienceUtils"
echo "=========================================================="
skillevaluator --version

# Isolate reports output to a fresh temp directory so repo root reports/ is never touched
TIER2_TEMP_DIR=$(mktemp -d)
trap 'rm -rf "$TIER2_TEMP_DIR"' EXIT

# Parallel indexed arrays for bash 3.2 compatibility
CHECK_NAMES=()
CHECK_STATUSES=()
CHECK_FINDINGS=()
OVERALL_FAILED=0

# Helper to parse JSON report status
parse_status() {
  local report_file="$1"
  local exit_code="$2"
  python3 -c "
import json, sys
try:
    with open('$report_file') as f:
        data = json.load(f)
    status = data.get('overall_status', '').lower()
    if status == 'passed':
        print('PASS')
    elif status == 'failed':
        print('FAIL')
    elif status == 'incomplete':
        print('INCOMPLETE')
    else:
        sys.exit(1)
except Exception:
    if $exit_code == 0:
        print('PASS')
    else:
        print('FAIL')
" 2>/dev/null || {
    if [ "$exit_code" -eq 0 ]; then
      echo "PASS"
    else
      echo "FAIL"
    fi
  }
}

# Helper to extract findings from JSON report
extract_findings() {
  local report_file="$1"
  python3 -c "
import json
try:
    with open('$report_file') as f:
        data = json.load(f)
    findings = []
    for r in data.get('results', []):
        for f in r.get('findings', []):
            cat = f.get('category', 'FINDING')
            fp = f.get('file_path') or 'unknown'
            meta = f.get('metadata', {})
            srcs = meta.get('source_files', [])
            if srcs:
                fp = ', '.join(srcs)
            msg = f.get('message', '').splitlines()[0] if f.get('message') else ''
            findings.append(f'[{cat}] {fp}: {msg}')
    print('\n'.join(findings))
except Exception:
    pass
" 2>/dev/null
}

# 1. Similarity Check (Inter-skill)
echo ""
echo "----------------------------------------------------------"
echo ">>> Running inter-skill similarity-check on $SKILLS_DIR_TARGET"
echo "----------------------------------------------------------"
SIM_OUT="$TIER2_TEMP_DIR/similarity"
mkdir -p "$SIM_OUT"

skillevaluator similarity-check "$SKILLS_DIR_TARGET" --type skill -r cli,json -o "$SIM_OUT"
sim_rc=$?

sim_status=$(parse_status "$SIM_OUT/skillevaluator-similarity.json" "$sim_rc")
sim_findings=""
[ "$sim_status" = "FAIL" ] && sim_findings=$(extract_findings "$SIM_OUT/skillevaluator-similarity.json")

CHECK_NAMES+=("similarity-check")
CHECK_STATUSES+=("$sim_status")
CHECK_FINDINGS+=("$sim_findings")

if [ "$sim_status" != "PASS" ]; then
  OVERALL_FAILED=1
fi

# 2. Intra-skill Dedup Scan per skill directory
for s_dir in "$SKILLS_DIR_TARGET"/*/; do
  [ -d "$s_dir" ] || continue
  s_name=$(basename "$s_dir")
  echo ""
  echo "----------------------------------------------------------"
  echo ">>> Running dedup-scan on $s_name"
  echo "----------------------------------------------------------"

  dedup_out="$TIER2_TEMP_DIR/dedup-$s_name"
  mkdir -p "$dedup_out"

  skillevaluator dedup-scan "$s_dir" -r cli,json -o "$dedup_out"
  dedup_rc=$?

  dedup_status=$(parse_status "$dedup_out/skillevaluator-dedup.json" "$dedup_rc")
  dedup_findings=""
  [ "$dedup_status" = "FAIL" ] && dedup_findings=$(extract_findings "$dedup_out/skillevaluator-dedup.json")

  CHECK_NAMES+=("dedup:$s_name")
  CHECK_STATUSES+=("$dedup_status")
  CHECK_FINDINGS+=("$dedup_findings")

  if [ "$dedup_status" != "PASS" ]; then
    OVERALL_FAILED=1
  fi
done

echo ""
echo "=========================================================="
echo "                  TIER 2 EVALUATION SUMMARY"
echo "=========================================================="
printf "%-35s | %-15s\n" "Check / Skill" "Status"
echo "----------------------------------------------------------"
for i in "${!CHECK_NAMES[@]}"; do
  c_name="${CHECK_NAMES[$i]}"
  c_stat="${CHECK_STATUSES[$i]}"
  printf "%-35s | %-15s\n" "$c_name" "$c_stat"
done
echo "=========================================================="

# If there were any failures, print finding details
HAS_FINDINGS=0
for i in "${!CHECK_FINDINGS[@]}"; do
  f_text="${CHECK_FINDINGS[$i]}"
  if [ -n "$f_text" ]; then
    if [ $HAS_FINDINGS -eq 0 ]; then
      echo ""
      echo "Findings summary:"
      HAS_FINDINGS=1
    fi
    echo "  ${CHECK_NAMES[$i]}:"
    while IFS= read -r f_line; do
      [ -n "$f_line" ] && echo "    $f_line"
    done <<< "$f_text"
  fi
done

if [ $OVERALL_FAILED -ne 0 ]; then
  echo ""
  echo "Tier 2 evaluation completed with one or more failures or incomplete scans." >&2
  exit 1
else
  echo ""
  echo "All Tier 2 similarity and deduplication checks successfully passed!"
  exit 0
fi
