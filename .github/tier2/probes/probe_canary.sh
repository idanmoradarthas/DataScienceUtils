#!/usr/bin/env bash
#
# Probe D: Canary Positive Control Verification.
#
# Runs eval-skills-tier2.sh against tests/resources/tier2_canary.
# Asserts:
# 1. similarity-check flags skill-a and skill-b.
# 2. dedup-scan flags skill-c with DUPLICATE.
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
CANARY_DIR="$REPO_ROOT/tests/resources/tier2_canary"

echo "=========================================================="
echo "Probe D: Running Canary Positive Control on $CANARY_DIR"
echo "=========================================================="

TEMP_OUT=$(mktemp -d)
trap 'rm -rf "$TEMP_OUT"' EXIT

export SKILLS_DIR="$CANARY_DIR"

# Run similarity-check on canary
echo "Running similarity-check on canary..."
skillevaluator similarity-check "$CANARY_DIR" --type skill -r json -o "$TEMP_OUT/similarity" || true

SIM_REPORT="$TEMP_OUT/similarity/skillevaluator-similarity.json"
if [ ! -f "$SIM_REPORT" ]; then
  echo "Error: Similarity report was not generated for canary!" >&2
  exit 1
fi

python3 -c "
import json, sys
with open('$SIM_REPORT') as f:
    data = json.load(f)

overall_status = data.get('overall_status')
incomplete_scans = data.get('incomplete_scans', [])

if overall_status != 'failed':
    print(
        f'Error: Expected similarity-check overall_status == \"failed\", got \"{overall_status}\". '
        f'incomplete_scans: {incomplete_scans}',
        file=sys.stderr,
    )
    sys.exit(1)

found_ab = False
for r in data.get('results', []):
    for finding in r.get('findings', []):
        if finding.get('category') == 'SIMILARITY':
            meta = finding.get('metadata', {})
            entry_a = meta.get('entry_a') or meta.get('path_a')
            entry_b = meta.get('entry_b') or meta.get('path_b')
            if {entry_a, entry_b} == {'skill-a', 'skill-b'}:
                found_ab = True
                break
    if found_ab:
        break

if not found_ab:
    print(
        f'Error: similarity-check did not report a SIMILARITY finding for skill-a and skill-b! '
        f'overall_status: \"{overall_status}\", incomplete_scans: {incomplete_scans}',
        file=sys.stderr,
    )
    sys.exit(1)

print('✓ Canary similarity check successfully identified content overlap between skill-a and skill-b.')
"

# Run dedup-scan on skill-c
echo "Running dedup-scan on skill-c..."
skillevaluator dedup-scan "$CANARY_DIR/skill-c" -r json -o "$TEMP_OUT/dedup-c" || true

DEDUP_REPORT="$TEMP_OUT/dedup-c/skillevaluator-dedup.json"
if [ ! -f "$DEDUP_REPORT" ]; then
  echo "Error: Dedup report was not generated for skill-c!" >&2
  exit 1
fi

python3 -c "
import json, sys, os
with open('$DEDUP_REPORT') as f:
    data = json.load(f)
found_dup = False
llm_model = os.environ.get('SKILL_EVAL_LLM_MODEL', 'unknown')
for r in data.get('results', []):
    for finding in r.get('findings', []):
        if finding.get('category') == 'DUPLICATE':
            found_dup = True
            print(f'Found duplicate finding: {finding.get(\"message\", \"\")[:120]}...')
            break
if not found_dup:
    print(f'Error: LLM {llm_model} missed the repeated section in skill-c! Findings were: {data.get(\"findings\")}', file=sys.stderr)
    sys.exit(1)
print(f'✓ Canary dedup check successfully flagged repeated section in skill-c using {llm_model}.')
"

echo "✓ Probe D passed: All canary positive control checks verified."
