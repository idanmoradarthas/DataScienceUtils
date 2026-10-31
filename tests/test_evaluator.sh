#!/bin/bash
# Test verifying eval-skills.sh behavior on INCOMPLETE scanner status
set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "=== Running Controlled Evaluator INCOMPLETE Test ==="

OUTPUT=$(cd "$REPO_ROOT" && ./eval-skills.sh 2>&1)
EXIT_CODE=$?

echo "eval-skills.sh exited with: $EXIT_CODE"

# 1. Must exit non-zero because Gitleaks is missing and SkillSpector is partial (INCOMPLETE)
if [ $EXIT_CODE -eq 0 ]; then
  echo "FAILED: Expected non-zero exit code due to incomplete scanner evidence, got 0!" >&2
  exit 1
fi

# 2. Must evaluate all six skills (none skipped due to premature exit)
for skill in ds-utils-metrics ds-utils-preprocess ds-utils-strings ds-utils-transformers ds-utils-unsupervised ds-utils-xai; do
  if ! echo "$OUTPUT" | grep -q "Evaluating: ${skill}"; then
    echo "FAILED: Skill ${skill} was not evaluated!" >&2
    exit 1
  fi
done

# 3. Must display EVALUATION SUMMARY table
if ! echo "$OUTPUT" | grep -q "EVALUATION SUMMARY"; then
  echo "FAILED: Summary table missing from output!" >&2
  exit 1
fi

# 4. Summary table must honestly disclose FAIL/INCOMPL rather than claiming PASS
if ! echo "$OUTPUT" | grep "ds-utils-metrics" | grep -q "FAIL/INCOMPL"; then
  echo "FAILED: Summary falsely claimed full Tier 1 completion!" >&2
  exit 1
fi

echo "✓ Controlled evaluator INCOMPLETE test passed: all 6 skills evaluated, genuine exit status captured, and incomplete scanner coverage truthfully disclosed!"
