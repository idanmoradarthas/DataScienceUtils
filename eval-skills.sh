#!/usr/bin/env bash
#
# Evaluates Agent Skills under skills/ using NVIDIA SkillEvaluator.
# Executes quality checks and Tier 1 static validation across all skills,
# reports individual statuses, and exits non-zero if any required check fails.
# For full setup instructions and scanner prerequisites, see docs/source/ai_skills.rst.
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

# 1. Verify prerequisite CLI
if ! command -v skillevaluator >/dev/null 2>&1; then
  echo "Error: 'skillevaluator' executable not found in PATH." >&2
  echo "Please install dependencies listed in requirements-skills.txt:" >&2
  echo '  uv tool install --python 3.13 "skillevaluator @ git+https://github.com/NVIDIA/SkillEvaluator.git@a61e1f3ff609118c0c07e5f569aaf58fa5a76be0"' >&2
  exit 1
fi

echo "=========================================================="
echo "    NVIDIA SkillEvaluator Runner - DataScienceUtils"
echo "=========================================================="
skillevaluator --version

# Use parallel indexed arrays for bash 3.2 compatibility (macOS default)
SKILL_NAMES=()
Q_CODES=()
T_CODES=()
OVERALL_FAILED=0

# Isolate reports output to a fresh temp directory so repo root reports/ is never deleted
EVAL_TEMP_DIR=$(mktemp -d)
trap 'rm -rf "$EVAL_TEMP_DIR"' EXIT

for skill_dir in "$SCRIPT_DIR"/skills/*/; do
  [ -d "$skill_dir" ] || continue
  skill_name=$(basename "$skill_dir")
  SKILL_NAMES+=("$skill_name")

  echo ""
  echo "----------------------------------------------------------"
  echo ">>> Evaluating: ${skill_name}"
  echo "----------------------------------------------------------"

  # [1/2] Quality Check (Default threshold >= 70)
  echo "Running quality-check on ${skill_name}..."
  (
    cd "$EVAL_TEMP_DIR" || exit 1
    skillevaluator quality-check "$skill_dir" -o "$EVAL_TEMP_DIR/quality-$skill_name"
  )
  q_exit=$?
  Q_CODES+=("$q_exit")
  if [ $q_exit -ne 0 ]; then
    echo "! quality-check failed for ${skill_name} (exit=${q_exit})"
    OVERALL_FAILED=1
  else
    echo "✓ quality-check passed for ${skill_name}"
  fi

  # [2/2] Tier 1 Static Validation (--no-llm keyless mode)
  echo "Running tier1 --no-llm on ${skill_name}..."
  (
    cd "$EVAL_TEMP_DIR" || exit 1
    skillevaluator tier1 --no-llm "$skill_dir" -o "$EVAL_TEMP_DIR/tier1-$skill_name"
  )
  t_exit=$?
  T_CODES+=("$t_exit")
  if [ $t_exit -ne 0 ]; then
    echo "! tier1 check incomplete or failed for ${skill_name} (exit=${t_exit})"
    OVERALL_FAILED=1
  else
    echo "✓ tier1 check passed for ${skill_name}"
  fi
done

echo ""
echo "=========================================================="
echo "                  EVALUATION SUMMARY"
echo "=========================================================="
printf "%-26s | %-15s | %-15s\n" "Skill" "Quality Check" "Tier 1 Static"
echo "----------------------------------------------------------"
for i in "${!SKILL_NAMES[@]}"; do
  s_name="${SKILL_NAMES[$i]}"
  q_code="${Q_CODES[$i]}"
  t_code="${T_CODES[$i]}"
  q_str="PASS (0)"
  [ "$q_code" -ne 0 ] && q_str="FAIL ($q_code)"
  t_str="PASS (0)"
  [ "$t_code" -ne 0 ] && t_str="FAIL/INCOMPL ($t_code)"
  printf "%-26s | %-15s | %-15s\n" "$s_name" "$q_str" "$t_str"
done
echo "=========================================================="

if [ $OVERALL_FAILED -ne 0 ]; then
  echo "Evaluation completed with one or more check failures or incomplete coverage." >&2
  exit 1
else
  echo "All skills successfully passed quality and Tier 1 validation!"
  exit 0
fi
