#!/bin/bash
#
# Evaluates Agent Skills under skills/ using NVIDIA SkillEvaluator.
# Executes quality checks and Tier 1 static validation across all skills,
# reports individual statuses, and exits non-zero if any required check fails.
# For full setup instructions and scanner prerequisites, see skills/README.md.
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

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

# Track per-skill exit statuses
declare -A QUALITY_STATUS
declare -A TIER1_STATUS
OVERALL_FAILED=0

for skill_dir in skills/*/; do
  [ -d "$skill_dir" ] || continue
  skill_name=$(basename "$skill_dir")
  echo ""
  echo "----------------------------------------------------------"
  echo ">>> Evaluating: ${skill_name}"
  echo "----------------------------------------------------------"

  # [1/2] Quality Check (Default threshold >= 70)
  echo "Running quality-check on ${skill_name}..."
  skillevaluator quality-check "$skill_dir"
  q_exit=$?
  QUALITY_STATUS["$skill_name"]=$q_exit
  if [ $q_exit -ne 0 ]; then
    echo "! quality-check failed for ${skill_name} (exit=${q_exit})"
    OVERALL_FAILED=1
  else
    echo "✓ quality-check passed for ${skill_name}"
  fi

  # [2/2] Tier 1 Static Validation (--no-llm keyless mode)
  echo "Running tier1 --no-llm on ${skill_name}..."
  skillevaluator tier1 --no-llm "$skill_dir"
  t_exit=$?
  TIER1_STATUS["$skill_name"]=$t_exit
  if [ $t_exit -ne 0 ]; then
    echo "! tier1 check incomplete or failed for ${skill_name} (exit=${t_exit})"
    OVERALL_FAILED=1
  else
    echo "✓ tier1 check passed for ${skill_name}"
  fi
done

# Clean up generated local reports directory if created
rm -rf reports

echo ""
echo "=========================================================="
echo "                  EVALUATION SUMMARY"
echo "=========================================================="
printf "%-26s | %-15s | %-15s\n" "Skill" "Quality Check" "Tier 1 Static"
echo "----------------------------------------------------------"
for skill_dir in skills/*/; do
  [ -d "$skill_dir" ] || continue
  s_name=$(basename "$skill_dir")
  q_code=${QUALITY_STATUS["$s_name"]}
  t_code=${TIER1_STATUS["$s_name"]}
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
