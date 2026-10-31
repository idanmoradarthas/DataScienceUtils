#!/bin/bash
# Run NVIDIA SkillEvaluator checks on all skills in skills/
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Running NVIDIA SkillEvaluator on Agent Skills ==="

for skill_dir in skills/*/; do
  skill_name=$(basename "$skill_dir")
  echo ""
  echo ">>> Evaluating ${skill_name}..."

  echo "  [1/2] Quality Check (Score >= 70)..."
  uv run skillevaluator quality-check "$skill_dir"

  echo "  [2/2] Tier 1 Static Validation (--no-llm)..."
  uv run skillevaluator tier1 --no-llm "$skill_dir" || true
done

echo ""
echo "=== Skill evaluation completed! ==="
