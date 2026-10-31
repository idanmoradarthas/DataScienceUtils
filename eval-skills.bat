@echo off
REM Run NVIDIA SkillEvaluator checks on all skills in skills\

echo === Running NVIDIA SkillEvaluator on Agent Skills ===

for /d %%D in (skills\*) do (
    echo.
    echo >>> Evaluating %%~nxD...

    echo   [1/2] Quality Check (Score ^>= 70)...
    uv run skillevaluator quality-check "%%D"

    echo   [2/2] Tier 1 Static Validation (--no-llm)...
    uv run skillevaluator tier1 --no-llm "%%D"
)

echo.
echo === Skill evaluation completed! ===
