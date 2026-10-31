@echo off
setlocal enabledelayedexpansion

REM Evaluates Agent Skills under skills\ using NVIDIA SkillEvaluator.
REM Executes quality checks and Tier 1 static validation across all skills,
REM reports individual statuses, and exits non-zero if any required check fails.
REM For full setup instructions and scanner prerequisites, see skills\README.md.

where skillevaluator >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Error: 'skillevaluator' executable not found in PATH. >&2
    echo Please install dependencies listed in requirements-skills.txt: >&2
    echo   uv tool install --python 3.13 "skillevaluator @ git+https://github.com/NVIDIA/SkillEvaluator.git@a61e1f3ff609118c0c07e5f569aaf58fa5a76be0" >&2
    exit /b 1
)

echo ==========================================================
echo     NVIDIA SkillEvaluator Runner - DataScienceUtils
echo ==========================================================
skillevaluator --version

set OVERALL_FAILED=0

for /d %%D in (skills\*) do (
    set "SKILL_NAME=%%~nxD"
    echo.
    echo ----------------------------------------------------------
    echo ^>^>^> Evaluating: !SKILL_NAME!
    echo ----------------------------------------------------------

    echo Running quality-check on !SKILL_NAME!...
    skillevaluator quality-check "%%D"
    set "Q_EXIT=!ERRORLEVEL!"
    if !Q_EXIT! neq 0 (
        echo ! quality-check failed for !SKILL_NAME! ^(exit=!Q_EXIT!^)
        set OVERALL_FAILED=1
    ) else (
        echo [OK] quality-check passed for !SKILL_NAME!
    )

    echo Running tier1 --no-llm on !SKILL_NAME!...
    skillevaluator tier1 --no-llm "%%D"
    set "T_EXIT=!ERRORLEVEL!"
    if !T_EXIT! neq 0 (
        echo ! tier1 check incomplete or failed for !SKILL_NAME! ^(exit=!T_EXIT!^)
        set OVERALL_FAILED=1
    ) else (
        echo [OK] tier1 check passed for !SKILL_NAME!
    )
)

if exist reports rmdir /s /q reports

echo.
echo ==========================================================
if %OVERALL_FAILED% neq 0 (
    echo Evaluation completed with one or more check failures or incomplete coverage. >&2
    exit /b 1
) else (
    echo All skills successfully passed quality and Tier 1 validation!
    exit /b 0
)
