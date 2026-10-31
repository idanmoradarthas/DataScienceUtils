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
set TEMP_EVAL_DIR=%TEMP%\eval_skills_%RANDOM%_%RANDOM%
mkdir "%TEMP_EVAL_DIR%" >nul 2>nul

set COUNT=0

for /d %%D in (skills\*) do (
    set "SKILL_NAME=%%~nxD"
    echo.
    echo ----------------------------------------------------------
    echo ^>^>^> Evaluating: !SKILL_NAME!
    echo ----------------------------------------------------------

    echo Running quality-check on !SKILL_NAME!...
    skillevaluator quality-check "%%D" -o "%TEMP_EVAL_DIR%\quality-!SKILL_NAME!"
    set "Q_EXIT=!ERRORLEVEL!"

    echo Running tier1 --no-llm on !SKILL_NAME!...
    skillevaluator tier1 --no-llm "%%D" -o "%TEMP_EVAL_DIR%\tier1-!SKILL_NAME!"
    set "T_EXIT=!ERRORLEVEL!"

    set /a COUNT+=1
    set "SKILL_NAME_!COUNT!=!SKILL_NAME!"
    set "Q_EXIT_!COUNT!=!Q_EXIT!"
    set "T_EXIT_!COUNT!=!T_EXIT!"

    if !Q_EXIT! neq 0 set OVERALL_FAILED=1
    if !T_EXIT! neq 0 set OVERALL_FAILED=1
)

REM Clean up only our temp evaluation directory
rmdir /s /q "%TEMP_EVAL_DIR%" >nul 2>nul

echo.
echo ==========================================================
echo                   EVALUATION SUMMARY
echo ==========================================================
echo Skill                      ^| Quality Check   ^| Tier 1 Static
echo ----------------------------------------------------------
for /l %%I in (1,1,%COUNT%) do (
    set "S_NAME=!SKILL_NAME_%%I!                          "
    set "S_NAME=!S_NAME:~0,26!"
    set "Q_STATUS=PASS (0)       "
    if !Q_EXIT_%%I! neq 0 set "Q_STATUS=FAIL (!Q_EXIT_%%I!)        "
    set "Q_STATUS=!Q_STATUS:~0,15!"
    set "T_STATUS=PASS (0)       "
    if !T_EXIT_%%I! neq 0 set "T_STATUS=FAIL/INCOMPL (!T_EXIT_%%I!)"
    set "T_STATUS=!T_STATUS:~0,15!"
    echo !S_NAME! ^| !Q_STATUS! ^| !T_STATUS!
)
echo ==========================================================

if %OVERALL_FAILED% neq 0 (
    echo Evaluation completed with one or more check failures or incomplete coverage. >&2
    exit /b 1
) else (
    echo All skills successfully passed quality and Tier 1 validation!
    exit /b 0
)
