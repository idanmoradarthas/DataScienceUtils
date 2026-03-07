@echo off
echo Building Docker image ds_utils_mpl_gen...
docker build -t ds_utils_mpl_gen -f Dockerfile.mpl_hash_gen .
if %errorlevel% neq 0 (
    echo Docker build failed.
    exit /b %errorlevel%
)

echo.
echo Running Docker container to generate baseline images...
docker run --rm -v "%cd%:/app" ds_utils_mpl_gen
if %errorlevel% neq 0 (
    echo Container execution failed.
    exit /b %errorlevel%
)

echo.
echo Baseline images successfully generated in tests/baseline_images/mpl3108_ft261.json
