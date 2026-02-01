@echo off
REM Build and deploy DataScienceUtils conda package using Docker
REM Usage: deploy-docker.bat [--skip-upload]
REM The script will prompt for your Anaconda password when needed unless --skip-upload is used

setlocal

set IMAGE_NAME=datascienceutils-conda-deploy
set VOLUME_NAME=datascienceutils-conda-cache
set SKIP_UPLOAD_FLAG=

REM Parse arguments
if "%~1"=="--skip-upload" (
    set SKIP_UPLOAD_FLAG=-e SKIP_UPLOAD=true
    echo Running in test mode - no upload
)

REM Check if Docker is installed
docker --version >nul 2>&1
if errorlevel 1 (
    echo Error: Docker is not installed
    exit /b 1
)

REM Check if we're in the right directory
if not exist "pyproject.toml" (
    echo Error: Run this script from the project root directory
    exit /b 1
)

REM Get numpy version from host environment
echo Detecting numpy version from host environment...
for /f "delims=" %%v in ('python -c "import numpy; print(numpy.__version__)" 2^>nul') do (
    set NUMPY_VERSION=%%v
)

if defined NUMPY_VERSION (
    echo Using numpy version: %NUMPY_VERSION%
    set NUMPY_VERSION_FLAG=-e NUMPY_VERSION=%NUMPY_VERSION%
) else (
    echo Warning: Could not detect numpy version from host. Using version from container.
    set NUMPY_VERSION_FLAG=
)

REM Build Docker image
echo Building Docker image...
docker build -t %IMAGE_NAME% .

if errorlevel 1 (
    echo Error: Failed to build Docker image
    exit /b 1
)

REM Run the deployment
echo Running deployment...
docker run -it --rm ^
  -v "%cd%:/workspace" ^
  -v "%VOLUME_NAME%:/opt/conda/pkgs" ^
  %SKIP_UPLOAD_FLAG% ^
  %NUMPY_VERSION_FLAG% ^
  %IMAGE_NAME%

echo Deployment complete! Packages are in .\outputdir

endlocal
