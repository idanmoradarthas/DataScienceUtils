@echo off
REM Build and deploy DataScienceUtils conda package using Docker
REM Usage: deploy-docker.bat [--skip-upload]
REM The script will prompt for your Anaconda password when needed

setlocal

set IMAGE_NAME=datascienceutils-conda-deploy
set VOLUME_NAME=datascienceutils-conda-cache
set SKIP_UPLOAD_ENV=

REM Parse arguments
if "%~1"=="--skip-upload" (
    set SKIP_UPLOAD_ENV=-e SKIP_UPLOAD=true
    echo Test mode: SKIP_UPLOAD is set to true
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
  %SKIP_UPLOAD_ENV% ^
  %IMAGE_NAME%

echo Deployment complete! Packages are in .\outputdir

endlocal
