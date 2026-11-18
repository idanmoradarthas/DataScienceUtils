@echo off
REM Build and upload DataScienceUtils to PyPI with uv.
REM The upload token is read from the UV_PUBLISH_TOKEN environment variable; it is never stored in the repository.
REM   PowerShell: $env:UV_PUBLISH_TOKEN = "pypi-..."      cmd: set UV_PUBLISH_TOKEN=pypi-...
IF EXIST dist rmdir /s /q dist
uv build --out-dir dist
IF ERRORLEVEL 1 EXIT /B 1
uv publish dist\*
