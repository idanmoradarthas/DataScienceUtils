:: taken from: https://medium.com/@giswqs/building-a-conda-package-and-uploading-it-to-anaconda-cloud-6a3abd1c5c52

:: run this line of code if you don't have a skeleton directory (./data-science-utils).
:: conda skeleton pypi data-science-utils --python-version 3.6

:: Extract version from __init__.py
for /F "tokens=2 delims== " %%i in ('findstr /R "__version__[^=]*=" ds_utils\__init__.py') do (
    set version=%%i
)

set version=%version:"=%

:: Get numpy version
for /f "delims=" %%v in ('python -c "import numpy; print(numpy.__version__)"') do (
    set numpy_version=%%v
)

:: Create output directory if it doesn't exist
if not exist .\outputdir mkdir .\outputdir

:: Build for different Python versions
for %%v in (3.9 3.10 3.11 3.12) do (
    call conda build --python %%v data-science-utils --numpy %numpy_version% --output-folder outputdir\
)
call conda build purge

:: Convert packages for all Python versions and platforms
for %%v in (39 310 311 312) do (
    call conda convert -f --platform all outputdir\win-64\data-science-utils-%version%-py%%v_0.tar.bz2 -o outputdir\
)

:: Get available platforms from conda convert help
:: First, find the line containing -p or --platform
for /f "tokens=1,* delims={}" %%a in ('conda convert --help ^| findstr /C:"-p {" /C:"--platform {"') do (
    set "platforms_str=%%b"
)

:: Remove trailing text after } to get clean platform list
for /f "tokens=1 delims=}" %%a in ("%platforms_str%") do (
    set "platforms_str=%%a"
)

:: Login to Anaconda
anaconda login --username IdanMorad

:: Upload packages for all Python versions and platforms
for %%v in (39 310 311 312) do (
    for %%p in (%platforms_str%) do (
        if not "%%p"=="all" (
            call anaconda upload outputdir/%%p/data-science-utils-%version%-py%%v_0.tar.bz2
        )
    )
)

anaconda logout