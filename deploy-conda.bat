:: taken from: https://medium.com/@giswqs/building-a-conda-package-and-uploading-it-to-anaconda-cloud-6a3abd1c5c52

:: run this line of code if you don't have a skeleton directory (./data-science-utils).
:: conda skeleton pypi data-science-utils --python-version 3.6

for /F "tokens=2 delims== " %%i in ('findstr /R "__version__[^=]*=" ds_utils\__init__.py') do (
    set version=%%i
)

set version=%version:"=%

if not exist .\outputdir mkdir .\outputdir

for %%v in (3.9 3.10 3.11) do (
    call conda build --python %%v data-science-utils --numpy 1.26.3 --output-folder outputdir\
)
call conda build purge

for %%v in (39 310 311) do (
    call conda convert -f --platform all outputdir\win-64\data-science-utils-%version%-py%%v_0.tar.bz2 -o outputdir\
)

anaconda login

for %%v in (39 310 311) do (
    for %%i in (linux-32 linux-64 linux-aarch64 linux-armv6l linux-armv7l linux-ppc64le osx-64 win-32 win-64) do (
        call anaconda upload outputdir/%%i/data-science-utils-%version%-py%%v_0.tar.bz2
    )
)