conda skeleton pypi data-science-utils --python-version 3.6
FOR %G IN (3.6 3.7 3.8) DO(
conda-build --python %G data-science-utils
)
conda build purge
conda convert -f --platform all C:\Anaconda3\envs\DataScienceUtils\conda-bld\win-64\data-science-utils-1.4.1-py36_0.tar.bz2 -o outputdir\
conda convert -f --platform all C:\Anaconda3\envs\DataScienceUtils\conda-bld\win-64\data-science-utils-1.4.1-py37_0.tar.bz2 -o outputdir\
conda convert -f --platform all C:\Anaconda3\envs\DataScienceUtils\conda-bld\win-64\data-science-utils-1.4.1-py38_0.tar.bz2 -o outputdir\

anaconda upload outputdir/linux-32/data-science-utils-1.4.1-py36_0.tar.bz2 outputdir/linux-32/data-science-utils-1.4.1-py37_0.tar.bz2
anaconda upload outputdir/linux-64/data-science-utils-1.4.1-py36_0.tar.bz2 outputdir/linux-64/data-science-utils-1.4.1-py37_0.tar.bz2
anaconda upload outputdir/linux-aarch64/data-science-utils-1.4.1-py36_0.tar.bz2 outputdir/linux-aarch64/data-science-utils-1.4.1-py37_0.tar.bz2
anaconda upload outputdir/linux-armv6l/data-science-utils-1.4.1-py36_0.tar.bz2 outputdir/linux-armv6l/data-science-utils-1.4.1-py37_0.tar.bz2
anaconda upload outputdir/linux-armv7l/data-science-utils-1.4.1-py36_0.tar.bz2 outputdir/linux-armv7l/data-science-utils-1.4.1-py37_0.tar.bz2
anaconda upload outputdir/linux-ppc64le/data-science-utils-1.4.1-py36_0.tar.bz2 outputdir/linux-ppc64le/data-science-utils-1.4.1-py37_0.tar.bz2
anaconda upload outputdir/osx-64/data-science-utils-1.4.1-py36_0.tar.bz2 outputdir/osx-64/data-science-utils-1.4.1-py37_0.tar.bz2
anaconda upload outputdir/win-32/data-science-utils-1.4.1-py36_0.tar.bz2 outputdir/win-32/data-science-utils-1.4.1-py37_0.tar.bz2
anaconda upload outputdir/win-64/data-science-utils-1.4.1-py36_0.tar.bz2 outputdir/win-64/data-science-utils-1.4.1-py37_0.tar.bz2