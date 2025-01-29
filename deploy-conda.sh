#!/bin/bash

# Extract version from __init__.py
version=$(grep -E "__version__[^=]*=" ds_utils/__init__.py | cut -d'"' -f2)

# Get numpy version
numpy_version=$(python -c "import numpy; print(numpy.__version__)")

# Create output directory if it doesn't exist
mkdir -p ./outputdir

# Build for different Python versions
for pyver in 3.9 3.10 3.11 3.12; do
    conda build --python $pyver data-science-utils --numpy $numpy_version --output-folder outputdir/ --package-format 1
done
conda build purge

# Convert packages for all Python versions and platforms
for pyver in 39 310 311 312; do
    conda convert -f --platform all "outputdir/osx-arm64/data-science-utils-${version}-py${pyver}_0.tar.bz2" -o outputdir/
done

# Get available platforms
platforms=$(conda convert --help | grep -E "(-p|--platform)" | grep -o "{.*}" | sed 's/[{}]//g' | tr ',' ' ')

# Login to Anaconda
anaconda login --username IdanMorad

# Upload packages for all Python versions and platforms
for pyver in 39 310 311 312; do
    for platform in $platforms; do
        if [ "$platform" != "all" ]; then
            anaconda upload "outputdir/${platform}/data-science-utils-${version}-py${pyver}_0.tar.bz2"
        fi
    done
done

anaconda logout