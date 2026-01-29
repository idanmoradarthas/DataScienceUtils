#!/bin/bash

set -e

# Extract version from __init__.py
version=$(grep -E "__version__[^=]*=" ds_utils/__init__.py | cut -d'"' -f2)

# Get numpy version
numpy_version=$(python -c "import numpy; print(numpy.__version__)")

# Create output directory if it doesn't exist
mkdir -p ./outputdir

# Build for different Python versions
for pyver in 3.10 3.11 3.12 3.13; do
    conda build --python $pyver data-science-utils --numpy $numpy_version --output-folder outputdir/ --package-format 1
done
conda build purge

# Determine the build platform based on the architecture
PLATFORM=$(uname -m)
if [ "$PLATFORM" = "x86_64" ]; then
    BUILD_PLATFORM="linux-64"
elif [ "$PLATFORM" = "aarch64" ]; then
    BUILD_PLATFORM="linux-aarch64"
elif [ "$PLATFORM" = "arm64" ]; then
    BUILD_PLATFORM="osx-arm64"
else
    # Default fallback - try to find which directory was created
    for dir in outputdir/*/; do
        if [ -d "$dir" ] && [ "$(basename "$dir")" != "outputdir" ]; then
            BUILD_PLATFORM=$(basename "$dir")
            break
        fi
    done
fi

# Convert packages for all Python versions and platforms
for pyver in 310 311 312 313; do
    conda convert -f --platform all "outputdir/${BUILD_PLATFORM}/data-science-utils-${version}-py${pyver}_0.tar.bz2" -o outputdir/
done

# Check if upload should be skipped
if [ "$SKIP_UPLOAD" = "true" ]; then
    echo "SKIP_UPLOAD is set. Skipping Anaconda upload."
    echo "Built packages are available in ./outputdir"
    exit 0
fi

# Get available platforms
platforms=$(conda convert --help | grep -E "(-p|--platform)" | grep -o "{.*}" | sed 's/[{}]//g' | tr ',' ' ')

# Login to Anaconda
anaconda login --username IdanMorad

# Upload packages for all Python versions and platforms
for pyver in 310 311 312 313; do
    for platform in $platforms; do
        if [ "$platform" != "all" ]; then
            anaconda upload "outputdir/${platform}/data-science-utils-${version}-py${pyver}_0.tar.bz2"
        fi
    done
done

anaconda logout
