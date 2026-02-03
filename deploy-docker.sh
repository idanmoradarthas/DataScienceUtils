#!/bin/bash
# Build and deploy DataScienceUtils conda package using Docker
# Usage: ./deploy-docker.sh [--skip-upload]
# The script will prompt for your Anaconda password when needed unless --skip-upload is used

set -e

IMAGE_NAME="datascienceutils-conda-deploy"
VOLUME_NAME="datascienceutils-conda-cache"

# Parse arguments
SKIP_UPLOAD_FLAG=""
for arg in "$@"; do
  if [ "$arg" == "--skip-upload" ]; then
    SKIP_UPLOAD_FLAG="-e SKIP_UPLOAD=true"
    echo "Running in test mode (no upload)"
  fi
done

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed"
    exit 1
fi

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo "Error: Run this script from the project root directory"
    exit 1
fi

# Get numpy version from host environment
echo "Detecting numpy version from host environment..."
NUMPY_VERSION=$(python -c "import numpy; print(numpy.__version__)" 2>/dev/null || echo "")
if [ -z "$NUMPY_VERSION" ]; then
    echo "Warning: Could not detect numpy version from host. Using version from container."
    NUMPY_VERSION_FLAG=""
else
    echo "Using numpy version: $NUMPY_VERSION"
    NUMPY_VERSION_FLAG="-e NUMPY_VERSION=$NUMPY_VERSION"
fi

# Build Docker image
echo "Building Docker image..."
docker build -t $IMAGE_NAME .

# Run the deployment
echo "Running deployment..."
docker run -it --rm \
  -v "$(pwd):/workspace" \
  -v "$VOLUME_NAME:/opt/conda/pkgs" \
  $SKIP_UPLOAD_FLAG \
  $NUMPY_VERSION_FLAG \
  $IMAGE_NAME

echo "Deployment complete! Packages are in ./outputdir"
