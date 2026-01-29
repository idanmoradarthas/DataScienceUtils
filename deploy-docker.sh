#!/bin/bash
# Build and deploy DataScienceUtils conda package using Docker
# Usage: ./deploy-docker.sh [--skip-upload]
# The script will prompt for your Anaconda password when needed

set -e

IMAGE_NAME="datascienceutils-conda-deploy"
VOLUME_NAME="datascienceutils-conda-cache"
SKIP_UPLOAD_ENV=""

# Parse arguments
for arg in "$@"; do
  if [ "$arg" == "--skip-upload" ]; then
    SKIP_UPLOAD_ENV="-e SKIP_UPLOAD=true"
    echo "Test mode: SKIP_UPLOAD is set to true"
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

# Build Docker image
echo "Building Docker image..."
docker build -t $IMAGE_NAME .

# Run the deployment
echo "Running deployment..."
docker run -it --rm \
  -v "$(pwd):/workspace" \
  -v "$VOLUME_NAME:/opt/conda/pkgs" \
  $SKIP_UPLOAD_ENV \
  $IMAGE_NAME

echo "Deployment complete! Packages are in ./outputdir"
