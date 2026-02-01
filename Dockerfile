# DataScienceUtils Conda Package Deployment
#
# USAGE:
#   1. Build: docker build -t datascienceutils-conda-deploy .
#   2. Run:   docker run -it --rm -v $(pwd):/workspace -v datascienceutils-conda-cache:/opt/conda/pkgs datascienceutils-conda-deploy
#      Windows PowerShell: docker run -it --rm -v ${PWD}:/workspace -v datascienceutils-conda-cache:/opt/conda/pkgs datascienceutils-conda-deploy
#      Windows CMD: docker run -it --rm -v %cd%:/workspace -v datascienceutils-conda-cache:/opt/conda/pkgs datascienceutils-conda-deploy
#   3. Test (no upload): Add -e SKIP_UPLOAD=true to the run command
#   4. Specify numpy version: Add -e NUMPY_VERSION=1.26.3 to the run command
#   5. Output: Built packages will be in ./outputdir
#
# The script will prompt for your Anaconda password when running.

FROM continuumio/miniconda3:latest

WORKDIR /workspace

# Install conda build dependencies
COPY requirements-conda.txt /tmp/requirements-conda.txt
RUN conda install --yes -c conda-forge --file /tmp/requirements-conda.txt && rm /tmp/requirements-conda.txt

# Install numpy - will be overridden at runtime if NUMPY_VERSION is provided
# but we need some version for the conda build command to work
RUN conda install --yes -c conda-forge numpy

# Copy and prepare the deployment script
COPY deploy-conda.sh /usr/local/bin/deploy-conda.sh
RUN sed -i 's/\r$//' /usr/local/bin/deploy-conda.sh
RUN chmod +x /usr/local/bin/deploy-conda.sh

# Set the entrypoint to run the deployment script
ENTRYPOINT ["/usr/local/bin/deploy-conda.sh"]
