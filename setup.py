from pathlib import Path

from setuptools import setup, find_packages

version = Path(__file__).parents[0].joinpath(".version").read_text().split("==")[1]
long_description = Path(__file__).parents[0].joinpath("README.md").read_text()
requirements = Path(__file__).parents[0].joinpath("requirements.txt").read_text().splitlines()
project_license = Path(__file__).parents[0].joinpath("LICENSE").read_text().splitlines()[0]

setup(name="data_science_utils",
      version=version,
      author="Idan Morad",
      description="This project is an ensemble of methods which are frequently used in python ML projects.",
      long_description=long_description,
      long_description_content_type="text/markdown",
      classifiers=["Development Status :: 5 - Production/Stable",
                   "Framework :: tox",
                   "Intended Audience :: Developers",
                   "Intended Audience :: Education",
                   "Intended Audience :: Science/Research",
                   "License :: OSI Approved :: MIT License",
                   "Programming Language :: Python :: 3.6",
                   "Topic :: Scientific/Engineering :: Artificial Intelligence"],
      keywords="machine-learning ml scikit-learn",
      packages=find_packages(exclude=['contrib', 'docs', 'tests']),
      install_requires=requirements,
      python_requires='>=3.6',
      license=project_license)
