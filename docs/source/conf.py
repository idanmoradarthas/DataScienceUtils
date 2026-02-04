"""Configuration file for the Sphinx documentation builder."""

# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import datetime
import os
import sys
from pathlib import Path

# -- Path setup --------------------------------------------------------------
# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("../../ds_utils"))

# -- Project information -----------------------------------------------------

project = "Data Science Utils"
copyright = f"{datetime.date.today().year}, Idan Morad"
author = "Idan Morad"


# Dynamic version extraction with fallback
def get_version():
    """Extract version from package.

    First tries to import the package directly (works when package is installed).
    Falls back to parsing __init__.py file if import fails.

    Returns:
        str: Version string or "unknown" if version cannot be determined

    """
    # Method 1: Try importing the package directly (preferred)
    # This works when the package is installed via pip
    try:
        import ds_utils

        if hasattr(ds_utils, "__version__"):
            return ds_utils.__version__
    except ImportError:
        pass

    # Method 2: Fallback to parsing __init__.py file
    # This works during development when package isn't installed
    init_file = Path(__file__).parents[2] / "ds_utils" / "__init__.py"
    try:
        with open(init_file, "r", encoding="utf-8") as f:
            content = f.read()
            # Match patterns like: __version__ = "1.2.3" or __version__ = '1.2.3'
            import re

            match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', content)
            if match:
                return match.group(1)
    except (FileNotFoundError, AttributeError, OSError):
        pass

    # Method 3: Final fallback if all else fails
    return "unknown"


# The full version, including alpha/beta/rc tags
release = get_version()

# The short X.Y version (extract from full version)
version = ".".join(release.split(".")[:2]) if release != "unknown" else "unknown"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx_rtd_theme",
    "sphinx.ext.viewcode",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
]

# Intersphinx mapping - allows linking to other project's documentation
intersphinx_mapping = {
    # Python standard library
    "python": ("https://docs.python.org/3/", None),
    # Core scientific computing
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    # Data manipulation and analysis
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    # Machine learning
    "sklearn": ("https://scikit-learn.org/stable/", None),
    # Visualization
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "plotly": ("https://plotly.com/python-api-reference/", None),
    "seaborn": ("https://seaborn.pydata.org/", None),
    # Other useful libraries
    "joblib": ("https://joblib.readthedocs.io/en/latest/", None),
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_static']

master_doc = "index"
