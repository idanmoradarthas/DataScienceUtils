##################
Installation Guide
##################
.. highlight:: bash

Data Science Utils is compatible with |py_version|. The preferred way to install
the package is using the AI skills installer script, because it installs both the
``data-science-utils`` package and the tool skills in one guided flow. See
:doc:`AI Coding Skills <ai_skills>`.

Here are several ways to install the package:

0. Preferred: Install using the skills installer script
=======================================================

**Mac / Linux**::

    bash <(curl -sL https://raw.githubusercontent.com/idanmoradarthas/DataScienceUtils/master/install.sh)

**Windows (PowerShell)**::

    irm https://raw.githubusercontent.com/idanmoradarthas/DataScienceUtils/master/install.ps1 | iex

For full details, interactive options, and verification steps, see
:doc:`AI Coding Skills <ai_skills>`.

1. Install from PyPI
====================
The simplest way to install Data Science Utils and its dependencies is from PyPI using pip, Python's preferred package
installer |pypi_version|::

    pip install data-science-utils

To install with optional dependencies (like NLP features that require ``sentence-transformers``), use the extras syntax::

    pip install "data-science-utils[nlp]"

To upgrade Data Science Utils to the latest version, use::

    pip install -U data-science-utils


2. Install from Source
======================
If you prefer to install from source, you can clone the repository and install |github_release|::

    git clone https://github.com/idanmoradarthas/DataScienceUtils.git
    cd DataScienceUtils
    pip install .

To include optional dependencies when installing from source, use::

    pip install ".[nlp]"

Alternatively, you can install directly from GitHub using pip::

    pip install git+https://github.com/idanmoradarthas/DataScienceUtils.git

3. Install using Anaconda
=========================
If you're using Anaconda, you can install using conda |conda_version|::

    conda install idanmorad::data-science-utils

To install the optional NLP features (``sentence-transformers``) via conda, you need to separately install them from the ``conda-forge`` channel::

    conda install conda-forge::sentence-transformers


Note on Dependencies
====================

Data Science Utils has several core dependencies, including numpy, pandas, matplotlib, plotly and scikit-learn. These will be
automatically installed when you install the package using the methods above. 

Optional features (such as ``SentenceEmbeddingTransformer``) require additional dependencies (``sentence-transformers``) which must be installed explicitly using the ``[nlp]`` extra or separately via conda.

Staying Updated
===============

Data Science Utils is an active project that routinely publishes new releases with additional methods and improvements.
We recommend periodically checking for updates to access the latest features and bug fixes.

If you encounter any issues during installation, please check our
GitHub `issues <https://github.com/idanmoradarthas/DataScienceUtils/issues>`_ page or open a new issue for assistance.

.. |py_version| image:: https://img.shields.io/pypi/pyversions/data-science-utils
.. |pypi_version| image:: https://badge.fury.io/py/data-science-utils.svg
.. |github_release| image:: https://img.shields.io/github/v/release/idanmoradarthas/DataScienceUtils
.. |conda_version| image:: https://anaconda.org/idanmorad/data-science-utils/badges/version.svg