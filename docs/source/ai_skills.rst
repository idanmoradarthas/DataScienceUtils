================
AI Coding Skills
================

DataScienceUtils ships a set of *AI skills* — structured knowledge files that
teach AI coding assistants (Claude Code, Cursor, GitHub Copilot, Gemini CLI)
how to use this library correctly. Skills eliminate the most common mistakes:
wrong import paths, dtype-dependent behaviour, argument ordering pitfalls, and
API differences between matplotlib and Plotly outputs.

What Are Skills?
----------------
Skills are specialized knowledge files (a folder containing a SKILL.md file) that an AI
coding assistant reads automatically when it detects a relevant task. For more details on
how skills work, you can refer to the
`Anthropic's Skills Guide <https://resources.anthropic.com/hubfs/The-Complete-Guide-to-Building-Skill-for-Claude.pdf>`_.

Quick Install
-------------

**Mac / Linux:**

.. code-block:: bash

    bash <(curl -sL https://raw.githubusercontent.com/idanmoradarthas/DataScienceUtils/master/install.sh)

**Windows (PowerShell):**

.. code-block:: powershell

    irm https://raw.githubusercontent.com/idanmoradarthas/DataScienceUtils/master/install.ps1 | iex

The installer automatically detects your installed tools, project scope, and package manager.

What the Installer Does
-----------------------
The installer interactively guides you through the setup process. It performs tool detection
to locate AI editors, displays a checkbox selector for choosing which tools to inject skills into,
offers a project vs. global scope radio selector, auto-detects pip, conda, or a
source install from the cloned repository, and finally deploys
the skill files to the correct locations such as ``.claude/skills/``, ``.cursor/rules/``,
``.github/instructions/``, and ``.gemini/skills/``.

Step-by-Step Installation Guide
-------------------------------

**Step 1 — Run the command**

Open your terminal in your project directory and run the install command.

.. code-block:: bash

    bash <(curl -sL https://raw.githubusercontent.com/idanmoradarthas/DataScienceUtils/master/install.sh)

Windows users:

.. code-block:: powershell

    irm https://raw.githubusercontent.com/idanmoradarthas/DataScienceUtils/master/install.ps1 | iex

**Step 2 — Installer header + prerequisite check**

.. code-block:: text

    DataScienceUtils AI Skills Installer
    ────────────────────────────────────────

    Checking prerequisites
    ✓ git
    ✓ pip3

    Detecting AI coding tools

The installer checks that required tools (git and a Python package manager) are present before proceeding.

**Step 3 — Tool detection checkbox**

.. code-block:: text

    Select AI tools to install skills for:

      ↑/↓ navigate · space toggle · enter confirm

    ❯ [✓] Claude Code          detected
      [✓] Cursor               detected
      [ ] GitHub Copilot       not found
      [ ] Gemini CLI           not found

        [ Confirm ]

- Arrow keys ``↑`` / ``↓`` move the cursor between items
- ``Space`` toggles the checkbox on or off
- Navigate down to ``[ Confirm ]`` and press ``Enter`` (or press ``Enter`` while on any item to toggle it)
- Tools marked ``detected`` were found automatically; ``not found`` means the installer did not detect them but you can still select them manually
- At least one tool must be selected

On **Windows (PowerShell)**, the same selector appears with slightly different characters:

.. code-block:: text

    Select AI tools to install skills for:

      (Up/Down: navigate, Space/Enter: toggle/confirm)

    > [✓] Claude Code          detected
      [✓] Cursor               detected
      [ ] GitHub Copilot       not found
      [ ] Gemini CLI           not found

        [ Confirm ]

The controls are identical: arrow keys to navigate, ``Space`` or ``Enter`` to toggle, move to ``[ Confirm ]`` and press ``Enter`` to proceed.

**Step 4 — Scope radio selector**

.. code-block:: text

    Install scope:

      ↑/↓ navigate · enter select

    ❯ ● Project (current directory)   skills in .claude/skills, .cursor/rules, etc.
      ○ Global (home directory)        skills available across all projects

        [ Confirm ]

- **Project** installs skill files inside the current directory only (e.g. ``<your-project>/.claude/skills/``). This is the recommended default — skills are scoped to this repo and checked into version control.
- **Global** installs into your home directory (``~/.claude/skills/``), making the skills available in every project you open. Choose this if you use DataScienceUtils across many projects and do not want per-repo setup.

On **Windows (PowerShell)**, the same selector appears with slightly different characters:

.. code-block:: text

    Install scope:

      (Up/Down: navigate, Enter: select)

    > * Project (current directory)   skills in .claude\skills, .cursor\rules, etc.
      o Global (home directory)        skills available across all projects

        [ Confirm ]

**Step 5 — Package manager radio selector**

.. code-block:: text

    Install data-science-utils using:

      ↑/↓ navigate · enter select

    ❯ ● pip (PyPI)                           recommended
      ○ conda (idanmorad channel)             available
      ○ Install from source (git clone)       clone repo and pip install .
      ○ Skip (install skills only)            do not install the python package

        [ Confirm ]

- **pip** installs from PyPI (``pip install data-science-utils``). This is the default for most Python environments.
- **conda** installs from the ``idanmorad`` Anaconda channel (``conda install -c idanmorad data-science-utils``). If you are inside an active conda environment (i.e. ``$CONDA_DEFAULT_ENV`` is set), conda will be pre-selected automatically.
- **Install from source** clones the GitHub repository to a temporary directory, runs ``pip install .``, then removes the clone. Use this if you need the absolute latest unreleased code. If you are already running the installer from inside a cloned DataScienceUtils repo, this option will be pre-selected automatically.
- **Skip (install skills only)** bypasses the package installation step completely. Use this if you already have the package installed or are running in an environment that manages its own dependencies.

.. note::
   If you installed ``data-science-utils`` from source (``git clone`` +
   ``pip install .``) or already have it installed, skip the package install
   step by passing ``--skills-only`` to the installer. When ``--skills-only``
   is active, Step 5 is skipped entirely and the installer jumps straight to
   the confirmation summary.

   **Mac / Linux:**

   .. code-block:: bash

       bash <(curl -sL https://raw.githubusercontent.com/idanmoradarthas/DataScienceUtils/master/install.sh) --skills-only

   **Windows (PowerShell):**

   .. code-block:: powershell

       irm https://raw.githubusercontent.com/idanmoradarthas/DataScienceUtils/master/install.ps1 -OutFile install.ps1
       .\install.ps1 -SkillsOnly

On **Windows (PowerShell)**, the same selector appears with slightly different characters:

.. code-block:: text

    Install data-science-utils using:

      (Up/Down: navigate, Enter: select)

    > * pip (PyPI)                           recommended
      o conda (idanmorad channel)             available
      o Install from source (git clone)       clone repo and pip install .
      o Skip (install skills only)            do not install the python package

        [ Confirm ]

**Step 6 — Confirmation summary**

.. code-block:: text

    Summary
    ──────────────────────────────────────
    Tools:    Claude Code, Cursor
    Scope:    project (/home/user/my-project)
    Package:  pip install
    Skills:   metrics preprocess unsupervised strings xai

    Proceed with installation? (y/n) [y]:

Review every line of this summary before pressing ``Enter``. Type ``n`` and press ``Enter`` to cancel without making any changes. Press ``Enter`` (or type ``y``) to proceed. (The ``Package`` line will read "conda install" or "skip" depending on your selections).

**Step 7 — Installation progress**

.. code-block:: text

    Installing data-science-utils Python package
    Using: pip3
    ✓ data-science-utils installed

    Installing ds_utils skills
    ✓ metrics    → .claude/skills/ds-utils-metrics
    ✓ preprocess → .claude/skills/ds-utils-preprocess
    ✓ unsupervised → .claude/skills/ds-utils-unsupervised
    ✓ strings    → .claude/skills/ds-utils-strings
    ✓ xai        → .claude/skills/ds-utils-xai
    ✓ metrics    → .cursor/rules/ds-utils-metrics
    ✓ preprocess → .cursor/rules/ds-utils-preprocess
    ✓ unsupervised → .cursor/rules/ds-utils-unsupervised
    ✓ strings    → .cursor/rules/ds-utils-strings
    ✓ xai        → .cursor/rules/ds-utils-xai

Each ``✓`` line confirms a skill file was downloaded and placed correctly. If any line shows ``!`` (warning) instead, see Troubleshooting.

**Step 8 — Completion message**

.. code-block:: text

    Installation complete!
    ────────────────────────────────────────
    Package:  installed
    Scope:    project
    Tools:    Claude Code, Cursor

    Skills installed:
      ds-utils-metrics
      ds-utils-preprocess
      ds-utils-unsupervised
      ds-utils-strings
      ds-utils-xai

    Next steps:
    1. Open your project in your AI coding tool
    2. Skills are auto-loaded — no config needed
    3. Try: "Use ds_utils to plot a confusion matrix for my classifier"

The skills are now active. No further configuration is needed — open your project in Claude Code or Cursor and the skills will load automatically whenever you ask something related to DataScienceUtils.

**Step 9 — Verify the installation**

After the installer exits, confirm the files are present:

.. code-block:: bash

    find . -path "*/ds-utils-*/SKILL.md"

Expected output (for a project-scoped Claude Code + Cursor install):

.. code-block:: text

    ./.claude/skills/ds-utils-metrics/SKILL.md
    ./.claude/skills/ds-utils-preprocess/SKILL.md
    ./.claude/skills/ds-utils-unsupervised/SKILL.md
    ./.claude/skills/ds-utils-strings/SKILL.md
    ./.claude/skills/ds-utils-xai/SKILL.md
    ./.cursor/rules/ds-utils-metrics/SKILL.md
    ./.cursor/rules/ds-utils-preprocess/SKILL.md
    ./.cursor/rules/ds-utils-unsupervised/SKILL.md
    ./.cursor/rules/ds-utils-strings/SKILL.md
    ./.cursor/rules/ds-utils-xai/SKILL.md

On **Windows (PowerShell)**:

.. code-block:: powershell

    Get-ChildItem -Recurse -Filter "SKILL.md" | Where-Object { $_.DirectoryName -match "ds-utils-" }

Expected output:

.. code-block:: text

        Directory: C:\Users\you\my-project\.claude\skills\ds-utils-metrics

    Mode  LastWriteTime    Length Name
    ----  -------------    ------ ----
    -a--- 01/01/2026 ...     2048 SKILL.md

    (repeated for each of the 5 skills × number of tools selected)

On Mac / Linux, if ``find`` returns nothing, the skill files were not placed
correctly. On Windows, if ``Get-ChildItem`` returns nothing, the same applies.
In either case, re-run the installer or follow the Manual Installation section below.

**Step 10 — First use**

Open your project in your AI coding tool — no restart needed. Then give two concrete example prompts to try:

- ``"Plot a confusion matrix for my classifier using ds_utils"``

The AI will automatically load the relevant skill and follow the correct import paths, function signatures, and usage patterns for DataScienceUtils.

.. _CLI Flags Reference:

CLI Flags Reference
-------------------

.. list-table::
   :widths: 25 15 60
   :header-rows: 1

   * - Flag
     - Platform
     - Description
   * - ``--global`` / ``-g``
     - bash
     - Install into home dir (``~/.claude/skills/`` etc.) instead of the current project directory
   * - ``-Global``
     - PowerShell
     - Same as ``--global``
   * - ``--skills-only``
     - bash
     - Skip the Python package install; only deploy skill files
   * - ``-SkillsOnly``
     - PowerShell
     - Same as ``--skills-only``
   * - ``--from-source``
     - bash
     - Install the package by cloning the repo and running ``pip install .`` instead of PyPI/conda
   * - ``-FromSource``
     - PowerShell
     - Same as ``--from-source``
   * - ``--tools claude,cursor``
     - bash
     - Pre-select tools without the interactive checkbox; comma-separated values: ``claude``, ``cursor``, ``copilot``, ``gemini``
   * - ``-Tools "claude,cursor"``
     - PowerShell
     - Same as ``--tools``
   * - ``--force`` / ``-f``
     - bash
     - Overwrite already-installed skill files
   * - ``-Force``
     - PowerShell
     - Same as ``--force``
   * - ``--help`` / ``-h``
     - bash
     - Print usage and exit

Install skills only for Claude Code, globally, force-overwriting existing files:

.. code-block:: bash

    bash <(curl -sL https://raw.githubusercontent.com/idanmoradarthas/DataScienceUtils/master/install.sh) \
        --skills-only --global --tools claude --force

PowerShell equivalent — download first, then run with parameters:

.. code-block:: powershell

    irm https://raw.githubusercontent.com/idanmoradarthas/DataScienceUtils/master/install.ps1 -OutFile install.ps1
    .\install.ps1 -SkillsOnly -Global -Tools "claude" -Force

Manual Installation
-------------------
If you prefer not to use the automated installer, you can install skills manually.
Download the ``SKILL.md`` files from the
`skills directory on GitHub <https://github.com/idanmoradarthas/DataScienceUtils/tree/master/skills>`_
and place them in the following paths depending on your tool:

* **Claude Code**: ``.claude/skills/ds-utils-<module>/SKILL.md``
* **Cursor**: ``.cursor/rules/ds-utils-<module>/SKILL.md``
* **GitHub Copilot**: ``.github/instructions/ds-utils-<module>/SKILL.md``
* **Gemini CLI**: ``.gemini/skills/ds-utils-<module>/SKILL.md``

Available Skills
----------------

.. list-table:: Available Skills
   :widths: 25 25 50
   :header-rows: 1

   * - Skill name
     - Module
     - What it covers
   * - ds-utils-metrics
     - ds_utils.metrics
     - Confusion matrix, ROC, PR, learning curves, probability analysis
   * - ds-utils-preprocess
     - ds_utils.preprocess
     - Feature visualization, correlation, statistics, mutual information
   * - ds-utils-unsupervised
     - ds_utils.unsupervised
     - Cluster cardinality, magnitude, optimal k
   * - ds-utils-strings
     - ds_utils.strings
     - Tag encoding, significant term extraction
   * - ds-utils-xai
     - ds_utils.xai
     - Feature importance visualization

Updating Skills
---------------
To update your skills to the latest version, run the installer with the ``--force`` flag:

.. code-block:: bash

    bash <(curl -sL https://raw.githubusercontent.com/idanmoradarthas/DataScienceUtils/master/install.sh) --force

On **Windows (PowerShell)**:

.. code-block:: powershell

    irm https://raw.githubusercontent.com/idanmoradarthas/DataScienceUtils/master/install.ps1 -OutFile install.ps1
    .\install.ps1 -Force

Troubleshooting
---------------
* **Skills not triggering**: The description field guides the AI on when to trigger the skill. Try an explicit trigger by asking your AI assistant to "use the ds-utils-metrics skill".
* **Wrong directory**: Ensure the skills are in the correct dot-directory corresponding to your tool (e.g., ``.cursor/rules/`` for Cursor).
* **Package not installed**: The AI tools expect the package to be present in your environment. Please follow the :doc:`Installation Guide <installation>` if it is missing.
* **Installed from source and don't want pip/conda to reinstall the package**:
  Use the ``--skills-only`` flag (bash) or ``-SkillsOnly`` (PowerShell).
  This deploys only the skill files and skips the package manager step
  entirely. See :ref:`CLI Flags Reference` for details.
