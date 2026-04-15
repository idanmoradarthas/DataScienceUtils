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
Skills are structured knowledge files — a folder containing a ``SKILL.md``
file — that an AI coding assistant reads automatically when it detects a
relevant task. Skills follow the open `Agent Skills
<https://agentskills.io>`_ standard, which defines a common format portable
across all compliant AI coding tools.

For further reading:

* `Agent Skills specification <https://agentskills.io/specification>`_ — the canonical open standard defining the ``SKILL.md`` format, frontmatter fields, and directory conventions
* `Adding skills support <https://agentskills.io/client-implementation/adding-skills-support>`_ — how compliant clients discover, parse, and activate skills
* `Anthropic's Skills Guide <https://resources.anthropic.com/hubfs/The-Complete-Guide-to-Building-Skill-for-Claude.pdf>`_ — Anthropic's guide to building skills for Claude
* `Google Antigravity — Skills <https://antigravity.google/docs/skills>`_ — Antigravity-specific skills reference

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
The installer interactively guides you through the setup process. It
performs tool detection to locate AI editors (Claude Code, Cursor, GitHub
Copilot, Gemini CLI, Antigravity), displays a checkbox selector for
choosing which tools to inject skills into, offers a project vs. global
scope radio selector, auto-detects pip, conda, or a source install from
the cloned repository, optionally installs extra dependencies (like nlp),
and finally deploys skill files to:

* **Tool-specific directories**: ``.claude/skills/``, ``.cursor/rules/``,
  ``.github/instructions/``, ``.gemini/skills/``
* **Antigravity**: ``.agents/skills/`` (project) or
  ``~/.gemini/antigravity/skills/`` (global)
* **Cross-client path**: ``.agents/skills/`` — always written regardless
  of which tools you select, so skills are visible to any
  `Agent Skills <https://agentskills.io>`_-compatible tool

Cross-Client Compatibility
--------------------------

DataScienceUtils skills follow the open `Agent Skills
<https://agentskills.io>`_ standard. This means they work with any
skills-compatible AI coding tool — not only the ones you select during
installation.

The installer always writes skills to two locations in parallel:

1. **Tool-specific directories** for each tool you select (e.g.
   ``.claude/skills/``, ``.cursor/rules/``, ``.github/instructions/``,
   ``.gemini/skills/``)
2. **The cross-client path** (``.agents/skills/``) — the standard location
   that all Agent Skills-compatible clients scan automatically at session
   start

This means if you adopt a new AI coding tool later, it will discover the
DataScienceUtils skills automatically from ``.agents/skills/`` without
needing to re-run the installer.

Tools known to scan ``.agents/skills/``:

* **Claude Code** — also reads ``.claude/skills/`` (tool-specific)
* **Cursor** — also reads ``.cursor/rules/`` (tool-specific)
* **Antigravity** — defaults to ``.agents/skills/`` as its primary skill
  path (global: ``~/.gemini/antigravity/skills/``)
* **Any tool implementing the** `Agent Skills specification
  <https://agentskills.io/specification>`_

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
      [ ] Antigravity          not found

        [ Confirm ]

- Arrow keys ``↑`` / ``↓`` move the cursor between items
- ``Space`` toggles the checkbox on or off
- Navigate down to ``[ Confirm ]`` and press ``Enter`` (or press ``Enter`` while on any item to toggle it)
- Tools marked ``detected`` were found automatically; ``not found`` means the installer did not detect them but you can still select them manually
- At least one tool must be selected
- If you select **Antigravity**, skills are written directly to
  ``.agents/skills/`` (its primary path). The separate cross-client pass
  is skipped for Antigravity to avoid writing the same directory twice.

On **Windows (PowerShell)**, the same selector appears with slightly different characters:

.. code-block:: text

    Select AI tools to install skills for:

      (Up/Down: navigate, Space/Enter: toggle/confirm)

    > [✓] Claude Code          detected
      [✓] Cursor               detected
      [ ] GitHub Copilot       not found
      [ ] Gemini CLI           not found
      [ ] Antigravity          not found

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

**Step 6 — Optional dependencies checkbox**

.. code-block:: text

    Optional dependencies:

      ↑/↓ navigate · space toggle · enter confirm

    ❯ [ ] NLP (sentence-transformers)         enables SentenceEmbeddingTransformer

        [ Confirm ]

This selector lets you optionally install extra dependency groups (such as ``nlp``, which provides ``sentence-transformers``) along with the package. Skipping this installs only the core dependencies. Note: conda installations do not support this prompt and will skip it.

On **Windows (PowerShell)**, the same selector appears with slightly different characters, but identical controls.

**Step 7 — Confirmation summary**

.. code-block:: text

    Summary
    ──────────────────────────────────────
    Tools:    Claude Code, Cursor
    Scope:    project (/home/user/my-project)
    Package:  pip install
    Extras:   none
    Skills:   metrics preprocess unsupervised strings transformers xai

    Proceed with installation? (y/n) [y]:

Review every line of this summary before pressing ``Enter``. Type ``n`` and press ``Enter`` to cancel without making any changes. Press ``Enter`` (or type ``y``) to proceed. (The ``Package`` line will read "conda install" or "skip" depending on your selections).

**Step 8 — Installation progress**

.. code-block:: text

    Installing data-science-utils Python package
    Using: pip3
    ✓ data-science-utils installed

    Installing ds_utils skills
    ✓ metrics    → .claude/skills/ds-utils-metrics
    ✓ preprocess → .claude/skills/ds-utils-preprocess
    ✓ unsupervised → .claude/skills/ds-utils-unsupervised
    ✓ strings    → .claude/skills/ds-utils-strings
    ✓ transformers → .claude/skills/ds-utils-transformers
    ✓ xai        → .claude/skills/ds-utils-xai
    ✓ metrics    → .cursor/rules/ds-utils-metrics
    ✓ preprocess → .cursor/rules/ds-utils-preprocess
    ✓ unsupervised → .cursor/rules/ds-utils-unsupervised
    ✓ strings    → .cursor/rules/ds-utils-strings
    ✓ transformers → .cursor/rules/ds-utils-transformers
    ✓ xai        → .cursor/rules/ds-utils-xai

    Installing to cross-client path (.agents/skills)
    ✓ metrics    → .agents/skills/ds-utils-metrics
    ✓ preprocess → .agents/skills/ds-utils-preprocess
    ✓ unsupervised → .agents/skills/ds-utils-unsupervised
    ✓ strings    → .agents/skills/ds-utils-strings
    ✓ transformers → .agents/skills/ds-utils-transformers
    ✓ xai        → .agents/skills/ds-utils-xai

Each ``✓`` line confirms a skill file was downloaded and placed correctly.
The tool-specific paths ensure your selected tools load skills immediately.
The ``.agents/skills/`` path makes skills visible to any other
`Agent Skills <https://agentskills.io>`_-compatible tool automatically.
If any line shows ``!`` (warning) instead, see Troubleshooting.

**Step 9 — Completion message**

.. code-block:: text

    Installation complete!
    ────────────────────────────────────────
    Package:  installed
    Extras:   none
    Scope:    project
    Tools:    Claude Code, Cursor

    Skills installed:
      ds-utils-metrics
      ds-utils-preprocess
      ds-utils-unsupervised
      ds-utils-strings
      ds-utils-transformers
      ds-utils-xai

    Cross-client path: .agents/skills/  (all Agent Skills-compatible tools)

    Next steps:
    1. Open your project in your AI coding tool
    2. Skills are auto-loaded — no config needed
    3. Try: "Use ds_utils to plot a confusion matrix for my classifier"

The skills are now active. No further configuration is needed — open your project in Claude Code or Cursor and the skills will load automatically whenever you ask something related to DataScienceUtils.

**Step 10 — Verify the installation**

After the installer exits, confirm the files are present:

.. code-block:: bash

    find . -path "*/ds-utils-*/SKILL.md"

Expected output (for a project-scoped Claude Code + Cursor install):

.. code-block:: text

    ./.claude/skills/ds-utils-metrics/SKILL.md
    ./.claude/skills/ds-utils-preprocess/SKILL.md
    ./.claude/skills/ds-utils-unsupervised/SKILL.md
    ./.claude/skills/ds-utils-strings/SKILL.md
    ./.claude/skills/ds-utils-transformers/SKILL.md
    ./.claude/skills/ds-utils-xai/SKILL.md
    ./.cursor/rules/ds-utils-metrics/SKILL.md
    ./.cursor/rules/ds-utils-preprocess/SKILL.md
    ./.cursor/rules/ds-utils-unsupervised/SKILL.md
    ./.cursor/rules/ds-utils-strings/SKILL.md
    ./.cursor/rules/ds-utils-transformers/SKILL.md
    ./.cursor/rules/ds-utils-xai/SKILL.md

To also verify the cross-client path:

.. code-block:: bash

    find .agents/skills -name "SKILL.md"

Expected output:

.. code-block:: text

    .agents/skills/ds-utils-metrics/SKILL.md
    .agents/skills/ds-utils-preprocess/SKILL.md
    .agents/skills/ds-utils-unsupervised/SKILL.md
    .agents/skills/ds-utils-strings/SKILL.md
    .agents/skills/ds-utils-transformers/SKILL.md
    .agents/skills/ds-utils-xai/SKILL.md

On **Windows (PowerShell)**, verify both tool-specific and cross-client
paths:

.. code-block:: powershell

    # Tool-specific paths
    Get-ChildItem -Recurse -Filter "SKILL.md" | Where-Object { $_.DirectoryName -match "ds-utils-" }

    # Cross-client path
    Get-ChildItem -Path ".agents\skills" -Recurse -Filter "SKILL.md"

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

**Step 11 — First use**

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
     - Pre-select tools without the interactive checkbox; comma-separated values: ``claude``, ``cursor``, ``copilot``, ``gemini``, ``antigravity``
   * - ``-Tools "claude,cursor"``
     - PowerShell
     - Pre-select tools without the interactive checkbox; comma-separated values: ``claude``, ``cursor``, ``copilot``, ``gemini``, ``antigravity``
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
If you prefer not to use the automated installer, you can install skills
manually. Download the ``SKILL.md`` files from the
`skills directory on GitHub
<https://github.com/idanmoradarthas/DataScienceUtils/tree/master/skills>`_
and place them in the paths below. Installing to ``.agents/skills/`` is
recommended as a baseline — it ensures skills are discoverable by any
present or future `Agent Skills <https://agentskills.io>`_-compatible tool.

* **Cross-client (recommended)**: ``.agents/skills/ds-utils-<module>/SKILL.md``
* **Claude Code**: ``.claude/skills/ds-utils-<module>/SKILL.md``
* **Cursor**: ``.cursor/rules/ds-utils-<module>/SKILL.md``
* **GitHub Copilot**: ``.github/instructions/ds-utils-<module>/SKILL.md``
* **Gemini CLI**: ``.gemini/skills/ds-utils-<module>/SKILL.md``
* **Antigravity (project)**: ``.agents/skills/ds-utils-<module>/SKILL.md``
* **Antigravity (global)**: ``~/.gemini/antigravity/skills/ds-utils-<module>/SKILL.md``

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
   * - ds-utils-transformers
     - ds_utils.transformers
     - Multi-label binarization for pipelines, feature names, pandas output
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
