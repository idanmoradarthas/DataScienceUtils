================
AI Coding Skills
================

DataScienceUtils ships a set of *AI skills* — structured knowledge files that
teach AI coding assistants (Claude Code, Cursor, GitHub Copilot, Gemini CLI,
and Antigravity) how to use this library as end-to-end workflows: choosing the
right function, using correct import paths, following runnable examples, and
troubleshooting common pitfalls.

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
       powershell -ExecutionPolicy Bypass -File .\install.ps1 -SkillsOnly

   Windows PowerShell blocks downloaded scripts under the default execution policy; ``-ExecutionPolicy Bypass`` applies to that single run only without altering persistent system settings.

On **Windows (PowerShell)**, the same selector appears with slightly different characters:

.. code-block:: text

    Install data-science-utils using:

      (Up/Down: navigate, Enter: select)

    > * pip (PyPI)                           recommended
      o conda (idanmorad channel)             available
      o Install from source (git clone)       clone repo and pip install .
      o Skip (install skills only)            do not install the python package

        [ Confirm ]

**Step 6 — Optional dependencies prompt**

.. code-block:: text

    Install optional NLP extras (sentence-transformers)? (y/N) [n]

This prompt lets you optionally install extra dependency groups (such as ``nlp``, which provides ``sentence-transformers``) along with the package. Pressing ``Enter`` defaults to ``no`` and installs only the core dependencies. Note: conda installations do not support this prompt and will skip it.

On **Windows (PowerShell)**, the prompt is functionally identical.

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
    Scope:    project (/home/user/my-project)
    Tools:    Claude Code, Cursor

    Installation Outcome (12 total checks):
      Newly installed:          12
      Already present (skipped): 0
      Failed downloads:         0

    Newly installed skills:
      ✓ claude: ds-utils-metrics → .claude/skills/ds-utils-metrics
      ✓ claude: ds-utils-preprocess → .claude/skills/ds-utils-preprocess
      ✓ claude: ds-utils-unsupervised → .claude/skills/ds-utils-unsupervised
      ✓ claude: ds-utils-strings → .claude/skills/ds-utils-strings
      ✓ claude: ds-utils-transformers → .claude/skills/ds-utils-transformers
      ✓ claude: ds-utils-xai → .claude/skills/ds-utils-xai
      ✓ cross-client: ds-utils-metrics → .agents/skills/ds-utils-metrics
      ✓ cross-client: ds-utils-preprocess → .agents/skills/ds-utils-preprocess
      ✓ cross-client: ds-utils-unsupervised → .agents/skills/ds-utils-unsupervised
      ✓ cross-client: ds-utils-strings → .agents/skills/ds-utils-strings
      ✓ cross-client: ds-utils-transformers → .agents/skills/ds-utils-transformers
      ✓ cross-client: ds-utils-xai → .agents/skills/ds-utils-xai

    Cross-client path:
      .agents/skills/ (readable by all Agent Skills-compatible tools)

    Next steps:
    1. Open your project in your AI coding tool
    2. The skills will be auto-loaded — no configuration needed
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

    (repeated for each of the 6 skills × number of tools selected)

On Mac / Linux, if ``find`` returns nothing, the skill files were not placed
correctly. On Windows, if ``Get-ChildItem`` returns nothing, the same applies.
In either case, re-run the installer or follow the Manual Installation section below.

**Step 11 — First use**

Open your project in your AI coding tool — no restart needed. Then give two concrete example prompts to try:

- ``"Plot a confusion matrix for my classifier using ds_utils"``
- ``"Use ds_utils to find highly correlated features in my DataFrame"``

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
    powershell -ExecutionPolicy Bypass -File .\install.ps1 -SkillsOnly -Global -Tools "claude" -Force

Windows PowerShell blocks downloaded scripts under the default execution policy; ``-ExecutionPolicy Bypass`` applies to that single run only without altering persistent system settings.

Manual Installation
-------------------
If you prefer not to use the automated installer, you can install skills
manually. Download both the ``SKILL.md`` and ``references/api_reference.md`` files
for each skill from the `skills directory on GitHub
<https://github.com/idanmoradarthas/DataScienceUtils/tree/master/skills>`_
and place them in the paths below. Installing to ``.agents/skills/`` is
recommended as a baseline — it ensures skills are discoverable by any
present or future `Agent Skills <https://agentskills.io>`_-compatible tool.
For instructions on validating and testing skills with NVIDIA SkillEvaluator,
see the :ref:`Validating Skills with NVIDIA SkillEvaluator` section below.

* **Cross-client (recommended)**:

  * ``.agents/skills/ds-utils-<module>/SKILL.md``
  * ``.agents/skills/ds-utils-<module>/references/api_reference.md``

* **Claude Code**:

  * ``.claude/skills/ds-utils-<module>/SKILL.md``
  * ``.claude/skills/ds-utils-<module>/references/api_reference.md``

* **Cursor**:

  * ``.cursor/rules/ds-utils-<module>/SKILL.md``
  * ``.cursor/rules/ds-utils-<module>/references/api_reference.md``

* **GitHub Copilot**:

  * ``.github/instructions/ds-utils-<module>/SKILL.md``
  * ``.github/instructions/ds-utils-<module>/references/api_reference.md``

* **Gemini CLI**:

  * ``.gemini/skills/ds-utils-<module>/SKILL.md``
  * ``.gemini/skills/ds-utils-<module>/references/api_reference.md``

* **Antigravity (project)**:

  * ``.agents/skills/ds-utils-<module>/SKILL.md``
  * ``.agents/skills/ds-utils-<module>/references/api_reference.md``

* **Antigravity (global)**:

  * ``~/.gemini/antigravity/skills/ds-utils-<module>/SKILL.md``
  * ``~/.gemini/antigravity/skills/ds-utils-<module>/references/api_reference.md``

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
     - Confusion matrix, ROC and precision-recall curves with threshold annotations, REC curves, directional metrics, probability and error analysis, learning curves
   * - ds-utils-preprocess
     - ds_utils.preprocess
     - Feature visualization, correlation and dendrogram, statistics, mutual information, PCA explained variance
   * - ds-utils-unsupervised
     - ds_utils.unsupervised
     - Cluster cardinality, magnitude, optimal k, 2D cluster scatter plots (PCA/t-SNE/UMAP)
   * - ds-utils-strings
     - ds_utils.strings
     - Tag encoding, significant term extraction
   * - ds-utils-transformers
     - ds_utils.transformers
     - Multi-label binarization and sentence embeddings as scikit-learn pipeline steps (feature names, pandas output)
   * - ds-utils-xai
     - ds_utils.xai
     - Feature importance plots, Graphviz decision-tree diagrams and text rule export

Updating Skills
---------------
To update your skills to the latest version, run the installer with the ``--force`` flag:

.. code-block:: bash

    bash <(curl -sL https://raw.githubusercontent.com/idanmoradarthas/DataScienceUtils/master/install.sh) --force

On **Windows (PowerShell)**:

.. code-block:: powershell

    irm https://raw.githubusercontent.com/idanmoradarthas/DataScienceUtils/master/install.ps1 -OutFile install.ps1
    powershell -ExecutionPolicy Bypass -File .\install.ps1 -Force

Windows PowerShell blocks downloaded scripts under the default execution policy; ``-ExecutionPolicy Bypass`` applies to that single run only without altering persistent system settings.

.. _Validating Skills with NVIDIA SkillEvaluator:

Validating Skills with NVIDIA SkillEvaluator
--------------------------------------------

All skills in DataScienceUtils are validated using `NVIDIA SkillEvaluator <https://github.com/NVIDIA/SkillEvaluator>`_ for Tier 1 static validation and quality grading.

**Prerequisites and Installation**

Install the pinned versions of SkillEvaluator (a PyPI release) and SkillSpector (a git tag) matching ``requirements-skills.txt`` in Python 3.13 via ``uv``:

.. code-block:: bash

    uv tool install --python 3.13 "skillevaluator[tier2]==0.7.2"
    uv tool install git+https://github.com/NVIDIA/SkillSpector.git@v2.12.0

External security scanners provide static security coverage: Gitleaks is required for secret detection (Tier 1 reports ``INCOMPLETE`` without it, together with ``skillspector``). Bandit and Semgrep are only invoked for skills that bundle code files. The current skills are Markdown-only, so they are not needed here. If you add code files to a skill, install both, otherwise Tier 1 reports ``INCOMPLETE`` and exits 1:

.. code-block:: bash

    # macOS: Homebrew (its version may differ from the pinned CI release)
    brew install gitleaks

    # Linux x86_64: pinned v8.28.0 release, installed without sudo
    mkdir -p "$HOME/.local/bin"
    curl -sL https://github.com/gitleaks/gitleaks/releases/download/v8.28.0/gitleaks_8.28.0_linux_x64.tar.gz | tar -xz -C "$HOME/.local/bin" gitleaks

    # Only needed if a skill bundles code files
    pip install bandit semgrep

Only ``quality-check`` and Tier 1 static validation (``tier1 --no-llm``) are run; Tier 2 and Tier 3 validation stages are not executed. With both ``skillspector`` and ``gitleaks`` installed, all six skills pass static checks completely without reporting ``INCOMPLETE``. To maintain complete Tier 1 static validation, ``SKILL.md`` prose and frontmatter must avoid backticked or quoted ``name.ext`` tokens that could be misinterpreted as missing bundled files (guarded by ``test_skill_md_has_no_unresolved_path_like_tokens`` in ``tests/test_skills_docs.py``).

To upgrade SkillEvaluator, change the version in ``requirements-skills.txt`` and in the install snippets in this page and in the ``eval-skills*`` hints together, then run ``./eval-skills.sh`` and the manual Tier 2 workflow. The upstream ``skillevaluator ci init`` workflow generator is intentionally not used: Tier 1 already gates pull requests in ``test.yml`` and Tier 2 runs through the workflows described below.

**Running Static Validation and Quality Checks**

To evaluate an individual skill locally:

.. code-block:: bash

    skillevaluator quality-check skills/ds-utils-metrics
    skillevaluator tier1 --no-llm skills/ds-utils-metrics

Both commands support writing validation reports to a custom output directory via ``-o`` or ``--output-dir`` (e.g., ``skillevaluator quality-check -o ./reports skills/ds-utils-metrics``).

**Automated Evaluation Runner Scripts**

The repository provides runner scripts at the root that execute ``quality-check`` and ``tier1 --no-llm`` across all six skills, write reports to a temporary directory cleaned up on exit, and exit non-zero if any check fails or reports incomplete evidence:

On Linux / macOS:

.. code-block:: bash

    ./eval-skills.sh

On Windows (Command Prompt / PowerShell):

.. code-block:: bat

    .\eval-skills.bat

.. note::

   PowerShell requires the explicit ``.\`` prefix to run scripts in the current directory;
   ``cmd.exe`` accepts either ``eval-skills.bat`` or ``.\eval-skills.bat``.

Optional Tier 2 Deduplication Checks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In addition to Tier 1 static validation and quality grading, NVIDIA SkillEvaluator provides Tier 2 validation to detect semantic duplication:

* **Inter-skill similarity** (``similarity-check``): Compares skills against each other using vector embedding cosine similarity to detect overlapping responsibilities or descriptions.
* **Intra-skill deduplication** (``dedup-scan``): Chunks files within each skill directory, clusters semantically similar passages using embeddings (threshold 0.80), and queries an LLM to identify verbatim or near-verbatim redundancies.

Tier 2 checks require both an embedding provider and an LLM provider configured via environment variables.

**Running Tier 2 Checks Locally**

You can run the Tier 2 evaluation runner script ``eval-skills-tier2.sh`` locally against any OpenAI-compatible server (such as a local Ollama instance):

.. code-block:: bash

    uv tool install --python 3.13 --force "skillevaluator[tier2]==0.7.2"

The ``[tier2]`` extra installs the OpenAI client used by the Tier 2 checks, so no extra ``--with`` flag is needed.

.. code-block:: bash

    # Ollama must already be running (another terminal or a service) with OLLAMA_CONTEXT_LENGTH=32768, e.g.:
    # OLLAMA_CONTEXT_LENGTH=32768 ollama serve

    # Start passage-prefix proxy (listens on 127.0.0.1:11435, forwards to 127.0.0.1:11434;
    # configurable via TIER2_EMBED_PROXY_PORT, TIER2_UPSTREAM, TIER2_EMBED_PREFIX)
    python .github/tier2/embed_prefix_proxy.py &
    PROXY_PID=$!

    export SKILL_EVAL_EMBEDDING_PROVIDER=openai-compatible
    export SKILL_EVAL_EMBEDDING_BASE_URL=http://127.0.0.1:11435/v1
    export SKILL_EVAL_EMBEDDING_API_KEY=ollama
    export SKILL_EVAL_EMBEDDING_MODEL=nemotron-3-embed-1b

    export SKILL_EVAL_LLM_PROVIDER=openai-compatible
    export SKILL_EVAL_LLM_BASE_URL=http://127.0.0.1:11434/v1
    export SKILL_EVAL_LLM_API_KEY=ollama
    export SKILL_EVAL_LLM_MODEL=granite4.1:3b

    ./eval-skills-tier2.sh
    kill "$PROXY_PID"

The passage-prefix proxy is necessary because ``nemotron-3-embed-1b`` expects ``"passage: "`` on every document. While NVIDIA's hosted endpoint honours ``input_type=passage`` sent by SkillEvaluator, Ollama ignores it; without the proxy the chunks are embedded unprefixed, similarities are inflated and unrelated chunks merge into large clusters (false DUPLICATE reports). Ollama's default 4096-token context truncates large cluster prompts, hence ``OLLAMA_CONTEXT_LENGTH=32768``.

The community GGUF of ``nemotron-3-embed-1b`` does not set ``tokenizer.ggml.add_bos_token``, so llama.cpp/Ollama silently prepends a BOS token that the BF16 reference does not use. Embeddings are then consistently offset (on the Probe B gate: pairwise max abs diff 0.071, Spearman 0.982, one extra token per text). Derive a GGUF with the key set to false before ``ollama create``: ``pip install gguf==0.19.0`` and then ``python .github/tier2/patch_gguf_add_bos.py SRC.gguf DST.gguf`` (tensors are copied byte-identically). With the derived file the gate numbers are 0.003 / 0.9997 with exact token counts.

The local dedup LLM is advisory. The judge is shown only the file, heading, line range and text of each cluster member, never the function name, so sections of different functions that share a heading and structure (``### Parameters``, ``### Example``) can be reported as ``DUPLICATE``, and a verdict can differ between runs on unchanged content. Treat ``DUPLICATE`` findings as candidates for manual review. The default judge is ``granite4.1:3b``; ``phi4-mini:3.8b`` is a pinned alternate for a second opinion (pass it as the ``llm_model`` workflow input or as ``SKILL_EVAL_LLM_MODEL``). Both have their Ollama manifest digest pinned in ``.github/tier2/pins.env``; any other model tag runs unpinned and its digest is printed for provenance.

Choosing the judge
^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 20 25 15 40

   * - Judge
     - Runs with a DUPLICATE flag
     - Scan time
     - Note
   * - gemma4:e2b
     - 3 of 3
     - ~9 min
     - confidence was always 0.95
   * - gemma4:e4b
     - 1 of 2
     - ~25 min
     -
   * - ministral-3:3b
     - 1 of 1
     - ~7 min
     - flagged the ROC/PR ``### Parameters`` pair
   * - granite4.1:3b
     - 1 of 3
     - 6-13 min
     - flagged one ``### Example`` pair of sibling functions; labelled the identical-sentence pair of Probe C ``DUPLICATE``
   * - phi4-mini:3.8b
     - 0 of 3
     - 8-9 min
     - most stable verdicts; labelled the Probe C identical-sentence pair ``RELATED_BUT_DISTINCT``

Some early flags were genuine (fixed in #149 and #152); the later ones were sibling-function sections that share a heading. Counts are small and the skills contain no true duplicates, so the table measures false positives, not recall.

The script runs ``similarity-check`` across all skills, followed by ``dedup-scan`` for each individual skill directory in an isolated temporary location. Each check is mapped to ``PASS``, ``FAIL``, or ``INCOMPLETE``. If any duplicate findings are reported, details are displayed and the script exits 1.

Pull request checks
^^^^^^^^^^^^^^^^^^^

Pull requests touching skills or the Tier 2 pipeline are evaluated by the automated workflow ``Skills Tier 2 (PR)`` (``.github/workflows/skills-tier2-pr.yml``), comprising two distinct jobs:

* **Tier 2 similarity (blocking)**: Executes cross-skill embedding similarity at SkillEvaluator's default threshold (0.75). Only embeddings are used (no LLM). Any score meeting or exceeding the threshold fails the check and blocks the PR. In branch protection rules, **only this job should be marked as a required status check**.
* **Tier 2 dedup (advisory)**: Executes intra-skill LLM deduplication scanning **only on skills touched by the PR**. Findings are surfaced exclusively as GitHub warning annotations on changed lines and summarized in the GitHub Actions step summary (never posted as PR comments). Because sibling functions can share common headings and structural patterns, this job is strictly advisory and **never fails the PR** due to reported findings or incomplete scans; it must never be made a required status check.

The LLM judge used for the advisory scan is read dynamically from ``.github/tier2/pins.env`` (default ``granite4.1:3b``) and verified against its pinned sha256 manifest digest.

To reproduce these checks locally:

.. code-block:: bash

    # Run only the blocking similarity check:
    TIER2_MODE=similarity ./eval-skills-tier2.sh

    # Run the advisory dedup check on specific skills:
    TIER2_MODE=dedup TIER2_SKILLS="ds-utils-metrics,ds-utils-preprocess" ./eval-skills-tier2.sh

**Manual CI Workflow**

An optional, non-blocking CI workflow (``.github/workflows/skills-tier2.yml``) is available to execute Tier 2 validation in a GitHub Actions runner. Key characteristics:

* **Manual trigger only**: The workflow never runs on ``push`` or ``pull_request``; it is invoked manually via ``workflow_dispatch``.
* **Embedding prefix input**: ``embedding_prefix`` (default ``passage``) routes embeddings through the proxy; ``none`` bypasses it and is meant for A/B diagnostics only.
* **Embedding BOS input**: ``embedding_bos`` (default ``disabled``) derives a GGUF with ``add_bos_token=false`` before ``ollama create`` so the embeddings match the BF16 reference; ``default`` uses the community GGUF as published and is meant for A/B diagnostics only.
* **Runtime downloads**: Model weights (such as GGUF embeddings and Ollama LLM tags) are downloaded and verified dynamically at runtime; they are not bundled or redistributed within the repository.
* **Embedding verification**: The community GGUF embedding conversion (``NeoRoth/nemotron-3-embed-1b-gguf``) is verified by an automated probe comparing its pairwise cosine matrix against reference scores computed from the official upstream BF16 checkpoint (``nvidia/Nemotron-3-Embed-1B-BF16``). Probe B reports which reference tokenizer variant (plain, bos, eos, bos_eos) the Ollama vectors match best.
* **Positive control canary**: Synthetic canary fixtures with intentional duplicates are scanned to confirm that both inter-skill and intra-skill duplication detectors are fully operational before repository skills are evaluated.

Troubleshooting
---------------
* **Skills not triggering**: The description field guides the AI on when to trigger the skill. Try an explicit trigger by asking your AI assistant to "use the ds-utils-metrics skill".
* **Wrong directory**: Ensure the skills are in the correct dot-directory corresponding to your tool (e.g., ``.cursor/rules/`` for Cursor).
* **Package not installed**: The AI tools expect the package to be present in your environment. Please follow the :doc:`Installation Guide <installation>` if it is missing.
* **Installed from source and don't want pip/conda to reinstall the package**:
  Use the ``--skills-only`` flag (bash) or ``-SkillsOnly`` (PowerShell).
  This deploys only the skill files and skips the package manager step
  entirely. See :ref:`CLI Flags Reference` for details.
