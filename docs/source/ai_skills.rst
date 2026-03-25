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
offers a project vs. global scope radio selector, auto-detects pip or conda, and finally deploys
the skill files to the correct locations such as ``.claude/skills/``, ``.cursor/rules/``,
``.github/instructions/``, and ``.gemini/skills/``.

Step-by-Step Installation Guide
-------------------------------
1. **Open your terminal**: Navigate to your project directory.
2. **Run the install command**:

   *Mac / Linux:*

   .. code-block:: bash

       bash <(curl -sL https://raw.githubusercontent.com/idanmoradarthas/DataScienceUtils/master/install.sh)

   *Windows (PowerShell):*

   .. code-block:: powershell

       irm https://raw.githubusercontent.com/idanmoradarthas/DataScienceUtils/master/install.ps1 | iex

3. **Tool detection**: The installer scans your environment for Claude Code, Cursor, VS Code (GitHub Copilot), and Gemini CLI.
4. **Checkbox selection**: You will see a list of detected tools. Use the arrow keys up and down to navigate, press the Spacebar to toggle a selection, and press Enter to confirm your choices.
5. **Scope selection**: You will be prompted to choose between a "project" installation (local to current directory) or a "global" installation (for all projects). Use arrow keys to select a radio button.
6. **Package manager selection**: The installer automatically detects if you are using pip or conda. You can confirm or change it via the radio buttons.
7. **Confirmation summary**: The installer displays a summary of the actions it is about to perform.
8. **Deployment**: Skills are securely downloaded from the repository and placed in the appropriate tool directories.
9. **Verification**: Verify the installation by checking your project structure for files like ``.cursor/rules/ds-utils-metrics/SKILL.md``.
10. **First use**: Try out your newly installed skills. Test it by asking your AI assistant: "Use the ds-utils-metrics skill to plot a confusion matrix."

Manual Installation
-------------------
If you prefer not to use the automated installer, you can install skills manually. Download the `SKILL.md` files from the GitHub repository and place them in the following paths depending on your tool:

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

Troubleshooting
---------------
* **Skills not triggering**: The description field guides the AI on when to trigger the skill. Try an explicit trigger by asking your AI assistant to "use the ds-utils-metrics skill".
* **Wrong directory**: Ensure the skills are in the correct dot-directory corresponding to your tool (e.g., ``.cursor/rules/`` for Cursor).
* **Package not installed**: The AI tools expect the package to be present in your environment. Run ``pip install data-science-utils`` manually if it is missing.
