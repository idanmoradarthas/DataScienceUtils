#!/bin/bash
#
# DataScienceUtils - AI Skills Installer
#
# Installs ds_utils skills for Claude Code, Cursor, GitHub Copilot, Gemini CLI, and Antigravity.
# Also installs the data-science-utils Python package via pip or conda.
#
# Usage:
#   bash <(curl -sL https://raw.githubusercontent.com/idanmoradarthas/DataScienceUtils/master/install.sh)
#
# Options:
#   -g, --global          Install globally (home dir) instead of current project
#   --skills-only         Skip Python package installation
#   --tools LIST          Comma-separated: claude,cursor,copilot,gemini,antigravity
#   -f, --force           Force reinstall even if already installed
#   -h, --help            Show this help
#

set -e

# ── Config ───────────────────────────────────────────────────────
REPO="idanmoradarthas/DataScienceUtils"
RAW_URL="https://raw.githubusercontent.com/${REPO}/master"
SKILLS="metrics preprocess unsupervised strings transformers xai"

# ── Colors ───────────────────────────────────────────────────────
G='\033[0;32m'   # green
Y='\033[1;33m'   # yellow
R='\033[0;31m'   # red
BL='\033[0;34m'  # blue
B='\033[1m'      # bold
D='\033[2m'      # dim
N='\033[0m'      # reset

# ── Defaults ─────────────────────────────────────────────────────
SCOPE="project"
INSTALL_PKG=true
FORCE=false
SILENT=false
USER_TOOLS=""
TOOLS=""
PKG_MANAGER_OVERRIDE=""

# ── Output helpers ───────────────────────────────────────────────
msg()  { [ "$SILENT" = false ] && echo -e "  $*"; }
ok()   { [ "$SILENT" = false ] && echo -e "  ${G}✓${N} $*"; }
warn() { [ "$SILENT" = false ] && echo -e "  ${Y}!${N} $*"; }
die()  { echo -e "  ${R}✗${N} $*" >&2; exit 1; }
step() { [ "$SILENT" = false ] && echo -e "\n${B}$*${N}"; }

# ── Argument parsing ─────────────────────────────────────────────
while [ $# -gt 0 ]; do
    case $1 in
        -g|--global)      SCOPE="global"; shift ;;
        --skills-only)    INSTALL_PKG=false; shift ;;
        --from-source)    PKG_MANAGER_OVERRIDE="source"; shift ;;
        --tools)          USER_TOOLS="$2"; shift 2 ;;
        -f|--force)       FORCE=true; shift ;;
        --silent)         SILENT=true; shift ;;
        -h|--help)
            echo ""
            echo "DataScienceUtils AI Skills Installer"
            echo ""
            echo "Usage: bash <(curl -sL ${RAW_URL}/install.sh) [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -g, --global       Install globally (~/) instead of current project dir"
            echo "  --skills-only      Skip Python package install (skills only)"
            echo "  --from-source      Install from git clone instead of PyPI/conda"
            echo "  --tools LIST       Comma-separated list: claude,cursor,copilot,gemini,antigravity"
            echo "  -f, --force        Force reinstall"
            echo "  -h, --help         Show this help"
            echo ""
            exit 0 ;;
        *) die "Unknown option: $1 (use -h for help)" ;;
    esac
done

# ── Interactive helpers (work even when piped via curl | bash) ───

prompt() {
    local text=$1 default=$2 result=""
    [ "$SILENT" = true ] && echo "$default" && return
    if [ -e /dev/tty ]; then
        printf "  %b [%s]: " "$text" "$default" > /dev/tty
        read -r result < /dev/tty
    fi
    [ -z "$result" ] && echo "$default" || echo "$result"
}

# Interactive multi-select checkboxes
# Args: "Label|value|on_or_off|hint" ...
checkbox_select() {
    local -a labels=() values=() states=() hints=()
    local count=0

    for item in "$@"; do
        IFS='|' read -r label value state hint <<< "$item"
        labels+=("$label"); values+=("$value"); hints+=("$hint")
        [ "$state" = "on" ] && states+=(1) || states+=(0)
        count=$((count + 1))
    done

    local cursor=0
    local total_rows=$((count + 2))

    _draw() {
        for i in $(seq 0 $((count - 1))); do
            local check=" "
            [ "${states[$i]}" = "1" ] && check="\033[0;32m✓\033[0m"
            local arrow="  "
            [ "$i" = "$cursor" ] && arrow="\033[0;34m❯\033[0m "
            local hint_style="\033[2m"
            [ "${states[$i]}" = "1" ] && hint_style="\033[0;32m"
            printf "\033[2K  %b[%b] %-22s %b%s\033[0m\n" "$arrow" "$check" "${labels[$i]}" "$hint_style" "${hints[$i]}" > /dev/tty
        done
        printf "\033[2K\n" > /dev/tty
        if [ "$cursor" = "$count" ]; then
            printf "\033[2K  \033[0;34m❯\033[0m \033[1;32m[ Confirm ]\033[0m\n" > /dev/tty
        else
            printf "\033[2K    \033[2m[ Confirm ]\033[0m\n" > /dev/tty
        fi
    }

    printf "\n  \033[2m↑/↓ navigate · space toggle · enter confirm\033[0m\n\n" > /dev/tty
    printf "\033[?25l" > /dev/tty
    trap 'printf "\033[?25h" > /dev/tty 2>/dev/null' EXIT
    _draw

    while true; do
        printf "\033[%dA" "$total_rows" > /dev/tty
        _draw

        local key=""
        IFS= read -rsn1 key < /dev/tty 2>/dev/null

        if [ "$key" = $'\x1b' ]; then
            local s1="" s2=""
            read -rsn1 s1 < /dev/tty 2>/dev/null
            read -rsn1 s2 < /dev/tty 2>/dev/null
            if [ "$s1" = "[" ]; then
                case "$s2" in
                    A) [ "$cursor" -gt 0 ] && cursor=$((cursor - 1)) ;;
                    B) [ "$cursor" -lt "$count" ] && cursor=$((cursor + 1)) ;;
                esac
            fi
        elif [ "$key" = " " ] || [ "$key" = "" ]; then
            if [ "$cursor" -lt "$count" ]; then
                [ "${states[$cursor]}" = "1" ] && states[$cursor]=0 || states[$cursor]=1
            else
                printf "\033[%dA" "$total_rows" > /dev/tty
                _draw
                break
            fi
        fi
    done

    printf "\033[?25h" > /dev/tty
    trap - EXIT

    local selected=""
    for i in $(seq 0 $((count - 1))); do
        [ "${states[$i]}" = "1" ] && selected="${selected:+$selected }${values[$i]}"
    done
    echo "$selected"
}

# Interactive single-select radio buttons
# Args: "Label|value|on_or_off|hint" ...
radio_select() {
    local -a labels=() values=() hints=()
    local count=0 selected=0

    for item in "$@"; do
        IFS='|' read -r label value state hint <<< "$item"
        labels+=("$label"); values+=("$value"); hints+=("$hint")
        [ "$state" = "on" ] && selected=$count
        count=$((count + 1))
    done

    local cursor=0
    local total_rows=$((count + 2))

    _radio_draw() {
        for i in $(seq 0 $((count - 1))); do
            local dot="○" dot_color="\033[2m"
            [ "$i" = "$selected" ] && dot="●" && dot_color="\033[0;32m"
            local arrow="  "
            [ "$i" = "$cursor" ] && arrow="\033[0;34m❯\033[0m "
            local hint_style="\033[2m"
            [ "$i" = "$selected" ] && hint_style="\033[0;32m"
            printf "\033[2K  %b%b%b %-22s %b%s\033[0m\n" "$arrow" "$dot_color" "$dot" "${labels[$i]}" "$hint_style" "${hints[$i]}" > /dev/tty
        done
        printf "\033[2K\n" > /dev/tty
        if [ "$cursor" = "$count" ]; then
            printf "\033[2K  \033[0;34m❯\033[0m \033[1;32m[ Confirm ]\033[0m\n" > /dev/tty
        else
            printf "\033[2K    \033[2m[ Confirm ]\033[0m\n" > /dev/tty
        fi
    }

    printf "\n  \033[2m↑/↓ navigate · enter select\033[0m\n\n" > /dev/tty
    printf "\033[?25l" > /dev/tty
    trap 'printf "\033[?25h" > /dev/tty 2>/dev/null' EXIT
    _radio_draw

    while true; do
        printf "\033[%dA" "$total_rows" > /dev/tty
        _radio_draw

        local key=""
        IFS= read -rsn1 key < /dev/tty 2>/dev/null

        if [ "$key" = $'\x1b' ]; then
            local s1="" s2=""
            read -rsn1 s1 < /dev/tty 2>/dev/null
            read -rsn1 s2 < /dev/tty 2>/dev/null
            if [ "$s1" = "[" ]; then
                case "$s2" in
                    A) [ "$cursor" -gt 0 ] && cursor=$((cursor - 1)) ;;
                    B) [ "$cursor" -lt "$count" ] && cursor=$((cursor + 1)) ;;
                esac
            fi
        elif [ "$key" = "" ]; then
            [ "$cursor" -lt "$count" ] && selected=$cursor
            printf "\033[%dA" "$total_rows" > /dev/tty
            _radio_draw
            break
        elif [ "$key" = " " ]; then
            [ "$cursor" -lt "$count" ] && selected=$cursor
        fi
    done

    printf "\033[?25h" > /dev/tty
    trap - EXIT
    echo "${values[$selected]}"
}

# ── Tool detection ────────────────────────────────────────────────
detect_tools() {
    if [ -n "$USER_TOOLS" ]; then
        TOOLS=$(echo "$USER_TOOLS" | tr ',' ' ')
        return
    fi

    local has_claude=false has_cursor=false has_copilot=false has_gemini=false has_antigravity=false
    local claude_hint="not found" cursor_hint="not found" copilot_hint="not found" gemini_hint="not found" antigravity_hint="not found"
    local claude_state="off"  cursor_state="off" copilot_state="off" gemini_state="off" antigravity_state="off"

    command -v claude >/dev/null 2>&1          && has_claude=true  && claude_hint="detected"  && claude_state="on"
    { command -v cursor >/dev/null 2>&1 || [ -d "/Applications/Cursor.app" ]; } \
                                               && has_cursor=true  && cursor_hint="detected"  && cursor_state="on"
    { command -v code >/dev/null 2>&1 || [ -d "/Applications/Visual Studio Code.app" ]; } \
                                               && has_copilot=true && copilot_hint="detected" && copilot_state="on"
    command -v gemini >/dev/null 2>&1          && has_gemini=true  && gemini_hint="detected"  && gemini_state="on"
    command -v antigravity >/dev/null 2>&1 \
        && has_antigravity=true \
        && antigravity_hint="detected" \
        && antigravity_state="on"

    # If nothing found, default to claude
    if [ "$has_claude" = false ] && [ "$has_cursor" = false ] && \
       [ "$has_copilot" = false ] && [ "$has_gemini" = false ] && \
       [ "$has_antigravity" = false ]; then
        claude_state="on"; claude_hint="default"
    fi

    if [ "$SILENT" = false ] && [ -e /dev/tty ]; then
        echo ""
        echo -e "  ${B}Select AI tools to install skills for:${N}"
        TOOLS=$(checkbox_select \
            "Claude Code|claude|${claude_state}|${claude_hint}" \
            "Cursor|cursor|${cursor_state}|${cursor_hint}" \
            "GitHub Copilot|copilot|${copilot_state}|${copilot_hint}" \
            "Gemini CLI|gemini|${gemini_state}|${gemini_hint}" \
            "Antigravity|antigravity|${antigravity_state}|${antigravity_hint}" \
        )
    else
        local tools=""
        [ "$has_claude" = true ]  && tools="claude"
        [ "$has_cursor" = true ]  && tools="${tools:+$tools }cursor"
        [ "$has_copilot" = true ] && tools="${tools:+$tools }copilot"
        [ "$has_gemini" = true ]  && tools="${tools:+$tools }gemini"
        [ "$has_antigravity" = true ] && tools="${tools:+$tools }antigravity"
        [ -z "$tools" ] && tools="claude"
        TOOLS="$tools"
    fi

    [ -z "$TOOLS" ] && warn "No tools selected, defaulting to Claude Code" && TOOLS="claude"
}

# ── Scope selection ───────────────────────────────────────────────
select_scope() {
    [ "$SCOPE" != "project" ] && return   # already set via --global flag
    [ "$SILENT" = true ] || [ ! -e /dev/tty ] && return

    echo ""
    echo -e "  ${B}Install scope:${N}"
    SCOPE=$(radio_select \
        "Project (current directory)|project|on|skills go into .claude/skills, .cursor/rules, etc." \
        "Global (home directory)|global|off|skills available across all projects" \
    )
}

# ── Package manager selection ──────────────────────────────────────
select_package() {
    [ "$INSTALL_PKG" = false ] && return

    # Honour CLI override — skip interactive selector
    if [ -n "$PKG_MANAGER_OVERRIDE" ]; then
        PKG_MANAGER="$PKG_MANAGER_OVERRIDE"
        return
    fi

    # Detect active conda env first
    if [ -n "$CONDA_DEFAULT_ENV" ] && command -v conda >/dev/null 2>&1; then
        PKG_MANAGER="conda"
    elif command -v conda >/dev/null 2>&1 && conda info --envs 2>/dev/null | grep -q "^\*"; then
        PKG_MANAGER="conda"
    elif command -v pip3 >/dev/null 2>&1; then
        PKG_MANAGER="pip3"
    elif command -v pip >/dev/null 2>&1; then
        PKG_MANAGER="pip"
    else
        die "No Python package manager found. Install pip or conda first."
    fi

    if [ "$SILENT" = false ] && [ -e /dev/tty ]; then
        local detected_hint="detected"
        local other_hint="available"
        local pip_state="off" conda_state="off"

        local source_state="off"
        local source_hint="clone repo and pip install ."
        if [ -f "pyproject.toml" ] && grep -q "data-science-utils" "pyproject.toml" 2>/dev/null; then
            source_state="on"
            source_hint="detected: running inside repo"
            pip_state="off"
            conda_state="off"
        elif [ "$PKG_MANAGER" = "conda" ]; then
            conda_state="on"
            detected_hint="active conda env: ${CONDA_DEFAULT_ENV:-detected}"
        else
            pip_state="on"
        fi

        echo ""
        echo -e "  ${B}Install data-science-utils using:${N}"

        local chosen
        chosen=$(radio_select \
            "pip (PyPI)|pip|${pip_state}|${detected_hint}" \
            "conda (idanmorad channel)|conda|${conda_state}|${other_hint}" \
            "Install from source (git clone)|source|${source_state}|${source_hint}" \
            "Skip (install skills only)|skip|off|do not install the python package" \
        )
        if [ "$chosen" = "skip" ]; then
            INSTALL_PKG=false
        else
            PKG_MANAGER="$chosen"
        fi
    fi
}

# ── Package manager install ────────────────────────────────────────
install_package() {
    [ "$INSTALL_PKG" = false ] && return

    step "Installing data-science-utils Python package"
    local pkg_manager="$PKG_MANAGER"

    msg "Using: ${B}${pkg_manager}${N}"

    if [ "$pkg_manager" = "conda" ]; then
        conda install -y -c idanmorad data-science-utils 2>/dev/null \
            || die "conda install failed. Try: conda install -c idanmorad data-science-utils"
    elif [ "$pkg_manager" = "source" ]; then
        # Clone the repo to a temp dir and install from source
        local tmp_dir
        tmp_dir=$(mktemp -d)
        msg "Cloning DataScienceUtils into ${tmp_dir}..."
        git clone -q --depth 1 "https://github.com/idanmoradarthas/DataScienceUtils.git" "$tmp_dir" \
            || die "git clone failed. Check your internet connection."
        local pip_cmd="pip3"
        command -v pip3 >/dev/null 2>&1 || pip_cmd="pip"
        $pip_cmd install -q "$tmp_dir" \
            || die "pip install from source failed. Try manually: git clone ... && pip install ."
        rm -rf "$tmp_dir"
    else
        local pip_cmd="$pkg_manager"
        $pip_cmd install -U data-science-utils \
            || die "pip install failed. Try: pip install data-science-utils"
    fi

    ok "data-science-utils installed"
}

# ── Skills installation ───────────────────────────────────────────
install_skills() {
    step "Installing ds_utils skills"

    local base_dir
    [ "$SCOPE" = "global" ] && base_dir="$HOME" || base_dir="$(pwd)"

    for tool in $TOOLS; do
        local skills_dir=""
        case $tool in
            claude)  skills_dir="$base_dir/.claude/skills" ;;
            cursor)  skills_dir="$base_dir/.cursor/rules" ;;
            copilot) skills_dir="$base_dir/.github/instructions" ;;
            gemini)  skills_dir="$base_dir/.gemini/skills" ;;
            antigravity)
                if [ "$SCOPE" = "global" ]; then
                    skills_dir="$HOME/.gemini/antigravity/skills"
                else
                    skills_dir="$base_dir/.agents/skills"
                fi ;;
        esac

        [ -z "$skills_dir" ] && continue
        mkdir -p "$skills_dir"

        for skill in $SKILLS; do
            local dest="$skills_dir/ds-utils-${skill}"
            if [ -d "$dest" ] && [ "$FORCE" = false ]; then
                ok "${skill} skill already present for ${tool} (use --force to overwrite)"
                continue
            fi
            mkdir -p "$dest"
            local url="${RAW_URL}/skills/${skill}/SKILL.md"
            if curl -fsSL "$url" -o "$dest/SKILL.md" 2>/dev/null; then
                ok "${skill} → ${dest#$HOME/}"
            else
                warn "Could not fetch ${skill} skill (URL: ${url})"
                rmdir "$dest" 2>/dev/null || true
            fi
        done
    done

    # ── Cross-client: Agent Skills standard path (.agents/skills/) ────
    # Skip if antigravity was selected — it already installs to .agents/skills/
    if ! echo "$TOOLS" | grep -qw "antigravity"; then
        step "Installing to cross-client path (.agents/skills/)"
        local agents_dir
        [ "$SCOPE" = "global" ] && agents_dir="$HOME/.agents/skills" \
                                 || agents_dir="$base_dir/.agents/skills"
        mkdir -p "$agents_dir"

        for skill in $SKILLS; do
            local dest="$agents_dir/ds-utils-${skill}"
            if [ -d "$dest" ] && [ "$FORCE" = false ]; then
                ok "${skill} already present in .agents/skills (use --force to overwrite)"
                continue
            fi
            mkdir -p "$dest"
            local url="${RAW_URL}/skills/${skill}/SKILL.md"
            if curl -fsSL "$url" -o "$dest/SKILL.md" 2>/dev/null; then
                ok "${skill} → ${dest#$HOME/}"
            else
                warn "Could not fetch ${skill} skill"
                rmdir "$dest" 2>/dev/null || true
            fi
        done
    fi
}

# ── Summary ───────────────────────────────────────────────────────
summary() {
    [ "$SILENT" = true ] && return
    local base_dir
    [ "$SCOPE" = "global" ] && base_dir="~" || base_dir="."

    echo ""
    echo -e "${G}${B}Installation complete!${N}"
    echo "────────────────────────────────────────"
    msg "Package:  data-science-utils $([ "$INSTALL_PKG" = true ] && echo "installed" || echo "skipped")"
    msg "Scope:    ${SCOPE} (${base_dir})"
    msg "Tools:    $(echo "$TOOLS" | tr ' ' ', ')"
    echo ""
    msg "${B}Skills installed:${N}"
    for skill in $SKILLS; do
        msg "  ${BL}ds-utils-${skill}${N}"
    done
    msg ""
    msg "${B}Cross-client path:${N}"
    local agents_base
    [ "$SCOPE" = "global" ] && agents_base="~/.agents/skills" \
                            || agents_base=".agents/skills"
    msg "  ${BL}${agents_base}/${N} ${D}(readable by all Agent Skills-compatible tools)${N}"
    echo ""
    msg "${B}Next steps:${N}"
    msg "1. Open your project in your AI coding tool"
    msg "2. The skills will be auto-loaded — no configuration needed"
    msg "3. Try: ${D}\"Use ds_utils to plot a confusion matrix for my classifier\"${N}"
    echo ""
}

# ── Main ──────────────────────────────────────────────────────────
main() {
    if [ "$SILENT" = false ]; then
        echo ""
        echo -e "${B}DataScienceUtils AI Skills Installer${N}"
        echo "────────────────────────────────────────"
    fi

    # Detect & select tools
    step "Detecting AI coding tools"
    detect_tools
    ok "Selected: $(echo "$TOOLS" | tr ' ' ', ')"

    # Select scope
    step "Install scope"
    select_scope
    ok "Scope: ${SCOPE}"

    # Select package manager
    step "Package manager"
    select_package
    ok "Package: $([ "$INSTALL_PKG" = true ] && echo "$PKG_MANAGER" || echo "skip")"

    # Confirm
    if [ "$SILENT" = false ] && [ -e /dev/tty ]; then
        local base_dir
        [ "$SCOPE" = "global" ] && base_dir="~" || base_dir="$(pwd)"
        echo ""
        echo -e "  ${B}Summary${N}"
        echo -e "  ──────────────────────────────────────"
        echo -e "  Tools:    ${G}$(echo "$TOOLS" | tr ' ' ', ')${N}"
        echo -e "  Scope:    ${G}${SCOPE} (${base_dir})${N}"
        echo -e "  Package:  ${G}$([ "$INSTALL_PKG" = true ] && echo "pip/conda install" || echo "skip")${N}"
        echo -e "  Skills:   ${G}${SKILLS}${N}"
        echo ""
        local confirm
        confirm=$(prompt "Proceed with installation? (y/n)" "y")
        if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ] && [ "$confirm" != "yes" ]; then
            msg "Installation cancelled."
            exit 0
        fi
    fi

    # Install package
    install_package

    # Install skills
    install_skills

    # Done
    summary
}

main "$@"