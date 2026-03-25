# DataScienceUtils - AI Skills Installer (Windows PowerShell)
#
# Installs ds_utils skills for Claude Code, Cursor, GitHub Copilot, and Gemini CLI.
# Also installs the data-science-utils Python package via pip or conda.
#
# Usage:
#   irm https://raw.githubusercontent.com/idanmoradarthas/DataScienceUtils/master/install.ps1 | iex
#
# Options:
#   -Global           Install globally (home dir) instead of current project
#   -SkillsOnly       Skip Python package installation
#   -Tools "LIST"     Comma-separated: claude,cursor,copilot,gemini
#   -Force            Force reinstall even if already installed
#

param(
    [switch]$Global,
    [switch]$SkillsOnly,
    [switch]$Force,
    [switch]$Silent,
    [string]$Tools = ""
)

$ErrorActionPreference = "Stop"

$RAW_URL = "https://raw.githubusercontent.com/idanmoradarthas/DataScienceUtils/master"
$SKILLS  = @("metrics", "preprocess", "unsupervised", "strings", "xai")
$SCOPE   = if ($Global) { "global" } else { "project" }

# ── Colors ───────────────────────────────────────────────────────
function Write-Ok($msg)   { Write-Host "  " -NoNewline; Write-Host "✓ " -ForegroundColor Green -NoNewline; Write-Host $msg }
function Write-Warn($msg) { Write-Host "  " -NoNewline; Write-Host "! " -ForegroundColor Yellow -NoNewline; Write-Host $msg }
function Write-Err($msg)  { Write-Host "  " -NoNewline; Write-Host "✗ " -ForegroundColor Red -NoNewline; Write-Host $msg; exit 1 }
function Write-Step($msg) { Write-Host ""; Write-Host $msg -ForegroundColor White }
function Write-Msg($msg)  { if (-not $Silent) { Write-Host "  $msg" } }
function Write-Dim($msg)  { if (-not $Silent) { Write-Host "  $msg" -ForegroundColor DarkGray } }

# ── Interactive checkbox selector ────────────────────────────────
function Show-Checkbox {
    param([array]$Items)   # Each item: @{Label; Value; Selected; Hint}

    $cursor = 0
    $count  = $Items.Count

    function Render-Checkboxes {
        $i = 0
        foreach ($item in $Items) {
            $check  = if ($item.Selected) { [char]0x2713 } else { " " }
            $arrow  = if ($i -eq $cursor) { "> " } else { "  " }
            $color  = if ($item.Selected) { "Green" } else { "DarkGray" }
            $acolor = if ($i -eq $cursor) { "Cyan" }  else { "DarkGray" }
            Write-Host ("  " + $arrow) -NoNewline -ForegroundColor $acolor
            Write-Host ("[" + $check + "] ") -NoNewline -ForegroundColor $color
            Write-Host ("{0,-22}" -f $item.Label) -NoNewline
            Write-Host $item.Hint -ForegroundColor $color
            $i++
        }
        Write-Host ""
        $doneColor = if ($cursor -eq $count) { "Cyan" } else { "DarkGray" }
        $doneArrow = if ($cursor -eq $count) { "> " } else { "  " }
        Write-Host ("  " + $doneArrow + "[ Confirm ]") -ForegroundColor $doneColor
    }

    Write-Host ""
    Write-Host "  " -NoNewline
    Write-Host "(Up/Down: navigate, Space/Enter: toggle/confirm)" -ForegroundColor DarkGray
    Write-Host ""

    $startRow = [Console]::CursorTop
    Render-Checkboxes

    while ($true) {
        [Console]::SetCursorPosition(0, $startRow)
        Render-Checkboxes

        $key = [Console]::ReadKey($true)

        switch ($key.Key) {
            "UpArrow"   { if ($cursor -gt 0) { $cursor-- } }
            "DownArrow" { if ($cursor -lt $count) { $cursor++ } }
            "Spacebar" {
                if ($cursor -lt $count) {
                    $Items[$cursor].Selected = -not $Items[$cursor].Selected
                } else {
                    break
                }
            }
            "Enter" {
                if ($cursor -lt $count) {
                    $Items[$cursor].Selected = -not $Items[$cursor].Selected
                } else {
                    [Console]::SetCursorPosition(0, $startRow)
                    Render-Checkboxes
                    break
                }
            }
        }
        if ($key.Key -eq "Enter" -and $cursor -eq $count) { break }
    }

    Write-Host ""
    return ($Items | Where-Object { $_.Selected } | ForEach-Object { $_.Value })
}

# ── Interactive radio selector ────────────────────────────────────
function Show-Radio {
    param([array]$Items)   # Each item: @{Label; Value; Selected; Hint}

    $cursor   = 0
    $selected = ($Items | Select-Object -ExpandProperty Selected).IndexOf($true)
    if ($selected -lt 0) { $selected = 0 }
    $count = $Items.Count

    function Render-Radio {
        $i = 0
        foreach ($item in $Items) {
            $dot    = if ($i -eq $selected) { [char]0x25CF } else { [char]0x25CB }
            $arrow  = if ($i -eq $cursor) { "> " } else { "  " }
            $color  = if ($i -eq $selected) { "Green" } else { "DarkGray" }
            $acolor = if ($i -eq $cursor) { "Cyan" }  else { "DarkGray" }
            Write-Host ("  " + $arrow) -NoNewline -ForegroundColor $acolor
            Write-Host ($dot + " ") -NoNewline -ForegroundColor $color
            Write-Host ("{0,-28}" -f $item.Label) -NoNewline
            Write-Host $item.Hint -ForegroundColor $color
            $i++
        }
        Write-Host ""
        $doneColor = if ($cursor -eq $count) { "Cyan" } else { "DarkGray" }
        $doneArrow = if ($cursor -eq $count) { "> " } else { "  " }
        Write-Host ("  " + $doneArrow + "[ Confirm ]") -ForegroundColor $doneColor
    }

    Write-Host ""
    Write-Host "  (Up/Down: navigate, Enter: select)" -ForegroundColor DarkGray
    Write-Host ""

    $startRow = [Console]::CursorTop
    Render-Radio

    while ($true) {
        [Console]::SetCursorPosition(0, $startRow)
        Render-Radio

        $key = [Console]::ReadKey($true)

        switch ($key.Key) {
            "UpArrow"   { if ($cursor -gt 0) { $cursor-- } }
            "DownArrow" { if ($cursor -lt $count) { $cursor++ } }
            "Enter" {
                if ($cursor -lt $count) { $selected = $cursor }
                [Console]::SetCursorPosition(0, $startRow)
                Render-Radio
                break
            }
            "Spacebar" { if ($cursor -lt $count) { $selected = $cursor } }
        }
        if ($key.Key -eq "Enter") { break }
    }

    Write-Host ""
    return $Items[$selected].Value
}

# ── Tool detection ────────────────────────────────────────────────
function Get-Tools {
    if ($Tools -ne "") {
        return ($Tools -split ",") | ForEach-Object { $_.Trim() }
    }

    $detected = @()
    $items = @()

    $hasClaude  = $null -ne (Get-Command claude  -ErrorAction SilentlyContinue)
    $hasCursor  = $null -ne (Get-Command cursor  -ErrorAction SilentlyContinue) -or (Test-Path "$env:LOCALAPPDATA\Programs\cursor\Cursor.exe")
    $hasCopilot = $null -ne (Get-Command code    -ErrorAction SilentlyContinue) -or (Test-Path "$env:LOCALAPPDATA\Programs\Microsoft VS Code\Code.exe")
    $hasGemini  = $null -ne (Get-Command gemini  -ErrorAction SilentlyContinue)

    $items += @{ Label = "Claude Code";    Value = "claude";  Selected = $hasClaude;  Hint = if ($hasClaude)  { "detected" } else { "not found" } }
    $items += @{ Label = "Cursor";         Value = "cursor";  Selected = $hasCursor;  Hint = if ($hasCursor)  { "detected" } else { "not found" } }
    $items += @{ Label = "GitHub Copilot"; Value = "copilot"; Selected = $hasCopilot; Hint = if ($hasCopilot) { "detected" } else { "not found" } }
    $items += @{ Label = "Gemini CLI";     Value = "gemini";  Selected = $hasGemini;  Hint = if ($hasGemini)  { "detected" } else { "not found" } }

    # Default to Claude if nothing found
    if (-not ($hasClaude -or $hasCursor -or $hasCopilot -or $hasGemini)) {
        $items[0].Selected = $true
        $items[0].Hint = "default"
    }

    Write-Host ""
    Write-Host "  Select AI tools to install skills for:" -ForegroundColor White
    return Show-Checkbox -Items $items
}

# ── Scope selection ───────────────────────────────────────────────
function Get-Scope {
    if ($Global) { return "global" }
    if ($Silent) { return "project" }

    Write-Host ""
    Write-Host "  Install scope:" -ForegroundColor White

    $items = @(
        @{ Label = "Project (current directory)"; Value = "project"; Selected = $true;  Hint = "skills in .claude\skills, .cursor\rules, etc." },
        @{ Label = "Global (home directory)";     Value = "global";  Selected = $false; Hint = "skills available across all projects" }
    )
    return Show-Radio -Items $items
}

# ── Package manager detection & install ──────────────────────────
function Install-Package {
    if ($SkillsOnly) { return }

    Write-Step "Installing data-science-utils Python package"

    $hasCondaEnv = $env:CONDA_DEFAULT_ENV -ne $null
    $hasConda    = $null -ne (Get-Command conda -ErrorAction SilentlyContinue)
    $hasPip      = $null -ne (Get-Command pip   -ErrorAction SilentlyContinue)
    $hasPip3     = $null -ne (Get-Command pip3  -ErrorAction SilentlyContinue)

    if (-not ($hasConda -or $hasPip -or $hasPip3)) {
        Write-Err "No Python package manager found. Install pip or conda first."
    }

    $defaultPkg = if ($hasCondaEnv -and $hasConda) { "conda" } else { "pip" }

    $pkgManager = $defaultPkg
    if (-not $Silent) {
        Write-Host ""
        Write-Host "  Install data-science-utils using:" -ForegroundColor White

        $items = @(
            @{ Label = "pip (PyPI)";             Value = "pip";   Selected = ($defaultPkg -eq "pip");   Hint = if ($defaultPkg -eq "pip")   { "recommended" } else { "available" } },
            @{ Label = "conda (idanmorad channel)"; Value = "conda"; Selected = ($defaultPkg -eq "conda"); Hint = if ($defaultPkg -eq "conda") { "active env: $env:CONDA_DEFAULT_ENV" } else { "available" } }
        )
        $pkgManager = Show-Radio -Items $items
    }

    Write-Msg "Using: $pkgManager"

    if ($pkgManager -eq "conda") {
        conda install -y -c idanmorad data-science-utils
    } else {
        $pipCmd = if ($hasPip3) { "pip3" } else { "pip" }
        & $pipCmd install -U data-science-utils
    }

    Write-Ok "data-science-utils installed"
}

# ── Skills installation ───────────────────────────────────────────
function Install-Skills {
    param([array]$SelectedTools, [string]$InstallScope)

    Write-Step "Installing ds_utils skills"

    $baseDir = if ($InstallScope -eq "global") { $HOME } else { (Get-Location).Path }

    foreach ($tool in $SelectedTools) {
        $skillsDir = switch ($tool) {
            "claude"  { Join-Path $baseDir ".claude\skills" }
            "cursor"  { Join-Path $baseDir ".cursor\rules" }
            "copilot" { Join-Path $baseDir ".github\instructions" }
            "gemini"  { Join-Path $baseDir ".gemini\skills" }
            default   { $null }
        }

        if (-not $skillsDir) { continue }
        New-Item -ItemType Directory -Force -Path $skillsDir | Out-Null

        foreach ($skill in $SKILLS) {
            $dest = Join-Path $skillsDir "ds-utils-$skill"

            if ((Test-Path $dest) -and -not $Force) {
                Write-Ok "$skill skill already present for $tool (use -Force to overwrite)"
                continue
            }

            New-Item -ItemType Directory -Force -Path $dest | Out-Null
            $url     = "$RAW_URL/skills/$skill/SKILL.md"
            $outFile = Join-Path $dest "SKILL.md"

            try {
                Invoke-WebRequest -Uri $url -OutFile $outFile -UseBasicParsing -ErrorAction Stop
                $relPath = $dest.Replace($HOME, "~")
                Write-Ok "$skill → $relPath"
            } catch {
                Write-Warn "Could not fetch $skill skill"
                Remove-Item $dest -Recurse -Force -ErrorAction SilentlyContinue
            }
        }
    }
}

# ── Main ─────────────────────────────────────────────────────────
Write-Host ""
Write-Host "DataScienceUtils AI Skills Installer" -ForegroundColor White
Write-Host "────────────────────────────────────────"

Write-Step "Detecting AI coding tools"
$selectedTools = Get-Tools
Write-Ok "Selected: $($selectedTools -join ', ')"

Write-Step "Install scope"
$SCOPE = Get-Scope
Write-Ok "Scope: $SCOPE"

# Confirm
if (-not $Silent) {
    $baseDir = if ($SCOPE -eq "global") { "~" } else { (Get-Location).Path }
    Write-Host ""
    Write-Host "  Summary" -ForegroundColor White
    Write-Host "  ──────────────────────────────────────"
    Write-Host ("  Tools:    " + ($selectedTools -join ", ")) -ForegroundColor Green
    Write-Host "  Scope:    $SCOPE ($baseDir)" -ForegroundColor Green
    Write-Host ("  Package:  " + (if ($SkillsOnly) { "skip" } else { "pip/conda install" })) -ForegroundColor Green
    Write-Host "  Skills:   $($SKILLS -join ', ')" -ForegroundColor Green
    Write-Host ""
    $confirm = Read-Host "  Proceed with installation? (y/n) [y]"
    if ($confirm -ne "" -and $confirm -notin @("y","Y","yes")) {
        Write-Msg "Installation cancelled."
        exit 0
    }
}

Install-Package
Install-Skills -SelectedTools $selectedTools -InstallScope $SCOPE

Write-Host ""
Write-Host "Installation complete!" -ForegroundColor Green
Write-Host "────────────────────────────────────────"
Write-Msg "Package:  $(if ($SkillsOnly) { 'skipped' } else { 'installed' })"
Write-Msg "Scope:    $SCOPE"
Write-Msg "Tools:    $($selectedTools -join ', ')"
Write-Host ""
Write-Msg "Skills installed:"
foreach ($s in $SKILLS) { Write-Host "    ds-utils-$s" -ForegroundColor Cyan }
Write-Host ""
Write-Msg "Next steps:"
Write-Msg "1. Open your project in your AI coding tool"
Write-Msg "2. Skills are auto-loaded — no config needed"
Write-Msg "3. Try: `"Use ds_utils to plot a confusion matrix for my classifier`""
Write-Host ""