"""Build the release-notes context pack: the facts about a release range, extracted by code instead of by a model.

Why these sources, in this order (decided while writing the 1.10.0 release notes by hand):

1. The net public API change between the two refs is the most reliable source. It is computed from the syntax tree of
   the package at both refs, so it sees through refactors: a function that moved to another module is one "moved"
   entry, not a removal plus an addition, and a parameter that was inserted or renamed shows up as such.
2. Whether an old import path still resolves is checked against the head tree. This is what turns a "refactor" into a
   breaking change for notebooks and scripts.
3. Existing public functions whose code changed are listed with a trimmed source diff, because bug fixes and behavior
   changes hide there and are rarely named correctly in commit messages.
4. Support matrix and dependency changes come from ``pyproject.toml`` (Python versions, minimum dependency versions,
   optional extras, build backend); CI changes from the workflow files; new top-level paths from ``git diff``.
5. Pull requests come last, oldest first, pre-classified (user facing / maintainer / noise) from the files they touch.
   Commit bodies carry intent and wording, but they describe intermediate states: they are cleaned, capped, and
   checked against the head tree, and a claim such as "added X to module A" is flagged when X lives in module B.

Everything here is standard library only and read-only. The pack is a prompt-ready text (``context_pack.md``) and a
machine-readable ``facts.json`` that later steps use to verify the model's answer (every name it mentions must exist).
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import copy
from dataclasses import asdict, dataclass, field
import difflib
import json
from pathlib import Path
import re
import subprocess
import sys
import tomllib

CHARS_PER_TOKEN = 3.5
DEFAULT_BUDGET_TOKENS = 16000
USER_CATEGORIES = {"library", "skills"}
LOCK_FILES = {"uv.lock", "poetry.lock", "Pipfile.lock", "package-lock.json", "pnpm-lock.yaml", "yarn.lock"}
PACKAGING_NAMES = {"pyproject.toml", "setup.py", "setup.cfg", "MANIFEST.in", "meta.yaml", ".dockerignore"}
PR_SUFFIX = re.compile(r"\s*\(#(\d+)\)")
TAG = re.compile(r"^v?(\d+)\.(\d+)\.(\d+)$")
DEPENDENCY_BUMP = re.compile(r"^(?:\w+(?:\([^)]*\))?:\s*)?bump\s+\S+\s+from\s+\S+\s+to\s+\S+", re.IGNORECASE)
VERSION_TITLE = re.compile(r"\b(version|rc\d+|bump(?:ed)?)\b", re.IGNORECASE)
TRAILER = re.compile(r"^\s*(co-authored-by|signed-off-by|claude-session|reviewed-by)\b|generated with", re.IGNORECASE)
VERSION_LINE = re.compile(
    r"\b\d+\.\d+\.\d+(?:rc|a|b)\d+\b|\b(?:update[ds]?|bump(?:ed|s)?)\b[^.\n]*\bversion\b", re.IGNORECASE
)
RULE_LINE = re.compile(r"^\s*-{5,}\s*$")
ELLIPSIS = "..."
HOUSEKEEPING_BLOCK = re.compile(
    r"^\*\s*(update|upgrade|bump)\w*\b.*\b(dependenc\w*|versioning|version)\b", re.IGNORECASE
)
MODULE_WORD = re.compile(r"`?([A-Za-z_][A-Za-z0-9_]*)`?\s+(?:module|package)\b")
NOT_A_MODULE = {"the", "a", "an", "this", "that", "new", "python", "each", "every", "same", "main", "test", "tests"}
WORKFLOW_JOB = re.compile(r"^ {2}([A-Za-z0-9_-]+):\s*$")
WORKFLOW_JOB_NAME = re.compile(r"^ {4}name:\s*(.+?)\s*$")
WORKFLOW_NAME = re.compile(r"^name:\s*(.+?)\s*$")
MATRIX_LINE = re.compile(r"^\s*(os|python-version):\s*(\[.*\])\s*$")


def git(repo: Path, *args: str, check: bool = True) -> str:
    """Run a read-only git command and return its standard output."""
    result = subprocess.run(
        ["git", "-C", str(repo), "-c", "core.quotepath=off", *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if check and result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout


def git_show(repo: Path, ref: str, path: str) -> str | None:
    """Return the content of ``path`` at ``ref``, or None when it does not exist there."""
    result = subprocess.run(
        ["git", "-C", str(repo), "show", f"{ref}:{path}"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    return result.stdout if result.returncode == 0 else None


def latest_release_tag(repo: Path, exclude: str | None = None) -> str | None:
    """Return the newest ``X.Y.Z`` or ``vX.Y.Z`` tag (numeric order), skipping ``exclude`` when given."""
    best: tuple[tuple[int, int, int], str] | None = None
    for tag in git(repo, "tag", "--list").split():
        match = TAG.match(tag)
        if not match or (exclude and tag.lstrip("v") == exclude.lstrip("v")):
            continue
        key = tuple(int(part) for part in match.groups())
        if best is None or key > best[0]:
            best = (key, tag)
    return best[1] if best else None


# ----------------------------------------------------------------------------------------------------------------
# Public API extraction
# ----------------------------------------------------------------------------------------------------------------


KIND_NAMES = {
    "posonly": "positional-only",
    "pos": "positional-or-keyword",
    "var": "*args",
    "kwonly": "keyword-only",
    "varkw": "**kwargs",
}


@dataclass
class Param:
    """One parameter of a public callable."""

    name: str
    kind: str  # posonly | pos | var | kwonly | varkw
    annotation: str | None
    default: str | None  # source text of the default value, None when the parameter is required

    def text(self) -> str:
        """Render the parameter like a signature fragment."""
        prefix = {"var": "*", "varkw": "**"}.get(self.kind, "")
        rendered = prefix + self.name + (f": {self.annotation}" if self.annotation else "")
        return rendered + (f" = {self.default}" if self.default is not None else "")


@dataclass
class Symbol:
    """A public function or class of the package."""

    qualname: str
    module: str
    name: str
    kind: str  # function | class
    params: list[Param]
    doc: str
    deprecated: bool
    methods: list[str]
    body_hash: str
    code: str  # docstring-free, normalized source (used for code diffs)
    doc_full: str

    def signature(self) -> str:
        """Render ``(a, b=1, *, c)``."""
        params = list(self.params)
        parts: list[str] = []
        for index, param in enumerate(params):
            if param.kind == "kwonly" and not any(p.kind == "var" for p in params[:index]):
                if not any(p.text() == "*" for p in parts_as_params(parts)):
                    parts.append("*")
            parts.append(param.text())
        return "(" + ", ".join(parts) + ")"


def parts_as_params(parts: list[str]) -> list[Param]:
    """Wrap rendered fragments so ``signature`` can look for an existing bare ``*``."""
    return [Param(part, "pos", None, None) for part in parts]


@dataclass
class ModuleInfo:
    """Names a module defines or imports at its top level (what ``from module import name`` can find)."""

    names: set[str] = field(default_factory=set)
    star_imports: list[str] = field(default_factory=list)
    has_getattr: bool = False
    is_package: bool = False


@dataclass
class Api:
    """The public API of a package at one ref."""

    symbols: dict[str, Symbol] = field(default_factory=dict)
    modules: dict[str, ModuleInfo] = field(default_factory=dict)
    private: dict[str, str] = field(default_factory=dict)  # private helper name -> body hash


def _unparse(node: ast.AST | None) -> str | None:
    return ast.unparse(node) if node is not None else None


def extract_params(args: ast.arguments, *, skip_first: bool = False) -> list[Param]:
    """Convert an ``ast.arguments`` node into a flat list of parameters."""
    positional = args.posonlyargs + args.args
    defaults: list[ast.expr | None] = [None] * (len(positional) - len(args.defaults)) + list(args.defaults)
    params: list[Param] = []
    for index, (arg, default) in enumerate(zip(positional, defaults, strict=True)):
        kind = "posonly" if index < len(args.posonlyargs) else "pos"
        params.append(Param(arg.arg, kind, _unparse(arg.annotation), _unparse(default)))
    if args.vararg:
        params.append(Param(args.vararg.arg, "var", _unparse(args.vararg.annotation), None))
    for arg, default in zip(args.kwonlyargs, args.kw_defaults, strict=True):
        params.append(Param(arg.arg, "kwonly", _unparse(arg.annotation), _unparse(default)))
    if args.kwarg:
        params.append(Param(args.kwarg.arg, "varkw", _unparse(args.kwarg.annotation), None))
    if skip_first and params and params[0].name in {"self", "cls"}:
        params = params[1:]
    return params


def _split_docstring(node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> tuple[str, list[ast.stmt]]:
    body = list(node.body)
    doc = ast.get_docstring(node) or ""
    if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant):
        if isinstance(body[0].value.value, str):
            body = body[1:]
    return doc, body


def _is_deprecated(node: ast.AST) -> bool:
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            function = child.func
            called = function.attr if isinstance(function, ast.Attribute) else getattr(function, "id", "")
            if called == "warn" and any(
                "DeprecationWarning" in ast.unparse(argument) or "FutureWarning" in ast.unparse(argument)
                for argument in [*child.args, *[keyword.value for keyword in child.keywords]]
            ):
                return True
        if isinstance(child, (ast.FunctionDef, ast.ClassDef)):
            for decorator in child.decorator_list:
                if "deprecated" in ast.unparse(decorator).lower():
                    return True
    return False


def _code_text(node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef, body: list[ast.stmt]) -> str:
    """Return the normalized source of ``node`` without its docstring."""
    clone = copy.copy(node)
    clone.body = body or [ast.Pass()]
    return ast.unparse(clone)


def _first_paragraph(doc: str) -> str:
    return " ".join(doc.strip().split("\n\n")[0].split())


def _module_names(tree: ast.Module, module: str, *, is_package: bool) -> ModuleInfo:
    info = ModuleInfo(is_package=is_package)
    stack: list[ast.stmt] = list(tree.body)
    base = module if is_package else module.rpartition(".")[0]
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            info.names.add(node.name)
            info.has_getattr = info.has_getattr or node.name == "__getattr__"
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            info.names.update(target.id for target in targets if isinstance(target, ast.Name))
        elif isinstance(node, ast.Import):
            info.names.update((alias.asname or alias.name.split(".")[0]) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name == "*":
                    anchor = base.split(".")
                    if node.level:
                        anchor = anchor[: len(anchor) - (node.level - 1)] if node.level > 1 else anchor
                        source = ".".join([*anchor, *([node.module] if node.module else [])])
                    else:
                        source = node.module or ""
                    info.star_imports.append(source)
                else:
                    info.names.add(alias.asname or alias.name)
        elif isinstance(node, (ast.If, ast.Try, ast.With)):
            for field_name in ("body", "orelse", "finalbody"):
                stack.extend(getattr(node, field_name, []))
            for handler in getattr(node, "handlers", []):
                stack.extend(handler.body)
    return info


def collect_api(repo: Path, ref: str, package: str) -> Api:
    """Parse every module of ``package`` at ``ref`` and return its public API and module namespaces."""
    api = Api()
    for path in git(repo, "ls-tree", "-r", "--name-only", ref, "--", package).splitlines():
        if not path.endswith(".py"):
            continue
        text = git_show(repo, ref, path)
        if text is None:
            continue
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        parts = path[:-3].split("/")
        is_package = parts[-1] == "__init__"
        module = ".".join(parts[:-1] if is_package else parts)
        api.modules[module] = _module_names(tree, module, is_package=is_package)
        private_module = any(part.startswith("_") for part in module.split(".")[1:])
        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            doc, body = _split_docstring(node)
            body_hash = "".join(ast.dump(item) for item in body)
            if node.name.startswith("_") or private_module:
                api.private.setdefault(node.name, body_hash)
                continue
            if isinstance(node, ast.ClassDef):
                init = next(
                    (item for item in body if isinstance(item, ast.FunctionDef) and item.name == "__init__"), None
                )
                params = extract_params(init.args, skip_first=True) if init else []
                methods = [
                    item.name for item in body if isinstance(item, ast.FunctionDef) and not item.name.startswith("_")
                ]
                kind = "class"
            else:
                params, methods, kind = extract_params(node.args), [], "function"
            qualname = f"{module}.{node.name}"
            api.symbols[qualname] = Symbol(
                qualname=qualname,
                module=module,
                name=node.name,
                kind=kind,
                params=params,
                doc=_first_paragraph(doc),
                deprecated=_is_deprecated(node),
                methods=methods,
                body_hash=body_hash,
                code=_code_text(node, body),
                doc_full=doc,
            )
    return api


# ----------------------------------------------------------------------------------------------------------------
# API diff
# ----------------------------------------------------------------------------------------------------------------


@dataclass
class ParamDiff:
    """Differences between two parameter lists of the same callable."""

    removed: list[str] = field(default_factory=list)
    added: list[str] = field(default_factory=list)  # rendered with default, e.g. "sparse: bool = False"
    default_changed: list[str] = field(default_factory=list)  # "name: old -> new"
    possible_renames: list[str] = field(default_factory=list)  # "old -> new"
    positional_shift: list[str] = field(default_factory=list)  # "sparse inserted before tokenizer"
    became_required: list[str] = field(default_factory=list)
    kind_changed: list[str] = field(default_factory=list)  # "sparse: positional-or-keyword -> keyword-only"

    def empty(self) -> bool:
        """Return True when nothing differs."""
        return not any(asdict(self).values())


def diff_params(old: list[Param], new: list[Param]) -> ParamDiff:
    """Compare two parameter lists: removals, additions, default and kind changes, renames and positional shifts."""
    diff = ParamDiff()
    old_by_name = {param.name: param for param in old}
    new_by_name = {param.name: param for param in new}
    removed = [param for param in old if param.name not in new_by_name]
    added = [param for param in new if param.name not in old_by_name]
    diff.removed = [param.name for param in removed]
    diff.added = [param.text() for param in added]
    for param in old:
        other = new_by_name.get(param.name)
        if other is None:
            continue
        if param.default != other.default:
            if param.default is not None and other.default is None:
                diff.became_required.append(param.name)
            else:
                diff.default_changed.append(f"{param.name}: {param.default} -> {other.default}")
        if param.kind != other.kind:  # for example a parameter that callers could pass by position became keyword-only
            diff.kind_changed.append(f"{param.name}: {KIND_NAMES[param.kind]} -> {KIND_NAMES[other.kind]}")
    old_index = {param.name: index for index, param in enumerate(old)}
    new_index = {param.name: index for index, param in enumerate(new)}
    for gone in removed:
        for fresh in added:
            if old_index[gone.name] == new_index[fresh.name] and gone.annotation == fresh.annotation:
                diff.possible_renames.append(f"{gone.name} -> {fresh.name}")
    positional = ("posonly", "pos")
    for fresh in added:
        if fresh.kind not in positional:
            continue
        later = [
            param.name
            for param in new[new_index[fresh.name] + 1 :]
            if param.kind in positional and param.name in old_by_name and old_by_name[param.name].kind in positional
        ]
        if later:
            diff.positional_shift.append(f"{fresh.name} inserted before {later[0]}")
    return diff


@dataclass
class Move:
    """A public symbol that kept its name but changed module."""

    old: str
    new: str


@dataclass
class SignatureChange:
    """A public symbol whose parameters changed (possibly while moving)."""

    old: str
    new: str
    diff: ParamDiff


@dataclass
class ImplementationChange:
    """An existing public symbol whose code or docstring changed."""

    old: str
    new: str
    code_changed: bool
    doc_changed: bool
    diff_lines: list[str]


@dataclass
class LegacyImport:
    """``from module import name`` as it worked at the base ref, and what happens at head."""

    statement: str
    outcome: str  # ok | ImportError | ModuleNotFoundError | unknown


@dataclass
class ApiDiff:
    """Net public API change between two refs."""

    moved: list[Move]
    added: list[str]
    removed: list[str]
    signature_changes: list[SignatureChange]
    implementation_changes: list[ImplementationChange]
    private_changed: list[str]
    legacy_imports: list[LegacyImport]
    deprecated_now: list[str]


def _same_params(first: Symbol, second: Symbol) -> bool:
    return [param.text() for param in first.params] == [param.text() for param in second.params]


def resolve_import(api: Api, module: str, name: str, *, depth: int = 0) -> str:
    """Say whether ``from module import name`` works against ``api`` (static check, nothing is executed)."""
    info = api.modules.get(module)
    if info is None:
        return "ModuleNotFoundError"
    if name in info.names:
        return "ok"
    if info.has_getattr:
        return "unknown"
    if depth < 2:
        for source in info.star_imports:
            if resolve_import(api, source, name, depth=depth + 1) == "ok":
                return "ok"
    return "ImportError"


DOC_PARAM_LINE = re.compile(r"^[+-]\s*:(param|raises|return|returns)\b")


def _implementation_diff(old: Symbol, new: Symbol, limit: int) -> list[str]:
    """Return the changed code lines (docstring excluded) followed by changed ``:param``/``:raises`` doc lines."""
    if not limit:
        return []
    code = difflib.unified_diff(old.code.splitlines(), new.code.splitlines(), lineterm="", n=0)
    lines = [line for line in code if line[:1] in "+-" and not line.startswith(("+++", "---"))]
    doc = difflib.unified_diff(old.doc_full.splitlines(), new.doc_full.splitlines(), lineterm="", n=0)
    doc_lines = [f"doc {line.strip()}" for line in doc if DOC_PARAM_LINE.match(line)]
    return [*lines[:limit], *doc_lines[: max(limit // 4, 2)]]


def diff_api(old: Api, new: Api, *, diff_limit: int = 60) -> ApiDiff:
    """Compute moved, added, removed and changed public symbols plus the legacy import check."""
    added = sorted(set(new.symbols) - set(old.symbols))
    removed = sorted(set(old.symbols) - set(new.symbols))
    by_name: dict[str, list[str]] = defaultdict(list)
    for qualname in added:
        by_name[new.symbols[qualname].name].append(qualname)
    moves: list[Move] = []
    taken: set[str] = set()
    for qualname in removed:
        symbol = old.symbols[qualname]
        candidates = [c for c in by_name.get(symbol.name, []) if new.symbols[c].kind == symbol.kind and c not in taken]
        exact = [c for c in candidates if _same_params(symbol, new.symbols[c])]
        pool = exact or candidates
        if len(pool) == 1:
            moves.append(Move(qualname, pool[0]))
            taken.add(pool[0])
    moved_old = {move.old for move in moves}
    pairs = [(q, q) for q in sorted(set(old.symbols) & set(new.symbols))] + [(m.old, m.new) for m in moves]

    signature_changes: list[SignatureChange] = []
    implementation: list[ImplementationChange] = []
    for old_name, new_name in pairs:
        before, after = old.symbols[old_name], new.symbols[new_name]
        params = diff_params(before.params, after.params)
        if not params.empty():
            signature_changes.append(SignatureChange(old_name, new_name, params))
        code_changed, doc_changed = before.body_hash != after.body_hash, before.doc_full != after.doc_full
        if code_changed or doc_changed:
            implementation.append(
                ImplementationChange(
                    old_name, new_name, code_changed, doc_changed, _implementation_diff(before, after, diff_limit)
                )
            )

    legacy: list[LegacyImport] = []
    for qualname in sorted({*moved_old, *(r for r in removed if r not in moved_old)}):
        symbol = old.symbols[qualname]
        outcome = resolve_import(new, symbol.module, symbol.name)
        legacy.append(LegacyImport(f"from {symbol.module} import {symbol.name}", outcome))

    private_changed = sorted(
        name for name, body in old.private.items() if name in new.private and new.private[name] != body
    )
    deprecated_now = sorted(q for q, s in new.symbols.items() if s.deprecated and not old.symbols.get(q, s).deprecated)
    return ApiDiff(
        moved=moves,
        added=[q for q in added if q not in taken],
        removed=[q for q in removed if q not in moved_old],
        signature_changes=signature_changes,
        implementation_changes=implementation,
        private_changed=private_changed,
        legacy_imports=legacy,
        deprecated_now=deprecated_now,
    )


# ----------------------------------------------------------------------------------------------------------------
# Packaging, CI and layout
# ----------------------------------------------------------------------------------------------------------------


def _requirement_name(requirement: str) -> str:
    match = re.match(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)", requirement)
    return re.sub(r"[-_.]+", "-", match.group(1)).lower() if match else requirement.strip()


def _requirement_spec(requirement: str) -> str:
    return re.sub(r"^\s*[A-Za-z0-9][A-Za-z0-9._-]*\s*(\[[^\]]*\])?\s*", r"\1", requirement).strip() or "(any)"


def _read_pyproject(repo: Path, ref: str) -> dict:
    text = git_show(repo, ref, "pyproject.toml")
    if text is None:
        return {}
    try:
        return tomllib.loads(text)
    except tomllib.TOMLDecodeError:
        return {}


def _python_versions(classifiers: list[str]) -> list[str]:
    prefix = "Programming Language :: Python :: "
    versions = [
        c[len(prefix) :] for c in classifiers if c.startswith(prefix) and re.fullmatch(r"3\.\d+", c[len(prefix) :])
    ]
    return sorted(versions, key=lambda value: int(value.split(".")[1]))


def _requirement_map(requirements: list[str]) -> dict[str, str]:
    return {_requirement_name(requirement): _requirement_spec(requirement) for requirement in requirements}


def _diff_requirements(label: str, old: dict[str, str], new: dict[str, str]) -> list[str]:
    lines = [f"{label} added: {name} {new[name]}" for name in sorted(set(new) - set(old))]
    lines += [f"{label} removed: {name} {old[name]}" for name in sorted(set(old) - set(new))]
    lines += [
        f"{label} changed: {name} {old[name]} -> {new[name]}"
        for name in sorted(old)
        if name in new and old[name] != new[name]
    ]
    return lines


def diff_packaging(repo: Path, base: str, head: str) -> list[str]:
    """Describe support-matrix and dependency changes between two refs from ``pyproject.toml``."""
    old, new = _read_pyproject(repo, base), _read_pyproject(repo, head)
    old_project, new_project = old.get("project", {}), new.get("project", {})
    lines: list[str] = []
    if old_project.get("requires-python") != new_project.get("requires-python"):
        lines.append(f"requires-python: {old_project.get('requires-python')} -> {new_project.get('requires-python')}")
    old_versions, new_versions = (
        _python_versions(old_project.get("classifiers", [])),
        _python_versions(new_project.get("classifiers", [])),
    )
    if old_versions != new_versions:
        lines.append(f"python classifiers: {', '.join(old_versions)} -> {', '.join(new_versions)}")
    old_backend = old.get("build-system", {}).get("build-backend")
    new_backend = new.get("build-system", {}).get("build-backend")
    if old_backend != new_backend:
        lines.append(f"build backend: {old_backend} -> {new_backend}")
    lines += _diff_requirements(
        "dependency",
        _requirement_map(old_project.get("dependencies", [])),
        _requirement_map(new_project.get("dependencies", [])),
    )
    old_extras, new_extras = old_project.get("optional-dependencies", {}), new_project.get("optional-dependencies", {})
    for extra in sorted(set(old_extras) | set(new_extras)):
        if extra not in old_extras:
            lines.append(f"optional extra added: {extra} = {', '.join(new_extras[extra])}")
        elif extra not in new_extras:
            lines.append(f"optional extra removed: {extra}")
        else:
            lines += _diff_requirements(
                f"extra [{extra}]", _requirement_map(old_extras[extra]), _requirement_map(new_extras[extra])
            )
    return lines


@dataclass
class Workflow:
    """A GitHub Actions workflow reduced to the names a reader sees."""

    path: str
    name: str
    jobs: dict[str, str]
    matrix: list[str]


def _parse_workflow(path: str, text: str) -> Workflow:
    name, jobs, matrix, current_job, in_jobs = Path(path).stem, {}, [], None, False
    for line in text.splitlines():
        if line.startswith("jobs:"):
            in_jobs = True
            continue
        if not in_jobs and (match := WORKFLOW_NAME.match(line)):
            name = match.group(1).strip("'\"")
        elif in_jobs and (match := WORKFLOW_JOB.match(line)):
            current_job = match.group(1)
            jobs[current_job] = current_job
        elif in_jobs and current_job and (match := WORKFLOW_JOB_NAME.match(line)):
            if jobs[current_job] == current_job:
                jobs[current_job] = match.group(1).strip("'\"")
        if match := MATRIX_LINE.match(line):
            entry = f"{match.group(1)}: {match.group(2)}"
            if entry not in matrix:
                matrix.append(entry)
    return Workflow(path, name, jobs, matrix)


def _workflows(repo: Path, ref: str) -> dict[str, Workflow]:
    found: dict[str, Workflow] = {}
    for path in git(repo, "ls-tree", "-r", "--name-only", ref, "--", ".github/workflows").splitlines():
        if path.endswith((".yml", ".yaml")):
            text = git_show(repo, ref, path)
            if text is not None:
                found[path] = _parse_workflow(path, text)
    return found


def diff_workflows(repo: Path, base: str, head: str) -> list[str]:
    """Describe added, removed and changed GitHub Actions workflows, jobs and test matrices."""
    old, new = _workflows(repo, base), _workflows(repo, head)
    lines: list[str] = []
    for path in sorted(set(new) - set(old)):
        jobs = "; ".join(f"{job_id}: {title}" for job_id, title in new[path].jobs.items())
        lines.append(f"workflow added: {new[path].name} ({path}); jobs: {jobs}")
    for path in sorted(set(old) - set(new)):
        lines.append(f"workflow removed: {old[path].name} ({path})")
    for path in sorted(set(old) & set(new)):
        before, after = old[path], new[path]
        for job_id in sorted(set(after.jobs) - set(before.jobs)):
            lines.append(f"job added in {after.name}: {job_id}: {after.jobs[job_id]}")
        for job_id in sorted(set(before.jobs) - set(after.jobs)):
            lines.append(f"job removed from {before.name}: {job_id}")
        if before.matrix != after.matrix:
            lines.append(f"matrix in {after.name}: {' / '.join(before.matrix)} -> {' / '.join(after.matrix)}")
    return lines


def new_top_level_paths(repo: Path, base: str, head: str) -> list[str]:
    """List top-level files and directories that did not exist at ``base``, with file counts."""
    # NUL-separated output: without it git puts a name with a quote sign, a tab or a newline in quotes
    existing = set(git(repo, "ls-tree", "-z", "--name-only", base).split("\0"))
    counts: Counter[str] = Counter()
    folders: set[str] = set()  # a path with a slash in it lives in a folder, as the new tree says (not the checkout)
    changes = git(repo, "diff", "-z", "--name-status", "--no-renames", base, head).split("\0")
    for status, path in zip(changes[0::2], changes[1::2]):
        top = path.split("/")[0]
        if status == "A" and top not in existing:
            counts[top] += 1
            if "/" in path:
                folders.add(top)
    return [f"{top}/ ({count} files)" if top in folders else top for top, count in sorted(counts.items())]


def previous_changelog_section(repo: Path, head: str, limit: int = 3500) -> str:
    """Return the newest section of CHANGELOG.md at ``head`` as a style example."""
    text = git_show(repo, head, "CHANGELOG.md") or ""
    sections = re.split(r"(?m)^(?=## )", text)
    section = next((s for s in sections if s.startswith("## ")), "")
    return section.strip()[:limit]


# ----------------------------------------------------------------------------------------------------------------
# Pull requests
# ----------------------------------------------------------------------------------------------------------------


def categorize(path: str, package: str) -> str:
    """Put a changed file into one coarse category used to judge who a change matters to."""
    name = path.rsplit("/", 1)[-1]
    if name in LOCK_FILES:
        return "lock"
    if path.startswith(f"{package}/"):
        return "library"
    if path.startswith("skills/") or name.startswith("install."):
        return "skills"
    if path.startswith(("tests/", "test/")):
        return "tests"
    if path.startswith(".github/"):
        return "ci"
    if path.startswith("docs/") or name in {"README.md", "README.rst"}:
        return "docs"
    if name == "CHANGELOG.md":
        return "changelog"
    if name in PACKAGING_NAMES or re.match(r"requirements.*\.txt$", name) or name.startswith(("deploy-", "Dockerfile")):
        return "packaging"
    return "other"


def _is_version_file(path: str, package: str) -> bool:
    name = path.rsplit("/", 1)[-1]
    return path == f"{package}/__init__.py" or name in {
        "test_version.py",
        "pyproject.toml",
        "meta.yaml",
        "CHANGELOG.md",
        *LOCK_FILES,
    }


def classify_change(title: str, files: list[str], package: str) -> tuple[str, str]:
    """Return ``(relevance, reason)``: relevance is ``user``, ``maintainer`` or ``noise``."""
    categories = {categorize(path, package) for path in files}
    if DEPENDENCY_BUMP.match(title):
        return "noise", "dependency bump"
    if files and categories <= {"lock"}:
        return "noise", "lockfile only"
    if files and categories <= {"tests"}:
        return "noise", "tests only"
    if files and all(_is_version_file(path, package) for path in files) and VERSION_TITLE.search(title):
        return "noise", "version bump"
    if categories & USER_CATEGORIES:
        return "user", ", ".join(sorted(categories & USER_CATEGORIES))
    return "maintainer", ", ".join(sorted(categories)) or "no files"


def clean_body(body: str, cap: int) -> str:
    """Drop trailers, version and dependency-bump blocks and repeated lines from a commit body, then cap its length."""
    seen: set[str] = set()
    kept: list[str] = []
    skipping = False
    for raw in body.splitlines():
        line = raw.rstrip()
        key = line.strip()
        if raw.startswith("* "):
            skipping = bool(HOUSEKEEPING_BLOCK.search(key))
        if skipping or not key or TRAILER.search(line) or VERSION_LINE.search(line) or RULE_LINE.match(line):
            continue
        if key in seen:
            continue
        seen.add(key)
        kept.append(line)
    text = "\n".join(kept)
    if len(text) <= cap:
        return text
    if cap <= len(ELLIPSIS):
        return text[: max(cap, 0)]
    return text[: cap - len(ELLIPSIS)].rstrip() + ELLIPSIS


def find_stale_claims(body: str, api: Api, package: str) -> list[str]:
    """Flag lines that say a public symbol lives in a module where it does not live at head (one claim per symbol)."""
    by_name: dict[str, set[str]] = defaultdict(set)
    for symbol in api.symbols.values():
        by_name[symbol.name].add(symbol.module)
    dotted = re.compile(rf"{re.escape(package)}(?:\.[A-Za-z_][A-Za-z0-9_]*)+")
    claims: dict[str, str] = {}
    for line in body.splitlines():
        names = [name for name in set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", line)) if len(by_name.get(name, ())) == 1]
        if not names:
            continue
        mentions = [m.lower() for m in dotted.findall(line)]
        mentions += [w.lower() for w in MODULE_WORD.findall(line) if w.lower() not in NOT_A_MODULE]
        for name in names:
            (home,) = by_name[name]
            for mention in mentions:
                if "." in mention:
                    consistent = home == mention or home.startswith(mention + ".")
                else:
                    consistent = any(s.startswith(mention) or mention.startswith(s) for s in home.lower().split("."))
                if not consistent and name not in claims:
                    claims[name] = f"{name}: the message says '{mention}', at head it lives in {home}"
    return [claims[name] for name in sorted(claims)]


@dataclass
class Change:
    """One merged pull request (first-parent commit) of the release range."""

    sha: str
    numbers: list[int]
    title: str
    body: str
    files: list[str]
    categories: dict[str, int]
    relevance: str
    reason: str
    stale_claims: list[str] = field(default_factory=list)


def collect_changes(repo: Path, base: str, head: str, package: str, api_head: Api) -> list[Change]:
    """List the first-parent commits of ``base..head`` oldest first, classified and checked against ``api_head``."""
    changes: list[Change] = []
    log = git(repo, "log", "--first-parent", "--reverse", "--format=%H%x1f%s%x1f%b%x1e", f"{base}..{head}")
    for record in log.split("\x1e"):
        if not record.strip():
            continue
        sha, title, body = (record.strip("\n").split("\x1f") + ["", ""])[:3]
        sha = sha.strip()
        parent = git(repo, "rev-parse", "--verify", "--quiet", f"{sha}^1", check=False).strip()
        numstat = (
            git(repo, "diff", "--numstat", "--no-renames", parent, sha)
            if parent
            else git(repo, "show", "--numstat", "--no-renames", "--format=", sha)
        )
        files = [line.split("\t")[2] for line in numstat.splitlines() if line.count("\t") >= 2]
        numbers = [int(number) for number in PR_SUFFIX.findall(title)]
        clean_title = PR_SUFFIX.sub("", title).strip()
        relevance, reason = classify_change(clean_title, files, package)
        change = Change(
            sha=sha[:7],
            numbers=numbers,
            title=clean_title,
            body=body.strip(),
            files=files,
            categories=dict(Counter(categorize(path, package) for path in files)),
            relevance=relevance,
            reason=reason,
        )
        if relevance != "noise":
            change.stale_claims = find_stale_claims(f"{clean_title}\n{body}", api_head, package)
        changes.append(change)
    return changes


# ----------------------------------------------------------------------------------------------------------------
# Analysis and rendering
# ----------------------------------------------------------------------------------------------------------------


@dataclass
class Analysis:
    """Everything extracted for one release range."""

    repo: Path
    base: str
    head: str
    base_sha: str
    head_sha: str
    package: str
    api_old: Api
    api_new: Api
    api_diff: ApiDiff
    packaging: list[str]
    workflows: list[str]
    new_paths: list[str]
    changes: list[Change]
    previous_changelog: str
    shortstat: str


def analyze(repo: Path, base: str, head: str, package: str) -> Analysis:
    """Extract every fact of the release range ``base..head``."""
    api_old, api_new = collect_api(repo, base, package), collect_api(repo, head, package)
    return Analysis(
        repo=repo,
        base=base,
        head=head,
        base_sha=git(repo, "rev-parse", "--short", base).strip(),
        head_sha=git(repo, "rev-parse", "--short", head).strip(),
        package=package,
        api_old=api_old,
        api_new=api_new,
        api_diff=diff_api(api_old, api_new),
        packaging=diff_packaging(repo, base, head),
        workflows=diff_workflows(repo, base, head),
        new_paths=new_top_level_paths(repo, base, head),
        changes=collect_changes(repo, base, head, package, api_new),
        previous_changelog=previous_changelog_section(repo, head),
        shortstat=git(repo, "diff", "--shortstat", base, head).strip(),
    )


@dataclass
class Caps:
    """Size limits applied while rendering; they shrink step by step until the pack fits the token budget."""

    body: int
    diff_lines: int
    doc: int
    maintainer_bodies: bool


CAP_STEPS = (
    Caps(body=1200, diff_lines=40, doc=600, maintainer_bodies=True),
    Caps(body=700, diff_lines=24, doc=400, maintainer_bodies=True),
    Caps(body=400, diff_lines=12, doc=250, maintainer_bodies=False),
    Caps(body=200, diff_lines=6, doc=160, maintainer_bodies=False),
    Caps(body=0, diff_lines=0, doc=100, maintainer_bodies=False),
)


def estimate_tokens(text: str) -> int:
    """Rough token count (the real count depends on the model's tokenizer)."""
    return int(len(text) / CHARS_PER_TOKEN)


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: max(limit - 3, 0)].rstrip() + "..."


def _param_diff_text(diff: ParamDiff) -> str:
    parts = []
    if diff.possible_renames:
        parts.append("possible rename: " + ", ".join(diff.possible_renames))
    removed = [
        name for name in diff.removed if not any(pair.startswith(f"{name} ->") for pair in diff.possible_renames)
    ]
    if removed:
        parts.append("removed: " + ", ".join(removed))
    renamed_to = {pair.split(" -> ")[1] for pair in diff.possible_renames}
    added = [text for text in diff.added if text.split(":")[0].split(" =")[0] not in renamed_to]
    if added:
        parts.append("added: " + ", ".join(added))
    if diff.positional_shift:
        parts.append("positional order changed: " + ", ".join(diff.positional_shift))
    if diff.kind_changed:
        parts.append("parameter kind changed: " + ", ".join(diff.kind_changed))
    if diff.default_changed:
        parts.append("default changed: " + ", ".join(diff.default_changed))
    if diff.became_required:
        parts.append("now required: " + ", ".join(diff.became_required))
    return "; ".join(parts)


def _section_range(analysis: Analysis) -> str:
    counts: Counter[str] = Counter()
    for change in analysis.changes:
        counts[change.relevance] += 1
    mix = ", ".join(f"{counts[key]} {key}" for key in ("user", "maintainer", "noise") if counts[key])
    return (
        f"base: {analysis.base} ({analysis.base_sha})\nhead: {analysis.head} ({analysis.head_sha})\n"
        f"merged pull requests: {len(analysis.changes)} ({mix})\n{analysis.shortstat}"
    )


def _section_api(analysis: Analysis, caps: Caps) -> str:
    diff, new, old = analysis.api_diff, analysis.api_new, analysis.api_old
    lines: list[str] = []
    broken = [item for item in diff.legacy_imports if item.outcome in {"ImportError", "ModuleNotFoundError"}]
    lines.append("### Old import paths that no longer work at head")
    if broken:
        grouped: dict[tuple[str, str], list[str]] = defaultdict(list)
        for item in broken:
            module, _, name = item.statement[len("from ") :].partition(" import ")
            grouped[(module, item.outcome)].append(name)
        for (module, outcome), names in sorted(grouped.items()):
            lines.append(f"- from {module} import <{len(names)} names>: {outcome} at head ({', '.join(sorted(names))})")
    else:
        lines.append("- none")
    lines.append("\n### Moved (same name, new module)")
    groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    for move in diff.moved:
        groups[(old.symbols[move.old].module, new.symbols[move.new].module)].append(old.symbols[move.old].name)
    lines += [f"- {a} -> {b}: {', '.join(sorted(names))}" for (a, b), names in sorted(groups.items())] or ["- none"]
    lines.append("\n### Added public symbols")
    for qualname in diff.added:
        symbol = new.symbols[qualname]
        methods = f" methods: {', '.join(symbol.methods[:8])}" if symbol.methods else ""
        deprecated = " [deprecated]" if symbol.deprecated else ""
        lines.append(f"- [{symbol.kind}] {qualname}{symbol.signature()}{deprecated}{methods}")
        if symbol.doc and caps.doc:
            lines.append(f"    {_clip(symbol.doc, caps.doc)}")
    if not diff.added:
        lines.append("- none")
    lines.append("\n### Removed public symbols (not moved)")
    lines += [f"- {qualname}" for qualname in diff.removed] or ["- none"]
    lines.append("\n### Signature changes of existing public symbols")
    for change in diff.signature_changes:
        where = change.new if change.old == change.new else f"{change.old} -> {change.new}"
        lines.append(f"- {where}: {_param_diff_text(change.diff)}")
    if not diff.signature_changes:
        lines.append("- none")
    lines.append("\n### Existing public symbols whose code or docstring changed")
    for item in diff.implementation_changes:
        what = "code" if item.code_changed else "docstring only"
        where = item.new if item.old == item.new else f"{item.old} -> {item.new}"
        lines.append(f"- {where} ({what})")
        lines += [f"    {_clip(text, 160)}" for text in item.diff_lines[: caps.diff_lines]]
    if not diff.implementation_changes:
        lines.append("- none")
    if diff.private_changed:
        lines.append(f"\nPrivate helpers with changed code: {', '.join(diff.private_changed[:30])}")
    if diff.deprecated_now:
        lines.append(f"Deprecated at head (warn on use): {', '.join(diff.deprecated_now)}")
    still = sorted(q for q, s in new.symbols.items() if s.deprecated and q in old.symbols and old.symbols[q].deprecated)
    if still:
        lines.append(f"Deprecated before and still shipped: {', '.join(still)}")
    return "\n".join(lines)


def _section_changes(analysis: Analysis, caps: Caps) -> str:
    lines: list[str] = []
    for change in analysis.changes:
        refs = " ".join(f"#{number}" for number in change.numbers) or change.sha
        header = f"- {refs} [{change.relevance}: {change.reason}] {change.title}"
        lines.append(header)
        show_body = change.relevance == "user" or (change.relevance == "maintainer" and caps.maintainer_bodies)
        if show_body and caps.body:
            body = clean_body(change.body, caps.body)
            lines += [f"    {text}" for text in body.splitlines()]
        lines += [f"    (!) {claim}" for claim in change.stale_claims]
    return "\n".join(lines)


def render(analysis: Analysis, caps: Caps) -> dict[str, str]:
    """Render every section of the pack (name -> text) under the given caps."""
    api_title = "PUBLIC API CHANGES (net result between base and head, computed from the code; most reliable)"
    pr_title = (
        "PULL REQUESTS, oldest first (intent and wording; may describe intermediate states, (!) = contradicts head)"
    )
    packaging = "\n".join(f"- {line}" for line in analysis.packaging) or "- no change"
    workflows = "\n".join(f"- {line}" for line in analysis.workflows) or "- no change"
    new_paths = "\n".join(f"- {line}" for line in analysis.new_paths) or "- none"
    return {
        "RANGE": _section_range(analysis),
        api_title: _section_api(analysis, caps),
        "SUPPORT MATRIX AND DEPENDENCIES (pyproject.toml)": packaging,
        "CI AND TOOLING (workflow files)": workflows,
        "NEW TOP-LEVEL PATHS": new_paths,
        pr_title: _section_changes(analysis, caps),
        "PREVIOUS CHANGELOG SECTION (style example)": analysis.previous_changelog or "(none)",
    }


def build_pack(analysis: Analysis, budget_tokens: int = DEFAULT_BUDGET_TOKENS) -> tuple[str, dict]:
    """Render the pack with the largest caps that fit ``budget_tokens``; return ``(text, info)``."""
    sections: dict[str, str] = {}
    used = CAP_STEPS[-1]
    text = ""
    for caps in CAP_STEPS:
        sections = render(analysis, caps)
        text = _assemble(analysis, sections)
        used = caps
        if estimate_tokens(text) <= budget_tokens:
            break
    info = {
        "tokens": estimate_tokens(text),
        "budget": budget_tokens,
        "fits": estimate_tokens(text) <= budget_tokens,
        "caps": asdict(used),
        "reduced": used is not CAP_STEPS[0],
        "section_tokens": {name.split(" (")[0]: estimate_tokens(body) for name, body in sections.items()},
    }
    return text, info


def _assemble(analysis: Analysis, sections: dict[str, str]) -> str:
    parts = [f'<release_context base="{analysis.base}" head="{analysis.head_sha}">']
    for index, (name, body) in enumerate(sections.items(), start=1):
        parts.append(f"\n## {index}. {name}\n{body}")
    parts.append("\n</release_context>")
    return "\n".join(parts)


def facts_for_verification(analysis: Analysis) -> dict:
    """Machine-readable facts used to verify what a model wrote (every name must exist at head)."""
    diff = analysis.api_diff
    return {
        "base": analysis.base,
        "head": analysis.head,
        "package": analysis.package,
        "public_symbols": {
            qualname: {"kind": symbol.kind, "module": symbol.module, "params": [param.name for param in symbol.params]}
            for qualname, symbol in sorted(analysis.api_new.symbols.items())
        },
        "added": diff.added,
        "removed": diff.removed,
        "moved": [asdict(move) for move in diff.moved],
        "signature_changes": [asdict(change) for change in diff.signature_changes],
        "legacy_imports": [asdict(item) for item in diff.legacy_imports],
        "implementation_changes": [
            {"old": item.old, "new": item.new, "code_changed": item.code_changed}
            for item in diff.implementation_changes
        ],
        "packaging": analysis.packaging,
        "workflows": analysis.workflows,
        "new_paths": analysis.new_paths,
        "pull_requests": [
            {
                "sha": change.sha,
                "numbers": change.numbers,
                "title": change.title,
                "relevance": change.relevance,
                "reason": change.reason,
                "categories": change.categories,
                "stale_claims": change.stale_claims,
            }
            for change in analysis.changes
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    """Create the command line parser."""
    parser = argparse.ArgumentParser(description="Extract the release-notes context pack of a git range.")
    parser.add_argument("--repo", type=Path, default=Path("."), help="repository root")
    parser.add_argument("--base", help="previous release ref (default: newest X.Y.Z tag)")
    parser.add_argument("--head", default="HEAD", help="ref of the release")
    parser.add_argument("--package", default="ds_utils", help="importable package directory to analyze")
    parser.add_argument("--budget-tokens", type=int, default=DEFAULT_BUDGET_TOKENS, help="soft size limit of the pack")
    parser.add_argument("--out-dir", type=Path, default=Path("release-context"), help="where to write the files")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Write ``context_pack.md`` and ``facts.json``; return a process exit code."""
    args = build_parser().parse_args(argv)
    repo = args.repo.resolve()
    try:
        base = args.base or latest_release_tag(repo)
        if not base:
            print("error: no release tag found, pass --base", file=sys.stderr)
            return 2
        analysis = analyze(repo, base, args.head, args.package)
    except RuntimeError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    text, info = build_pack(analysis, args.budget_tokens)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "context_pack.md").write_text(text, encoding="utf-8")
    (args.out_dir / "facts.json").write_text(json.dumps(facts_for_verification(analysis), indent=2), encoding="utf-8")
    (args.out_dir / "pack_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"context pack: ~{info['tokens']} tokens (budget {info['budget']}, fits={info['fits']}) -> {args.out_dir}")
    for name, tokens in info["section_tokens"].items():
        print(f"  {tokens:6d}  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
