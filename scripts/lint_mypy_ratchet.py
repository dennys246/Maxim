#!/usr/bin/env python3
"""Repo-wide mypy per-file error ratchet (roadmap 1.3.2 item 4, part B).

**The rule** (owner decision 2026-10-04): under ``src/maxim/`` no file's mypy error
count may rise above its count at the merge-base with ``origin/main``. A new file
starts at zero, a deleted file is free, and a renamed file keeps its base count
(renames map through ``_lint_git.changed_files``, i.e. ``git diff -M``). Suppressions
are a SEPARATE per-file counter that may not rise either, even when the error count
falls: inline ``type: ignore`` comments; ``# mypy:`` directives, matched by mypy's
own rule (any raw line starting ``# mypy: ``, which includes one inside a string
literal); ``@no_type_check`` (and an aliased import of it); and every ``if`` /
``elif`` / ``while`` / ``assert`` / conditional expression whose test mentions
``TYPE_CHECKING`` or ``MYPY`` anywhere, except the plain import guard
``if TYPE_CHECKING:`` with no ``else``/``elif`` and no early exit at its body's top
level (``return``/``raise``/``continue``/``break``, a falsy-constant ``assert``, or a
call named ``exit``/``_exit``/``abort``/``quit``) — an early exit there makes the rest
of the enclosing block unreachable to mypy, so that guard counts too.
Three things fail outright: a ``.pyi`` under ``src/maxim/`` that was not there at
base, a NEW file-level ``# mypy: ignore-errors``, and a NEW file-level
``# type: ignore`` (one before the first statement, which mypy reads as "ignore the
whole module").

**No committed baseline.** Both trees are measured in the same job, so the
environment cancels out and there is no pinned number for a PR to raise. HEAD is the
working tree; base is exported with ``git archive`` into a ``mkdtemp`` outside the
repo. Each tree gets ONE fixed invocation (``MYPY_ARGS``): the pinned mypy, an
empty ``--config-file`` (so a ``mypy.ini`` / ``[tool.mypy]`` / ``setup.cfg`` added by
the PR has no effect), ``--no-site-packages`` (so a developer machine and the
lint job resolve imports identically), a throwaway cache with ``--no-incremental``,
and ``MYPYPATH`` / ``MYPY_CACHE_DIR`` / ``PYTHONPATH`` scrubbed from the env.

**Fail-closed parsing.** An instrument that silently under-reports is a vacuous
gate, so the lint exits 2 when: mypy exits with anything but 0 or 1; the output says
"prevented further checking" (a blocking error, e.g. a syntax error, hides every
other error in the tree); any stdout line is not an ``error:`` line, a ``note:`` line
or the one summary line; the summary's error count disagrees with the parsed lines;
anything is written to stderr; or the "checked N source files" count differs from
that tree's module count under ``src/maxim/`` (its ``.py`` files, plus any stub-only
``.pyi`` — mypy counts a stub-only module and lets a ``.pyi`` REPLACE its ``.py``).

It PRINTS both trees' totals and every per-file delta on every run, so the number
lives in the CI output, not in a doc. On a non-PR CI event (a push to main) it
measures HEAD only and prints its totals. On a pull request any git failure
(no merge-base, unreadable diff) is an error via ``_lint_git.must_not_skip``;
locally it skips the comparison with an INFO line.

**Residuals, stated:**

- The lint environment has no third-party packages and ``--no-site-packages`` hides
  any that are installed, so third-party types are ``Any``. Errors that only a real
  ``numpy``/``httpx`` stub would expose are not counted.
- ``cast(Any, ...)``, ``Any`` annotations and dropping a signature's annotations are
  not counted as suppressions, and neither is any code mypy treats as unreachable
  (a literal ``False``/``0`` test, code after a ``NoReturn`` call such as
  ``sys.exit()``, platform/version guards); review reads them. That includes a
  plain ``if TYPE_CHECKING:`` whose body ends in a call to a USER-defined
  ``NoReturn`` function (only the stdlib exit names above are detected).
- The ratchet is per file against EACH PR's merge-base, not a repo total: errors
  cross files, so two PRs that each pass can together raise the total.
- The base is the merge-base, not ``main``'s tip: a branch cut before a burn-down can
  merge counts back up without failing (the property every diff-scoped lint here has).
- Catches FORGETTING, not determined evasion: the suppression counter is a lexical
  count, and over-counts rather than under-counts (any comment containing
  ``type: ignore`` counts).

**Decomposition.** Renames come from ``git diff -M`` with no copy detection, so: a
file that keeps its path stays modified and keeps its count; every module EXTRACTED
from it is added, starts at 0, and any error moved into it fails; and a file turned
into a package with no piece at least 50% similar to the original makes every piece
new. That is intended — the roadmap's "every module the decomposition creates enters
CI's mypy set" — so a slice of ``runtime/agent_loop.py`` (19 errors today) must land
its moved code clean, or leave it where it was.

Regression guard: tests/unit/test_lint_mypy_ratchet.py (drives ``main()`` on fixture
git repos; each mechanism deletion-proven) and the "mypy per-file ratchet (roadmap
1.3.2 item 4)" step in .github/workflows/test.yml (lint job).

Exits: 0 clean; 1 a file's count rose (stderr); 2 the instrument or git could not
answer (stderr).
"""

from __future__ import annotations

import ast
import io
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import tokenize
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lint_git import GitUnavailable, base_ref, changed_files, must_not_skip  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
SCOPE = "src/maxim"

# The ONE invocation. --config-file and --cache-dir are appended per run.
MYPY_ARGS: tuple[str, ...] = (
    SCOPE,
    "--ignore-missing-imports",
    "--check-untyped-defs",
    "--no-incremental",
    "--no-site-packages",
    "--show-error-codes",
    "--no-pretty",
    "--no-color-output",
)
SCRUBBED_ENV = ("MYPYPATH", "MYPY_CACHE_DIR", "PYTHONPATH", "MYPY_FORCE_COLOR", "FORCE_COLOR")

_ERROR_RE = re.compile(r"^(src/maxim/[^:]+\.pyi?):(\d+): error: .+$")
_NOTE_RE = re.compile(r"^(src/maxim/[^:]+\.pyi?):(\d+): note: .*$")
_SUMMARY_RE = re.compile(
    r"^(?:Found (\d+) errors? in (\d+) files? \(checked (\d+) source files?\)"
    r"|Success: no issues found in (\d+) source files?)$"
)
_TYPE_IGNORE_RE = re.compile(r"#\s*type\s*:\s*ignore")
# mypy 1.20.0 `mypy/util.py::get_mypy_comments`: `source.split("\n")`, then `line.startswith("# mypy: ")` —
# on RAW lines, so a directive inside a string literal is honoured too. Matched exactly, not via tokenize.
_MYPY_DIRECTIVE_PREFIX = "# mypy: "
_IGNORE_ERRORS_RE = re.compile(r"ignore[-_]errors")
_TC_NAMES = frozenset({"TYPE_CHECKING", "MYPY"})
_NTC_NAMES = frozenset({"no_type_check", "no_type_check_decorator"})


class InstrumentError(RuntimeError):
    """mypy's output could not be trusted — the lint exits 2, never passes."""


# ── suppression counter (tokenize/ast; `# mypy:` lines by mypy's raw-line rule, strings included) ──


def _is_tc(node: ast.expr) -> bool:
    """``TYPE_CHECKING`` and ``MYPY``, as mypy; NOT coverage.py's set (see ``_lint_ledger.py``)."""
    if isinstance(node, ast.Name):
        return node.id in _TC_NAMES
    return isinstance(node, ast.Attribute) and node.attr in _TC_NAMES


_EXIT_CALLS = frozenset({"exit", "_exit", "abort", "quit"})


def _exits_early(body: list[ast.stmt]) -> bool:
    """A top-level statement in ``body`` after which mypy treats the rest of the enclosing block as unreachable."""
    for st in body:
        if isinstance(st, (ast.Return, ast.Raise, ast.Continue, ast.Break)):
            return True
        if isinstance(st, ast.Assert) and isinstance(st.test, ast.Constant) and not st.test.value:
            return True
        if isinstance(st, ast.Expr) and isinstance(st.value, ast.Call) and _decorator_name(st.value) in _EXIT_CALLS:
            return True
    return False


def _mentions_tc(node: ast.expr) -> bool:
    """Any TYPE_CHECKING / MYPY name ANYWHERE in the expression (`TYPE_CHECKING or False`, `not X and ...`)."""
    return any(_is_tc(n) for n in ast.walk(node) if isinstance(n, ast.expr))


def read_source(raw: bytes) -> str:
    """Decode as mypy does (`mypy/util.py::decode_python_encoding`): a UTF-8 BOM stripped, else the PEP 263
    coding line honoured; newlines NOT translated, because the directive rule splits on "\n" only."""
    if raw.startswith(b"\xef\xbb\xbf"):
        return raw[3:].decode("utf-8", errors="replace")
    try:
        encoding, _ = tokenize.detect_encoding(io.BytesIO(raw).readline)
    except SyntaxError:
        encoding = "utf-8"
    return raw.decode(encoding, errors="replace")


def mypy_directives(text: str) -> list[str]:
    """The `# mypy:` directive lines mypy itself reads (see ``_MYPY_DIRECTIVE_PREFIX``)."""
    return [ln for ln in text.split("\n") if ln.startswith(_MYPY_DIRECTIVE_PREFIX)]


def _decorator_name(node: ast.expr) -> str:
    if isinstance(node, ast.Call):
        node = node.func
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _comments(text: str) -> list[tokenize.TokenInfo]:
    try:
        return [t for t in tokenize.generate_tokens(io.StringIO(text).readline) if t.type == tokenize.COMMENT]
    except (tokenize.TokenError, SyntaxError) as exc:
        raise InstrumentError(f"cannot tokenize: {exc}") from exc


def suppression_count(text: str) -> int:
    """Per-file count of the ways to make mypy not report: see the module docstring."""
    n = 0
    for tok in _comments(text):
        if _TYPE_IGNORE_RE.search(tok.string):
            n += 1
    n += len(mypy_directives(text))
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        raise InstrumentError(f"cannot parse: {exc}") from exc
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            n += sum(1 for d in node.decorator_list if _decorator_name(d) in _NTC_NAMES)
        elif isinstance(node, ast.ImportFrom):
            # `from typing import no_type_check as ntc` would hide the decorator's name.
            n += sum(1 for a in node.names if a.name in _NTC_NAMES and a.asname)
        elif isinstance(node, (ast.If, ast.While, ast.IfExp, ast.Assert)) and _mentions_tc(node.test):
            # Any test that mentions TYPE_CHECKING/MYPY can make mypy skip a branch (`if not TYPE_CHECKING:`,
            # `if TYPE_CHECKING or False: ... elif True: <never checked>`, `assert not TYPE_CHECKING`). The one
            # shape exempt is the idiomatic import guard: a plain `if TYPE_CHECKING:` with no else/elif, whose
            # body mypy DOES check. Over-counting is fine — both trees are counted the same way.
            if isinstance(node, ast.If) and _is_tc(node.test) and not node.orelse and not _exits_early(node.body):
                continue
            n += 1
    return n


def file_level_suppressed(text: str) -> bool:
    """True if the file carries a whole-module suppression mypy honours."""
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(text).readline))
    except (tokenize.TokenError, SyntaxError) as exc:
        raise InstrumentError(f"cannot tokenize: {exc}") from exc
    if any(_IGNORE_ERRORS_RE.search(d) for d in mypy_directives(text)):
        return True
    seen_code = False
    for tok in tokens:
        if tok.type == tokenize.COMMENT:
            if not seen_code and _TYPE_IGNORE_RE.search(tok.string):
                return True
        elif tok.type not in (tokenize.NL, tokenize.NEWLINE, tokenize.ENCODING, tokenize.ENDMARKER):
            seen_code = True
    return False


# ── one tree, measured ────────────────────────────────────────────────────────


@dataclass
class TreeMeasure:
    errors: Counter[str] = field(default_factory=Counter)
    suppressions: Counter[str] = field(default_factory=Counter)
    file_level: set[str] = field(default_factory=set)
    pyi: set[str] = field(default_factory=set)
    checked: int = 0


def run_mypy(tree: Path) -> tuple[int, str, str]:
    """The single fixed invocation, from ``tree``'s root. Returns (exit code, stdout, stderr)."""
    scratch = Path(tempfile.mkdtemp(prefix="mypy-ratchet-run-"))
    try:
        cfg = scratch / "empty.ini"
        cfg.write_text("[mypy]\n")  # a section header only; without it mypy warns on stderr
        env = {k: v for k, v in os.environ.items() if k not in SCRUBBED_ENV}
        cmd = [
            sys.executable,
            "-m",
            "mypy",
            *MYPY_ARGS,
            f"--config-file={cfg}",
            f"--cache-dir={scratch / 'cache'}",
        ]
        try:
            r = subprocess.run(cmd, cwd=tree, env=env, capture_output=True, text=True, timeout=600)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise InstrumentError(f"mypy did not run: {exc}") from exc
        return r.returncode, r.stdout, r.stderr
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def parse_mypy(rc: int, stdout: str, stderr: str, n_modules: int) -> tuple[Counter[str], int]:
    """Per-file error counts and the checked-file count, or :class:`InstrumentError`."""
    if rc not in (0, 1):
        raise InstrumentError(f"mypy exited {rc}: {(stderr or stdout).strip()[-2000:]}")
    if "prevented further checking" in stdout:
        raise InstrumentError(
            "a blocking error prevented further checking — every other error in the tree is hidden:\n"
            + "\n".join(ln for ln in stdout.splitlines() if ": error:" in ln)
        )
    if stderr.strip():
        raise InstrumentError(f"unexpected mypy stderr: {stderr.strip()[-2000:]}")
    errors: Counter[str] = Counter()
    summary: re.Match[str] | None = None
    for ln in stdout.splitlines():
        if summary is not None:
            raise InstrumentError(f"output after the summary line: {ln!r}")
        m = _ERROR_RE.match(ln)
        if m:
            errors[m.group(1)] += 1
            continue
        if _NOTE_RE.match(ln):
            continue
        s = _SUMMARY_RE.match(ln)
        if s:
            summary = s
            continue
        raise InstrumentError(f"unrecognised mypy output line: {ln!r}")
    if summary is None:
        raise InstrumentError("no mypy summary line")
    if summary.group(4) is not None:
        reported, checked = 0, int(summary.group(4))
    else:
        reported, checked = int(summary.group(1)), int(summary.group(3))
    if reported != sum(errors.values()):
        raise InstrumentError(f"summary says {reported} errors, parsed {sum(errors.values())} error lines")
    if (rc == 0) != (reported == 0):
        raise InstrumentError(f"exit code {rc} disagrees with {reported} reported errors")
    if checked != n_modules:
        raise InstrumentError(
            f"mypy checked {checked} source files but the tree has {n_modules} modules (.py/.pyi) under {SCOPE}"
        )
    return errors, checked


def measure(tree: Path) -> TreeMeasure:
    src = tree / SCOPE
    m = TreeMeasure()
    py = sorted(src.rglob("*.py"))
    # mypy FIRST: a file mypy cannot parse is a blocking error it must report itself (the
    # suppression scanner below would otherwise be the one to trip, and the fail-closed parse
    # of mypy's output would go unexercised).
    # mypy counts MODULES: a stub-only .pyi adds one, a .pyi beside its .py replaces it (the .py is then
    # never checked — which is why a new .pyi fails outright in compare()).
    modules = {p.with_suffix("") for p in py} | {p.with_suffix("") for p in src.rglob("*.pyi")}
    m.errors, m.checked = parse_mypy(*run_mypy(tree), n_modules=len(modules))
    for p in py:
        rel = p.relative_to(tree).as_posix()
        text = read_source(p.read_bytes())
        try:
            m.suppressions[rel] = suppression_count(text)
            if file_level_suppressed(text):
                m.file_level.add(rel)
        except InstrumentError as exc:
            raise InstrumentError(f"{rel}: {exc}") from exc
    m.pyi = {p.relative_to(tree).as_posix() for p in src.rglob("*.pyi")}
    return m


def export_tree(repo: Path, ref: str, dest: Path) -> None:
    """``git archive <ref> -- src/maxim`` unpacked into ``dest``."""
    try:
        r = subprocess.run(
            ["git", "archive", "--format=tar", ref, "--", SCOPE], cwd=repo, capture_output=True, timeout=120
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise GitUnavailable(f"git archive {ref}: {exc}") from exc
    if r.returncode != 0:
        raise GitUnavailable(f"git archive {ref}: {r.stderr.decode(errors='replace').strip()}")
    with tarfile.open(fileobj=io.BytesIO(r.stdout)) as tf:
        tf.extractall(dest, filter="data")


# ── comparison ────────────────────────────────────────────────────────────────


def _totals(label: str, m: TreeMeasure) -> str:
    return (
        f"mypy-ratchet: {label}: {sum(m.errors.values())} error(s) in {len(m.errors)} file(s), "
        f"{sum(m.suppressions.values())} suppression(s) in {sum(1 for v in m.suppressions.values() if v)} "
        f"file(s) (checked {m.checked} source files)"
    )


def compare(head: TreeMeasure, base: TreeMeasure, renamed_from: dict[str, str]) -> tuple[list[str], list[str]]:
    """(violations, delta lines). ``renamed_from`` maps a HEAD path to its base path for renames
    and genuinely-new files (``""``); every other HEAD path is its own base path."""
    failures: list[str] = []
    deltas: list[str] = []
    for rel in sorted(set(head.suppressions) | set(head.errors)):
        old = renamed_from.get(rel, rel)
        note = f" (renamed from {old})" if old and old != rel else (" (new file)" if not old else "")
        e_new, e_old = head.errors.get(rel, 0), base.errors.get(old, 0) if old else 0
        s_new, s_old = head.suppressions.get(rel, 0), base.suppressions.get(old, 0) if old else 0
        if e_new != e_old or s_new != s_old:
            deltas.append(f"  {rel}{note}: errors {e_old} → {e_new}, suppressions {s_old} → {s_new}")
        if e_new > e_old:
            failures.append(
                f"{rel}{note}: mypy error count rose {e_old} → {e_new} — fix the new errors"
                + (" (a new file starts at zero)" if not old else "")
            )
        if s_new > s_old:
            failures.append(
                f"{rel}{note}: suppression count rose {s_old} → {s_new} (type: ignore / # mypy: / "
                "@no_type_check / a test mentioning TYPE_CHECKING/MYPY) — a suppression costs the same as an error"
            )
        if rel in head.file_level and not (old and old in base.file_level):
            failures.append(
                f"{rel}{note}: NEW file-level suppression (# mypy: ignore-errors or a leading # type: ignore)"
            )
    for gone in sorted(set(base.errors) - set(head.errors) - set(renamed_from.values())):
        if gone not in head.suppressions:
            deltas.append(f"  {gone} (deleted): errors {base.errors[gone]} → 0")
    for rel in sorted(head.pyi - base.pyi):
        failures.append(f"{rel}: NEW .pyi stub under {SCOPE} — a stub replaces the checked source")
    return failures, deltas


def main() -> int:
    started = time.monotonic()
    event = os.environ.get("GITHUB_EVENT_NAME")
    # pull_request and push (against the last green push, _lint_git.push_base, #1089) are gates; every other
    # event is totals-only and exits 0. If the workflow ever gains a `merge_group` trigger (a merge queue), it must
    # be handled here as a gate, or the queue merges ungated.
    try:
        if event and event not in ("pull_request", "push"):
            head = measure(REPO_ROOT)
            print(_totals("HEAD", head))
            print(f"mypy-ratchet: {event} event — totals only ({time.monotonic() - started:.1f}s)")
            return 0
        try:
            base = base_ref(REPO_ROOT)
            renamed_from = {new: old for new, old in changed_files(REPO_ROOT, base, f"{SCOPE}/") if new != old}
        except GitUnavailable as e:
            if must_not_skip(str(e)):
                return 2
            head = measure(REPO_ROOT)
            print(_totals("HEAD", head))
            print(f"INFO: no base ref available; skipping the per-file comparison ({e})")
            return 0
        head = measure(REPO_ROOT)
        base_dir = Path(tempfile.mkdtemp(prefix="mypy-ratchet-base-"))
        try:
            try:
                export_tree(REPO_ROOT, base, base_dir)
            except GitUnavailable as e:
                print(f"ERROR: could not export the base tree ({e})", file=sys.stderr)
                return 2
            base_m = measure(base_dir)
        finally:
            shutil.rmtree(base_dir, ignore_errors=True)
    except InstrumentError as e:
        print(f"mypy-ratchet: ERROR — mypy's result cannot be trusted, failing closed: {e}", file=sys.stderr)
        return 2

    print(_totals("HEAD", head))
    print(_totals(f"base {base[:12]}", base_m))
    failures, deltas = compare(head, base_m, renamed_from)
    if deltas:
        print(f"mypy-ratchet: per-file deltas ({len(deltas)}):")
        print("\n".join(deltas))
    else:
        print("mypy-ratchet: no per-file deltas")
    print(f"mypy-ratchet: {time.monotonic() - started:.1f}s")
    if failures:
        print("mypy per-file ratchet FAILED:", file=sys.stderr)
        for f in failures:
            print(f"  {f}", file=sys.stderr)
        return 1
    print("mypy per-file ratchet: clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
