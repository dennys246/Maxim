#!/usr/bin/env python3
"""Lint NEW tests that touch per-agent state for the multi-agent-modes marker.

Per CLAUDE.md L43 (P4 multi-agent rule), any per-agent runtime stash MUST
be a ``dict[agent_id, value]`` from day one. This lint turns the rule
into a forcing function on every NEW test that touches per-agent state:
the test author must declare either

    @pytest.mark.multi_agent_modes      # opts into the 3-mode parametrization
    @pytest.mark.single_agent_only      # explicit opt-out, this test is genuinely 1-agent

on the test function (or its enclosing class).

Scope:

- Lint scans ONLY NEW tests added on the current branch relative to
  ``origin/main`` (or the merge-base if origin/main is unavailable).
  Existing tests are grandfathered; migration is done in a separate
  sweep, not as a side effect of unrelated PRs.
- Lint targets test files matching the patterns named in the plan:
  ``tests/unit/test_nac*.py``, ``tests/unit/test_hippocampus*.py``,
  ``tests/integration/test_memory_hub.py``. Plus any test file with
  a body that references one of the per-agent stash fields
  (``_temporal_anchors``, ``_reward_bias``, ``_event_outcome_welford``,
  ``_cluster_reward_bias``, ``_goal_reward_bias``, ``_percept_valences``).
- The lint is **heuristic on AST** — it finds test function definitions
  newly added in the diff and checks for the marker decorator on the
  function or its class. False positives can be silenced by adding the
  explicit ``single_agent_only`` marker.

Exits:
  0 — no violations (no new test touches per-agent state without the marker).
  1 — one or more violations; details printed to stderr.
  2 — git command failed or repo state malformed.

See ``docs/plans/archive/structural_invariant_tests.md`` Stage 3 +
``tests/multi_agent_fixtures.py``.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Per-agent stash field names — heuristic indicator that a test touches
# the P4 surface.
PER_AGENT_FIELDS = {
    "_temporal_anchors",
    "_reward_bias",
    "_event_outcome_welford",
    "_cluster_reward_bias",
    "_goal_reward_bias",
    "_percept_valences",
}

# Targeted file patterns from the plan (Stage 3 scope).
TARGET_FILE_GLOBS = (
    "tests/unit/test_nac*.py",
    "tests/unit/test_hippocampus*.py",
    "tests/integration/test_memory_hub.py",
)

# Accepted markers.
ACCEPTED_MARKERS = {"multi_agent_modes", "single_agent_only"}


def _run_git(args: list[str]) -> str:
    """Run a git command and return stdout. Empty string on failure."""
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    if out.returncode != 0:
        return ""
    return out.stdout


def _resolve_base_ref() -> str | None:
    """Pick a base ref to diff against — origin/main, then main, then None."""
    for ref in ("origin/main", "main"):
        if _run_git(["rev-parse", "--verify", "--quiet", ref]).strip():
            mb = _run_git(["merge-base", ref, "HEAD"]).strip()
            if mb:
                return mb
    return None


def _changed_test_files(base_ref: str) -> list[Path]:
    """Return test files that changed between base_ref and HEAD."""
    out = _run_git(["diff", "--name-only", "--diff-filter=AM", f"{base_ref}...HEAD"])
    files: list[Path] = []
    for line in out.splitlines():
        line = line.strip()
        if not line or not line.startswith("tests/") or not line.endswith(".py"):
            continue
        path = REPO_ROOT / line
        if path.exists():
            files.append(path)
    return files


def _changed_line_numbers(base_ref: str, rel_path: str) -> set[int]:
    """Return line numbers in `rel_path` that were ADDED in the diff vs base_ref."""
    out = _run_git(["diff", "--unified=0", f"{base_ref}...HEAD", "--", rel_path])
    added: set[int] = set()
    current_new_start = 0
    for line in out.splitlines():
        if line.startswith("@@"):
            # @@ -old,oldcount +new,newcount @@
            parts = line.split()
            for tok in parts:
                if tok.startswith("+") and not tok.startswith("+++"):
                    body = tok[1:]
                    if "," in body:
                        start_s, count_s = body.split(",", 1)
                    else:
                        start_s, count_s = body, "1"
                    try:
                        current_new_start = int(start_s)
                        count = int(count_s)
                    except ValueError:
                        continue
                    for n in range(current_new_start, current_new_start + count):
                        added.add(n)
    return added


def _matches_target_glob(path: Path) -> bool:
    rel = path.relative_to(REPO_ROOT).as_posix()
    from fnmatch import fnmatch

    return any(fnmatch(rel, glob) for glob in TARGET_FILE_GLOBS)


def _file_touches_per_agent_state(path: Path) -> bool:
    """Heuristic: file content references any per-agent stash field."""
    try:
        text = path.read_text()
    except OSError:
        return False
    return any(field in text for field in PER_AGENT_FIELDS)


def _has_accepted_marker(node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> bool:
    for dec in node.decorator_list:
        # @pytest.mark.<name>
        # ast.Attribute(value=ast.Attribute(value=ast.Name("pytest"), attr="mark"), attr=<name>)
        target = dec.func if isinstance(dec, ast.Call) else dec
        if (
            isinstance(target, ast.Attribute)
            and target.attr in ACCEPTED_MARKERS
            and isinstance(target.value, ast.Attribute)
            and target.value.attr == "mark"
            and isinstance(target.value.value, ast.Name)
            and target.value.value.id == "pytest"
        ):
            return True
    return False


def _function_touches_per_agent_state(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Walk fn body and look for per-agent field references."""
    for node in ast.walk(fn):
        if isinstance(node, ast.Attribute) and node.attr in PER_AGENT_FIELDS:
            return True
        if isinstance(node, ast.Name) and node.id in PER_AGENT_FIELDS:
            return True
    return False


def _lint_file(
    path: Path,
    added_lines: set[int],
    only_changed_functions: bool,
) -> list[tuple[int, str, str]]:
    """Return list of (lineno, function_name, reason) violations in the file."""
    try:
        tree = ast.parse(path.read_text(), filename=str(path))
    except (OSError, SyntaxError):
        return []

    violations: list[tuple[int, str, str]] = []

    def _check_fn(fn: ast.FunctionDef | ast.AsyncFunctionDef, cls_decorated: bool) -> None:
        # Only flag functions that are newly added in the diff (heuristic:
        # the `def` line falls inside the added-line set).
        if only_changed_functions and fn.lineno not in added_lines:
            return
        if not fn.name.startswith("test_"):
            return
        if not _function_touches_per_agent_state(fn):
            return
        if cls_decorated or _has_accepted_marker(fn):
            return
        violations.append(
            (
                fn.lineno,
                fn.name,
                "touches per-agent stash field but declares neither "
                "@pytest.mark.multi_agent_modes nor @pytest.mark.single_agent_only",
            )
        )

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            _check_fn(node, cls_decorated=False)
        elif isinstance(node, ast.ClassDef):
            cls_decorated = _has_accepted_marker(node)
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    _check_fn(sub, cls_decorated=cls_decorated)

    return violations


def main() -> int:
    base_ref = _resolve_base_ref()
    if base_ref is None:
        # No diff base — a fresh clone without origin/main, or an out-of-tree run.
        # On a PULL REQUEST this lint IS the gate, so skipping is a vacuous guard:
        # verified 2026-08-29 (CI run 33259722155) that this branch had been taken on
        # every PR since the lint shipped, because the lint job checked out at depth 1.
        sys.path.insert(0, str(REPO_ROOT / "scripts"))
        from _lint_git import must_not_skip

        if must_not_skip("no origin/main or main to diff against"):
            return 2
        print("INFO: no base ref (origin/main) available; skipping multi-agent marker lint")
        return 0

    changed = _changed_test_files(base_ref)
    if not changed:
        print("PASS: no changed test files touch per-agent state on this branch.")
        return 0

    all_violations: list[tuple[Path, int, str, str]] = []

    for path in changed:
        # Two scoping rules combine OR-style:
        # 1. The file matches the named-target glob (test_nac*, test_hippocampus*, test_memory_hub).
        # 2. The file body references per-agent stash fields.
        # If neither hits, skip the file entirely.
        if not (_matches_target_glob(path) or _file_touches_per_agent_state(path)):
            continue
        rel = path.relative_to(REPO_ROOT).as_posix()
        added_lines = _changed_line_numbers(base_ref, rel)
        violations = _lint_file(path, added_lines, only_changed_functions=True)
        for line, name, reason in violations:
            all_violations.append((path, line, name, reason))

    if not all_violations:
        print("PASS: all NEW per-agent-state tests on this branch declare the required marker.")
        return 0

    print(
        f"FAIL: {len(all_violations)} new test(s) touch per-agent state without the required marker.",
        file=sys.stderr,
    )
    print("", file=sys.stderr)
    for path, line, name, reason in all_violations:
        rel = path.relative_to(REPO_ROOT).as_posix()
        print(f"  {rel}:{line}: {name}", file=sys.stderr)
        print(f"    {reason}", file=sys.stderr)
        print("", file=sys.stderr)
    print(
        "Per CLAUDE.md L43 (P4 multi-agent rule), declare ONE of the two markers:\n"
        "  @pytest.mark.multi_agent_modes      # opts into the 3-mode parametrization\n"
        "  @pytest.mark.single_agent_only      # explicit opt-out\n"
        "See tests/multi_agent_fixtures.py for the fixture + the tripwire warning,\n"
        "and docs/plans/archive/structural_invariant_tests.md Stage 3 for rationale.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
