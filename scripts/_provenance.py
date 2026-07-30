"""Shared provenance guard for experiment harnesses that spawn `maxim` sub-sims.

WHY THIS EXISTS (2026-07-28, Exp 42b)
-------------------------------------
A 40-sub-sim behavioural re-validation was invalidated because the sub-sims
imported a DIFFERENT checkout than the one under test. Three things conspired,
and each of them is silent on its own:

1. `maxim` is a console script; it resolves `import maxim` purely through
   `sys.path`. A venv can carry stale editable `.pth` files pointing at other
   checkouts (or deleted worktrees) from an old `pip install -e`.
2. `PYTHONPATH=src` beats those `.pth` entries — but it is RELATIVE, so it
   silently resolves to nothing unless the launch cwd is the repo root.
3. The shell that launched the run used
   `source .venv/bin/activate && export PYTHONPATH=src`; the `source` failed,
   `&&` short-circuited, and the export never ran.

Nothing errored. Every sub-sim "succeeded". And the `git_hash` recorded in the
run records came from the *harness file's* directory, so the JSONL looked
authoritative while describing code that was never executed. The mistake only
surfaced days later via an unrelated missing-artifact symptom — by which point
the run could not be proven either way.

THE RULE
--------
`git_hash` answers "where does the harness live?". That is NOT the question.
The question is "which code did the sub-sims execute?" — and a harness that
cannot answer it produces results that mean nothing, whether or not they
happen to be correct.

So: every harness that spawns `maxim` MUST call :func:`assert_repo_interpreter`
before its first sub-sim, and SHOULD record :func:`executed_code_provenance`
into each run record so the artifact is self-auditing forever after.

This module is deliberately stdlib-only and does NOT import `maxim` — it is
imported by path from the harness's own directory, so it is guaranteed to come
from the same tree as the harness that calls it.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

__all__ = ["ProvenanceError", "assert_repo_interpreter", "executed_code_provenance", "resolved_maxim_file"]


class ProvenanceError(RuntimeError):
    """The interpreter would import a `maxim` outside the harness's repo."""


def _shebang_interpreter(binary: str) -> str:
    """The interpreter the console script itself runs under.

    Probing with ``sys.executable`` would test the HARNESS's interpreter, which
    need not be the one `maxim` uses — that gap is exactly where a mismatch hides.
    """
    try:
        first = Path(binary).read_text().splitlines()[0]
    except (OSError, IndexError):
        return sys.executable
    return first.lstrip("#!").strip() or sys.executable


def resolved_maxim_file(binary: str, *, timeout: float = 60.0) -> str | None:
    """Return `maxim.__file__` as the sub-sims would resolve it, or None."""
    interp = _shebang_interpreter(binary)
    try:
        probe = subprocess.run(
            [interp, "-c", "import maxim,sys; sys.stdout.write(maxim.__file__)"],
            env=os.environ.copy(),  # same env the sub-sims inherit
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except Exception:
        return None
    out = probe.stdout.strip()
    return out if probe.returncode == 0 and out else None


def _git_hash(cwd: Path) -> str:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=15,
        )
        return r.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def assert_repo_interpreter(repo_root: Path | str, binary: str, *, exempt: bool = False) -> str | None:
    """Raise :class:`ProvenanceError` unless `maxim` resolves inside ``repo_root``.

    ``exempt`` is for mock/dry runs that never spawn a sub-sim. Returns the
    resolved ``maxim.__file__`` on success (None when exempt).
    """
    if exempt:
        return None
    root = Path(repo_root).resolve()
    src = (root / "src").resolve()
    resolved = resolved_maxim_file(binary)
    if resolved is None:
        raise ProvenanceError(
            f"cannot import `maxim` with the interpreter behind {binary}.\n"
            f"  Activate the right venv, then: export PYTHONPATH={src}"
        )
    imported_root = Path(resolved).resolve().parent.parent
    if imported_root == src:
        return resolved
    raise ProvenanceError(
        "the `maxim` package the sub-sims would import is NOT this repo's src.\n"
        f"  harness repo src : {src}\n"
        f"  imported maxim   : {imported_root}\n"
        "  → the run would measure the WRONG CODE and its results would be meaningless.\n"
        f"  Fix: export PYTHONPATH={src}\n"
        "       (ABSOLUTE — a relative 'src' silently resolves to nothing off the repo root;\n"
        "        and put it on its own line, never `source ... && export ...`, because a\n"
        "        failing `source` short-circuits the export without erroring.)\n"
        "  Also check your venv's site-packages for stale `__editable__*.pth` files left by\n"
        "  an old `pip install -e` from another/deleted checkout. Best cure: give each\n"
        "  worktree its own venv + editable install so PYTHONPATH is never load-bearing."
    )


def executed_code_provenance(repo_root: Path | str, binary: str) -> dict[str, str]:
    """Provenance describing the CODE THAT RAN, for embedding in run records.

    ``harness_git_hash`` is where the harness file lives; ``executed_git_hash``
    is the tree the sub-sims actually import. When they disagree, the run is
    suspect — record both so the artifact can be audited long after the shell
    history is gone.
    """
    root = Path(repo_root).resolve()
    resolved = resolved_maxim_file(binary)
    executed_root = Path(resolved).resolve().parent.parent.parent if resolved else None
    return {
        "harness_repo": str(root),
        "harness_git_hash": _git_hash(root),
        "executed_maxim_file": resolved or "unresolved",
        "executed_git_hash": _git_hash(executed_root) if executed_root else "unknown",
        "pythonpath": os.environ.get("PYTHONPATH", ""),
    }