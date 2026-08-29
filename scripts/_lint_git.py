"""Shared git plumbing for the diff-scoped lints (extracted 2026-08-29).

Four lints now run the same shape — resolve a base ref, diff against it, and refuse
to let a per-file count RISE — and the block had been copy-pasted four times
(``lint_multi_agent_marker.py``, ``lint_no_silent_swallows.py`` and the two ratchets
added by roadmap item 16, whose own preamble says "every piece rides an existing lint
or CI step — no new mechanism"). This module is that shared piece; ``scripts/
_provenance.py`` is the precedent for a stdlib-only helper imported by path.

The graceful-skip rule is load-bearing and is why this is shared rather than
re-derived: a shallow CI clone can lack a merge-base entirely, and a lint whose
OTHER checks found real violations must not discard them because git could not
answer (the pre-fold swallow lint returned 2 here and made every PR red with its
findings unprinted — caught by the #508 review).

**But a graceful skip in the one environment the lint exists for is a vacuous
guard.** Verified 2026-08-29 against run 33259722155: the `lint` job checks out at
depth 1 and `git fetch origin main --depth=1` leaves two disjoint shallow roots, so
`merge-base` fails and EVERY diff-scoped lint printed `INFO: no base ref … skipping`
— the multi-agent marker lint and the swallow lint's check 2 had been no-ops on
every pull request since they shipped. The workflow now checks out with
`fetch-depth: 0`, and :func:`must_not_skip` turns "no base ref" into a hard error
whenever the run IS a pull request, so this can never silently return.

Deliberately stdlib-only; does not import ``maxim``.
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

__all__ = [
    "GitUnavailable",
    "base_ref",
    "changed_files",
    "count_ratchet",
    "git",
    "must_not_skip",
    "show",
]


class GitUnavailable(RuntimeError):
    """Git could not answer — the caller SKIPS the diff-scoped check, never fails it."""


def git(repo_root: Path, *args: str, timeout: float = 60.0) -> str:
    try:
        r = subprocess.run(["git", *args], cwd=repo_root, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise GitUnavailable(f"git {' '.join(args)}: {exc}") from exc
    if r.returncode != 0:
        raise GitUnavailable(f"git {' '.join(args)}: {r.stderr.strip()}")
    return r.stdout


def base_ref(repo_root: Path) -> str:
    """The merge-base with origin/main (then main). Raises :class:`GitUnavailable`.

    Note what this is NOT: a statement about ``main``'s current totals. A branch cut
    before a burn-down landed can merge a count back up without failing — the same
    property the swallow lint has had since it shipped.
    """
    for ref in ("origin/main", "main"):
        try:
            mb = git(repo_root, "merge-base", ref, "HEAD").strip()
        except GitUnavailable:
            continue
        if mb:
            return mb
    raise GitUnavailable("no origin/main or main to diff against")


def must_not_skip(reason: str) -> bool:
    """True when a skipped diff-scoped check must be a hard ERROR instead.

    On a pull request the diff-scoped check IS the gate; skipping it silently is the
    vacuous-guard failure this repo keeps paying for. Locally (or on a push, where the
    range is empty by construction) the graceful skip stays.
    """
    if os.environ.get("GITHUB_EVENT_NAME") != "pull_request":
        return False
    print(
        f"ERROR: diff-scoped check cannot run on a pull request ({reason}). The lint job must check out "
        "with `fetch-depth: 0` — a depth-1 checkout plus `git fetch origin main --depth=1` leaves disjoint "
        "shallow roots and makes this guard a no-op (verified 2026-08-29, CI run 33259722155).",
        file=sys.stderr,
    )
    return True


def changed_files(repo_root: Path, base: str, scope: str, *, suffix: str = ".py") -> list[tuple[str, str]]:
    """[(path, path-at-base)] for files changed in ``scope``; renames map to their SOURCE.

    Without ``-M`` a renamed file looks new (``git show base:<newpath>`` fails), so its
    pre-existing sites read as freshly added and the ratchet fires on a pure move — which
    would make item 7's god-function decomposition (all moving code) impossible to land.
    """
    out: list[tuple[str, str]] = []
    for line in git(repo_root, "diff", "--name-status", "-M", base, "HEAD", "--", scope).splitlines():
        parts = line.split("\t")
        status = parts[0]
        if status.startswith("R") and len(parts) >= 3:
            old_path, new_path = parts[1], parts[2]
        elif len(parts) >= 2:
            old_path = new_path = parts[1]
            if status.startswith("A"):
                old_path = ""  # new file — grandfathered at zero
        else:
            continue
        if new_path.endswith(suffix):
            out.append((new_path, old_path))
    return out


def show(repo_root: Path, base: str, rel: str) -> str:
    """File content at ``base``. Empty only when ``rel`` genuinely did not exist there —
    a transient git failure RAISES rather than reading as "the file was empty", which
    would silently turn every pre-existing site into a new one."""
    if not rel:
        return ""
    # Resolve the ref FIRST: without this, a bad/unfetched base makes every path
    # look absent, i.e. every pre-existing site reads as newly added.
    git(repo_root, "rev-parse", "--verify", "--quiet", f"{base}^{{commit}}")
    try:
        git(repo_root, "cat-file", "-e", f"{base}:{rel}")
    except GitUnavailable:
        return ""  # genuinely absent at base
    return git(repo_root, "show", f"{base}:{rel}")


def count_ratchet(
    repo_root: Path,
    base: str,
    scope: str,
    hits: Callable[[str], list],
    *,
    exclude: frozenset[str] = frozenset(),
    what: str = "count",
    advice: str = "",
) -> list[str]:
    """Violations for every changed file whose ``len(hits(text))`` rose against ``base``.

    Per-file and count-based, so moving code within a file is free and a new file is
    grandfathered at zero. ``exclude`` holds repo-relative paths the caller checks
    another way (e.g. the canonical writer itself).
    """
    out: list[str] = []
    for rel, rel_at_base in changed_files(repo_root, base, scope):
        if rel in exclude or rel_at_base in exclude:
            continue
        path = repo_root / rel
        new = len(hits(path.read_text(errors="replace"))) if path.exists() else 0
        old = len(hits(show(repo_root, base, rel_at_base)))
        if new > old:
            moved = f" (renamed from {rel_at_base})" if rel_at_base and rel_at_base != rel else ""
            out.append(f"{rel}{moved}: {what} rose {old} → {new} on this branch{(' — ' + advice) if advice else ''}")
    return out
