#!/usr/bin/env python3
"""Which open PRs collide, and in what order to merge them: the merge-train stopgap (owner request 2026-10-27, M83).

``main`` requires branches to be up to date (classic protection, ``strict``), and merge queues are not offered on a
personal-account repository. So every merge makes each other open PR BEHIND, its update restarts a ~40 min CI run, and
a PR that shares a hot file with the one that landed (``CHANGELOG.md``, ``scripts/coverage_floors.json``,
``src/maxim/utils/function_length_baseline.json``) also conflicts or must re-pin. This script makes that visible
before anyone clicks merge. For every open PR against ``main`` it prints:

- whether the PR conflicts with ``main`` NOW, and in which files (``git merge-tree`` of ``origin/main`` and the head):
  a PR that already conflicts owes a resolution whatever lands next, and that is never blamed on another PR;
- its merge state (BEHIND/DIRTY/...) and the NEWEST ``unit-tests`` run's conclusion on its head (a superseded,
  cancelled run is not its state, #1442);
- for each other PR it shares files with, what it owes if that PR lands first. The landing is SIMULATED: the other
  head is merged into ``origin/main`` (``merge-tree`` + ``commit-tree``, loose objects only) and this head is merged
  onto that. Only the conflicts that adds beyond this PR's own conflicts with ``main`` are listed, plus a floors
  re-pin when both edit ``coverage_floors.json``. A PR that itself conflicts with ``main`` cannot be simulated
  landing, and is reported as such;
- a suggested order: drafts last; then PRs that conflict with ``main`` or have red CI; then PRs that collide with
  nothing; then fewest new conflicts caused, fewest hot-file overlaps, oldest.

File sets come from git (``git diff --name-only`` from the merge-base with ``origin/main``), never from ``gh``, which
returns at most 100 files per PR. PR heads are fetched to ``refs/pr-overlap/<pid>/<n>`` (one namespace per run, so
two sessions never collide) and deleted when the run ends. Nothing else is written: no branch, index or checkout
changes, only loose objects for the simulated merges, and ``origin/main`` is fetched (as any ``git fetch`` would).

It is advisory: it gates nothing and exits 0 whatever it finds (2 when ``gh`` or ``git`` cannot answer). Running it
before asking the owner to merge is a rule followed by attention, not enforced; the merge train (M83 step 2) retires
it. Residual: predictions are pairwise. A three-way interaction, or one that appears only after ``main`` moves again,
is not seen.

Usage:
    python scripts/pr_overlap.py            # every open PR against main
    python scripts/pr_overlap.py --json     # the same, machine-readable
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from itertools import permutations
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
HOT_FILES: tuple[str, ...] = (
    "CHANGELOG.md",
    "scripts/coverage_floors.json",
    "src/maxim/utils/function_length_baseline.json",
)
FLOORS = "scripts/coverage_floors.json"
BASE_REF = "refs/remotes/origin/main"
_PR_FIELDS = "number,title,headRefOid,baseRefName,isDraft,mergeStateStatus,statusCheckRollup,createdAt"
_IDENT = ["-c", "user.name=pr-overlap", "-c", "user.email=pr-overlap@localhost", "-c", "commit.gpgsign=false"]


class ToolError(RuntimeError):
    """``gh`` or ``git`` could not answer: no advice is better than advice over missing data."""


@dataclass
class PR:
    number: int
    title: str
    head: str
    files: set[str]
    merge_state: str
    unit_tests: str  # the NEWEST unit-tests run's conclusion on the head, "pending", or "absent"
    created: str
    draft: bool = False
    main_conflicts: list[str] = field(default_factory=list)


@dataclass
class Pair:
    a: int
    b: int
    shared: set[str]
    # {PR that lands first: the conflicts the OTHER gains}; None when the first cannot land cleanly on main
    owed: dict[int, list[str] | None] = field(default_factory=dict)


def _run(cmd: list[str], *, cwd: Path = REPO_ROOT, check: bool = True) -> subprocess.CompletedProcess[str]:
    try:
        r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=300)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ToolError(f"{' '.join(cmd[:3])}: {exc}") from exc
    if check and r.returncode != 0:
        raise ToolError(f"{' '.join(cmd[:3])}: {r.stderr.strip() or r.returncode}")
    return r


def _unit_tests(rollup: list[dict[str, Any]] | None) -> str:
    """The newest ``unit-tests`` run's state (by ``startedAt``; a run not started yet counts as newest, #1442). A
    CheckRun carries ``name``/``status``/``conclusion``; a StatusContext carries ``context``/``state``."""
    runs = [c for c in rollup or [] if (c.get("name") or c.get("context")) == "unit-tests"]
    if not runs:
        return "absent"
    newest = max(runs, key=lambda c: (not c.get("startedAt"), str(c.get("startedAt") or "")))
    if "state" in newest and "conclusion" not in newest:
        state = str(newest.get("state") or "").lower()
        return "pending" if state in ("pending", "expected") else state
    if newest.get("status") and newest.get("status") != "COMPLETED":
        return "pending"
    return str(newest.get("conclusion") or "pending").lower()


def parse_prs(raw: list[dict[str, Any]], base: str = "main") -> list[PR]:
    return [
        PR(
            number=int(p["number"]),
            title=str(p.get("title", "")),
            head=str(p["headRefOid"]),
            files=set(),
            merge_state=str(p.get("mergeStateStatus") or "UNKNOWN"),
            unit_tests=_unit_tests(p.get("statusCheckRollup")),
            created=str(p.get("createdAt", "")),
            draft=bool(p.get("isDraft")),
        )
        for p in raw
        if p.get("baseRefName") == base
    ]


def list_prs() -> list[PR]:
    r = _run(["gh", "pr", "list", "--state", "open", "--limit", "100", "--json", _PR_FIELDS])
    try:
        return parse_prs(json.loads(r.stdout))
    except (ValueError, KeyError, TypeError) as exc:
        raise ToolError(f"gh pr list: unreadable JSON ({exc})") from exc


def fetch_heads(prs: list[PR], root: Path = REPO_ROOT, ns: str | None = None) -> str:
    """Fetch ``main`` and each PR head into this run's namespace, and point each ``PR.head`` at the fetched commit
    (a push after ``gh pr list`` must not leave a head the repo lacks). Returns the namespace, for cleanup."""
    ns = ns or f"refs/pr-overlap/{os.getpid()}"
    specs = ["+refs/heads/main:" + BASE_REF] + [f"+pull/{p.number}/head:{ns}/{p.number}" for p in prs]
    _run(["git", "fetch", "-q", "origin", *specs], cwd=root)
    for p in prs:
        p.head = _run(["git", "rev-parse", f"{ns}/{p.number}"], cwd=root).stdout.strip()
    return ns


def drop_refs(ns: str, root: Path = REPO_ROOT) -> None:
    refs = _run(["git", "for-each-ref", "--format=%(refname)", ns + "/"], cwd=root, check=False).stdout.split()
    for ref in refs:
        _run(["git", "update-ref", "-d", ref], cwd=root, check=False)


def changed_files(head: str, root: Path = REPO_ROOT, base: str = BASE_REF) -> set[str]:
    mb = _run(["git", "merge-base", base, head], cwd=root).stdout.strip()
    return set(_run(["git", "diff", "--name-only", mb, head], cwd=root).stdout.split())


def merge_tree(a: str, b: str, root: Path = REPO_ROOT) -> tuple[str, list[str]]:
    """``(tree, conflicted files)`` of a three-way merge of commits ``a`` and ``b``; only objects are written. Exit 1
    means "conflicts" only when git wrote the tree: an unknown commit also exits 1, and its error is never reported
    as a list of conflicting files."""
    r = _run(["git", "merge-tree", "--write-tree", "--name-only", "--no-messages", a, b], cwd=root, check=False)
    out = r.stdout.splitlines()
    tree = out[0].strip() if out else ""
    if r.returncode not in (0, 1) or len(tree) not in (40, 64) or any(c not in "0123456789abcdef" for c in tree):
        raise ToolError(f"git merge-tree {a[:8]} {b[:8]}: {r.stderr.strip() or r.returncode}")
    return tree, ([] if r.returncode == 0 else [line for line in out[1:] if line.strip()])


def land(head: str, root: Path = REPO_ROOT, base: str = BASE_REF) -> str | None:
    """The commit ``main`` would be after ``head`` merges cleanly, or None when it conflicts with ``main``."""
    tree, conflicts = merge_tree(base, head, root)
    if conflicts:
        return None
    main_sha = _run(["git", "rev-parse", base], cwd=root).stdout.strip()
    cmd = ["git", *_IDENT, "commit-tree", tree, "-p", main_sha, "-p", head, "-m", "pr-overlap simulated landing"]
    return _run(cmd, cwd=root).stdout.strip()


def analyse(prs: list[PR], root: Path = REPO_ROOT, base: str = BASE_REF) -> list[Pair]:
    """Fill each PR's files and conflicts with ``main``, and return every pair sharing files with its owed lists."""
    for p in prs:
        p.files = changed_files(p.head, root, base)
        p.main_conflicts = merge_tree(base, p.head, root)[1]
    landed = {p.number: land(p.head, root, base) for p in prs}
    out: dict[tuple[int, int], Pair] = {}
    for first, other in permutations(sorted(prs, key=lambda p: p.number), 2):
        shared = first.files & other.files
        if not shared:
            continue
        key = (min(first.number, other.number), max(first.number, other.number))
        pair = out.setdefault(key, Pair(key[0], key[1], shared))
        after = landed[first.number]
        if after is None:
            pair.owed[first.number] = None
            continue
        pair.owed[first.number] = sorted(set(merge_tree(after, other.head, root)[1]) - set(other.main_conflicts))
    return list(out.values())


def merge_order(prs: list[PR], ps: list[Pair]) -> list[PR]:
    """Drafts last; then a PR that conflicts with main or has red CI; then collide-with-nothing; then fewest new
    conflicts it causes, fewest hot-file overlaps, oldest."""

    def key(p: PR) -> tuple:
        mine = [x for x in ps if p.number in (x.a, x.b)]
        causes = sum(1 for x in mine if x.owed.get(p.number))
        n_hot = sum(1 for x in mine if x.shared & set(HOT_FILES))
        blocked = bool(p.main_conflicts) or p.merge_state == "DIRTY" or p.unit_tests not in ("success", "pending")
        return (p.draft, blocked, bool(mine), causes, n_hot, p.created, p.number)

    return sorted(prs, key=key)


def owed_text(first: int, other: int, pair: Pair) -> str:
    gained = pair.owed.get(first)
    if gained is None:
        return f"#{first} conflicts with main itself, so its landing cannot be simulated"
    owed = ["merge main (a new CI run)"]
    if gained:
        owed.append("resolve " + ", ".join(gained))
    if FLOORS in pair.shared:
        owed.append("re-pin coverage floors from its own new CI run (the values lint_coverage.py prints)")
    return f"if #{first} lands first, #{other} needs: " + "; ".join(owed)


def report(prs: list[PR], ps: list[Pair]) -> str:
    lines = [f"{len(prs)} open PR(s) against main; {len(ps)} pair(s) share files."]
    for p in prs:
        flags = " [draft]" if p.draft else ""
        main = f"; CONFLICTS WITH MAIN NOW in {', '.join(p.main_conflicts)}" if p.main_conflicts else ""
        lines.append(f"  #{p.number} {p.title[:70]}{flags} — {p.merge_state}, unit-tests {p.unit_tests}{main}")
    if ps:
        lines.append("Overlaps:")
    for x in ps:
        hot = sorted(x.shared & set(HOT_FILES))
        cold = len(x.shared) - len(hot)
        what = (", ".join(hot) + (f" + {cold} other file(s)" if cold else "")) if hot else f"{cold} file(s)"
        lines.append(f"  #{x.a} × #{x.b}: share {what}")
        lines.append(f"    {owed_text(x.a, x.b, x)}")
        lines.append(f"    {owed_text(x.b, x.a, x)}")
    lines.append("Suggested merge order: " + " → ".join(f"#{p.number}" for p in merge_order(prs, ps)))
    return "\n".join(lines)


def as_json(prs: list[PR], ps: list[Pair]) -> dict[str, Any]:
    return {
        "prs": [
            {
                "number": p.number,
                "merge_state": p.merge_state,
                "unit_tests": p.unit_tests,
                "main_conflicts": p.main_conflicts,
            }
            for p in prs
        ],
        "pairs": [
            {"a": x.a, "b": x.b, "shared": sorted(x.shared), "owed": {str(k): v for k, v in x.owed.items()}} for x in ps
        ],
        "order": [p.number for p in merge_order(prs, ps)],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args(argv)
    ns = f"refs/pr-overlap/{os.getpid()}"  # chosen before the fetch, so a fetch that fails part-way is cleaned up too
    try:
        prs = list_prs()
        fetch_heads(prs, ns=ns)
        ps = analyse(prs)
    except ToolError as exc:
        print(f"pr_overlap: cannot answer: {exc}", file=sys.stderr)
        return 2
    finally:
        drop_refs(ns)
    print(json.dumps(as_json(prs, ps), indent=2) if args.json else report(prs, ps))
    return 0


if __name__ == "__main__":
    sys.exit(main())
