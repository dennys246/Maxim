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

**On a push to main** (#1089, owner decisions 2026-10-04) the merge-base with origin/main
is HEAD itself, so every diff-scoped lint compared HEAD with HEAD and passed vacuously. On
``push``, :func:`base_ref` returns :func:`push_base` instead: the newest first-parent
ancestor of HEAD whose ``lint`` job SUCCEEDED on a push run. Not the event's ``before``:
GitHub cancels a PENDING run in the concurrency group when a newer push queues, so with
pushes A, B, C the A..B range would otherwise never be diffed. Every input is validated and
any doubt raises :class:`GitUnavailable`, which :func:`must_not_skip` turns into exit 2 on a
push as on a pull request. :func:`push_units` splits that range into what landed (a merge, a
squash, a rebase, or a direct push), each with its merged PR when one exists, for the lints
whose rules read the PR (the ``[no-tests]`` opt-out, the ledger's branch point).

Residuals: a red push cannot be blocked, only flagged; it stays red until fixed forward,
because the next push's base is still the last green one. Some violations cannot be fixed
forward (a direct ``fix:`` push without tests, an edited append-only exception list, a format
migration), so the way out is an owner-approved, committed acceptance record,
``scripts/push_base_accepts.json`` (owner decision 2026-10-04): an entry ``{sha, reason, owner,
date}`` makes that first-parent commit count as green. An entry counts only when it reached main
through a merged PR, so a direct push cannot accept itself. A push's ranges are found
through the GitHub API (``gh api``, ``actions: read`` + ``pull-requests: read``); when it
cannot answer, the lint fails closed rather than guessing.

Deliberately stdlib-only (plus the ``gh`` CLI on push); does not import ``maxim``.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import re
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "GitUnavailable",
    "base_ref",
    "changed_files",
    "count_ratchet",
    "git",
    "must_not_skip",
    "push_base",
    "push_units",
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
    """The merge-base with origin/main (then main); on a push, :func:`push_base`. Raises :class:`GitUnavailable`.

    Note what this is NOT: a statement about ``main``'s current totals. A branch cut
    before a burn-down landed can merge a count back up without failing — the same
    property the swallow lint has had since it shipped.
    """
    if os.environ.get("GITHUB_EVENT_NAME") == "push":
        return push_base(repo_root)
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
    event = os.environ.get("GITHUB_EVENT_NAME")
    if event == "pull_request":
        print(
            f"ERROR: diff-scoped check cannot run on a pull request ({reason}). The lint job must check out "
            "with `fetch-depth: 0` — a depth-1 checkout plus `git fetch origin main --depth=1` leaves disjoint "
            "shallow roots and makes this guard a no-op (verified 2026-08-29, CI run 33259722155).",
            file=sys.stderr,
        )
        return True
    if event == "push":
        print(
            f"ERROR: diff-scoped check cannot run on this push ({reason}). A push to main is judged against the "
            "last push whose lint job passed (_lint_git.push_base); without that base the push would be unchecked. "
            "The lint job needs `fetch-depth: 0`, `actions: read`, `pull-requests: read` and GH_TOKEN.",
            file=sys.stderr,
        )
        return True
    return False


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
    include: frozenset[str] | None = None,
    what: str = "count",
    advice: str = "",
) -> list[str]:
    """Violations for every changed file whose ``len(hits(text))`` rose against ``base``.

    Per-file and count-based, so moving code within a file is free and a new file is
    grandfathered at zero. ``exclude`` holds repo-relative paths the caller checks
    another way (e.g. the canonical writer itself).

    ``include`` restricts the check to files whose NEW or OLD path is listed. Use it rather than
    passing each file as its own ``scope``: a single-file pathspec holds only one side of a rename,
    so ``-M`` cannot pair them and a pure move reads as a file whose every site is new — the
    ratchet then fires on the move it is supposed to allow.
    """
    out: list[str] = []
    for rel, rel_at_base in changed_files(repo_root, base, scope):
        if rel in exclude or rel_at_base in exclude:
            continue
        if include is not None and rel not in include and rel_at_base not in include:
            continue
        path = repo_root / rel
        new = len(hits(path.read_text(errors="replace"))) if path.exists() else 0
        old = len(hits(show(repo_root, base, rel_at_base)))
        if new > old:
            moved = f" (renamed from {rel_at_base})" if rel_at_base and rel_at_base != rel else ""
            out.append(f"{rel}{moved}: {what} rose {old} → {new} on this branch{(' — ' + advice) if advice else ''}")
    return out


# ── push events (#1089) ──────────────────────────────────────────────────────

_SHA_RE = re.compile(r"[0-9a-f]{40}")
_ZERO_SHA = "0" * 40
#: How far back along main's first-parent chain push_base looks for a green lint run before failing closed.
PUSH_BASE_SEARCH_LIMIT = 50
#: The committed acceptance record: first-parent commits of main the owner accepted as a push base (#1089).
ACCEPTS_REL = "scripts/push_base_accepts.json"
_ACCEPT_KEYS = {"sha", "reason", "owner", "date"}
_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")


def gh_api(path: str) -> Any:
    """``gh api <path>`` as JSON. Raises :class:`GitUnavailable`: a push base nobody can verify is no base."""
    try:
        r = subprocess.run(["gh", "api", path], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise GitUnavailable(f"gh api {path}: {exc}") from exc
    if r.returncode != 0:
        raise GitUnavailable(f"gh api {path}: {r.stderr.strip() or r.returncode}")
    try:
        return json.loads(r.stdout)
    except json.JSONDecodeError as exc:
        raise GitUnavailable(f"gh api {path}: unreadable JSON ({exc})") from exc


def _env(name: str) -> str:
    value = os.environ.get(name, "")
    if not value:
        raise GitUnavailable(f"push event without ${name}")
    return value


def _push_event(repo_root: Path) -> tuple[str, str]:
    """The pushed range's (before, after), validated against the checkout."""
    path = _env("GITHUB_EVENT_PATH")
    try:
        event = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise GitUnavailable(f"push event file {path} unreadable ({exc})") from exc
    before, after = (event.get(k) if isinstance(event, dict) else None for k in ("before", "after"))
    if not (isinstance(before, str) and _SHA_RE.fullmatch(before)):
        raise GitUnavailable(f"push event has no valid `before` ({before!r})")
    if not (isinstance(after, str) and _SHA_RE.fullmatch(after)):
        raise GitUnavailable(f"push event has no valid `after` ({after!r})")
    if before == _ZERO_SHA:
        raise GitUnavailable("the push created the branch: there is no base to diff against")
    head = git(repo_root, "rev-parse", "HEAD").strip()
    if head != after:
        raise GitUnavailable(f"HEAD {head[:12]} is not the pushed commit {after[:12]}: judging another range")
    return before, after


def _is_ancestor(repo_root: Path, ancestor: str, of: str) -> bool:
    try:
        git(repo_root, "merge-base", "--is-ancestor", ancestor, of)
    except GitUnavailable:
        return False
    return True


def push_base(repo_root: Path, *, api: Callable[[str], Any] | None = None) -> str:
    """The newest first-parent ancestor of the pushed commit whose ``lint`` job succeeded on a push run.

    The pushed ``before`` must exist and be an ancestor of HEAD (a non-fast-forward push fails), and the search
    starts there. Raises :class:`GitUnavailable` when no green run is found within
    :data:`PUSH_BASE_SEARCH_LIMIT` commits, or when the API cannot answer."""
    api = api or gh_api  # resolved per call, so a test can substitute the module's gh_api
    before, _after = _push_event(repo_root)
    git(repo_root, "rev-parse", "--verify", "--quiet", f"{before}^{{commit}}")
    if not _is_ancestor(repo_root, before, "HEAD"):
        raise GitUnavailable(f"`before` {before[:12]} is not an ancestor of HEAD: a non-fast-forward push")
    repo = _env("GITHUB_REPOSITORY")
    job = _env("GITHUB_JOB")
    workflow = _env("GITHUB_WORKFLOW_REF").split("@", 1)[0].rsplit("/", 1)[-1]
    candidates = git(repo_root, "rev-list", "--first-parent", f"--max-count={PUSH_BASE_SEARCH_LIMIT}", before).split()
    wanted = set(candidates)
    runs_by_sha: dict[str, list[int]] = {}
    for page in (1, 2, 3):
        data = api(f"repos/{repo}/actions/workflows/{workflow}/runs?branch=main&event=push&per_page=100&page={page}")
        runs = data.get("workflow_runs") if isinstance(data, dict) else None
        if not isinstance(runs, list):
            raise GitUnavailable(f"workflow runs for {workflow}: unexpected response")
        for run in runs:
            if isinstance(run, dict) and run.get("head_sha") in wanted and isinstance(run.get("id"), int):
                runs_by_sha.setdefault(run["head_sha"], []).append(run["id"])
        if len(runs) < 100:
            break
    accepted = _accepted(repo_root, repo, api)
    for sha in candidates:
        if sha in accepted:
            print(f"push base: {sha[:12]} by acceptance ({accepted[sha]})")
            return sha
        for run_id in runs_by_sha.get(sha, ()):
            data = api(f"repos/{repo}/actions/runs/{run_id}/jobs?per_page=100")
            jobs = data.get("jobs") if isinstance(data, dict) else None
            if not isinstance(jobs, list):
                raise GitUnavailable(f"jobs of run {run_id}: unexpected response")
            if any(isinstance(j, dict) and j.get("name") == job and j.get("conclusion") == "success" for j in jobs):
                return sha
    raise GitUnavailable(
        f"no push run with a green `{job}` job among the last {len(candidates)} first-parent commits of main"
    )


def _accepted(repo_root: Path, repo: str, api: Callable[[str], Any]) -> dict[str, str]:
    """{sha: "owner, date: reason"} for every honoured entry of :data:`ACCEPTS_REL` at HEAD. Fails closed on a
    malformed record, an entry that is not a first-parent commit of HEAD, or one that arrived without a merged PR."""
    path = repo_root / ACCEPTS_REL
    if not path.exists():
        return {}
    try:
        entries = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise GitUnavailable(f"{ACCEPTS_REL} unreadable ({exc})") from exc
    if not isinstance(entries, list):
        raise GitUnavailable(f"{ACCEPTS_REL} must be a list")
    chain = set(git(repo_root, "rev-list", "--first-parent", "HEAD").split())
    out: dict[str, str] = {}
    for e in entries:
        ok = isinstance(e, dict) and set(e) == _ACCEPT_KEYS and all(isinstance(e[k], str) and e[k].strip() for k in e)
        if not ok or not _SHA_RE.fullmatch(e["sha"]) or not _DATE_RE.fullmatch(e["date"]):
            raise GitUnavailable(f"{ACCEPTS_REL}: malformed entry {e!r} (needs exactly {sorted(_ACCEPT_KEYS)})")
        if e["sha"] not in chain:
            raise GitUnavailable(f"{ACCEPTS_REL}: {e['sha'][:12]} is not a first-parent commit of main")
        introduced = git(
            repo_root,
            "log",
            "--first-parent",
            "--diff-merges=first-parent",
            "--reverse",
            "--format=%H",
            f"-S{e['sha']}",
            "HEAD",
            "--",
            ACCEPTS_REL,
        ).split()
        if not introduced or _merged_pr(repo, introduced[0], api) is None:
            raise GitUnavailable(
                f"{ACCEPTS_REL}: the entry for {e['sha'][:12]} did not arrive through a merged PR (a direct push cannot "
                "accept itself)"
            )
        out[e["sha"]] = f"{e['owner']}, {e['date']}: {e['reason']}"
    return out


@dataclass(frozen=True)
class PushUnit:
    """One thing that landed on main inside a push range: its first-parent commit, the merged PR it belongs to
    (``None`` for a direct push), the commits it brought (a merge's branch commits, else itself), and the epoch
    when its work forked from main (the earliest such commit's parent, for the ledger's branch-point rule)."""

    sha: str
    pr: dict[str, Any] | None
    commits: tuple[str, ...]
    fork_epoch: int


def _merged_pr(repo: str, sha: str, api: Callable[[str], Any]) -> dict[str, Any] | None:
    data = api(f"repos/{repo}/commits/{sha}/pulls")
    if not isinstance(data, list):
        raise GitUnavailable(f"pulls for {sha[:12]}: unexpected response")
    for pr in data:
        if isinstance(pr, dict) and pr.get("merged_at") and (pr.get("base") or {}).get("ref") == "main":
            return {"number": pr.get("number"), "title": pr.get("title") or "", "body": pr.get("body") or ""}
    return None


def push_units(repo_root: Path, base: str, *, api: Callable[[str], Any] | None = None) -> list[PushUnit]:
    """Split ``base..HEAD`` along main's first-parent chain into what landed, oldest first.

    A rebase-merged PR lands as several first-parent commits; GitHub lands them as one contiguous block, which
    the fix->tests title rule relies on when it reads that PR's aggregate diff."""
    api = api or gh_api
    repo = _env("GITHUB_REPOSITORY")
    units = []
    for sha in git(repo_root, "rev-list", "--first-parent", "--reverse", f"{base}..HEAD").split():
        parents = git(repo_root, "rev-list", "--parents", "-n", "1", sha).split()[1:]
        if len(parents) > 1:
            commits = tuple(git(repo_root, "rev-list", "--reverse", f"{parents[0]}..{parents[1]}").split())
            fork = git(repo_root, "merge-base", parents[0], parents[1]).strip()
        else:
            commits = (sha,)
            fork = parents[0] if parents else sha
        pr = _merged_pr(repo, sha, api)
        if pr is not None and len(parents) <= 1:
            # A squash or rebase merge kept none of the branch's history; the PR's own first commit says when its
            # work began (the commits are on GitHub even after the branch is deleted).
            pr_commits = api(f"repos/{repo}/pulls/{pr['number']}/commits?per_page=100")
            if isinstance(pr_commits, list) and pr_commits:
                first = (pr_commits[0].get("commit") or {}).get("committer") or {}
                date = first.get("date")
                if isinstance(date, str):
                    epoch = int(_dt.datetime.fromisoformat(date.replace("Z", "+00:00")).timestamp())
                    units.append(PushUnit(sha, pr, commits, min(epoch, _commit_epoch(repo_root, fork))))
                    continue
            raise GitUnavailable(f"PR #{pr['number']}: its commits could not be read")
        units.append(PushUnit(sha, pr, commits, _commit_epoch(repo_root, fork)))
    return units


def _commit_epoch(repo_root: Path, sha: str) -> int:
    return int(git(repo_root, "show", "-s", "--format=%ct", sha).strip())
