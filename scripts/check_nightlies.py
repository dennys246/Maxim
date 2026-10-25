#!/usr/bin/env python3
"""Refuse to release while a nightly lane is red (roadmap 1.3.x: "the release procedure reads the nightlies").

The model-cache and slow lanes run only on schedule, never on a PR, so nothing on a release PR showed
that they were red — the model-cache lane was red 16 nights running while releases were being planned.
This reads the latest nightly run of the Tests workflow on ``main`` -- scheduled, or dispatched with
``gh workflow run test.yml --ref main`` -- and fails unless every ``(nightly)`` job in it succeeded, the
run is recent, AND it tested the commit ``main`` is at now: a green run from before the last merge says
nothing about the code being released.

Fails CLOSED: no scheduled run, a stale run, a run with no nightly jobs, or an unreadable API is a
refusal, never a pass — a check that cannot see the nightlies must not report them green.

``--only-when-releasing`` makes it a no-op unless a release is being cut: the version in
``pyproject.toml`` has no ``v<version>`` tag yet. Between releases ``pyproject`` carries the last
PUBLISHED (tagged) version by policy (CLAUDE.md §Versioning), so ordinary PRs skip it; the release PR,
which bumps the version, does not. The release-build CI job runs it that way; the publication guide
runs it plainly before publishing.

``--at <sha>`` (the publish workflow, #1372) reads the nightly run that tested THAT commit -- the tag being
published -- instead of the newest one: by the time a tag is pushed and approved, main may have moved on, and
"main's latest nightly" then says nothing about the released code. The run must be a non-cancelled scheduled or
dispatched run on ``main`` whose head is the commit, and the same green/age rules apply.

#938: the release-build job runs ``--only-when-releasing`` on ``pull_request`` only. On push and schedule a release
merge read red until its tag landed (the check ran inside the very nightly it waits for); the release PR and the
publish workflow are where it guards a release.

Usage: python3 scripts/check_nightlies.py [--only-when-releasing | --at <sha>] [--max-age-hours 48]
Exits: 0 green (or not releasing); 1 a nightly is red / stale / missing; 2 could not check.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tomllib
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
WORKFLOW = "test.yml"
NIGHTLY_MARKER = "(nightly)"


class CheckError(Exception):
    """The nightlies could not be read -- exit 2, never a pass."""


NIGHTLY_EVENTS = ("schedule", "workflow_dispatch")
DISPATCH_HINT = "run `gh workflow run test.yml --ref main` and re-check when it completes"
# `--at` (the publish job): a dispatch tests main's TIP, so once main has moved past the tag no re-run can pass.
AT_HINT = (
    "no green, recent nightly tested this commit. Before tagging (or on a rehearsal): dispatch one on main while "
    "main is still at this commit. After the approval wait (publish): if main has moved on, no re-run can test this "
    "commit -- reject this run and cut the release again from a commit a green nightly tested "
    "(docs/plans/archive/release_pipeline.md, SF4)"
)


def nightly_verdict(
    run: dict[str, Any] | None,
    jobs: list[dict[str, Any]],
    *,
    now: datetime,
    max_age: timedelta,
    main_sha: str | None = None,
) -> list[str]:
    """Why the nightlies are NOT green (empty list = green). Pure, so the rule is unit-testable."""
    if run is None:
        return [f"no completed nightly run of the Tests workflow on main -- {DISPATCH_HINT}"]
    problems = []
    if main_sha is not None and run.get("headSha") != main_sha:
        problems.append(
            f"run {run.get('databaseId')} tested {str(run.get('headSha'))[:12]}, but main is at {main_sha[:12]} -- "
            f"a nightly must test the code being released: {DISPATCH_HINT}"
        )
    created = datetime.fromisoformat(str(run["createdAt"]).replace("Z", "+00:00"))
    if now - created > max_age:
        problems.append(
            f"the latest scheduled run ({run.get('databaseId')}, {run['createdAt']}) is older than {max_age}"
        )
    nightly = [job for job in jobs if NIGHTLY_MARKER in str(job.get("name", ""))]
    if not nightly:
        problems.append(f"run {run.get('databaseId')} has no '{NIGHTLY_MARKER}' jobs -- nothing to read")
    for job in nightly:
        if job.get("conclusion") != "success":
            problems.append(f"{job.get('name')}: {job.get('conclusion') or job.get('status')}")
    return problems


def _gh(*args: str) -> Any:
    proc = subprocess.run(("gh", *args), cwd=REPO, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise CheckError(f"gh {' '.join(args[:3])}: {proc.stderr.strip() or proc.returncode}")
    try:
        return json.loads(proc.stdout) if proc.stdout.strip() else None
    except json.JSONDecodeError as exc:
        raise CheckError(f"gh {' '.join(args[:3])}: unreadable JSON ({exc})") from exc


def latest_nightly() -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    runs = _gh(
        "run", "list", "--workflow", WORKFLOW, "--branch", "main", "--status", "completed",
        "--limit", "30", "--json", "databaseId,createdAt,conclusion,event,headSha",
    )  # fmt: skip
    # Only runs whose nightly jobs execute (scheduled, or dispatched after a fix -- a re-run would
    # re-test the OLD commit). A run superseded in main's concurrency queue completes `cancelled`
    # without running a job and says nothing, so the newest run that is not cancelled is read.
    ran = [run for run in (runs or []) if run.get("event") in NIGHTLY_EVENTS and run.get("conclusion") != "cancelled"]
    if not ran:
        return None, []
    view = _gh("run", "view", str(ran[0]["databaseId"]), "--json", "jobs")
    return ran[0], list((view or {}).get("jobs") or [])


def nightly_at(sha: str) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """The newest non-cancelled scheduled/dispatched run on ``main`` that tested ``sha`` (#1372)."""
    runs = _gh(
        "run", "list", "--workflow", WORKFLOW, "--commit", sha, "--status", "completed",
        "--limit", "30", "--json", "databaseId,createdAt,conclusion,event,headSha,headBranch",
    )  # fmt: skip
    ran = [
        run
        for run in (runs or [])
        if run.get("event") in NIGHTLY_EVENTS
        and run.get("conclusion") != "cancelled"
        and run.get("headBranch") == "main"
        and run.get("headSha") == sha
    ]
    if not ran:
        return None, []
    view = _gh("run", "view", str(ran[0]["databaseId"]), "--json", "jobs")
    return ran[0], list((view or {}).get("jobs") or [])


def main_head() -> str:
    """The commit ``main`` is at on the remote."""
    proc = subprocess.run(
        ("git", "ls-remote", "origin", "refs/heads/main"), cwd=REPO, capture_output=True, text=True, check=False
    )
    if proc.returncode != 0 or not proc.stdout.strip():
        raise CheckError(f"git ls-remote origin main: {proc.stderr.strip() or 'no ref'}")
    return proc.stdout.split()[0]


def pyproject_version() -> str:
    return str(tomllib.loads((REPO / "pyproject.toml").read_text())["project"]["version"])


def is_tagged(version: str) -> bool:
    """Whether ``v<version>`` exists on the remote (the published-release marker)."""
    proc = subprocess.run(
        ("git", "ls-remote", "--tags", "origin", f"refs/tags/v{version}"),
        cwd=REPO, capture_output=True, text=True, check=False, timeout=60,
        env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},  # a credential prompt must fail, not hang the release
    )  # fmt: skip
    if proc.returncode != 0:
        raise CheckError(f"git ls-remote: {proc.stderr.strip() or proc.returncode}")
    return bool(proc.stdout.strip())


def releasing() -> tuple[bool, str]:
    """Is this tree cutting a release (pyproject's version has no ``v<version>`` tag yet)? Unreadable = yes.

    The one definition: ``--only-when-releasing`` here and the console gate (R4, ``_console_pin``) both use it, and
    both fail closed -- a tree whose release status cannot be read is treated as releasing.
    """
    try:
        version = pyproject_version()
        if is_tagged(version):
            return False, f"not a release: v{version} is already tagged"
        return True, f"releasing {version} (no v{version} tag yet)"
    except (CheckError, OSError, KeyError, TypeError, ValueError, subprocess.SubprocessError) as exc:
        return True, f"cannot tell whether this tree is releasing ({exc}); treating it as releasing"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--only-when-releasing", action="store_true")
    mode.add_argument("--at", metavar="SHA", help="judge the nightly that tested this commit (the publish workflow)")
    parser.add_argument("--max-age-hours", type=float, default=48.0)
    args = parser.parse_args(argv)
    try:
        if args.only_when_releasing:
            is_release, why = releasing()
            if not is_release:
                print(f"{why} -- nightlies not required here")
                return 0
            print(f"{why}: the nightlies must be green")
        if args.at:
            if len(args.at) != 40:
                raise CheckError(f"--at needs a full 40-character commit sha, got {args.at!r}")
            run, jobs = nightly_at(args.at)
            head = args.at
        else:
            run, jobs = latest_nightly()
            head = main_head()
    except CheckError as exc:
        print(f"UNVERIFIED: {exc}", file=sys.stderr)
        return 2
    problems = nightly_verdict(
        run, jobs, now=datetime.now(timezone.utc), max_age=timedelta(hours=args.max_age_hours), main_sha=head
    )
    if problems and args.at:
        problems.append(AT_HINT)
    if problems:
        print("REFUSED: a nightly lane is not green -- do not release:", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 1
    print(
        f"nightlies green at {'commit' if args.at else 'main'} {head[:12]} (run {run['databaseId']}, {run['createdAt']})"
    )  # type: ignore[index]
    return 0


if __name__ == "__main__":
    sys.exit(main())
