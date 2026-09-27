#!/usr/bin/env python3
"""Refuse to release while a nightly lane is red (roadmap 1.3.x: "the release procedure reads the nightlies").

The model-cache and slow lanes run only on schedule, never on a PR, so nothing on a release PR showed
that they were red — the model-cache lane was red 16 nights running while releases were being planned.
This reads the latest SCHEDULED run of the Tests workflow on ``main`` and fails unless every
``(nightly)`` job in it succeeded, and the run is recent.

Fails CLOSED: no scheduled run, a stale run, a run with no nightly jobs, or an unreadable API is a
refusal, never a pass — a check that cannot see the nightlies must not report them green.

``--only-when-releasing`` makes it a no-op unless a release is being cut: the version in
``pyproject.toml`` has no ``v<version>`` tag yet. Between releases ``pyproject`` carries the last
PUBLISHED (tagged) version by policy (CLAUDE.md §Versioning), so ordinary PRs skip it; the release PR,
which bumps the version, does not. The release-build CI job runs it that way; the publication guide
runs it plainly before publishing.

Usage: python3 scripts/check_nightlies.py [--only-when-releasing] [--max-age-hours 48]
Exits: 0 green (or not releasing); 1 a nightly is red / stale / missing; 2 could not check.
"""

from __future__ import annotations

import argparse
import json
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


def nightly_verdict(
    run: dict[str, Any] | None, jobs: list[dict[str, Any]], *, now: datetime, max_age: timedelta
) -> list[str]:
    """Why the nightlies are NOT green (empty list = green). Pure, so the rule is unit-testable."""
    if run is None:
        return ["no completed scheduled run of the Tests workflow on main"]
    problems = []
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
        "run", "list", "--workflow", WORKFLOW, "--branch", "main", "--event", "schedule",
        "--status", "completed", "--limit", "1", "--json", "databaseId,createdAt",
    )  # fmt: skip
    if not runs:
        return None, []
    view = _gh("run", "view", str(runs[0]["databaseId"]), "--json", "jobs")
    return runs[0], list((view or {}).get("jobs") or [])


def pyproject_version() -> str:
    return str(tomllib.loads((REPO / "pyproject.toml").read_text())["project"]["version"])


def is_tagged(version: str) -> bool:
    """Whether ``v<version>`` exists on the remote (the published-release marker)."""
    proc = subprocess.run(
        ("git", "ls-remote", "--tags", "origin", f"refs/tags/v{version}"),
        cwd=REPO, capture_output=True, text=True, check=False,
    )  # fmt: skip
    if proc.returncode != 0:
        raise CheckError(f"git ls-remote: {proc.stderr.strip() or proc.returncode}")
    return bool(proc.stdout.strip())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only-when-releasing", action="store_true")
    parser.add_argument("--max-age-hours", type=float, default=48.0)
    args = parser.parse_args(argv)
    try:
        if args.only_when_releasing:
            version = pyproject_version()
            if is_tagged(version):
                print(f"not a release: v{version} is already tagged -- nightlies not required here")
                return 0
            print(f"releasing {version} (no v{version} tag yet): the nightlies must be green")
        run, jobs = latest_nightly()
    except CheckError as exc:
        print(f"UNVERIFIED: {exc}", file=sys.stderr)
        return 2
    problems = nightly_verdict(run, jobs, now=datetime.now(timezone.utc), max_age=timedelta(hours=args.max_age_hours))
    if problems:
        print("REFUSED: a nightly lane is not green -- do not release:", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 1
    print(f"nightlies green (run {run['databaseId']}, {run['createdAt']})")  # type: ignore[index]
    return 0


if __name__ == "__main__":
    sys.exit(main())
