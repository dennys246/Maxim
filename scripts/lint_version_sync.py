#!/usr/bin/env python3
"""Version truth, in one place (roadmap 1.1.x item 16.1; score card 2026-08-27 Release governance + Documentation honesty).

Policy (CLAUDE.md §Versioning, decided 2026-08-29): **`main` is ahead of PyPI; the version
bump happens in the release transaction.** Between releases `pyproject.toml` /
`src/maxim/__init__.py` carry the LAST PUBLISHED version and `CHANGELOG.md` accumulates under
`## [Unreleased]`; the bump commit adds the `## [X.Y.Z] - <date>` section, and the tag +
publish follow the guide. Nothing in the repo asserts what PyPI serves — the three
version lines point at PyPI instead of describing it, because prose that describes PyPI
("pending", "rc", "serves") drifted on every release and the score card counted it.

Checks (all mechanical; every one fails loudly):

1. `pyproject.toml` version == `src/maxim/__init__.py` `__version__` (the original CI step).
2. `CHANGELOG.md` has a `## [<that version>]` section header — so a bump commit that forgot
   its CHANGELOG entry fails on the bump itself, and the top released header can never be
   ahead of or behind the code's version.
3. The three version lines — CLAUDE.md "Current version:", docs/plans/README.md "Current
   version:", docs/index.md "**Version:**" — name exactly the pyproject version and carry no
   PyPI-state prose (`pending`, `rc`, `serves`, `published`, `release candidate`); they link
   to PyPI for the served version. Each line must exist (a missing line is a failure, not a
   pass).

Exits: 0 clean; 1 violations (stderr).
"""

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PYPI_URL = "https://pypi.org/project/pymaxim/"

# (path, regex capturing the version on the line, human label)
VERSION_LINES = (
    ("CLAUDE.md", re.compile(r"^Current version: \*\*(?P<v>[^*]+)\*\*.*$", re.M), "CLAUDE.md 'Current version:' line"),
    (
        "docs/plans/README.md",
        re.compile(r"^Current version: \*\*(?P<v>[^*]+)\*\*.*$", re.M),
        "docs/plans/README.md 'Current version:' line",
    ),
    ("docs/index.md", re.compile(r"^\*\*Version:\*\* (?P<v>\S+).*$", re.M), "docs/index.md '**Version:**' line"),
)
FORBIDDEN_PROSE = re.compile(r"\b(pending|rc\d*|serves|published|release candidate|still on)\b", re.I)


def pyproject_version(repo_root: Path = REPO_ROOT) -> str:
    with open(repo_root / "pyproject.toml", "rb") as f:
        return tomllib.load(f)["project"]["version"]


def init_version(repo_root: Path = REPO_ROOT) -> str | None:
    m = re.search(r'^__version__\s*=\s*"([^"]+)"', (repo_root / "src/maxim/__init__.py").read_text(), re.M)
    return m.group(1) if m else None


def violations(repo_root: Path = REPO_ROOT) -> list[str]:
    out: list[str] = []
    version = pyproject_version(repo_root)
    iv = init_version(repo_root)
    if iv is None:
        out.append("src/maxim/__init__.py: __version__ not found")
    elif iv != version:
        out.append(
            f"version drift — pyproject.toml={version!r} vs src/maxim/__init__.py={iv!r}; "
            "bump both in the same commit (CLAUDE.md 'Versioning')"
        )
    changelog = (repo_root / "CHANGELOG.md").read_text()
    if not re.search(rf"^## \[{re.escape(version)}\]", changelog, re.M):
        out.append(
            f"CHANGELOG.md has no `## [{version}]` section — the bump commit adds it (the release transaction); "
            "between releases pyproject carries the last published version"
        )
    for rel, pattern, label in VERSION_LINES:
        text = (repo_root / rel).read_text()
        m = pattern.search(text)
        if m is None:
            out.append(f"{rel}: {label} is missing — it must exist and name the pyproject version")
            continue
        line = m.group(0)
        if m.group("v").strip() != version:
            out.append(f"{rel}: {label} says {m.group('v').strip()!r} but pyproject.toml says {version!r}")
        if FORBIDDEN_PROSE.search(line):
            out.append(
                f"{rel}: {label} carries PyPI-state prose ({FORBIDDEN_PROSE.search(line).group(0)!r}) — "
                f"the line names the version and links {PYPI_URL} for what is served; it does not describe PyPI"
            )
        if PYPI_URL not in line:
            out.append(f"{rel}: {label} must link {PYPI_URL} (the served version lives there, not in prose)")
    return out


def main() -> int:
    fails = violations()
    if fails:
        print("version-sync lint FAILED:", file=sys.stderr)
        for f in fails:
            print(f"  {f}", file=sys.stderr)
        return 1
    print(
        f"version-sync lint: OK — {pyproject_version()} in pyproject, __init__, CHANGELOG and the three version lines"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
