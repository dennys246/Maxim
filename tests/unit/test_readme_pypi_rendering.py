"""The README is the PyPI long_description — its links must work off-GitHub.

`readme = "README.md"` in pyproject.toml means this file is what pypi.org
renders on the project page. Repo-relative markdown links resolve on GitHub and
404 on PyPI, so the first thing a new user clicks is broken. Measured
2026-08-21 on the 1.0.9 candidate: 17 such links.

PyPI metadata is immutable per version, so a bad link ships until the next
release. This guard is cheap; the mistake is not.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
README = REPO / "README.md"

# ](target) where target is not absolute, an anchor, or a mail link
_RELATIVE_LINK = re.compile(r"\]\((?!https?://|#|mailto:)([^)]+)\)")


def test_readme_is_the_declared_long_description() -> None:
    """If this stops being true, the rest of this file is guarding nothing."""
    pyproject = tomllib.loads((REPO / "pyproject.toml").read_text())
    assert pyproject["project"]["readme"] == "README.md"


def test_no_repo_relative_links() -> None:
    offenders = _RELATIVE_LINK.findall(README.read_text())
    assert not offenders, (
        f"{len(offenders)} repo-relative link(s) in README.md will 404 on pypi.org: "
        f"{sorted(set(offenders))[:5]}. Use absolute "
        "https://github.com/dennys246/Maxim/blob/main/... URLs."
    )


def test_github_links_point_at_paths_that_exist() -> None:
    """An absolute URL that 404s is worse than a relative one — it looks fine.

    Checks the path component against the working tree; no network needed.
    """
    body = README.read_text()
    prefix = "https://github.com/dennys246/Maxim/blob/main/"
    missing = []
    for url in re.findall(rf"\]\({re.escape(prefix)}([^)]+)\)", body):
        target = url.split("#")[0]
        if not (REPO / target).exists():
            missing.append(target)
    assert not missing, f"README links to paths absent from the repo: {missing}"
