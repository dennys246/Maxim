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


# Hosts the README is allowed to link to. An entry here is a DECISION, not a
# default — adding one means someone judged that domain durable enough to sit
# on the PyPI landing page, where the link is immutable for that version.
_ALLOWED_HOSTS = {
    "github.com",  # canonical source; blob paths additionally existence-checked above
    "pymaxim.bio",  # canonical docs site
    "pypi.org",
    "img.shields.io",  # badges
    # Legacy long-form guides. 27 of 29 of these 404'd when 1.0.9 shipped —
    # they are only reachable because dennyschaedig.com now 301s them to
    # pymaxim.bio. Kept deliberately (1.0.9's README is immutable and still
    # links here), but this is DEBT: drop the "Legacy Website Guides" section
    # and remove this entry when the migration finishes.
    "dennyschaedig.com",
    "www.dennyschaedig.com",
}


def test_readme_links_only_to_approved_hosts() -> None:
    """Catch link ROT, not just link FORM.

    The 1.0.9 README passed every check we had — no relative links, all GitHub
    paths resolved — and still shipped 27 dead links, because every check
    asked whether a link was well-formed and none asked where it pointed.
    A new host slipping into the README should be a deliberate decision,
    reviewed once, rather than something discovered by a reader hitting a 404
    on a page that cannot be edited.
    """
    from urllib.parse import urlparse

    body = README.read_text()
    hosts = {urlparse(u).hostname for u in re.findall(r"\]\((https?://[^)]+)\)", body)}
    hosts.discard(None)
    unknown = sorted(h for h in hosts if h not in _ALLOWED_HOSTS)
    assert not unknown, (
        f"README links to unapproved host(s): {unknown}. PyPI renders this file as the "
        "project page and the metadata is immutable per version, so a dead link ships "
        "until the next release. Add to _ALLOWED_HOSTS only after deciding the domain "
        "is durable."
    )
