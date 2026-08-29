#!/usr/bin/env python3
"""Audit — and optionally reconstruct — the git tags for released versions.

Every version with a ``## [X.Y.Z]`` CHANGELOG section should have a matching
annotated ``vX.Y.Z`` tag on the commit that introduced the version bump. PyPI is
immutable but carries no git history, so an untagged release cannot be traced
back to the code that produced it.

This exists because that failure already happened twice: 1.0.1-1.0.6 were
reconstructed months later during the 1.1 release-truth pass, and the 2026-08-19
pre-publish review found the gap was wider still (pre-1.0 versions plus 1.0.7+).
The publication guide now tags at publish time; this script closes the backlog
and lets anyone re-check in one command.

Since 2026-08-29 (roadmap 1.1.x item 16.1 / score card Release governance) it also audits
the RELEASE OBJECTS, because a tag alone does not hand a reader the artifact:
``--check-releases`` asserts, for every version PyPI serves, that

* a ``vX.Y.Z`` git tag exists (locally or on the remote), and
* a GitHub Release exists on it, with **≥ 2 assets** (wheel + sdist) whose **sha256
  match PyPI's** for the same filename, and
* the Release notes contain **no repo-relative markdown links** — they render on
  github.com/<repo>/releases, where every ``../`` link 404s (v1.1.0 shipped with 7),
* and the in-repo notes source ``docs/announcements/release_<version>.md`` — what
  ``gh release create --notes-file`` publishes next time — has none either.

Historical failures are grandfathered BY EXPLICIT LIST with reasons (below) and reported
as still-failing on every run; an entry that starts passing fails as stale. Network or
``gh`` auth failure is exit 2 (cannot check), never a silent pass.

Usage:
    python scripts/audit_release_tags.py                 # tag↔changelog drift only (offline)
    python scripts/audit_release_tags.py --check-releases # + GitHub Release/PyPI integrity (network)
    python scripts/audit_release_tags.py --write-tags    # create missing tags LOCALLY

``--write-tags`` never pushes. Review with ``git tag -n99`` and push deliberately
(``git push origin --tags``) — a published tag is effectively permanent.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CHANGELOG = REPO / "CHANGELOG.md"
VERSION_FILES = ("pyproject.toml", "src/maxim/__init__.py")
PYPI_JSON = "https://pypi.org/pypi/pymaxim/json"
ANNOUNCEMENTS = Path("docs/announcements")

# Release objects that predate the rule, by version → reason. Reported as still-failing.
GRANDFATHERED_RELEASES: dict[str, str] = {
    "0.2.1": "pre-1.0 PyPI upload; no Release object was ever created (tag reconstructed 2026-08-20)",
    "0.3.0": "pre-1.0 PyPI upload; no Release object",
    "0.3.1": "pre-1.0 PyPI upload; no Release object",
    "0.3.2": "pre-1.0 PyPI upload; no Release object",
    "0.4.0": "pre-1.0 PyPI upload; no Release object",
    "0.5.0": "pre-1.0 PyPI upload; no Release object",
    "0.6.0": "0.6.0–0.8.1 shipped to PyPI without CHANGELOG entries or Release objects (CHANGELOG note, 2026-05-11)",
    "0.7.0": "see 0.6.0",
    "0.8.0": "see 0.6.0",
    "0.8.1": "see 0.6.0",
    "0.9.0": "Release object exists but predates the attach-the-artifacts rule (0 assets)",
    "0.9.1": "PyPI upload with no Release object; predates the rule",
    "1.0.0": "Release object exists ('Maxim 1.0.0 — The Honest Benchmark') but predates the rule (0 assets)",
    "1.0.9": (
        "Release object backfilled 2026-08-26 from the CHANGELOG with NO assets — the 1.0.9 wheel/sdist were "
        "not kept after the upload, so they cannot be attached without rebuilding (a rebuild would not be the "
        "published artifact). Named by the 2026-08-27 score card; fixing it means re-uploading provably-identical "
        "files, which is a deliberate operator act, not a lint fix."
    ),
    "1.1.0rc1": "pre-release backfilled 2026-08-26; its notes carry 2 repo-relative links (same defect as v1.1.0)",
    "1.1.0": (
        "published 2026-08-26 with the correct wheel + sdist attached (sha256 verified), but its PUBLISHED notes "
        "carry 7 repo-relative links, which 404 on the Releases page. The source "
        "(docs/announcements/release_1_1_0.md) was rewritten to absolute URLs on 2026-08-29, so the next release "
        "is clean and only the already-published body is wrong; editing it is an outward-facing act left to the "
        "operator (`gh release edit v1.1.0 --notes-file docs/announcements/release_1_1_0.md`)."
    ),
}


class AuditError(RuntimeError):
    """The release audit could not be performed (exit 2) — never a pass."""


_REL_LINK = re.compile(r"\[[^\]]*\]\((?!https?://|#|mailto:)([^)\s]+)\)")


def _pypi_files(timeout: float = 30.0) -> dict[str, dict[str, str]]:
    """{version: {filename: sha256}} from PyPI."""
    try:
        with urllib.request.urlopen(PYPI_JSON, timeout=timeout) as r:  # noqa: S310 - fixed https URL
            data = json.load(r)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
        raise AuditError(f"cannot read {PYPI_JSON}: {exc}") from exc
    return {v: {f["filename"]: f["digests"]["sha256"] for f in files} for v, files in data["releases"].items() if files}


def _gh_release(tag: str) -> dict | None:
    """The GitHub Release for ``tag`` (None when there is none); raises AuditError if gh is unusable."""
    proc = subprocess.run(
        ("gh", "release", "view", tag, "--json", "assets,body,tagName"),
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        err = proc.stderr.lower()
        if "release not found" in err or "not found" in err:
            return None
        raise AuditError(f"gh release view {tag}: {proc.stderr.strip()}")
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise AuditError(f"gh release view {tag}: unreadable JSON ({exc})") from exc


def release_problems(version: str, pypi: dict[str, str]) -> list[str]:
    """Everything wrong with the release object for ``version`` (empty = clean)."""
    tag = f"v{version}"
    problems: list[str] = []
    if not _git("tag", "-l", tag).strip() and not _git("ls-remote", "--tags", "origin", tag).strip():
        problems.append(f"PyPI serves {version} but there is no {tag} tag")
    rel = _gh_release(tag)
    if rel is None:
        problems.append(f"no GitHub Release on {tag} — a tag says which commit, a Release hands over the artifact")
        return problems
    assets = {a["name"]: (a.get("digest") or "").removeprefix("sha256:") for a in rel.get("assets", [])}
    if len(assets) < 2:
        problems.append(f"{tag} Release has {len(assets)} asset(s); the exact wheel + sdist (≥2) must be attached")
    for name, sha in assets.items():
        if name not in pypi:
            problems.append(f"{tag} asset {name} is not a file PyPI serves for {version}")
        elif sha and sha != pypi[name]:
            problems.append(
                f"{tag} asset {name} sha256 {sha[:12]}… != PyPI {pypi[name][:12]}… — NOT the published artifact"
            )
        elif not sha:
            problems.append(f"{tag} asset {name} exposes no digest — cannot verify it against PyPI")
    for missing in sorted(set(pypi) - set(assets)):
        problems.append(f"{tag} Release is missing the PyPI file {missing}")
    rel_links = _REL_LINK.findall(rel.get("body") or "")
    if rel_links:
        problems.append(
            f"{tag} Release notes carry {len(rel_links)} repo-relative link(s) that 404 on the Releases page "
            f"(e.g. {rel_links[0]}) — use absolute https://github.com/... URLs"
        )
    notes_src = REPO / ANNOUNCEMENTS / f"release_{version.replace('.', '_')}.md"
    if notes_src.exists():
        src_links = _REL_LINK.findall(notes_src.read_text())
        if src_links:
            problems.append(
                f"{notes_src.relative_to(REPO)} (the --notes-file source) carries {len(src_links)} repo-relative "
                f"link(s) (e.g. {src_links[0]}) — they will 404 when published"
            )
    return problems


def audit_releases(grandfathered: dict[str, str] | None = None) -> int:
    """0 clean; 1 violations; 2 cannot check."""
    grandfathered = GRANDFATHERED_RELEASES if grandfathered is None else grandfathered
    try:
        pypi = _pypi_files()
    except AuditError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if not pypi:
        print("ERROR: PyPI reports no released files — refusing to pass vacuously", file=sys.stderr)
        return 2
    failures: list[str] = []
    stale: list[str] = []
    checked = 0
    for version in sorted(pypi):
        try:
            problems = release_problems(version, pypi[version])
        except AuditError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
        checked += 1
        if problems and version in grandfathered:
            print(f"  {version:<9} GRANDFATHERED (still failing) — {grandfathered[version]}")
            for p in problems:
                print(f"      {p}")
        elif problems:
            failures.append(f"  {version}:")
            failures.extend(f"      {p}" for p in problems)
        elif version in grandfathered:
            stale.append(f"  {version}: listed as GRANDFATHERED but now PASSES — remove the stale entry")
    for version in grandfathered:
        if version not in pypi:
            stale.append(f"  {version}: GRANDFATHERED entry names a version PyPI does not serve — remove it")
    if failures or stale:
        print("release-object audit FAILED:", file=sys.stderr)
        for line in failures + stale:
            print(line, file=sys.stderr)
        print(
            "\nEvery version PyPI serves needs a tag, a GitHub Release, the exact wheel + sdist "
            "(sha256-matching), and notes with absolute links (docs/publication_guide.md "
            '§"Create the GitHub Release on the tag").',
            file=sys.stderr,
        )
        return 1
    print(f"release-object audit: clean — {checked} PyPI version(s) checked, {len(grandfathered)} grandfathered")
    return 0


_HEADING = re.compile(r"^## \[(?P<version>\d+\.\d+\.\d+)\](?:\s*-\s*(?P<date>\d{4}-\d{2}-\d{2}))?", re.M)


def _git(*args: str) -> str:
    return subprocess.run(("git", *args), cwd=REPO, capture_output=True, text=True, check=False).stdout.strip()


def changelog_releases() -> list[tuple[str, str | None, str]]:
    """(version, date, first summary line) for each released section, newest first."""
    text = CHANGELOG.read_text()
    out: list[tuple[str, str | None, str]] = []
    matches = list(_HEADING.finditer(text))
    for i, m in enumerate(matches):
        body = text[m.end() : matches[i + 1].start() if i + 1 < len(matches) else len(text)]
        summary = ""
        for line in body.splitlines():
            line = line.strip()
            if line.startswith("- **"):
                summary = line.lstrip("- ").strip()
                summary = re.sub(r"\*\*(.+?)\*\*", r"\1", summary).split(" — ")[0].split(". ")[0]
                break
        out.append((m.group("version"), m.group("date"), summary))
    return out


def version_bump_commit(version: str) -> str | None:
    """The earliest commit that introduced this version string into a version file."""
    result = _git("log", "--reverse", "--format=%H", "-S", f'"{version}"', "--", *VERSION_FILES)
    return result.splitlines()[0] if result else None


def existing_tags() -> set[str]:
    return {t.lstrip("v") for t in _git("tag", "-l", "v*").splitlines()}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write-tags", action="store_true", help="create the missing tags locally (never pushes)")
    ap.add_argument(
        "--check-releases",
        action="store_true",
        help="also audit GitHub Release objects against PyPI (assets, sha256, absolute links); needs network + gh",
    )
    ap.add_argument(
        "--include-current",
        action="store_true",
        help="also tag the version in pyproject.toml — only correct AT publish time",
    )
    args = ap.parse_args(argv)

    if args.check_releases:
        rc = audit_releases()
        if rc:
            return rc

    current = ""
    pyproject = (REPO / "pyproject.toml").read_text()
    if m := re.search(r'^version = "(.+?)"', pyproject, re.M):
        current = m.group(1)

    tagged = existing_tags()
    missing: list[tuple[str, str | None, str, str]] = []
    unresolved: list[str] = []
    seen: set[str] = set()

    for version, released, summary in changelog_releases():
        if version in tagged:
            continue
        if version in seen:
            # A version with two CHANGELOG sections is history, not a typo:
            # 0.2.0 was drafted as 1.0.0, relabelled to a research preview, and
            # its PyPI slot was later burned by an upload+delete cycle. Tag the
            # NEWEST section (sections arrive newest-first) and say so.
            print(f"  {version:<8} DUPLICATE CHANGELOG section ignored — tagging the newest one only")
            continue
        seen.add(version)
        if version == current and not args.include_current:
            print(f"  {version:<8} SKIPPED — current in-development version; tag it at publish time")
            continue
        commit = version_bump_commit(version)
        if commit is None:
            unresolved.append(version)
            continue
        missing.append((version, released, summary, commit))

    if not missing and not unresolved:
        print("All released CHANGELOG versions have tags.")
        return 0

    print(f"\n{len(missing)} released version(s) without a tag:\n")
    for version, released, summary, commit in missing:
        subject = _git("log", "-1", "--format=%s", commit)
        print(f"  v{version:<8} {commit[:8]}  {released or '????-??-??'}  {subject[:58]}")

    if unresolved:
        print(f"\n  Could not locate a version-bump commit for: {', '.join(unresolved)}")
        print("  (version string may predate the current version files — tag by hand)")

    if not args.write_tags:
        print("\nRe-run with --write-tags to create these locally, then review with `git tag -n99`.")
        return 1

    stamp = date.today().isoformat()
    for version, released, summary, commit in missing:
        message = f"pymaxim {version}"
        if released:
            message += f" ({released})"
        if summary:
            message += f" — {summary}"
        message += f". Tag reconstructed {stamp} from the version-bump commit."
        proc = subprocess.run(
            ("git", "tag", "-a", f"v{version}", commit, "-m", message),
            cwd=REPO,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            print(f"  FAILED v{version}: {proc.stderr.strip()}", file=sys.stderr)
            return 2
        print(f"  created v{version} -> {commit[:8]}")

    print("\nTags created LOCALLY. Review (`git tag -n99 | sort -V`), then push deliberately:")
    print("  git push origin --tags")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
