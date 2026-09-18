#!/usr/bin/env python3
"""Shared Paper-server setup helpers for the experiment world builders.

Used by ``scripts/exp56/setup_world.py`` and ``scripts/survival_world/setup_world.py`` (and the
next builder — Exp 62's two-pool world). Extracted 2026-09-18 during the Exp 56 re-baseline port,
when the two builders had already diverged in BOTH directions: exp56 verified the jar's sha256 and
survival did not; survival unlinked a truncated download and refused, exp56 warned and carried on
(so the NEXT run's ``jar.is_file()`` treated the truncated jar as complete). The band-aid trigger
("a fix that would need to be repeated elsewhere") was met literally. Nothing here is a science
constant — versions, ports and coordinates stay in the builders.

Imported by path from ``scripts/`` (the ``_provenance.py`` precedent), so it always comes from the
same tree as the builder that calls it. ``maxim`` imports are lazy so ``--help`` works without
``PYTHONPATH``.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

PAPER_API_TMPL = "https://fill.papermc.io/v3/projects/paper/versions/{mc}/builds/latest"
STAMP_NAME = "world_version.json"
_MAJOR_RE = re.compile(r'"(\d+)(?:\.(\d+))?')


def java_version_line() -> str | None:
    """The first line of ``java -version`` on PATH, or None when there is no java."""
    java = shutil.which("java")
    if not java:
        return None
    out = subprocess.run([java, "-version"], capture_output=True, text=True)
    text = out.stderr or out.stdout
    return text.splitlines()[0] if text else "unknown"


def java_major(version_line: str | None) -> int | None:
    """The Java major from a ``java -version`` first line: ``"1.8.0_392"`` → 8, ``"11.0.2"`` → 11,
    ``"16" 2021-03-16`` → 16, ``"17.0.9"`` → 17, ``"15-ea"`` → 15. None when unparseable (the macOS
    no-JDK stub prints a sentence, not a version)."""
    if not version_line:
        return None
    m = _MAJOR_RE.search(version_line)
    if not m:
        return None
    first, second = int(m.group(1)), m.group(2)
    if first == 1 and second is not None:  # the legacy "1.x" scheme (Java 8 and older)
        return int(second)
    return first


def java_start_command(server_dir: Path, jar_name: str, *, required_major: int = 17) -> str:
    """The start command with the Java binary PINNED per invocation — a shell whose default
    ``java`` is the exp56 Java 11 (or a Java 8 stub) must not boot a Paper that needs 17."""
    return f'(cd {server_dir} && "$(/usr/libexec/java_home -v {required_major})/bin/java" -jar {jar_name} nogui)'


def refuse_stale_world(server_dir: Path, world_dir_name: str, mc_version: str, *, force: bool) -> str | None:
    """The stale-world guard: booting a new Paper over a world another version generated silently
    UPGRADES it in place (mixed-generation chunks; the seed no longer reproduces it). Returns the
    refusal message when the world folder exists and is not stamped for ``mc_version``; None when
    setup may proceed. ``force`` accepts the in-place upgrade deliberately."""
    world_dir = server_dir / world_dir_name
    if not world_dir.exists() or force:
        return None
    stamp_path = server_dir / STAMP_NAME
    stamped: str | None = None
    corrupt = False
    if stamp_path.exists():
        try:
            stamped = json.loads(stamp_path.read_text()).get("mc_version")
        except (OSError, ValueError):
            corrupt = True
    if stamped == mc_version:
        return None
    if corrupt:
        held = "a world whose version stamp is unreadable"
    elif stamped:
        held = f"a world stamped MC {stamped}"
    else:
        held = "a world with no version stamp (it predates this guard)"
    return (
        f"Refusing setup: {world_dir} holds {held}, but this script targets MC {mc_version}.\n"
        f"Booting would silently upgrade the world in place (mixed-generation chunks; the seed no\n"
        f"longer reproduces it). Use a FRESH --dir (recommended), or pass --force-world to accept it."
    )


def write_world_stamp(server_dir: Path, mc_version: str, *, stamped_by: str) -> Path:
    """Write the version stamp the guard reads — atomically, with the repo's ``_format_version``
    (a dev-tool stamp, not maxim state, but the persisted-JSON rule still applies)."""
    from maxim.utils.atomic_io import atomic_write_json
    from maxim.utils.format_version import with_format_version

    path = server_dir / STAMP_NAME
    atomic_write_json(path, with_format_version({"mc_version": mc_version, "stamped_by": stamped_by}))
    return path


def download_paper(mc_version: str, jar: Path) -> bool:
    """Download the latest Paper build for ``mc_version`` to ``jar`` and verify its sha256.
    Returns True when the jar is present and verified. On ANY failure the partial jar is removed
    (so the next run re-downloads instead of booting a truncated jar) and False is returned after
    printing the manual-download hint."""
    if jar.is_file():
        print(f"paper jar already present: {jar}")
        return True
    try:
        from maxim.utils.http import download_to_file, fetch_url

        build = json.loads(fetch_url(PAPER_API_TMPL.format(mc=mc_version), timeout=30).content)
        dl = build["downloads"]["server:default"]
        print(f"downloading Paper {mc_version} build {build['id']} ({dl['size']} bytes) …")
        download_to_file(dl["url"], jar, expected_bytes=int(dl["size"]))
        digest = hashlib.sha256(jar.read_bytes()).hexdigest()
        want = dl["checksums"]["sha256"]
        if digest != want:
            raise RuntimeError(f"sha256 mismatch: got {digest}, want {want}")
        print(f"downloaded + sha256-verified: {jar}")
        return True
    except Exception as exc:  # noqa: BLE001 — every failure mode ends the same way: no jar, say so
        jar.unlink(missing_ok=True)
        print(
            f"WARNING: automatic Paper download failed ({type(exc).__name__}: {exc}).\n"
            f"Download Paper {mc_version} manually from https://papermc.io/downloads/all "
            f"and save it as {jar}"
        )
        return False


def server_version_matches(reply: str | None, mc_version: str) -> bool:
    """Whether Paper's RCON ``version`` reply names ``mc_version`` — e.g.
    ``"This server is running Paper version git-Paper-499 (MC: 1.20.4) …"``. A None/empty reply is
    a mismatch (fail closed): a campaign row must carry the platform it ran on, measured."""
    return bool(reply) and f"(MC: {mc_version})" in reply
