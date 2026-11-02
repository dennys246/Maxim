#!/usr/bin/env python3
"""Fetch and verify the GGUF fixtures the observer's gates need (#1315 item 1; owner decision V17, 2026-11-01).

The fixtures live under ``tests/fixtures/advisor/``. Two are committed, because they are small:
- ``tiny_runnable.gguf``, stories260K (about 1.2 MB), the runnable model;
- ``vocab/ggml-vocab-phi-3.gguf`` (about 0.7 MB).

The other three llama.cpp vocab files (about 25 MB together) are gitignored, and this script fetches them. Every
file, committed or fetched, is pinned by URL (at a fixed revision), size and sha256 in ``FIXTURES``, and a file
that does not match is refused, never used. The gates in ``tests/integration/test_advisor_gate.py`` FAIL, never
skip, when a fixture is missing, so a checkout runs this once before the advisor gates:

    python scripts/fetch_advisor_fixtures.py          # fetch what is missing, verify everything
    python scripts/fetch_advisor_fixtures.py --check  # verify only; exit 1 on anything missing or wrong

CI runs it in the unit-test job before the tests (and before the netns sandbox, which has no network).

Sources (pinned):
- llama.cpp ``models/ggml-vocab-*.gguf`` at commit 785322d17f2d51e5185cf996a1f4e46d87640841 (MIT);
- ``ggml-org/models-moved`` (formerly ``ggml-org/models``) ``tinyllamas/stories260K.gguf`` at revision
  499bc8821c6b12b4e53c5bffcb21ec206f212d81 (MIT; Karpathy's llama2.c TinyStories model).

Exits: 0 every fixture present and verified; 1 a fixture missing (``--check``) or failing verification; 2 a
download that failed.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
FIXTURE_DIR = REPO / "tests" / "fixtures" / "advisor"

_LLAMA_CPP = "https://raw.githubusercontent.com/ggml-org/llama.cpp/785322d17f2d51e5185cf996a1f4e46d87640841/models"
_TINY = (
    "https://huggingface.co/ggml-org/models-moved/resolve/499bc8821c6b12b4e53c5bffcb21ec206f212d81/"
    "tinyllamas/stories260K.gguf"
)


@dataclass(frozen=True)
class Fixture:
    """One pinned fixture file. Not persisted and never crosses a wire (CC3 does not apply)."""

    path: str  # relative to FIXTURE_DIR
    url: str
    size: int
    sha256: str
    committed: bool


FIXTURES: tuple[Fixture, ...] = (
    Fixture(
        "tiny_runnable.gguf",
        _TINY,
        1185376,
        "270cba1bd5109f42d03350f60406024560464db173c0e387d91f0426d3bd256d",
        committed=True,
    ),
    Fixture(
        "vocab/ggml-vocab-phi-3.gguf",
        f"{_LLAMA_CPP}/ggml-vocab-phi-3.gguf",
        726019,
        "967d7190d11c4842eab697079d98d56c2116e10eb617be355a2733bfc132e326",
        committed=True,
    ),
    Fixture(
        "vocab/ggml-vocab-command-r.gguf",
        f"{_LLAMA_CPP}/ggml-vocab-command-r.gguf",
        10874545,
        "a2f8cfea952ef7c391a6d92a1c309d0bd32e36384d9b9230569a7425732f27d9",
        committed=False,
    ),
    Fixture(
        "vocab/ggml-vocab-qwen2.gguf",
        f"{_LLAMA_CPP}/ggml-vocab-qwen2.gguf",
        5928681,
        "44c2f46b715f585c6ab513970e8a006bfa5badd6108560054921cf598d154d8c",
        committed=False,
    ),
    Fixture(
        "vocab/ggml-vocab-llama-bpe.gguf",
        f"{_LLAMA_CPP}/ggml-vocab-llama-bpe.gguf",
        7818140,
        "97272e430d53bc7688f52d5e0ad8ea8f163ede9f1bbd1694feaa504797d5d96e",
        committed=False,
    ),
)


def verify(fixture: Fixture, root: Path | None = None) -> str | None:
    """``None`` when the file is present with the pinned size and sha256; otherwise why it is not."""
    path = (root or FIXTURE_DIR) / fixture.path
    if not path.is_file():
        return "missing"
    size = path.stat().st_size
    if size != fixture.size:
        return f"size {size}, pinned {fixture.size}"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != fixture.sha256:
        return f"sha256 {digest}, pinned {fixture.sha256}"
    return None


def fetch(fixture: Fixture, root: Path | None = None, *, timeout_s: float = 120.0) -> None:
    """Download ``fixture`` to a ``.part`` file, verify it, then rename it into place. A bad download is removed."""
    root = root or FIXTURE_DIR
    dest = root / fixture.path
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    try:
        with urllib.request.urlopen(fixture.url, timeout=timeout_s) as resp, open(part, "wb") as out:
            while chunk := resp.read(1 << 20):
                out.write(chunk)
        partial = Fixture(part.relative_to(root).as_posix(), fixture.url, fixture.size, fixture.sha256, False)
        problem = verify(partial, root)
        if problem:
            raise ValueError(f"{fixture.path}: the download does not match its pin ({problem})")
        os.replace(part, dest)
    finally:
        part.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--check", action="store_true", help="verify only; never download")
    args = parser.parse_args(argv)
    rc = 0
    for fixture in FIXTURES:
        problem = verify(fixture)
        if problem == "missing" and not args.check and not fixture.committed:
            try:
                fetch(fixture)
            except (OSError, ValueError) as e:
                print(f"FAIL  {fixture.path}: {e}", file=sys.stderr)
                rc = max(rc, 2)
                continue
            problem = verify(fixture)
        if problem:
            hint = "" if not fixture.committed else " (a committed fixture: restore it from git)"
            print(f"FAIL  {fixture.path}: {problem}{hint}", file=sys.stderr)
            rc = max(rc, 1)
        else:
            print(f"ok    {fixture.path}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
