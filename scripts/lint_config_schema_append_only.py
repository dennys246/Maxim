#!/usr/bin/env python3
"""Diff-scoped lint: the pinned config.json schema is APPEND-ONLY (#856).

``tests/fixtures/config_schema_by_version.json`` pins every config field path per
``CONFIG_FORMAT_VERSION``; ``tests/unit/test_config_format_version_856.py`` fails when the live
schema differs from the entry for the current version. The obvious wrong response to that failure
is to add the new path to the EXISTING entry, which passes the test and silently re-creates #856
(a same-version file an older build refuses). This lint makes the append-only rule mechanical:
against the merge-base with origin/main, every version entry that existed there must be byte-for-
byte the same list. New versions may be added freely.

On a pull request a skip becomes a hard error (the check IS the gate).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lint_git import GitUnavailable, base_ref, must_not_skip, show  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE = "tests/fixtures/config_schema_by_version.json"
# What each version MEANS (how it resolves) is pinned the same way: a change of meaning is a new version too.
RESOLUTION_FIXTURE = "tests/fixtures/config_resolution_by_version.json"
FIXTURES = (FIXTURE, RESOLUTION_FIXTURE)


def _entries(text: str) -> dict[str, list[str]]:
    data = json.loads(text) if text.strip() else {}
    return {k: v for k, v in data.items() if not k.startswith("_")}


def verdict(base_text: str, head_text: str) -> tuple[bool, str]:
    """Pure decision (git-free, unit-testable): (ok, message)."""
    base, head = _entries(base_text), _entries(head_text)
    if not base:
        return True, "no pinned schema at the merge-base -- N/A"
    changed = [v for v in base if head.get(v) != base[v]]
    if changed:
        return False, (
            f"pinned config schema entries {changed} were edited or removed. The file is append-only: "
            "a schema change bumps CONFIG_FORMAT_VERSION and ADDS a new entry (#856)."
        )
    return True, f"{len(base)} existing entr{'y' if len(base) == 1 else 'ies'} unchanged"


def check() -> int:
    try:
        base = base_ref(REPO_ROOT)
    except GitUnavailable as exc:
        if must_not_skip(str(exc)):
            return 1
        print(f"INFO: no base ref available; skipping config-schema append-only check ({exc})")
        return 0
    failed = False
    for fixture in FIXTURES:
        ok, msg = verdict(show(REPO_ROOT, base, fixture), (REPO_ROOT / fixture).read_text(encoding="utf-8"))
        print(f"config-schema append-only ({fixture}): {'clean' if ok else 'FAIL'} -- {msg}")
        failed = failed or not ok
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(check())
