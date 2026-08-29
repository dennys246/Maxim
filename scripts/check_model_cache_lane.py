#!/usr/bin/env python3
"""Assert the nightly model-cache lane actually EXECUTED its tests (score card: Test/CI truthfulness).

Reads the lane's junit XML and fails unless every skip is on an EXPLICIT allow-list with a
reason — ``ALLOWED_SKIPS`` for test ids, ``ALLOWED_MODULE_SKIPS`` for module-scope
``importorskip`` collection skips (empty junit classname). The previous check was
`executed >= 12`, a floor that could not see WHICH tests were vacuous; now a skip is either
named-and-explained here or it fails the lane. Adding an entry is a reviewed, visible act,
and ``tests/unit/test_model_cache_names.py`` asserts every test-id entry is in the lane's
own ``-m requires_model_cache`` selection so a stale entry cannot sit here unnoticed.

Usage: python3 scripts/check_model_cache_lane.py model-cache-results.xml
Exits: 0 all executed or allow-listed; 1 unlisted skips / nothing executed; 2 unreadable XML.
"""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

# test id → why it may skip in the offline lane (dataset-gated tests only).
# Every entry must be in the lane's own selection (`-m requires_model_cache`) —
# tests/unit/test_model_cache_names.py asserts that against a live collection, because
# the first draft of this list carried two ids the lane never collects (they have no
# marker), written by inspection rather than from a real run.
ALLOWED_SKIPS: dict[str, str] = {
    "tests/substrate/test_p4_fixture_validation.py::TestFixtureImageLoader::test_fixture_loader_image_count_matches_descriptor": (
        "needs the Flowers102 dataset cache (scripts/p4_clip_calibration_sweep.py), not a model weight"
    ),
    "tests/substrate/test_p4_fixture_validation.py::TestFixtureImageLoader::test_fixture_loader_raises_on_class_index_drift": (
        "needs the Flowers102 dataset cache"
    ),
    "tests/substrate/test_p4_fixture_validation.py::TestFixtureRetrievalGate::test_canonical_fixture_meets_forward_and_reverse_gate": (
        "needs the Flowers102 dataset cache"
    ),
    "tests/substrate/test_p4_mug_test_roundtrip.py::TestP4MugTestRoundTrip::test_real_clip_mug_test_survives_session_snapshot_round_trip": (
        "needs the Flowers102 dataset cache"
    ),
}

# Module-scope skips (`pytest.importorskip` at import time) have an EMPTY junit classname
# and are reported by MODULE, not by test id. They arrive because the lane's `pytest tests/`
# collects every module before `-m` deselects; a module whose optional import is absent
# skips at collection. Listed by module with the dependency that gates them.
ALLOWED_MODULE_SKIPS: dict[str, str] = {
    "tests.unit.test_console_event_seam": "module-scope importorskip('fastapi') — the console extra is not installed in this lane",
    "tests.unit.test_console_identity": "module-scope importorskip('fastapi')",
    "tests.unit.test_console_launcher_seams": "module-scope importorskip('fastapi')",
    "tests.unit.test_console_server": "module-scope importorskip('fastapi')",
    "tests.unit.test_console_talk": "module-scope importorskip('fastapi')",
}


def test_id(case: ET.Element) -> tuple[str, bool]:
    """(id, is_module_scope). An empty classname means a COLLECTION skip, whose `name`
    is the module path — the first draft produced ".py::tests.unit.test_console_talk"
    for those and failed the lane on its own garbage ids."""
    cls = case.get("classname", "")
    name = case.get("name", "")
    if not cls:
        return name, True
    parts = cls.split(".")
    # tests.substrate.baselines.test_baselines.TestEmbeddingBaseline → path + class
    file_parts = [p for p in parts if not p[:1].isupper()]
    class_parts = [p for p in parts if p[:1].isupper()]
    path = "/".join(file_parts) + ".py"
    return "::".join([path, *class_parts, name]), False


def check(xml_path: Path) -> int:
    try:
        root = ET.parse(xml_path).getroot()
    except (OSError, ET.ParseError) as exc:
        print(f"ERROR: cannot read {xml_path}: {exc}", file=sys.stderr)
        return 2
    cases = list(root.iter("testcase"))
    skipped = [c for c in cases if c.find("skipped") is not None]
    executed = len(cases) - len(skipped)
    print(f"model-cache tests: {executed} executed, {len(skipped)} skipped, {len(cases)} collected")
    unlisted = []
    for c in skipped:
        tid, module_scope = test_id(c)
        reason = (c.find("skipped").get("message") or "").strip()
        allowed = ALLOWED_MODULE_SKIPS if module_scope else ALLOWED_SKIPS
        if tid in allowed:
            kind = "module" if module_scope else "test"
            print(f"  allowed {kind} skip: {tid} — {allowed[tid]}")
        else:
            unlisted.append((tid, reason))
    if executed == 0:
        print("FAIL: zero model-cache tests executed — the lane is vacuous", file=sys.stderr)
        return 1
    if unlisted:
        print(
            "FAIL: skipped tests not on the explicit allow-list (scripts/check_model_cache_lane.py):", file=sys.stderr
        )
        for tid, reason in unlisted:
            print(f"  {tid}: {reason[:120]}", file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print(__doc__, file=sys.stderr)
        return 2
    return check(Path(argv[0]))


if __name__ == "__main__":
    sys.exit(main())
