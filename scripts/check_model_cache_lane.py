#!/usr/bin/env python3
"""Assert the nightly model-cache lane actually EXECUTED its tests (score card: Test/CI truthfulness).

Reads the lane's junit XML and fails unless every skipped test is on the EXPLICIT allow-list
below with a reason. The previous check was `executed >= 12`, a floor that could not see
which tests were vacuous; now a skip is either named-and-explained here or it fails the
lane. Adding a test to the list is a reviewed, visible act.

Usage: python3 scripts/check_model_cache_lane.py model-cache-results.xml
Exits: 0 all executed or allow-listed; 1 unlisted skips / nothing executed; 2 unreadable XML.
"""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

# test id → why it may skip in the offline lane (dataset-gated / missing-dep-path tests).
ALLOWED_SKIPS: dict[str, str] = {
    "tests/substrate/baselines/test_baselines.py::TestEmbeddingBaseline::test_returns_result_without_deps": (
        "tests the MISSING-dependency path; sentence-transformers is installed in this lane by design"
    ),
    "tests/substrate/baselines/test_baselines.py::TestOpenCLIPBaseline::test_returns_result_without_deps": (
        "tests the MISSING-dependency path; skips whenever open_clip is importable"
    ),
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
    "tests/substrate/test_concept_decomposition_validation.py::TestDecompositionValidation::test_decomposition_improves_cross_modal_recall": (
        "needs spaCy + en_core_web_sm, which the lane does not install (a dataset-class dependency, listed so it is visible)"
    ),
}


def test_id(case: ET.Element) -> str:
    cls = case.get("classname", "")
    parts = cls.split(".")
    # tests.substrate.baselines.test_baselines.TestEmbeddingBaseline → path + class
    file_parts = [p for p in parts if not p[:1].isupper()]
    class_parts = [p for p in parts if p[:1].isupper()]
    path = "/".join(file_parts) + ".py"
    return "::".join([path, *class_parts, case.get("name", "")])


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
        tid = test_id(c)
        reason = (c.find("skipped").get("message") or "").strip()
        if tid in ALLOWED_SKIPS:
            print(f"  allowed skip: {tid} — {ALLOWED_SKIPS[tid]}")
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
