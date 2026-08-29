"""The nightly warm list is derived from source (scripts/model_cache_names.py) and must cover
every model a marked test or the production loaders name. Verified to fail on the pre-fix
workflow's hand-kept tuple, which lacked all-MiniLM-L6-v2 (every nightly run since 2026-08-21)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from scripts import check_model_cache_lane as C
from scripts import model_cache_names as M

PRODUCTION_DEFAULTS = {"all-mpnet-base-v2", "paraphrase-mpnet-base-v2", "clip-ViT-B-32", "all-MiniLM-L6-v2"}


def test_derived_list_covers_production_defaults_and_marked_tests() -> None:
    names = set(M.model_names())
    assert PRODUCTION_DEFAULTS <= names, names
    assert "paraphrase-MiniLM-L6-v2" in names  # the test_model_comparison sweep arm


def test_marked_test_files_are_found() -> None:
    files = {p.name for p in M.marked_test_files()}
    assert "test_baselines.py" in files and "test_clip_encoder.py" in files


def test_lane_check_reads_junit_ids(tmp_path: Path) -> None:
    xml = tmp_path / "r.xml"
    xml.write_text(
        '<testsuites><testsuite tests="3" skipped="2">'
        '<testcase classname="tests.substrate.test_p4_mug_test_roundtrip.TestP4MugTestRoundTrip" name="test_real_clip_mug_test_survives_session_snapshot_round_trip"><skipped message="Flowers102 cache not present"/></testcase>'
        '<testcase classname="tests.unit.test_clip_encoder" name="test_x"><skipped message="no reason"/></testcase>'
        '<testcase classname="tests.unit.test_clip_encoder" name="test_y"/>'
        "</testsuite></testsuites>"
    )
    assert C.check(xml) == 1  # test_x skipped and not allow-listed
    xml.write_text(
        '<testsuites><testsuite tests="2" skipped="1">'
        '<testcase classname="tests.substrate.test_p4_mug_test_roundtrip.TestP4MugTestRoundTrip" name="test_real_clip_mug_test_survives_session_snapshot_round_trip"><skipped message="Flowers102 cache not present"/></testcase>'
        '<testcase classname="tests.unit.test_clip_encoder" name="test_y"/>'
        "</testsuite></testsuites>"
    )
    assert C.check(xml) == 0


def test_module_scope_collection_skip_is_recognised(tmp_path: Path) -> None:
    """Empty classname = a collection skip whose `name` is the module path. The first
    draft produced ".py::tests.unit.test_console_talk" and failed the lane on its own id."""
    xml = tmp_path / "m.xml"
    xml.write_text(
        '<testsuite tests="2" skipped="1">'
        '<testcase classname="" name="tests.unit.test_console_talk"><skipped message="could not import fastapi"/></testcase>'
        '<testcase classname="tests.unit.test_clip_encoder" name="test_y"/>'
        "</testsuite>"
    )
    assert C.check(xml) == 0
    xml.write_text(
        '<testsuite tests="2" skipped="1">'
        '<testcase classname="" name="tests.unit.test_not_allow_listed"><skipped message="nope"/></testcase>'
        '<testcase classname="tests.unit.test_clip_encoder" name="test_y"/>'
        "</testsuite>"
    )
    assert C.check(xml) == 1


def test_lane_check_refuses_zero_executed(tmp_path: Path) -> None:
    xml = tmp_path / "r.xml"
    xml.write_text(
        '<testsuite tests="1" skipped="1"><testcase classname="tests.unit.t" name="a"><skipped/></testcase></testsuite>'
    )
    assert C.check(xml) == 1


def _lane_selection() -> set[str]:
    """The ids the nightly lane actually selects (`pytest -m requires_model_cache`)."""
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/",
            "-m",
            "requires_model_cache",
            "--collect-only",
            "-q",
            # clears the repo's `-v`, which turns --collect-only -q into a tree
            # view with no flat ids (the first draft silently collected nothing)
            "-o",
            "addopts=",
            "-p",
            "no:cacheprovider",
        ],
        cwd=M.REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return {line.split("[")[0].strip() for line in proc.stdout.splitlines() if "::" in line}


def test_every_allow_listed_test_is_actually_collected_by_the_lane() -> None:
    """The first draft listed two ids the lane never collects (no marker) — written by
    inspection instead of from a real run, and nothing could detect it."""
    selected = _lane_selection()
    assert selected, "collection produced no ids — the assertion below would be vacuous"
    missing = sorted(tid for tid in C.ALLOWED_SKIPS if tid not in selected)
    assert not missing, f"allow-listed ids the lane never collects (stale entries): {missing}"


def test_module_skip_allow_list_names_real_modules() -> None:
    for mod in C.ALLOWED_MODULE_SKIPS:
        assert (M.REPO_ROOT / (mod.replace(".", "/") + ".py")).exists(), mod
