"""Each nightly lane's roster matches collection, checked at PR time (roadmap 1.3.2 item 7, #940; #1117).

`scripts/check_lane_roster.py` holds each nightly lane to `scripts/lane_rosters/<lane>.json`; without this test a PR
that adds, removes or renames a lane's test would merge green and turn the NEXT nightly red on main. This runs in the
fast suite and compares file-scoped `pytest --collect-only -m <marker>` with the roster (adversarial design pass). A
marker applied some other way (a conftest hook, `getattr(pytest.mark, ...)`) is invisible to the file scan and is
caught only by the nightly, as "ran, not in the roster".
"""

from __future__ import annotations

import pytest

from scripts import check_lane_roster as C

# An excused skip is a reviewed act: raise the pin in the same PR and say why the test cannot run in the lane
# (the model-cache pin carries the 1.3.1 `ALLOWED_SKIPS <= 4` ratchet from tests/unit/test_model_cache_names.py).
ALLOWED_SKIP_PINS = {"slow": 1, "model-cache": 4}


@pytest.mark.parametrize("lane", sorted(C.LANES))
def test_the_roster_is_exactly_what_collection_selects(lane):
    nodeids, problems = C.collect(lane)
    expected, allowed = C.load_roster(C.roster_path(lane))
    assert not problems, problems
    missing, extra = sorted(set(expected) - set(nodeids)), sorted(set(nodeids) - set(expected))
    assert not missing and not extra, (
        f"{lane}-lane roster drift: not collected {missing}, not in the roster {extra}. "
        f"Run `python3 scripts/check_lane_roster.py --lane {lane} --generate` and review the diff."
    )
    assert set(allowed) <= set(expected) and all(r.strip() for r in allowed.values())


@pytest.mark.parametrize("lane", sorted(C.LANES))
def test_the_allowed_skips_cannot_grow_silently(lane):
    _expected, allowed = C.load_roster(C.roster_path(lane))
    assert len(allowed) <= ALLOWED_SKIP_PINS[lane], (
        f"the {lane} lane's allowed_skips grew: an excused skip is a reviewed decision -- raise ALLOWED_SKIP_PINS in "
        "the same PR and say why the test cannot run there"
    )


def test_the_model_cache_lane_excuses_only_dataset_gated_tests():
    """Carried from check_model_cache_lane.py (1.3.1): the lane installs every extra and warms every model, so the
    only excusable skip is a dataset it does not download."""
    _expected, allowed = C.load_roster(C.roster_path("model-cache"))
    assert all("dataset" in reason for reason in allowed.values()), allowed


def test_every_lane_has_a_roster_and_no_roster_is_orphaned():
    assert {p.stem for p in C.ROSTERS.glob("*.json")} == set(C.LANES)


def test_collection_is_scoped_to_files_that_apply_the_marker():
    """A module-level skip in a NON-lane file (the console modules on a box without the extra) must not fail the fast
    suite through this test (wire review); only files that apply the lane's marker are collected."""
    slow = C.marked_files("slow")
    assert "tests/integration/test_cradle_tool_routing.py" in slow
    assert "tests/unit/test_clip_encoder.py" in C.marked_files("requires_model_cache")
    for marker in C.LANES.values():
        files = C.marked_files(marker)
        assert not any("test_console_" in f for f in files)
        assert all(f.rsplit("/", 1)[-1].startswith("test_") for f in files)  # the configured python_files pattern


def test_a_module_skipped_at_collection_is_refused_from_real_collect_output(tmp_path, monkeypatch):
    """Executor review (#940): under `-q` pytest leaves the skip count out of its summary, so the first version never
    refused. Real `--collect-only` output from a repo holding a marked module that skips at import."""
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_ok.py").write_text("import pytest\n\n@pytest.mark.slow\ndef test_ok():\n    pass\n")
    (tests / "test_gone.py").write_text(
        "import pytest\npytest.importorskip('no_such_pkg_xyz')\n\n@pytest.mark.slow\ndef test_gone():\n    pass\n"
    )
    (tmp_path / "pytest.ini").write_text("[pytest]\nmarkers =\n    slow: slow\n")
    nodeids, problems = C.collect("slow", tmp_path)
    assert nodeids == ["tests/test_ok.py::test_ok"]
    assert any("test_gone.py" in p and "skipped or errored at collection" in p for p in problems), problems
