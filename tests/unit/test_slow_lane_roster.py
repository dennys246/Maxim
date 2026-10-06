"""The slow-lane roster matches collection, checked at PR time (roadmap 1.3.2 item 7, #940).

`scripts/check_slow_lane.py` holds the nightly lane to `scripts/slow_lane_roster.json`; without this test a PR
that adds, removes or renames a slow test would merge green and turn the NEXT nightly red on main. This runs in the
fast suite and compares `pytest --collect-only -m slow` with the roster (adversarial design pass).
"""

from __future__ import annotations

from scripts import check_slow_lane as C


def test_the_roster_is_exactly_what_collection_selects():
    nodeids, problems = C.collect_slow()
    expected, allowed = C.load_roster()
    assert not problems, problems
    missing, extra = sorted(set(expected) - set(nodeids)), sorted(set(nodeids) - set(expected))
    assert not missing and not extra, (
        f"slow-lane roster drift: not collected {missing}, not in the roster {extra}. "
        "Run `python3 scripts/check_slow_lane.py --generate` and review the diff."
    )
    assert set(allowed) <= set(expected) and all(r.strip() for r in allowed.values())


def test_collection_is_scoped_to_files_that_apply_the_marker():
    """A module-level skip in a NON-slow file (the console modules on a box without the extra) must not fail the fast
    suite through this test (wire review); only files that apply `mark.slow` are collected."""
    files = C.slow_files()
    assert "tests/integration/test_cradle_tool_routing.py" in files
    assert not any("test_console_" in f for f in files)
    assert all(f.rsplit("/", 1)[-1].startswith("test_") for f in files)  # the configured python_files pattern


def test_a_slow_module_skipped_at_collection_is_refused_from_real_collect_output(tmp_path):
    """Executor review: under `-q` pytest leaves the skip count out of its summary, so the first version never
    refused. Real `--collect-only` output from a repo holding a slow module that skips at import."""
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_ok.py").write_text("import pytest\n\n@pytest.mark.slow\ndef test_ok():\n    pass\n")
    (tests / "test_gone.py").write_text(
        "import pytest\npytest.importorskip('no_such_pkg_xyz')\n\n@pytest.mark.slow\ndef test_gone():\n    pass\n"
    )
    (tmp_path / "pytest.ini").write_text("[pytest]\nmarkers =\n    slow: slow\n")
    nodeids, problems = C.collect_slow(tmp_path)
    assert nodeids == ["tests/test_ok.py::test_ok"]
    assert any("test_gone.py" in p and "skipped or errored at collection" in p for p in problems), problems
