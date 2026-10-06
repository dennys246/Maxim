"""scripts/check_slow_lane.py — the nightly slow lane runs EXACTLY its roster (roadmap 1.3.2 item 7, #940).

The old check was `executed > 0`: on the 2026-10-05 nightly 46 slow tests were selected, 16 ran, and the lane was
green. These tests pin each way the roster check fails, against a synthetic JUnit report and roster.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import check_slow_lane as C

A = "tests/unit/test_a.py::test_one"
B = "tests/substrate/test_b.py::TestB::test_two[seed-1]"
SMOKE = "tests/integration/test_c.py::test_live_llm"


def _case(nodeid: str, outcome: str = "ran", message: str = "") -> str:
    classname, name = C.junit_key(nodeid)
    inner = {"ran": "", "skipped": f'<skipped message="{message}"/>', "error": '<error message="boom"/>'}[outcome]
    return f'<testcase classname="{classname}" name="{name}">{inner}</testcase>'


def _run(tmp_path: Path, cases: list[str], *, expected=(A, B, SMOKE), allowed=None) -> int:
    allowed = {SMOKE: "needs a live LLM"} if allowed is None else allowed
    roster = tmp_path / "roster.json"
    roster.write_text(json.dumps({"expected": list(expected), "allowed_skips": allowed}))
    xml = tmp_path / "r.xml"
    xml.write_text(f'<testsuites><testsuite name="pytest">{"".join(cases)}</testsuite></testsuites>')
    return C.check(xml, roster)


def test_the_exact_roster_with_its_allowed_skip_passes(tmp_path: Path) -> None:
    assert _run(tmp_path, [_case(A), _case(B), _case(SMOKE, "skipped", "no LLM")]) == 0


def test_an_unlisted_skip_FAILS(tmp_path: Path, capsys) -> None:
    """The 2026-10-05 shape: tests skipping for a missing extra while the lane stays green."""
    assert _run(tmp_path, [_case(A), _case(B, "skipped", "sentence-transformers not installed"), _case(SMOKE)]) == 1
    assert "unlisted skip" in capsys.readouterr().err


def test_a_roster_test_that_did_not_run_FAILS(tmp_path: Path, capsys) -> None:
    assert _run(tmp_path, [_case(A), _case(SMOKE, "skipped", "x")]) == 1
    assert f"in the roster, not run: {B}" in capsys.readouterr().err


def test_a_test_outside_the_roster_FAILS(tmp_path: Path, capsys) -> None:
    extra = "tests/unit/test_new.py::test_new"
    assert _run(tmp_path, [_case(A), _case(B), _case(SMOKE, "skipped", "x"), _case(extra)]) == 1
    assert "ran, not in the roster: tests.unit.test_new::test_new" in capsys.readouterr().err


def test_a_module_level_collection_skip_FAILS(tmp_path: Path, capsys) -> None:
    """A module skipped at collection drops out of both sets and would make the comparison vacuous."""
    module_skip = (
        '<testcase classname="" name="tests.unit.test_console_x"><skipped message="collection skipped"/></testcase>'
    )
    assert _run(tmp_path, [_case(A), _case(B), _case(SMOKE, "skipped", "x"), module_skip]) == 1
    assert "module-level collection skipped" in capsys.readouterr().err


def test_a_stale_or_reasonless_allowed_skip_FAILS(tmp_path: Path, capsys) -> None:
    cases = [_case(A), _case(B), _case(SMOKE, "skipped", "x")]
    assert _run(tmp_path, cases, allowed={SMOKE: "x", "tests/gone.py::test_gone": "x"}) == 1
    assert "is not in expected (stale)" in capsys.readouterr().err
    assert _run(tmp_path, cases, allowed={SMOKE: "  "}) == 1
    assert "has no reason" in capsys.readouterr().err


def test_nothing_executed_FAILS(tmp_path: Path) -> None:
    assert _run(tmp_path, [_case(A, "skipped", "x")], expected=(A,), allowed={A: "x"}) == 1


def test_an_errored_test_counts_as_reported_not_skipped(tmp_path: Path) -> None:
    """An error is the lane's own red (pytest exits non-zero); the roster check only asks that it ran."""
    assert _run(tmp_path, [_case(A, "error"), _case(B), _case(SMOKE, "skipped", "x")]) == 0


@pytest.mark.parametrize("name", ["absent.xml", "bad.xml"])
def test_an_unreadable_report_is_exit_2(tmp_path: Path, name: str) -> None:
    roster = tmp_path / "roster.json"
    roster.write_text(json.dumps({"expected": [A], "allowed_skips": {}}))
    if name == "bad.xml":
        (tmp_path / name).write_text("<not xml")
    assert C.check(tmp_path / name, roster) == 2


def test_junit_keys_match_pytests_own_writer() -> None:
    assert C.junit_key(B) == ("tests.substrate.test_b.TestB", "test_two[seed-1]")
    assert C.junit_key("tests/unit/x.py::test_y[a/b.py-c]") == ("tests.unit.x", "test_y[a/b.py-c]")


def test_bad_usage_returns_2() -> None:
    assert C.main([]) == 2


def test_an_xfail_is_a_run_not_a_skip(tmp_path: Path) -> None:
    """A strict red gate (#1120) runs and fails as expected; JUnit files it under <skipped type="pytest.xfail">."""
    classname, name = C.junit_key(B)
    xfail = f'<testcase classname="{classname}" name="{name}"><skipped type="pytest.xfail" message="#1120"/></testcase>'
    assert _run(tmp_path, [_case(A), xfail, _case(SMOKE, "skipped", "x")]) == 0
