"""scripts/check_slow_lane.py — the scheduled slow lane must not be vacuous.

Score card 2026-08-27, Test/CI truthfulness "Upgrade to B": a scheduled
`-m slow` lane with an `executed > 0` assert.

`pytest -m slow` exits 0 when it collects nothing AND when everything it
collected skipped. Both make the nightly job green while verifying nothing,
which is the shape `check_model_cache_lane.py` already exists to prevent for
the model lane. These tests pin that the checker actually distinguishes those
cases from a lane that ran.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts import check_slow_lane as C


def _xml(tmp_path: Path, *, tests: int, skips: int = 0, failures: int = 0, cases: str = "") -> Path:
    path = tmp_path / "results.xml"
    path.write_text(
        '<?xml version="1.0" encoding="utf-8"?>\n'
        "<testsuites>"
        f'<testsuite name="pytest" tests="{tests}" skips="{skips}" failures="{failures}" errors="0">'
        f"{cases}"
        "</testsuite></testsuites>\n",
        encoding="utf-8",
    )
    return path


def test_a_lane_that_ran_tests_passes(tmp_path: Path) -> None:
    assert C.check(_xml(tmp_path, tests=41)) == 0


def test_collecting_nothing_fails(tmp_path: Path) -> None:
    """The marker got removed from every test, or collection broke."""
    assert C.check(_xml(tmp_path, tests=0)) == 1


def test_everything_skipping_fails(tmp_path: Path) -> None:
    """41 collected, 41 skipped: pytest exits 0 and the lane verified nothing."""
    assert C.check(_xml(tmp_path, tests=41, skips=41)) == 1


def test_partial_skips_still_pass(tmp_path: Path) -> None:
    """A slow test skipping on a missing optional dep is not itself fatal."""
    assert C.check(_xml(tmp_path, tests=41, skips=40)) == 0


def test_failures_do_not_mask_the_executed_check(tmp_path: Path) -> None:
    """Tests that ran and failed still count as executed — the lane is doing
    its job; the failure is reported by pytest's own exit code."""
    assert C.check(_xml(tmp_path, tests=41, skips=0, failures=3)) == 0


def test_missing_report_fails(tmp_path: Path) -> None:
    """No XML means the lane died before writing one — never a pass."""
    assert C.check(tmp_path / "absent.xml") == 1


def test_unparsable_report_fails(tmp_path: Path) -> None:
    path = tmp_path / "results.xml"
    path.write_text("<not-xml", encoding="utf-8")
    assert C.check(path) == 1


def test_minimum_is_configurable(tmp_path: Path) -> None:
    assert C.check(_xml(tmp_path, tests=5), minimum=10) == 1
    assert C.check(_xml(tmp_path, tests=5), minimum=5) == 0


def test_skip_reasons_are_printed(tmp_path: Path, capsys) -> None:
    """An operator reading a red lane needs to see WHY things skipped."""
    case = '<testcase classname="t.test_x" name="test_y"><skipped message="needs torch"/></testcase>'
    C.check(_xml(tmp_path, tests=1, skips=1, cases=case))
    assert "needs torch" in capsys.readouterr().out


@pytest.mark.parametrize("argv,expected", [([], 2), (["a", "b", "c"], 2)])
def test_bad_usage_returns_2(argv, expected) -> None:
    assert C.main(argv) == expected
