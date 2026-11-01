"""scripts/rig_suite_verdict.py: a rig suite run is CALIBRATED only when every stage but the fast suite passed and the
fast suite's only failures are the known rig cadence tests; anything else keeps the suite's own exit code."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import rig_suite_verdict as V  # noqa: E402

KNOWN = V.RIG_KNOWN_FAILURES


def _log(fast: int = 1, gates: str = "0", hub: int = 0, ratchet: int = 0, failed: tuple[str, ...] = KNOWN[:2]) -> str:
    lines = [f"FAILED {f} - AssertionError: cadence" for f in failed]
    lines += [f"suite stage fast-suite: exit={fast}"]
    lines += [
        "suite stage release-gates: skipped" if gates == "skipped" else f"suite stage release-gates: exit={gates}"
    ]
    lines += [
        f"suite stage memory-hub: exit={hub}",
        f"suite stage ratchet: exit={ratchet}",
        "suite at abc: FAILED (exit 1)",
    ]
    return "\n".join(lines) + "\n"


def test_only_known_cadence_failures_with_every_other_stage_clean_is_calibrated():
    code, _, failures = V.verdict(_log(), 1)
    assert code == V.CALIBRATED and failures == sorted(KNOWN[:2])


def test_a_skipped_release_gate_twin_still_calibrates():
    assert V.verdict(_log(gates="skipped"), 1)[0] == V.CALIBRATED


@pytest.mark.parametrize(
    "log",
    [
        _log(gates="1"),  # the release gates failed: both reviewers' blocker, a fail the FAILED lines never show
        _log(gates="2"),
        _log(hub=1),
        _log(hub=3),  # INTERNALERROR, no summary line
        _log(ratchet=1),
        _log(fast=3),  # the fast suite crashed after some failures
        _log(fast=2),
        _log(failed=(*KNOWN[:1], "tests/unit/test_other.py::test_real_bug")),
        _log(failed=()),
    ],
    ids=["gates-1", "gates-2", "hub-1", "hub-internal", "ratchet", "fast-3", "fast-2", "unknown-failure", "no-failed"],
)
def test_anything_else_keeps_the_suites_own_exit_code(log):
    assert V.verdict(log, 1)[0] == 1


def test_a_missing_stage_line_is_never_calibrated():
    log = _log().replace("suite stage memory-hub: exit=0\n", "")
    code, message, _ = V.verdict(log, 1)
    assert code == 1 and "memory-hub" in message


def test_a_passing_suite_passes_and_calibrated_is_a_code_no_stage_produces():
    assert V.verdict("", 0)[0] == 0
    assert V.CALIBRATED not in range(0, 6)  # pytest 0-5, the lints 0-2


def test_the_cli_exits_with_the_verdict(tmp_path):
    log = tmp_path / "rig.log"
    log.write_text(_log())
    out = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "rig_suite_verdict.py"), str(log), "1"], capture_output=True, text=True
    )
    assert out.returncode == V.CALIBRATED and "CALIBRATED" in out.stdout and KNOWN[0] in out.stdout


def test_every_listed_cadence_test_exists_and_runs_in_the_fast_suite():
    """A stale id fails safe (that failure reads as unknown) but accumulates silently; a slow-marked one never runs on
    the rig's fast suite, so it can never be the failure it excuses."""
    files = sorted({k.split("::", 1)[0] for k in KNOWN})
    out = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "-m",
            "not slow",
            "-p",
            "no:cacheprovider",
            "-o",
            "addopts=",
            *files,
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    collected = set(out.stdout.split())
    assert not [k for k in KNOWN if k not in collected], out.stdout[-2000:]


def test_the_suite_script_prints_every_stage_the_verdict_needs():
    text = (ROOT / "scripts" / "suite_at_commit.sh").read_text(encoding="utf-8")
    assert 'stage() { echo "suite stage $1: exit=$2";' in text
    for name in V.STAGES:
        assert f"stage {name} " in text, name
    assert 'echo "suite stage release-gates: skipped"' in text
