#!/usr/bin/env python3
"""The verdict of a rig suite run (scripts/rig_suite.sh): PASS, CALIBRATED, or the suite's own failure.

    python scripts/rig_suite_verdict.py <log-file> <suite-exit>     # prints the verdict, exits with its code

The rig is a second machine that runs ``scripts/suite_at_commit.sh``. A few real-time cadence tests fail there even
alone and pass on the development box and in CI (macOS timer coalescing over SSH, calibrated 2026-10-27).
CALIBRATED (exit :data:`CALIBRATED`) means every stage passed except the fast suite, whose only failures are those
tests. It is not a pass: rig_suite.sh then re-runs exactly those tests locally on the same commit, and only that
re-run passing makes the run a pass.

The verdict reads only positive markers: the ``suite stage <name>: exit=N`` line suite_at_commit.sh prints for each
stage, and pytest's ``FAILED <nodeid>`` summary lines. A stage with no line, a stage other than the fast suite that
did not exit 0, a fast suite that ended other than 1 (tests failed), any ERROR line (a fixture failed), or any
failure outside the list yields the
suite's own exit code. Stdlib only.
"""

from __future__ import annotations

import re
import sys

# Exit code of a CALIBRATED verdict: no stage of suite_at_commit.sh produces it (pytest exits 0-5, the lints 0-2).
CALIBRATED = 10

# Fast-suite tests that fail on the rig even alone and pass on the development box and in CI. Kept honest by
# tests/unit/test_rig_suite_verdict.py: each id must exist and must not be slow (a slow test never runs there).
RIG_KNOWN_FAILURES: tuple[str, ...] = (
    "tests/unit/test_b1_floor_run.py::test_run_arm_seed_end_to_end_writes_a_serialisable_row_and_tears_down[q_explore_prior]",
    "tests/unit/test_b1_floor_run.py::test_run_arm_seed_end_to_end_writes_a_serialisable_row_and_tears_down[random]",
    "tests/unit/test_leader_proxy.py::TestKeepaliveEmitter::test_emits_at_configured_cadence",
    "tests/unit/test_water_trial_smoke.py::test_water_trial_ticks_acts_and_the_staging_close_persists_fear",
    "tests/unit/test_water_trial_smoke.py::test_donor_sequence_over_the_scripted_bridge_passes_the_staged_sanity",
)

# Every stage a rig run must report (the rig always measures coverage, so the ratchet stage is required).
STAGES = ("fast-suite", "release-gates", "memory-hub", "ratchet")
_STAGE = re.compile(r"^suite stage ([a-z-]+): (?:exit=(\d+)|(skipped))$")
_FAILED = re.compile(r"^FAILED (\S+)")
_ERROR = re.compile(r"^ERROR (\S+)")  # a fixture failed in setup or teardown: pytest exits 1 and prints ERROR


def _fast_suite_summary(log: str) -> list[str]:
    """The fast suite's ``short test summary info`` lines: after its header, up to the fast-suite stage line. Empty
    when there is none, which never calibrates."""
    out: list[str] = []
    inside = False
    for line in log.splitlines():
        if line.startswith("suite stage fast-suite:"):
            break
        if "short test summary info" in line:
            inside, out = True, []
        elif inside:
            out.append(line)
    return out


def verdict(log: str, suite_exit: int, known: tuple[str, ...] = RIG_KNOWN_FAILURES) -> tuple[int, str, list[str]]:
    """``(exit code, message, calibrated failures)``. The failures list is non-empty only for CALIBRATED."""
    if suite_exit == 0:
        return 0, "rig suite: PASS", []
    stages: dict[str, int | None] = {}
    for line in log.splitlines():
        m = _STAGE.match(line.rstrip())
        if m:
            stages[m.group(1)] = None if m.group(3) else int(m.group(2))
    missing = [s for s in STAGES if s not in stages]
    if missing:
        return suite_exit, f"rig suite: FAILED (exit {suite_exit}); no result for stage(s) {', '.join(missing)}", []
    # A skipped stage (None) is the release-gate twin at a commit from before it existed; any other nonzero fails.
    others = {s: stages[s] for s in STAGES if s != "fast-suite" and stages[s] not in (0, None)}
    if others:
        failed = ", ".join(f"{s} exit={c}" for s, c in others.items())
        return suite_exit, f"rig suite: FAILED (exit {suite_exit}); {failed}", []
    if stages["fast-suite"] != 1:
        return suite_exit, f"rig suite: FAILED (exit {suite_exit}); the fast suite exited {stages['fast-suite']}", []
    # Only the fast suite's short summary: a test's captured output may itself start a line with FAILED or ERROR.
    summary = _fast_suite_summary(log)
    failures = sorted({m.group(1) for line in summary if (m := _FAILED.match(line))})
    errors = sorted({m.group(1) for line in summary if (m := _ERROR.match(line))})
    # An ERROR is never calibrated, even on a listed test: the list excuses cadence assertions, not broken fixtures.
    unknown = [f for f in failures if f not in known] + [f"{e} (ERROR)" for e in errors]
    if not failures or unknown:
        why = (
            "no FAILED lines to judge"
            if not failures
            else "failures outside the calibrated list: " + ", ".join(unknown)
        )
        return suite_exit, f"rig suite: FAILED (exit {suite_exit}); {why}", []
    return CALIBRATED, f"rig suite: CALIBRATED (exit {CALIBRATED}): only known rig cadence tests failed", failures


def main(argv: list[str]) -> int:
    if len(argv) != 2 or not argv[1].isdigit():
        print("usage: rig_suite_verdict.py <log-file> <suite-exit>", file=sys.stderr)
        return 2
    with open(argv[0], encoding="utf-8", errors="replace") as f:
        code, message, failures = verdict(f.read(), int(argv[1]))
    print(message)
    for nodeid in failures:
        print(f"  {nodeid}")
    return code


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
