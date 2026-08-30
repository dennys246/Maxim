#!/usr/bin/env python
"""Assert the scheduled `-m slow` lane actually EXECUTED something.

Score card 2026-08-27, Test/CI-truthfulness "Upgrade to B": "a scheduled
`-m slow` lane with an `executed > 0` assert".

WHY THE ASSERT IS THE POINT
---------------------------
`@pytest.mark.slow` has been a registered marker and applied across ~10 files
for months, and the ONLY place the word appeared in CI was the fast suite's
`-m "not slow"` — a deselection. Nothing ever ran them. Adding a lane that runs
them is half the fix; the other half is making a GREEN lane mean something.

`pytest -m slow` exits 0 when it collects nothing at all, and exits 0 when every
test it collected skipped. Either way the lane is green and has verified nothing
— the same vacuous-guard shape `check_model_cache_lane.py` was written for, and
which that job's comments call "the vacuous-guard failure this lane exists to
catch". A lane that cannot go red for the reason it exists is decoration.

So: parse the JUnit XML, count tests that actually ran (total minus skipped),
and fail if that is zero. Skips are reported but not themselves fatal — a slow
test can legitimately skip on a missing optional dependency, and unlike the
model-cache lane there is no curated allow-list here yet. What must never happen
is the lane reporting success having run nothing.
"""

from __future__ import annotations

import sys
from pathlib import Path
from xml.etree import ElementTree


def check(xml_path: Path, *, minimum: int = 1) -> int:
    if not xml_path.is_file():
        print(f"ERROR: no JUnit XML at {xml_path} — the lane did not produce a report", file=sys.stderr)
        return 2

    try:
        root = ElementTree.parse(xml_path).getroot()
    except ElementTree.ParseError as exc:
        # Exit 2 == "could not evaluate"; exit 1 == "evaluated, and the lane
        # is vacuous". This matches check_model_cache_lane.py::check, which
        # also returns 2 for an unreadable report. The two checkers stay
        # separate mechanisms on purpose — that one carries a skip allow-list,
        # this one does not — but they must not disagree on what an outcome
        # MEANS (pre-merge review, architecture lens).
        print(f"ERROR: could not parse {xml_path}: {exc}", file=sys.stderr)
        return 2

    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    total = sum(int(s.get("tests", 0)) for s in suites)
    skipped = sum(int(s.get("skips", s.get("skipped", 0))) for s in suites)
    errors = sum(int(s.get("errors", 0)) for s in suites)
    failures = sum(int(s.get("failures", 0)) for s in suites)
    executed = total - skipped

    print(f"slow lane: {total} collected, {skipped} skipped, {executed} executed, {failures} failed, {errors} errored")

    if skipped:
        for case in root.iter("testcase"):
            if case.find("skipped") is not None:
                name = f"{case.get('classname', '')}::{case.get('name', '')}"
                reason = (case.find("skipped").get("message") or "").strip()
                print(f"  skipped: {name}: {reason[:120]}")

    if executed < minimum:
        print(
            f"FAIL: the slow lane executed {executed} test(s), expected at least {minimum}. "
            "A green lane that ran nothing verifies nothing — either the marker was removed "
            "from every test, collection broke, or everything skipped.",
            file=sys.stderr,
        )
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not 1 <= len(argv) <= 2:
        print(__doc__, file=sys.stderr)
        print("usage: check_slow_lane.py <junit.xml> [minimum_executed]", file=sys.stderr)
        return 2
    minimum = int(argv[1]) if len(argv) == 2 else 1
    return check(Path(argv[0]), minimum=minimum)


if __name__ == "__main__":
    sys.exit(main())
