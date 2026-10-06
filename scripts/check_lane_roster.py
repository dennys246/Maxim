#!/usr/bin/env python
"""Assert a scheduled test lane ran EXACTLY its roster (roadmap 1.3.2 item 7, #940; one checker for both lanes, #1117).

Lanes (``LANES``, lane → pytest marker): ``slow`` (``-m slow``) and ``model-cache`` (``-m requires_model_cache``).
The model-cache lane used to have its own checker, ``check_model_cache_lane.py``: a skip allow-list with no exact
set, so a marked test that dropped out of the lane (a renamed marker, broken collection) passed. Now both lanes
are held to the rules below.

The slow lane's history:

History: until 2026-08-30 nothing ran `@pytest.mark.slow` tests at all; then a nightly lane with an
`executed > 0` floor. The floor was too weak: on the 2026-10-05 nightly, 46 tests were selected and 16 ran.
The other 30 skipped, mostly for the semantic extra and the model cache the lane never installed. The lane
was green, and the 2026-09-27 Codex card named it ("the slow lane's execution floor is only one test, not its
expected roster"). Owner decision 2026-10-04: the lane INSTALLS and RUNS its tests, and this check holds it
to the exact roster.

``scripts/lane_rosters/<lane>.json`` = ``{"expected": [nodeid, …], "allowed_skips": {nodeid: reason}}`` and no
other key (there is no module-level allow-list: a module skipped at collection always fails).
Each roster nodeid is mapped FORWARD to JUnit's ``(classname, name)`` with pytest's own
``mangle_test_address`` (never the reverse, which needs a guess about classes vs paths). Fails when:

- a JUnit testcase is a module-level COLLECTION skip or error (empty classname). A module skipped at
  collection both when the roster is generated and in CI drops out of both sets, which would make the
  comparison vacuous;
- the collected set differs from ``expected`` (a missing test, an extra test, a renamed marker, broken
  collection);
- a skipped test is not in ``allowed_skips``, an ``allowed_skips`` key is not in ``expected``, or a reason is
  empty;
- nothing executed.

Exit 0 clean; 1 the lane did not run its roster; 2 the report or roster is unreadable.
``--lane <lane> --generate`` rewrites ``expected`` from ``pytest --collect-only -m <marker>`` (keeping
``allowed_skips``) and refuses while collection reports a module-level skip or error. The PR-time twin of the roster
comparison is ``tests/unit/test_lane_rosters.py``, which compares collection with ``expected`` in the fast suite, so
a PR adding or removing a lane's test fails before the nightly does.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from xml.etree import ElementTree

from _pytest.junitxml import mangle_test_address

REPO = Path(__file__).resolve().parent.parent
ROSTERS = REPO / "scripts" / "lane_rosters"
LANES = {"slow": "slow", "model-cache": "requires_model_cache"}  # lane -> the pytest marker that selects it
_ROSTER_KEYS = {"expected", "allowed_skips"}


def roster_path(lane: str) -> Path:
    return ROSTERS / f"{lane}.json"


def junit_key(nodeid: str) -> tuple[str, str]:
    """``(classname, name)`` exactly as pytest's JUnit writer derives them from ``nodeid``."""
    parts = mangle_test_address(nodeid)
    return ".".join(parts[:-1]), parts[-1]


def load_roster(path: Path) -> tuple[list[str], dict[str, str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or set(data) != _ROSTER_KEYS:
        raise ValueError(f"a roster holds exactly {sorted(_ROSTER_KEYS)}")
    expected, allowed = data["expected"], data["allowed_skips"]
    if not isinstance(expected, list) or not all(isinstance(n, str) for n in expected):
        raise ValueError("expected must be a list of nodeids")
    if not isinstance(allowed, dict):
        raise ValueError("allowed_skips must be a mapping of nodeid to reason")
    return expected, allowed


def check(xml_path: Path, roster: Path) -> int:
    try:
        expected, allowed = load_roster(roster)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"ERROR: roster {roster} unreadable: {exc}", file=sys.stderr)
        return 2
    if not xml_path.is_file():
        print(f"ERROR: no JUnit XML at {xml_path}: the lane did not produce a report", file=sys.stderr)
        return 2
    try:
        root = ElementTree.parse(xml_path).getroot()
    except ElementTree.ParseError as exc:
        print(f"ERROR: could not parse {xml_path}: {exc}", file=sys.stderr)
        return 2

    problems: list[str] = []
    for nodeid, reason in allowed.items():
        if nodeid not in expected:
            problems.append(f"allowed skip {nodeid} is not in expected (stale)")
        if not isinstance(reason, str) or not reason.strip():
            problems.append(f"allowed skip {nodeid} has no reason")

    want = {junit_key(n): n for n in expected}
    got: dict[tuple[str, str], str] = {}  # key -> outcome
    for case in root.iter("testcase"):
        classname, name = case.get("classname", ""), case.get("name", "")
        skipped, errored = case.find("skipped"), case.find("error")
        if skipped is not None and skipped.get("type") == "pytest.xfail":
            skipped = None  # an xfail RAN (a strict red gate, e.g. #1120); JUnit only files it under <skipped>
        outcome = "skipped" if skipped is not None else "error" if errored is not None else "ran"
        if not classname:
            problems.append(f"module-level collection {outcome}: {name}")
            continue
        got[(classname, name)] = outcome
        if skipped is not None:
            nodeid = want.get((classname, name))
            reason = (skipped.get("message") or "").strip()
            if nodeid is None or nodeid not in allowed:
                problems.append(f"unlisted skip: {classname}::{name}: {reason[:120]}")

    missing = sorted(want[k] for k in want if k not in got)
    extra = sorted(f"{c}::{n}" for c, n in got if (c, n) not in want)
    problems += [f"in the roster, not run: {n}" for n in missing]
    problems += [f"ran, not in the roster: {n}" for n in extra]
    executed = sum(1 for o in got.values() if o != "skipped")
    skipped_n = sum(1 for o in got.values() if o == "skipped")
    lane = roster.stem
    print(f"{lane} lane: {len(got)} of {len(expected)} roster tests reported, {executed} executed, {skipped_n} skipped")
    if not executed:
        problems.append("nothing executed: a green lane that ran nothing verifies nothing")
    if problems:
        print(
            f"FAIL: the {lane} lane did not run its roster ({roster.relative_to(REPO) if roster.is_relative_to(REPO) else roster}):",
            file=sys.stderr,
        )
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        return 1
    return 0


def marked_files(marker: str, repo: Path = REPO) -> list[str]:
    """Test files (``test_*.py``, the configured pattern) that apply ``marker`` themselves (``mark.<marker>``, as a
    decorator or in ``pytestmark``). No conftest applies either lane's marker (tests/conftest.py only adds a SKIP to
    items already carrying ``requires_model_cache``), so collection can be scoped to these: a module-level skip
    ELSEWHERE (the console modules without the extra) is not the lane's business, and must not fail the fast suite
    on a machine without that extra (wire review)."""
    return sorted(
        str(p.relative_to(repo))
        for p in (repo / "tests").rglob("test_*.py")
        if f"mark.{marker}" in p.read_text(encoding="utf-8", errors="replace")
    )


def collect(lane: str, repo: Path = REPO) -> tuple[list[str], list[str]]:
    """(nodeids, problems) from ``pytest --collect-only -q -m <marker>`` over :func:`marked_files`: a module-level
    skip or error in one of THOSE files is a problem (its tests would vanish from both the roster and the lane). A
    test marked some other way (a conftest hook, ``getattr(pytest.mark, ...)``) is missed here and caught by the
    nightly as "ran, not in the roster"."""
    marker = LANES[lane]
    files = marked_files(marker, repo)
    if not files:
        return [], [f"no test file applies the {marker} marker"]
    r = subprocess.run(
        # -o addopts="": pyproject's `-v` would print a tree instead of one nodeid per line.
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "-rsE",
            "-o",
            "addopts=",
            "-m",
            marker,
            "-p",
            "no:cacheprovider",
            *files,
        ],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    lines = r.stdout.splitlines()
    nodeids = [ln.strip() for ln in lines if "::" in ln and not ln.startswith(("=", " ", "SKIPPED", "ERROR"))]
    # Under -q the summary line leaves the skip count out (executor review), so read the short summary -rsE prints:
    # one `SKIPPED [n] path: reason` / `ERROR path` line per module skipped or broken at collection.
    problems = [
        f"a {lane}-lane module skipped or errored at collection: {ln.strip()} (on a box without an extra one of them "
        "needs, install it; the nightly installs semantic/console/sign)"
        for ln in lines
        if ln.startswith(("SKIPPED [", "ERROR "))
    ]
    if r.returncode not in (0, 5):
        problems.append(f"pytest --collect-only exited {r.returncode}: {r.stderr[-300:]}")
    return nodeids, problems


USAGE = f"usage: check_lane_roster.py --lane {{{'|'.join(LANES)}}} (<junit.xml> | --generate)"


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 3 or argv[0] != "--lane" or argv[1] not in LANES:
        print(USAGE, file=sys.stderr)
        return 2
    lane, arg = argv[1], argv[2]
    roster = roster_path(lane)
    if arg == "--generate":
        nodeids, problems = collect(lane)
        if problems or not nodeids:
            print(
                "refusing to write the roster:",
                *(problems or [f"no {lane} tests collected"]),
                sep="\n  ",
                file=sys.stderr,
            )
            return 1
        _expected, allowed = load_roster(roster) if roster.exists() else ([], {})
        payload = {
            "expected": sorted(nodeids),
            "allowed_skips": {k: allowed[k] for k in sorted(allowed) if k in nodeids},
        }
        roster.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {len(nodeids)} {lane} tests to {roster.relative_to(REPO)}")
        return 0
    return check(Path(arg), roster)


if __name__ == "__main__":
    sys.exit(main())
